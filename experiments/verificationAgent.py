"""
NGO Claim Verification Agent (FastAPI + LangGraph)

Endpoint:
  POST /verify  (multipart/form-data)
    - content: str (required)          NGO claim text (may include links)
    - vault_address: str (required)    EVM address holding ETH
    - images: UploadFile[] (optional)  Bills/receipts/proof images

Response:
  Strict JSON with a single numeric USD value:
    { "recommended_amount_usd": 1234.56 }

Notes:
  - This agent provides a RECOMMENDATION only.
  - It does NOT perform any on-chain transfer.
"""

from __future__ import annotations

import base64
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import requests
from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field
from web3 import Web3
from eth_account import Account
# Opik integration removed - evaluation mode doesn't use internal tracing

load_dotenv()


RPC_URL = os.getenv("RPC_URL", "https://ethereum-sepolia-rpc.publicnode.com")
COINGECKO_ETH_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"


def _extract_urls(text: str) -> List[str]:
    if not text:
        return []
    urls = re.findall(r"(https?://[^\s)>\]]+)", text)
    # de-dupe preserve order
    out: List[str] = []
    seen = set()
    for u in urls:
        u = u.strip().rstrip(".,;")
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _extract_json_obj(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON object extractor for LLM outputs that may include ```json fences.
    Returns {} on failure.
    """
    if not text:
        return {}

    # Direct parse
    try:
        import json

        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Fenced ```json ... ```
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        try:
            import json

            obj = json.loads(m.group(1))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass

    # Heuristic: first '{' to last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            import json

            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))


class _ImageInput(TypedDict):
    filename: str
    content_type: str
    data_b64: str  # base64-encoded bytes


class VerifyState(TypedDict):
    content: str
    vault_address: str
    images: List[_ImageInput]
    verbose: bool
    debug: List[Dict[str, Any]]

    urls: List[str]
    url_texts: List[Dict[str, str]]
    links_summary: str

    vault_balance_eth: float
    eth_price_usd: float
    vault_balance_usd: float

    claim_amount_usd: Optional[float]
    claim_summary: str
    evidence_summary: str
    images_summary: str

    decision_summary: str

    recommended_amount_usd: float


def _dbg(state: VerifyState, event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Append a debug event to state. If verbose, also print it.
    """
    debug = list(state.get("debug", []))
    debug.append(event)
    if state.get("verbose"):
        # Keep console output readable
        print(f"[VERIFY DEBUG] {event}")
    return {"debug": debug}


class NGOClaimVerifierAgent:
    def __init__(
        self,
        *,
        search_model: str = "gpt-4o-mini",
        reasoning_model: str = "gpt-4o",
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment/.env")
        self.client = OpenAI(api_key=api_key)
        self.search_model = search_model
        self.reasoning_model = reasoning_model
        
        self.graph = self._build_graph()

    def _node_prepare(self, state: VerifyState) -> Dict[str, Any]:
        urls = _extract_urls(state.get("content", ""))[:3]
        out: Dict[str, Any] = {"urls": urls, "url_texts": [], "links_summary": "", "images_summary": "", "decision_summary": ""}
        out.update(_dbg(state, {"step": "prepare", "urls_found": len(urls), "urls": urls}))
        return out

    def _node_fetch_vault_balance(self, state: VerifyState) -> Dict[str, Any]:
        vault_address = (state.get("vault_address") or "").strip()
        if not vault_address:
            raise ValueError("vault_address is required")

        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        if not w3.is_connected():
            raise RuntimeError("Failed to connect to RPC provider")

        try:
            checksum = Web3.to_checksum_address(vault_address)
        except Exception as e:
            raise ValueError(f"Invalid vault_address: {vault_address}") from e

        balance_wei = w3.eth.get_balance(checksum)
        balance_eth = balance_wei / 1e18

        r = requests.get(COINGECKO_ETH_PRICE_URL, timeout=15)
        r.raise_for_status()
        eth_price_usd = float(r.json()["ethereum"]["usd"])

        vault_usd = balance_eth * eth_price_usd
        out: Dict[str, Any] = {
            "vault_balance_eth": float(balance_eth),
            "eth_price_usd": float(eth_price_usd),
            "vault_balance_usd": float(vault_usd),
        }
        out.update(
            _dbg(
                state,
                {
                    "step": "vault",
                    "vault_address": vault_address,
                    "vault_balance_eth": float(balance_eth),
                    "eth_price_usd": float(eth_price_usd),
                    "vault_balance_usd": float(vault_usd),
                },
            )
        )
        return out

    def _node_fetch_links(self, state: VerifyState) -> Dict[str, Any]:
        urls = state.get("urls", [])[:3]
        collected: List[Dict[str, str]] = []
        statuses: List[Dict[str, Any]] = []
        for url in urls:
            try:
                resp = requests.get(url, timeout=15, headers={"User-Agent": "HandzVerifier/1.0"})
                resp.raise_for_status()
                text = resp.text
                # keep it compact; LLM can work with partial context
                snippet = text[:5000]
                collected.append({"url": url, "text_snippet": snippet})
                statuses.append({"url": url, "status": "ok", "http_status": resp.status_code})
            except Exception as e:
                collected.append({"url": url, "text_snippet": f"[fetch failed] {e}"})
                statuses.append({"url": url, "status": "failed", "error": str(e)})
        out: Dict[str, Any] = {"url_texts": collected}
        out.update(_dbg(state, {"step": "links", "urls_attempted": len(urls), "results": statuses}))
        return out

    def _node_summarize_links(self, state: VerifyState) -> Dict[str, Any]:
        """
        Summarize what the URLs say about the NGO's actions/impact (if anything).
        This is an explanation artifact (not hidden reasoning).
        """
        url_texts = state.get("url_texts", [])
        if not url_texts:
            out: Dict[str, Any] = {"links_summary": ""}
            out.update(_dbg(state, {"step": "links_summary", "note": "no urls"}))
            return out

        prompt = f"""Summarize what the following URL snippets say that is relevant to verifying an NGO claim.

URL SNIPPETS:
{url_texts}

OUTPUT:
- 3-8 bullet points, plain text
- If the snippets do NOT mention the NGO/expenses/impact, say so explicitly
"""
        # Get model and parameters override if specified
        model = getattr(self, '_llm_params_override', {}).get('model', self.reasoning_model)
        
        # Build LLM call parameters
        llm_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        
        # Override with variant_config if present
        override_params = getattr(self, '_llm_params_override', {})
        if 'temperature' in override_params:
            llm_params["temperature"] = override_params['temperature']
        for param in ['max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if param in override_params:
                llm_params[param] = override_params[param]
        
        resp = self.client.chat.completions.create(**llm_params)
        links_summary = (resp.choices[0].message.content or "").strip()
        out = {"links_summary": links_summary}
        out.update(_dbg(state, {"step": "links_summary", "summary": links_summary}))
        return out

    def _node_analyze_claim(self, state: VerifyState) -> Dict[str, Any]:
        """
        Extract requested claim amount (if any) + summarize asserted impact and evidence.
        """
        content = state.get("content", "")
        url_texts = state.get("url_texts", [])
        vault_usd = state.get("vault_balance_usd", 0.0)

        prompt = f"""You are verifying an NGO reimbursement claim.

VAULT FUNDS AVAILABLE (USD): {vault_usd}

CLAIM TEXT:
{content}

LINK CONTENT SNIPPETS (may be HTML):
{url_texts}

TASK:
1) Extract the requested amount in USD if stated (number only). If not stated, set to null.
2) Summarize what the NGO claims they did (impact + actions) in 3-6 bullet points.
3) Summarize what evidence is present/credible (receipts mentioned, links, etc.) in 3-6 bullet points.

OUTPUT STRICT JSON ONLY:
{{
  "claim_amount_usd": number|null,
  "claim_summary": "string",
  "evidence_summary": "string"
}}"""

        # Get model and parameters override if specified
        model = getattr(self, '_llm_params_override', {}).get('model', self.reasoning_model)
        
        # Build LLM call parameters
        llm_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        
        # Override with variant_config if present
        override_params = getattr(self, '_llm_params_override', {})
        if 'temperature' in override_params:
            llm_params["temperature"] = override_params['temperature']
        for param in ['max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if param in override_params:
                llm_params[param] = override_params[param]
        
        resp = self.client.chat.completions.create(**llm_params)
        raw = (resp.choices[0].message.content or "").strip()
        # best-effort parse
        try:
            import json
            data = json.loads(raw)
        except Exception:
            data = {}

        claim_amount = data.get("claim_amount_usd", None)
        claim_amount_f = None if claim_amount is None else _safe_float(claim_amount, default=0.0)

        out: Dict[str, Any] = {
            "claim_amount_usd": claim_amount_f,
            "claim_summary": (data.get("claim_summary") or "").strip(),
            "evidence_summary": (data.get("evidence_summary") or "").strip(),
        }
        out.update(
            _dbg(
                state,
                {
                    "step": "claim",
                    "claim_amount_usd": claim_amount_f,
                    "claim_summary": out["claim_summary"],
                    "evidence_summary": out["evidence_summary"],
                },
            )
        )
        return out

    def _node_analyze_images(self, state: VerifyState) -> Dict[str, Any]:
        """
        Use a vision-capable model to summarize the submitted proof images.
        """
        images = state.get("images", [])[:4]
        if not images:
            out: Dict[str, Any] = {"evidence_summary": state.get("evidence_summary", "")}
            out.update(_dbg(state, {"step": "images", "images_received": 0, "note": "skipped"}))
            return out

        # Build multimodal message: text + image data URLs
        parts: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Review these proof images (receipts/bills/invoices). "
                    "Summarize what they show and whether they support the claim. "
                    "Do not output any currency symbols. Output 3-6 bullet points as plain text."
                ),
            }
        ]

        for img in images:
            b64 = img["data_b64"]
            ctype = img.get("content_type") or "image/jpeg"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{ctype};base64,{b64}"},
                }
            )

        # Get model and parameters override if specified
        model = getattr(self, '_llm_params_override', {}).get('model', self.search_model)
        
        # Build LLM call parameters
        llm_params = {
            "model": model,
            "messages": [{"role": "user", "content": parts}],
            "temperature": 0.2
        }
        
        # Override with variant_config if present
        override_params = getattr(self, '_llm_params_override', {})
        if 'temperature' in override_params:
            llm_params["temperature"] = override_params['temperature']
        for param in ['max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if param in override_params:
                llm_params[param] = override_params[param]
        
        resp = self.client.chat.completions.create(**llm_params)
        img_summary = (resp.choices[0].message.content or "").strip()

        combined = (state.get("evidence_summary", "") + "\n" + img_summary).strip()
        out: Dict[str, Any] = {"evidence_summary": combined, "images_summary": img_summary}
        out.update(
            _dbg(
                state,
                {
                    "step": "images",
                    "images_received": len(images),
                    "image_filenames": [i.get("filename") for i in images],
                    "images_summary": img_summary,
                    "evidence_summary": combined,
                },
            )
        )
        return out

    def _node_recommend_amount(self, state: VerifyState) -> Dict[str, Any]:
        vault_usd = float(state.get("vault_balance_usd", 0.0))
        claim_amount = state.get("claim_amount_usd", None)
        claim_summary = state.get("claim_summary", "")
        evidence_summary = state.get("evidence_summary", "")

        prompt = f"""You recommend how much USD to release from a disaster relief vault to an NGO.

VAULT_BALANCE_USD: {vault_usd}
CLAIM_AMOUNT_USD: {claim_amount if claim_amount is not None else "null"}

CLAIM_SUMMARY:
{claim_summary}

EVIDENCE_SUMMARY:
{evidence_summary}

RULES:
- Recommend a single number in USD.
- The recommendation MUST be <= VAULT_BALANCE_USD.
- If evidence is weak or unclear, recommend a conservative amount.
- If claim amount is provided, the recommendation MUST be <= CLAIM_AMOUNT_USD.
- Output STRICT JSON ONLY with a single numeric field and NO currency symbols, NO commas:
{{ "recommended_amount_usd": 1234.56 }}
"""

        # Get model and parameters override if specified
        model = getattr(self, '_llm_params_override', {}).get('model', self.reasoning_model)
        
        # Build LLM call parameters
        llm_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        
        # Override with variant_config if present
        override_params = getattr(self, '_llm_params_override', {})
        if 'temperature' in override_params:
            llm_params["temperature"] = override_params['temperature']
        for param in ['max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if param in override_params:
                llm_params[param] = override_params[param]
        
        resp = self.client.chat.completions.create(**llm_params)
        
        raw = (resp.choices[0].message.content or "").strip()

        try:
            import json
            data = json.loads(raw)
            rec = _safe_float(data.get("recommended_amount_usd", 0.0), default=0.0)
        except Exception:
            rec = 0.0

        # enforce hard constraints
        rec_before_clamp = float(rec)
        rec = _clamp(rec, 0.0, vault_usd)
        rec_after_vault_clamp = float(rec)
        if claim_amount is not None:
            rec = min(rec, float(claim_amount))
        rec_after_claim_cap = float(rec)

        # keep stable numeric representation
        rec = round(float(rec), 2)
        out: Dict[str, Any] = {"recommended_amount_usd": rec}
        out.update(
            _dbg(
                state,
                {
                    "step": "recommend",
                    "vault_balance_usd": vault_usd,
                    "claim_amount_usd": claim_amount,
                    "model_output_usd": rec_before_clamp,
                    "after_vault_clamp_usd": rec_after_vault_clamp,
                    "after_claim_cap_usd": rec_after_claim_cap,
                    "final_recommended_amount_usd": rec,
                },
            )
        )
        
        return out

    def _node_explain_decision(self, state: VerifyState) -> Dict[str, Any]:
        """
        Provide a human-readable explanation of the recommendation (no chain-of-thought).
        """
        vault_usd = float(state.get("vault_balance_usd", 0.0))
        claim_amount = state.get("claim_amount_usd", None)
        recommended = float(state.get("recommended_amount_usd", 0.0))
        claim_summary = state.get("claim_summary", "")
        links_summary = state.get("links_summary", "")
        images_summary = state.get("images_summary", "")

        prompt = f"""Write a short justification for why the recommended payout amount was chosen.
This is NOT a chain-of-thought. Summarize the decision using only the provided facts.

FACTS:
- vault_balance_usd: {vault_usd}
- claim_amount_usd: {claim_amount if claim_amount is not None else "null"}
- recommended_amount_usd: {recommended}

claim_summary:
{claim_summary}

links_summary:
{links_summary}

images_summary:
{images_summary}

OUTPUT:
- 4-10 bullet points
- No question marks
- No currency symbols
"""

        # Get model and parameters override if specified
        model = getattr(self, '_llm_params_override', {}).get('model', self.reasoning_model)
        
        # Build LLM call parameters
        llm_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        
        # Override with variant_config if present
        override_params = getattr(self, '_llm_params_override', {})
        if 'temperature' in override_params:
            llm_params["temperature"] = override_params['temperature']
        for param in ['max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if param in override_params:
                llm_params[param] = override_params[param]
        
        resp = self.client.chat.completions.create(**llm_params)
        decision_summary = (resp.choices[0].message.content or "").strip()
        out: Dict[str, Any] = {"decision_summary": decision_summary}
        out.update(_dbg(state, {"step": "decision_summary", "decision_summary": decision_summary}))
        return out

    def _build_graph(self):
        g = StateGraph(VerifyState)
        g.add_node("prepare", self._node_prepare)
        g.add_node("vault", self._node_fetch_vault_balance)
        g.add_node("links", self._node_fetch_links)
        g.add_node("links_summary", self._node_summarize_links)
        g.add_node("claim", self._node_analyze_claim)
        g.add_node("images", self._node_analyze_images)
        g.add_node("recommend", self._node_recommend_amount)
        g.add_node("explain", self._node_explain_decision)

        g.set_entry_point("prepare")
        g.add_edge("prepare", "vault")
        g.add_edge("vault", "links")
        g.add_edge("links", "links_summary")
        g.add_edge("links_summary", "claim")
        g.add_edge("claim", "images")
        g.add_edge("images", "recommend")
        g.add_edge("recommend", "explain")
        g.add_edge("explain", END)
        return g.compile()

    def run(self, *, content: str, vault_address: str, images: List[_ImageInput], verbose: bool = True) -> VerifyState:
        state: VerifyState = {
            "content": content,
            "vault_address": vault_address,
            "images": images,
            "verbose": bool(verbose),
            "debug": [],
            "urls": [],
            "url_texts": [],
            "links_summary": "",
            "vault_balance_eth": 0.0,
            "eth_price_usd": 0.0,
            "vault_balance_usd": 0.0,
            "claim_amount_usd": None,
            "claim_summary": "",
            "evidence_summary": "",
            "images_summary": "",
            "decision_summary": "",
            "recommended_amount_usd": 0.0,
        }
        return self.graph.invoke(state)


app = FastAPI(title="NGO Claim Verification Agent", version="1.0.0")
agent = NGOClaimVerifierAgent()


@app.post("/verify")
async def verify(
    content: str = Form(...),
    vault_address: str = Form(...),
    images: List[UploadFile] = File(default=[]),
):
    image_inputs: List[_ImageInput] = []
    for f in images[:6]:
        b = await f.read()
        image_inputs.append(
            {
                "filename": f.filename or "image",
                "content_type": f.content_type or "application/octet-stream",
                "data_b64": base64.b64encode(b).decode("ascii"),
            }
        )

    # Always run in verbose mode and always return debug trace
    result_state = agent.run(content=content, vault_address=vault_address, images=image_inputs, verbose=True)

    response: Dict[str, Any] = {
        "recommended_amount_usd": float(result_state["recommended_amount_usd"]),
        "decision_summary": result_state.get("decision_summary", ""),
        "debug": result_state.get("debug", []),
    }
    return JSONResponse(content=response)


class VoteRequest(BaseModel):
    claim_submitted: str = Field(..., description="The NGO claim / what AI previously evaluated")
    relief_fund: float = Field(..., ge=0, description="Current relief fund amount (USD)")
    disaster_details: str = Field(..., description="Context about the disaster and overall relief situation")
    vote: str = Field(..., description='Majority verdict: "higher" or "lower"')

class ReleaseRequest(BaseModel):
    wallet_address: str = Field(..., description="Recipient NGO wallet address (EVM)")
    amount: float = Field(..., gt=0, description="Amount to release (USD)")
    vault_address: str = Field(..., description="Vault contract address (EVM)")
    vote: str = Field(..., description='Consensus verdict, e.g. "release"')


_VAULT_ABI = [
    {
        "inputs": [{"internalType": "address payable", "name": "_to", "type": "address"}, {"internalType": "uint256", "name": "_amount", "type": "uint256"}],
        "name": "withdraw",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "creator",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
]


@app.post("/vote")
async def vote(payload: Dict[str, Any] = Body(...)):
    """
    Raw JSON endpoint with two modes:

    1) Adjustment mode (per-org payout tweak):
      { claim_submitted, relief_fund, disaster_details, vote: "higher"|"lower" }

    2) Release mode (consensus to release funds from a vault to an NGO wallet):
      { wallet_address, amount, vault_address, vote: "release" }
    """
    # Detect release-mode request by its keys
    is_release_mode = all(k in payload for k in ("wallet_address", "amount", "vault_address", "vote"))
    
    if is_release_mode:
        req = ReleaseRequest.model_validate(payload)
        vote_dir = (req.vote or "").strip().lower()
        if vote_dir not in {"release", "approve", "approved", "yes"}:
            error_response = JSONResponse(
                status_code=400,
                content={"error": 'For release-mode, vote must indicate consensus (e.g. "release")'},
            )
            return error_response

        w3 = Web3(Web3.HTTPProvider(RPC_URL))
        if not w3.is_connected():
            error_response = JSONResponse(status_code=500, content={"error": "Failed to connect to RPC provider"})
            return error_response

            # Validate addresses
            try:
                vault = Web3.to_checksum_address(req.vault_address)
                to_wallet = Web3.to_checksum_address(req.wallet_address)
            except Exception:
                error_response = JSONResponse(status_code=400, content={"error": "Invalid wallet_address or vault_address"})
                return error_response

            # Fetch ETH price
            r = requests.get(COINGECKO_ETH_PRICE_URL, timeout=15)
            r.raise_for_status()
            eth_price_usd = float(r.json()["ethereum"]["usd"])

            # Vault balance
            vault_balance_wei = w3.eth.get_balance(vault)
            vault_balance_eth = vault_balance_wei / 1e18
            vault_balance_usd = vault_balance_eth * eth_price_usd

            # Convert amount USD -> ETH -> Wei
            amount_usd = float(req.amount)
            amount_eth = amount_usd / eth_price_usd if eth_price_usd > 0 else 0.0
            amount_wei = int(amount_eth * 1e18)

            # Require sufficient balance
            if amount_wei <= 0:
                error_response = JSONResponse(status_code=400, content={"error": "Computed amount_wei is 0; check amount and price feed"})
                return error_response
            if amount_wei > vault_balance_wei:
                error_response = JSONResponse(
                    status_code=400,
                    content={
                        "error": "Insufficient vault balance",
                        "vault_balance_usd": float(round(vault_balance_usd, 6)),
                        "vault_balance_eth": float(round(vault_balance_eth, 18)),
                        "requested_amount_usd": amount_usd,
                        "requested_amount_eth": float(round(amount_eth, 18)),
                    },
                )
                return error_response

            private_key = (os.getenv("PRIVATE_KEY") or "").strip()
            if not private_key:
                error_response = JSONResponse(status_code=500, content={"error": "PRIVATE_KEY not set in .env"})
                return error_response
            pk = private_key[2:] if private_key.startswith("0x") else private_key
            acct = Account.from_key(pk)

            vault_contract = w3.eth.contract(address=vault, abi=_VAULT_ABI)
            try:
                creator = vault_contract.functions.creator().call()
            except Exception as e:
                error_response = JSONResponse(status_code=500, content={"error": f"Failed to read vault creator: {e}"})
                return error_response

            if Web3.to_checksum_address(creator) != Web3.to_checksum_address(acct.address):
                error_response = JSONResponse(
                    status_code=403,
                    content={
                        "error": "This server is not the creator of the vault, cannot withdraw",
                        "vault_creator": creator,
                        "server_address": acct.address,
                    },
                )
                return error_response

            # Build & send withdraw transaction
            try:
                nonce = w3.eth.get_transaction_count(acct.address)
                gas_estimate = vault_contract.functions.withdraw(to_wallet, amount_wei).estimate_gas({"from": acct.address})
                gas_limit = int(gas_estimate * 1.2)

                base_fee = w3.eth.gas_price
                max_priority_fee = w3.to_wei(2, "gwei")
                max_fee = base_fee + max_priority_fee

                tx = vault_contract.functions.withdraw(to_wallet, amount_wei).build_transaction(
                    {
                        "from": acct.address,
                        "nonce": nonce,
                        "gas": gas_limit,
                        "maxFeePerGas": max_fee,
                        "maxPriorityFeePerGas": max_priority_fee,
                        "chainId": w3.eth.chain_id,
                    }
                )

                signed = acct.sign_transaction(tx)
                tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
                receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

                response_data = {
                    "status": "sent",
                    "tx_hash": tx_hash.hex(),
                    "block_number": int(receipt["blockNumber"]),
                    "to_wallet": to_wallet,
                    "vault_address": vault,
                    "amount_usd": float(round(amount_usd, 2)),
                    "amount_eth": float(round(amount_eth, 18)),
                    "vault_balance_usd": float(round(vault_balance_usd, 6)),
                    "vault_balance_eth": float(round(vault_balance_eth, 18)),
                }
                
                return JSONResponse(content=response_data)

            except Exception as e:
                error_response = JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})
                
                return error_response

        # Otherwise: adjustment mode
        req = VoteRequest.model_validate(payload)
        vote_dir = (req.vote or "").strip().lower()
        if vote_dir not in {"higher", "lower"}:
            error_response = JSONResponse(status_code=400, content={"error": 'vote must be either "higher" or "lower"'})
            return error_response

        client = agent.client

        prompt = f"""You are adjusting a SINGLE NGO payout amount based on a majority vote from people.

INPUTS:
- vote: {vote_dir}
- relief_fund_usd (this is the current payout amount to the NGO): {req.relief_fund}
- claim_submitted:
{req.claim_submitted}

- disaster_details:
{req.disaster_details}

TASK:
Recommend how much to adjust the relief fund in USD (delta_usd) based on the vote, and briefly explain why.

RULES:
- If vote is "higher": delta_usd MUST be positive.
- If vote is "lower": delta_usd MUST be negative.
- Keep the magnitude reasonable relative to relief_fund_usd:
  - Typical range: 5% to 25% of relief_fund_usd.
  - Never exceed 30% of relief_fund_usd in magnitude.
- Output STRICT JSON ONLY:
{{
  "delta_usd": -123.45,
  "reasoning_summary": "3-8 bullet points, plain text, no question marks"
}}
"""

        # Get model and parameters override if specified
        model = getattr(agent, '_llm_params_override', {}).get('model', agent.reasoning_model)
        
        # Build LLM call parameters
        llm_params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        
        # Override with variant_config if present
        override_params = getattr(agent, '_llm_params_override', {})
        if 'temperature' in override_params:
            llm_params["temperature"] = override_params['temperature']
        for param in ['max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if param in override_params:
                llm_params[param] = override_params[param]
        
        resp = client.chat.completions.create(**llm_params)

        raw = (resp.choices[0].message.content or "").strip()
        parsed = _extract_json_obj(raw)
        delta = _safe_float(parsed.get("delta_usd", 0.0), default=0.0)
        reasoning_val = parsed.get("reasoning_summary", "")
        if isinstance(reasoning_val, list):
            reasoning_summary = "\n".join([str(x) for x in reasoning_val if str(x).strip()]).strip()
        else:
            reasoning_summary = str(reasoning_val or "").strip()

        # Enforce constraints deterministically
        cap = abs(req.relief_fund) * 0.30
        delta_before_sign = float(delta)
        if vote_dir == "higher":
            delta = abs(delta)
        else:
            delta = -abs(delta)
        if cap > 0:
            delta = _clamp(delta, -cap, cap)
        delta_after_clamp = float(delta)

        # Avoid "-0.0" in output
        if abs(delta) < 0.005:
            delta = 0.0

        updated = max(0.0, float(req.relief_fund) + float(delta))

        response_data = {
            "updated_relief_fund": float(round(updated, 2)),
            "delta": float(round(delta, 2)),
            "reasoning_summary": reasoning_summary,
            "debug": {
                "vote": vote_dir,
                "relief_fund_input": float(req.relief_fund),
                "cap_30pct": float(round(cap, 6)),
                "model_raw": raw,
                "parsed_json": parsed,
                "parsed_delta": float(delta_before_sign),
                "delta_after_sign_and_cap": float(round(delta_after_clamp, 6)),
                "updated_relief_fund": float(round(updated, 6)),
            },
        }
        
        return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VERIFY_API_PORT", "8080")))

