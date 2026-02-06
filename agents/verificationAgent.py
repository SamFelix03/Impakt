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
from opik_integration import (
    OpikTracer, OpikSpan, trace_agent_execution,
    llm_judge, OpikAgentOptimizer, OPIK_AVAILABLE, OPIK_PROJECT
)
from opik_langgraph_helper import update_node_span_with_io, update_trace_with_feedback

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
        
        # Initialize Opik components
        self.opik_optimizer = OpikAgentOptimizer("claim_verification") if (OPIK_AVAILABLE and OpikAgentOptimizer) else None
        self.evaluation_results = []
        
        self.graph = self._build_graph()
        
        # Opik tracer will be created after graph compilation (per official Opik docs)
        # Store reference for later initialization
        self._opik_tracer_class = None
        print(f"[OPIK DEBUG] Initializing Opik tracer in __init__: OPIK_AVAILABLE={OPIK_AVAILABLE}")
        if OPIK_AVAILABLE:
            try:
                from opik_integration import LangGraphOpikTracer, OPIK_PROJECT
                print(f"[OPIK DEBUG] Imported LangGraphOpikTracer: {LangGraphOpikTracer}")
                print(f"[OPIK DEBUG] Imported OPIK_PROJECT: {OPIK_PROJECT}")
                if LangGraphOpikTracer is None:
                    print(f"[INIT] [WARNING] LangGraphOpikTracer not available")
                else:
                    self._opik_tracer_class = LangGraphOpikTracer
                    print(f"[INIT] [OK] Opik tracer class ready (will initialize after graph compilation)")
                    print(f"[OPIK DEBUG] Tracer class stored: {self._opik_tracer_class}")
            except Exception as e:
                print(f"[INIT] [WARNING] Failed to load Opik tracer class: {e}")
                import traceback
                traceback.print_exc()
                self._opik_tracer_class = None
        else:
            print(f"[OPIK DEBUG] OPIK_AVAILABLE is False, skipping tracer initialization")
        self.opik_tracer = None  # Will be created in run() after graph is compiled
        print(f"[OPIK DEBUG] __init__ complete: _opik_tracer_class={self._opik_tracer_class}, opik_tracer={self.opik_tracer}")

    def _node_prepare(self, state: VerifyState) -> Dict[str, Any]:
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("PREPARE")
        urls = _extract_urls(state.get("content", ""))[:3]
        out: Dict[str, Any] = {"urls": urls, "url_texts": [], "links_summary": "", "images_summary": "", "decision_summary": ""}
        out.update(_dbg(state, {"step": "prepare", "urls_found": len(urls), "urls": urls}))
        return out

    def _node_fetch_vault_balance(self, state: VerifyState) -> Dict[str, Any]:
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("FETCH_VAULT_BALANCE")
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
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("FETCH_LINKS")
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
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("SUMMARIZE_LINKS")
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
        resp = self.client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        links_summary = (resp.choices[0].message.content or "").strip()
        out = {"links_summary": links_summary}
        out.update(_dbg(state, {"step": "links_summary", "summary": links_summary}))
        return out

    def _node_analyze_claim(self, state: VerifyState) -> Dict[str, Any]:
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("ANALYZE_CLAIM")
        """
        Extract requested claim amount (if any) + summarize asserted impact and evidence.
        """
        content = state.get("content", "")
        url_texts = state.get("url_texts", [])
        vault_usd = state.get("vault_balance_usd", 0.0)

        prompt = f"""You are verifying an NGO reimbursement claim. The entire request comes from the CLAIM TEXT below—derive all amounts and summaries from it.

VAULT FUNDS AVAILABLE (USD): {vault_usd}

CLAIM TEXT:
{content}

LINK CONTENT SNIPPETS (may be HTML):
{url_texts}

TASK:
1) Extract the requested amount in USD (claim_amount_usd) from the CLAIM TEXT:
   - If the claim states a specific dollar amount (e.g. "we need $500", "requesting 200 USD"), use that number.
   - If the claim states a percentage of an amount (e.g. "20% of the amount", "20% of 9 dollars", "we request 20% of donations"), compute it: apply the percentage to the base amount mentioned (e.g. "9 dollars" and "20%" → 9 * 0.20 = 1.80). If they say "the amount" or "donations" and a dollar figure appears in the text, use that as the base; if the only available base is the vault, use VAULT FUNDS AVAILABLE.
   - Output the final number as claim_amount_usd (e.g. 1.8). Only set to null if there is no mention of any requested amount or percentage.
2) Summarize what the NGO claims they did (impact + actions) in 3-6 bullet points (claim_summary). Use the CLAIM TEXT; do not leave empty.
3) Summarize what evidence is present/credible (links, receipts, proof) in 3-6 bullet points (evidence_summary). Use the CLAIM TEXT and links; do not leave empty.

OUTPUT STRICT JSON ONLY:
{{
  "claim_amount_usd": number|null,
  "claim_summary": "string",
  "evidence_summary": "string"
}}"""

        resp = self.client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _extract_json_obj(raw)

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
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("ANALYZE_IMAGES")
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

        resp = self.client.chat.completions.create(
            model=self.search_model,  # smaller/cheaper vision model
            messages=[{"role": "user", "content": parts}],
            temperature=0.2,
        )
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
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("RECOMMEND_AMOUNT")
        # Prepare input data
        input_data = {
            "vault_balance_usd": state.get("vault_balance_usd", 0.0),
            "claim_amount_usd": state.get("claim_amount_usd"),
            "claim_summary_length": len(state.get("claim_summary", "")),
            "evidence_summary_length": len(state.get("evidence_summary", ""))
        }
        
        # Update LangGraph node span with input data
        update_node_span_with_io(input_data=input_data, metadata={"node": "recommend_amount"})
        
        # LangGraph already creates a trace - we don't need a custom OpikTracer here
        # Just execute the logic directly within the LangGraph trace context
        vault_usd = float(state.get("vault_balance_usd", 0.0))
        claim_amount = state.get("claim_amount_usd", None)
        claim_summary = state.get("claim_summary", "")
        evidence_summary = state.get("evidence_summary", "")
        links_summary = state.get("links_summary", "")

        # All amounts are USD. Spell out the check so the model cannot "preserve" by outputting 0.
        vault_covers_claim = claim_amount is not None and vault_usd >= float(claim_amount)
        prompt = f"""You recommend how much to release from a disaster relief vault to an NGO. All amounts are in USD (dollars).

VAULT_BALANCE_USD: {vault_usd} USD
CLAIM_AMOUNT_USD: {claim_amount if claim_amount is not None else "null (unspecified)"} USD

{"The vault has enough to cover the full claim (" + str(claim_amount) + " <= " + str(round(vault_usd, 2)) + "). You MUST recommend a positive amount in USD (e.g. " + str(claim_amount) + " or the full claim). Outputting 0 is wrong." if vault_covers_claim else "The vault has " + str(round(vault_usd, 2)) + " USD. You MUST recommend a positive amount in USD (never 0)."}

CLAIM_SUMMARY:
{claim_summary or "(none)"}

EVIDENCE_SUMMARY:
{evidence_summary or "(none)"}

LINKS_SUMMARY:
{links_summary or "(none)"}

RULES (all amounts in USD):
- Output ONLY valid JSON: {{ "recommended_amount_usd": 1234.56 }}. No other text. No currency symbols or commas in the number.
- recommended_amount_usd MUST be a positive number whenever the vault balance is positive. Never output 0.
- recommended_amount_usd <= VAULT_BALANCE_USD. If CLAIM_AMOUNT_USD is given, recommended_amount_usd <= CLAIM_AMOUNT_USD.
- Do NOT recommend 0 to "preserve" or "reserve" funds. Your task is to set the release amount in USD; it must be positive when the vault has funds.
"""

        # Add input/output tracking
        with OpikSpan("recommend_amount_llm",
                     input_data={
                         "vault_balance": vault_usd,
                         "claim_amount": claim_amount,
                         "prompt_preview": prompt[:150]
                     },
                     output_data={}):
            resp = self.client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            # Update span with output
            if OPIK_AVAILABLE:
                try:
                    from opik.opik_context import update_current_span
                    raw = (resp.choices[0].message.content or "").strip()
                    update_current_span(metadata={
                        "output": {
                            "raw_response": raw[:200],
                            "model": self.reasoning_model
                        }
                    })
                except:
                    pass
        raw = (resp.choices[0].message.content or "").strip()

        data = _extract_json_obj(raw)
        rec = _safe_float(data.get("recommended_amount_usd", 0.0), default=0.0)

        # If model returned 0 despite vault having funds, retry once with a minimal prompt (agent still decides the amount)
        if rec <= 0 and vault_usd > 0:
            retry_prompt = f"""All amounts are in USD. Vault balance: {vault_usd} USD. Claim amount: {claim_amount if claim_amount is not None else "unspecified"} USD.
Output ONLY this JSON with a positive number: {{ "recommended_amount_usd": N }} where N is the amount in USD to release (positive, never 0)."""
            retry_resp = self.client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": retry_prompt}],
                temperature=0.0,
            )
            retry_raw = (retry_resp.choices[0].message.content or "").strip()
            retry_data = _extract_json_obj(retry_raw)
            rec = _safe_float(retry_data.get("recommended_amount_usd", 0.0), default=0.0)

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
        
        # Prepare output data
        output_data = {
            "recommended_amount_usd": rec,
            "vault_balance_usd": vault_usd,
            "claim_amount_usd": claim_amount,
            "within_vault_balance": rec <= vault_usd,
            "within_claim_amount": rec <= claim_amount if claim_amount else True
        }
        
        # Evaluate claim verification quality
        evaluation = None
        if llm_judge:
            with OpikSpan("evaluate_claim_verification",
                         input_data={"recommended_amount": rec, "vault_balance": vault_usd},
                         output_data={}):
                evaluation = llm_judge.evaluate_claim_verification(
                    claim_text=state.get("content", ""),
                    recommended_amount=rec,
                    vault_balance=vault_usd,
                    evidence_summary=evidence_summary,
                    send_to_opik=True
                )
                print(f"\n[OPIK EVALUATION] Claim Verification Score: {evaluation.get('overall_score', 0):.2f}/10")
                
                # Explicitly send feedback scores to the current trace (LangGraph trace)
                if OPIK_AVAILABLE and evaluation.get('overall_score'):
                    try:
                        from opik.opik_context import update_current_trace
                        feedback_scores_list = [{
                            "name": "claim_verification_quality",
                            "value": float(evaluation.get('overall_score', 0)),
                            "reason": evaluation.get('reasoning', 'Claim verification quality evaluation')
                        }]
                        # Add individual criterion scores
                        for key, value in evaluation.items():
                            if key not in ['overall_score', 'reasoning'] and isinstance(value, (int, float)):
                                feedback_scores_list.append({
                                    "name": f"claim_verification_quality_{key}",
                                    "value": float(value),
                                    "reason": f"{key.replace('_', ' ').title()} criterion score"
                                })
                        
                        update_current_trace(feedback_scores=feedback_scores_list)
                        print(f"[OPIK] Sent {len(feedback_scores_list)} feedback scores to trace")
                    except Exception as e:
                        print(f"[OPIK] Failed to send feedback scores to trace: {e}")
                        import traceback
                        traceback.print_exc()
                
                self.evaluation_results.append({
                    "step": "claim_verification",
                    "recommended_amount": rec,
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat()
                })
                out["evaluation"] = evaluation
                output_data["evaluation_score"] = evaluation.get('overall_score', 0)
        
        # Update LangGraph node span with output data
        update_node_span_with_io(
            output_data={
                **output_data,
                "recommended_amount_usd": rec,
                "recommendation_valid": rec <= vault_usd
            },
            metadata={"node": "recommend_amount"}
        )
        
        return out

    def _node_explain_decision(self, state: VerifyState) -> Dict[str, Any]:
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("EXPLAIN_DECISION")
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
This is NOT a chain-of-thought. Summarize the decision using only the provided facts. All amounts are in USD (dollars)—always refer to them as "X USD" or "X dollars", never as "units".

FACTS (all in USD):
- vault_balance_usd: {vault_usd} USD
- claim_amount_usd: {claim_amount if claim_amount is not None else "null"} USD
- recommended_amount_usd: {recommended} USD

claim_summary:
{claim_summary}

links_summary:
{links_summary}

images_summary:
{images_summary}

OUTPUT:
- 4-10 bullet points. Always say amounts in USD (e.g. "8 USD", "1.8 dollars"). Never use "units".
- No question marks.
"""

        resp = self.client.chat.completions.create(
            model=self.reasoning_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
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
        # Set project name before trace creation
        from opik_integration import OPIK_AVAILABLE, OPIK_PROJECT
        if OPIK_AVAILABLE:
            import os
            os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT
        
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
        # Create Opik tracer AFTER graph compilation (per official Opik docs)
        print(f"[OPIK DEBUG] Creating tracer: _opik_tracer_class={self._opik_tracer_class}, opik_tracer={self.opik_tracer}")
        if self._opik_tracer_class and not self.opik_tracer:
            try:
                from opik_integration import OPIK_PROJECT, OPIK_AVAILABLE
                print(f"[OPIK DEBUG] OPIK_AVAILABLE={OPIK_AVAILABLE}, OPIK_PROJECT={OPIK_PROJECT}")
                
                # Get compiled graph with xray=True (per official example)
                print(f"[OPIK DEBUG] Getting graph with xray=True...")
                graph = self.graph.get_graph(xray=True)
                print(f"[OPIK DEBUG] Graph obtained: {type(graph)}")
                
                print(f"[OPIK DEBUG] Creating OpikTracer instance...")
                self.opik_tracer = self._opik_tracer_class(graph=graph)
                print(f"[OPIK DEBUG] Tracer created: {type(self.opik_tracer)}")
                print(f"[OPIK] [OK] Opik tracer created for project: {OPIK_PROJECT}")
            except Exception as e:
                print(f"[OPIK] [ERROR] Failed to create Opik tracer: {e}")
                import traceback
                traceback.print_exc()
                self.opik_tracer = None
        
        # Use Opik tracer callback (official approach)
        print(f"[OPIK DEBUG] About to invoke graph: opik_tracer={self.opik_tracer}")
        if self.opik_tracer:
            print(f"[OPIK DEBUG] Invoking graph WITH Opik tracer callback")
            print(f"[OPIK DEBUG] Tracer type: {type(self.opik_tracer)}")
            print(f"[OPIK DEBUG] Config callbacks: {[type(cb) for cb in [self.opik_tracer]]}")
            
            # Invoke graph with tracer
            result = self.graph.invoke(
                state,
                config={"callbacks": [self.opik_tracer]}
            )
            
            # Extract and log trace ID from tracer
            trace_id = None
            try:
                # Method 1: Check tracer's created_traces (most reliable)
                if hasattr(self.opik_tracer, 'created_traces') and self.opik_tracer.created_traces:
                    trace_ids = list(self.opik_tracer.created_traces)
                    if trace_ids:
                        trace_id = trace_ids[-1]  # Get the most recent trace ID
                        print(f"[OPIK] [TRACE ID] {trace_id}")
                        print(f"[OPIK] [TRACE URL] https://app.opik.com/impakt/projects/{OPIK_PROJECT}/traces/{trace_id}")
                
                # Method 2: Check _created_traces internal map
                if not trace_id and hasattr(self.opik_tracer, '_created_traces'):
                    created_traces = self.opik_tracer._created_traces
                    if created_traces:
                        trace_ids = list(created_traces.keys())
                        if trace_ids:
                            trace_id = trace_ids[-1]
                            print(f"[OPIK] [TRACE ID] {trace_id}")
                            print(f"[OPIK] [TRACE URL] https://app.opik.com/impakt/projects/{OPIK_PROJECT}/traces/{trace_id}")
                
                # Method 3: Check tracer attributes
                if not trace_id:
                    if hasattr(self.opik_tracer, 'trace_id'):
                        trace_id = self.opik_tracer.trace_id
                    elif hasattr(self.opik_tracer, 'run_manager'):
                        run_mgr = self.opik_tracer.run_manager
                        if hasattr(run_mgr, 'run_id'):
                            trace_id = run_mgr.run_id
                        elif hasattr(run_mgr, 'trace_id'):
                            trace_id = run_mgr.trace_id
                
                # Method 4: Try Opik context
                if not trace_id:
                    try:
                        from opik.opik_context import get_current_trace_data
                        trace_data = get_current_trace_data()
                        if trace_data:
                            if hasattr(trace_data, 'trace_id'):
                                trace_id = trace_data.trace_id
                            elif isinstance(trace_data, dict) and 'trace_id' in trace_data:
                                trace_id = trace_data['trace_id']
                            elif hasattr(trace_data, 'id'):
                                trace_id = trace_data.id
                    except:
                        pass
                
                # Method 5: Check tracer's internal __dict__
                if not trace_id and hasattr(self.opik_tracer, '__dict__'):
                    tracer_dict = self.opik_tracer.__dict__
                    for key in ['trace_id', 'run_id', 'current_trace_id']:
                        if key in tracer_dict:
                            trace_id = tracer_dict[key]
                            break
                
                if trace_id:
                    print(f"[OPIK] [TRACE ID] {trace_id}")
                    print(f"[OPIK] [TRACE URL] https://app.opik.com/impakt/projects/{OPIK_PROJECT}/traces/{trace_id}")
                    
                    # Also try to flush the tracer to ensure all data is sent
                    try:
                        if hasattr(self.opik_tracer, 'flush'):
                            print(f"[OPIK DEBUG] Flushing tracer to ensure feedback scores are sent...")
                            self.opik_tracer.flush()
                            print(f"[OPIK DEBUG] Tracer flushed successfully")
                    except Exception as flush_error:
                        print(f"[OPIK DEBUG] Flush error (non-critical): {flush_error}")
                else:
                    print(f"[OPIK DEBUG] Trace ID not found - checking tracer internals...")
                    if hasattr(self.opik_tracer, 'created_traces'):
                        print(f"[OPIK DEBUG] created_traces: {self.opik_tracer.created_traces}")
                    if hasattr(self.opik_tracer, '_created_traces'):
                        print(f"[OPIK DEBUG] _created_traces keys: {list(self.opik_tracer._created_traces.keys()) if self.opik_tracer._created_traces else 'empty'}")
                    if hasattr(self.opik_tracer, '_created_traces_data_map'):
                        print(f"[OPIK DEBUG] _created_traces_data_map keys: {list(self.opik_tracer._created_traces_data_map.keys()) if hasattr(self.opik_tracer, '_created_traces_data_map') and self.opik_tracer._created_traces_data_map else 'empty'}")
            except Exception as e:
                print(f"[OPIK DEBUG] Error extracting trace ID: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"[OPIK DEBUG] Graph invoke completed successfully")
            return result
        else:
            print(f"[OPIK DEBUG] Invoking graph WITHOUT Opik tracer (no tracer available)")
            result = self.graph.invoke(state)
            print(f"[OPIK DEBUG] Graph invoke completed (no tracer)")
            return result


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


def _normalize_vote_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept camelCase or snake_case for release-mode fields."""
    out = dict(payload)
    if "walletAddress" in out and "wallet_address" not in out:
        out["wallet_address"] = out["walletAddress"]
    if "vaultAddress" in out and "vault_address" not in out:
        out["vault_address"] = out["vaultAddress"]
    return out


@app.post("/vote")
async def vote(payload: Dict[str, Any] = Body(...)):
    """
    Raw JSON endpoint with two modes:

    1) Adjustment mode (per-org payout tweak):
      { claim_submitted, relief_fund, disaster_details, vote: "higher"|"lower" }

    2) Release mode (consensus to release funds from a vault to an NGO wallet):
      { wallet_address, amount, vault_address, vote: "release" }
      (camelCase walletAddress / vaultAddress also accepted)
    """
    # Set project name before trace creation
    if OPIK_AVAILABLE:
        os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT

    payload = _normalize_vote_payload(payload)

    # Detect release-mode request by its keys (snake_case after normalization)
    is_release_mode = all(k in payload for k in ("wallet_address", "amount", "vault_address", "vote"))
    
    # Wrap entire endpoint with Opik tracing
    trace_name = "vote.release" if is_release_mode else "vote.adjustment"
    with OpikTracer(trace_name, {
        "project": OPIK_PROJECT,
        "tags": [OPIK_PROJECT, "vote", "claim_verification"],
        "mode": "release" if is_release_mode else "adjustment",
        "input": payload
    }, project_tag=OPIK_PROJECT):
        
        if is_release_mode:
            req = ReleaseRequest.model_validate(payload)
            vote_dir = (req.vote or "").strip().lower()
            if vote_dir not in {"release", "approve", "approved", "yes"}:
                error_response = JSONResponse(
                    status_code=400,
                    content={"error": 'For release-mode, vote must indicate consensus (e.g. "release")'},
                )
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={"error": "Invalid vote for release mode", "success": False})
                    except:
                        pass
                return error_response

            w3 = Web3(Web3.HTTPProvider(RPC_URL))
            if not w3.is_connected():
                error_response = JSONResponse(status_code=500, content={"error": "Failed to connect to RPC provider"})
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={"error": "RPC connection failed", "success": False})
                    except:
                        pass
                return error_response

            # Validate addresses
            try:
                vault = Web3.to_checksum_address(req.vault_address)
                to_wallet = Web3.to_checksum_address(req.wallet_address)
            except Exception:
                error_response = JSONResponse(status_code=400, content={"error": "Invalid wallet_address or vault_address"})
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={"error": "Invalid addresses", "success": False})
                    except:
                        pass
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
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={"error": "Amount is zero", "success": False})
                    except:
                        pass
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
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={"error": "Insufficient balance", "success": False})
                    except:
                        pass
                return error_response

            private_key = (os.getenv("PRIVATE_KEY") or "").strip()
            if not private_key:
                error_response = JSONResponse(status_code=500, content={"error": "PRIVATE_KEY not set in .env"})
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={"error": "Private key not set", "success": False})
                    except:
                        pass
                return error_response
            pk = private_key[2:] if private_key.startswith("0x") else private_key
            acct = Account.from_key(pk)

            vault_contract = w3.eth.contract(address=vault, abi=_VAULT_ABI)
            try:
                creator = vault_contract.functions.creator().call()
            except Exception as e:
                error_response = JSONResponse(status_code=500, content={"error": f"Failed to read vault creator: {e}"})
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={"error": f"Failed to read creator: {e}", "success": False})
                    except:
                        pass
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
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={"error": "Not vault creator", "success": False})
                    except:
                        pass
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
                
                # Update trace with output
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={
                            "output": response_data,
                            "success": True
                        })
                    except:
                        pass
                
                return JSONResponse(content=response_data)

            except Exception as e:
                error_response = JSONResponse(status_code=500, content={"status": "failed", "error": str(e)})
                
                # Update trace with error
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_trace
                        update_current_trace(metadata={
                            "error": str(e),
                            "success": False
                        })
                    except:
                        pass
                
                return error_response

        # Otherwise: adjustment mode (only if payload has adjustment-mode keys)
        adjustment_keys = ("claim_submitted", "relief_fund", "disaster_details", "vote")
        is_adjustment_mode = all(k in payload for k in adjustment_keys)
        if not is_adjustment_mode:
            error_response = JSONResponse(
                status_code=400,
                content={
                    "error": "Invalid vote payload. Use one of: (1) Release mode: wallet_address, amount, vault_address, vote (e.g. \"release\"); (2) Adjustment mode: claim_submitted, relief_fund, disaster_details, vote (\"higher\" or \"lower\").",
                    "received_keys": list(payload.keys()),
                },
            )
            if OPIK_AVAILABLE:
                try:
                    from opik.opik_context import update_current_trace
                    update_current_trace(metadata={"error": "Invalid payload keys", "success": False})
                except:
                    pass
            return error_response

        req = VoteRequest.model_validate(payload)
        vote_dir = (req.vote or "").strip().lower()
        if vote_dir not in {"higher", "lower"}:
            error_response = JSONResponse(status_code=400, content={"error": 'vote must be either "higher" or "lower"'})
            if OPIK_AVAILABLE:
                try:
                    from opik.opik_context import update_current_trace
                    update_current_trace(metadata={"error": "Invalid vote direction", "success": False})
                except:
                    pass
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

        # LLM call with OpikSpan
        with OpikSpan("vote_adjustment_llm",
                     input_data={
                         "vote": vote_dir,
                         "relief_fund": req.relief_fund,
                         "claim_preview": req.claim_submitted[:200],
                         "disaster_preview": req.disaster_details[:200]
                     },
                     output_data={}):
            resp = client.chat.completions.create(
                model=agent.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            
            # Update span with output
            if OPIK_AVAILABLE:
                try:
                    from opik.opik_context import update_current_span
                    raw_preview = (resp.choices[0].message.content or "").strip()[:200]
                    update_current_span(metadata={
                        "output": {
                            "raw_response_preview": raw_preview,
                            "model": agent.reasoning_model
                        }
                    })
                except:
                    pass

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
        
        # LLM-as-judge evaluation
        evaluation = None
        if llm_judge:
            with OpikSpan("evaluate_vote_adjustment",
                         input_data={
                             "vote": vote_dir,
                             "original_relief_fund": req.relief_fund,
                             "delta": delta,
                             "updated_relief_fund": updated
                         },
                         output_data={}):
                evaluation = llm_judge.evaluate_vote_adjustment(
                    vote_direction=vote_dir,
                    original_relief_fund=req.relief_fund,
                    delta_usd=delta,
                    updated_relief_fund=updated,
                    claim_submitted=req.claim_submitted,
                    disaster_details=req.disaster_details,
                    reasoning_summary=reasoning_summary,
                    send_to_opik=True
                )
                print(f"\n[OPIK EVALUATION] Vote Adjustment Score: {evaluation.get('overall_score', 0):.2f}/10")
                
                # Explicitly send feedback scores to the current trace
                if OPIK_AVAILABLE and evaluation.get('overall_score'):
                    try:
                        from opik.opik_context import update_current_trace
                        feedback_scores_list = [{
                            "name": "vote_adjustment_quality",
                            "value": float(evaluation.get('overall_score', 0)),
                            "reason": evaluation.get('reasoning', 'Vote adjustment quality evaluation')
                        }]
                        # Add individual criterion scores
                        for key, value in evaluation.items():
                            if key not in ['overall_score', 'reasoning'] and isinstance(value, (int, float)):
                                feedback_scores_list.append({
                                    "name": f"vote_adjustment_quality_{key}",
                                    "value": float(value),
                                    "reason": f"{key.replace('_', ' ').title()} criterion score"
                                })
                        
                        update_current_trace(feedback_scores=feedback_scores_list)
                        print(f"[OPIK] Sent {len(feedback_scores_list)} feedback scores to trace")
                    except Exception as e:
                        print(f"[OPIK] Failed to send feedback scores to trace: {e}")
                        import traceback
                        traceback.print_exc()

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
        
        # Add evaluation to response if available
        if evaluation:
            response_data["evaluation"] = evaluation
        
        # Update trace with output
        if OPIK_AVAILABLE:
            try:
                from opik.opik_context import update_current_trace
                update_current_trace(metadata={
                    "output": {
                        "updated_relief_fund": updated,
                        "delta": delta,
                        "success": True
                    }
                })
            except:
                pass

        return JSONResponse(content=response_data)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("VERIFY_API_PORT", "8080")))

