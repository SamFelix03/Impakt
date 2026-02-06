"""
Response Coordination Agent

Takes disaster details (location + summary + sources) and uses OpenAI web search
to find relevant local people/services (agencies/NGOs/hospitals/relief orgs),
then drafts personalized outreach emails and sends them via SMTP.

Requires: openai langgraph python-dotenv
"""

import json
import os
import re
import mimetypes
import smtplib
from datetime import datetime
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI

load_dotenv()

# Opik integration removed - evaluation mode doesn't use internal tracing

def _load_env_local() -> None:
    """
    Ensure `.env` (next to this file) is loaded.
    Uses python-dotenv (already a dependency in this repo).
    """
    env_path = Path(__file__).resolve().parent / ".env"
    try:
        from dotenv import load_dotenv as _load_dotenv  # type: ignore
        _load_dotenv(env_path)
    except Exception:
        # If dotenv is not available for some reason, rely on environment variables.
        pass


@dataclass(frozen=True)
class _SMTPConfig:
    server: str
    port: int
    username: str
    password: str
    from_email: str

    @staticmethod
    def from_env() -> "_SMTPConfig":
        _load_env_local()

        server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        port = int(os.getenv("SMTP_PORT", "587"))
        username = (os.getenv("SMTP_USERNAME") or "").strip()
        password = (os.getenv("SMTP_PASSWORD") or "").strip()
        from_email = (os.getenv("SMTP_FROM_EMAIL") or username).strip()

        missing = [k for k, v in {"SMTP_USERNAME": username, "SMTP_PASSWORD": password}.items() if not v]
        if missing:
            raise RuntimeError(
                "Missing SMTP configuration: "
                + ", ".join(missing)
                + ". Set them in agents/.env or your environment."
            )

        return _SMTPConfig(server=server, port=port, username=username, password=password, from_email=from_email)


def _send_email_smtp(*, to_email: str, subject: str, body_text: str, body_html: Optional[str] = None,
                     attachments: Optional[Sequence[str]] = None) -> None:
    """
    Inline SMTP email sender (STARTTLS). No external imports/files required.
    """
    smtp_cfg = _SMTPConfig.from_env()
    to_email = (to_email or "").strip()
    if not to_email:
        raise ValueError("to_email is required")

    msg = EmailMessage()
    msg["From"] = smtp_cfg.from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.set_content(body_text or "")
    if body_html:
        msg.add_alternative(body_html, subtype="html")

    for p in list(attachments or []):
        path = Path(p)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Attachment not found: {path}")

        ctype, encoding = mimetypes.guess_type(str(path))
        if ctype is None or encoding is not None:
            ctype = "application/octet-stream"
        maintype, subtype = ctype.split("/", 1)
        msg.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)

    with smtplib.SMTP(smtp_cfg.server, smtp_cfg.port) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(smtp_cfg.username, smtp_cfg.password)
        server.send_message(msg, from_addr=smtp_cfg.from_email, to_addrs=[to_email])


def _extract_json(text: str) -> Optional[Any]:
    """
    Best-effort JSON extractor:
    - tries direct json.loads
    - tries fenced ```json blocks
    - tries substring from first '{'/'[' to last '}'/']'
    """
    if not text:
        return None

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fenced block
    fenced = re.search(r"```json\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # Heuristic substring
    start_candidates = [text.find("{"), text.find("[")]
    start_candidates = [i for i in start_candidates if i != -1]
    if not start_candidates:
        return None
    start = min(start_candidates)
    end_candidates = [text.rfind("}"), text.rfind("]")]
    end_candidates = [i for i in end_candidates if i != -1]
    if not end_candidates:
        return None
    end = max(end_candidates)

    if end <= start:
        return None

    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


class CoordinatorState(TypedDict):
    # Input packet
    location: str
    disaster_summary: str
    sources: List[Dict[str, str]]
    relief_amount_usd: Optional[int]
    relief_amount_eth: Optional[float]
    vault_address: Optional[str]

    # Working fields
    contacts_raw: Any
    contacts: List[Dict[str, Any]]
    email_drafts_raw: Any
    email_drafts: List[Dict[str, Any]]
    send_results: List[Dict[str, Any]]

    # Metadata
    created_at: str


class ResponseCoordinatorAgent:
    """
    A separate agent responsible for:
    1) discovering relevant local responders via web search
    2) drafting personalized outreach emails
    """

    def __init__(self, search_model: str = "gpt-4o-search-preview", reasoning_model: str = "gpt-4o"):
        self.search_model = search_model
        self.reasoning_model = reasoning_model

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment/.env")

        self.search_client = OpenAI(api_key=api_key)
        self.reasoning_client = OpenAI(api_key=api_key)
        
        self.graph = self._build_graph()

    def _search_contacts(self, state: CoordinatorState):
        print(f"\n{'='*80}")
        print(f"[COORD STEP 1: SEARCH CONTACTS] Finding local responders...")
        print(f"{'='*80}")

        location = state["location"]
        summary = state["disaster_summary"]
        sources = state.get("sources", [])
        sources_text = "\n".join([f"- {s.get('title','')}: {s.get('url','')}" for s in sources[:5]])

        # We intentionally ask for concrete contact channels; if no direct emails exist,
        # accept "contact form URL" or a department inbox.
        query = f"""You are helping coordinate disaster response outreach.

DISASTER LOCATION (focus area): {location}

DISASTER SUMMARY:
{summary}

SOURCES (for context):
{sources_text if sources_text else "- (none provided)"}

TASK:
Find 6-10 reputable, relevant people/services/organizations in or responsible for {location} that can help with disaster response, such as:
- Official emergency management / disaster response agency
- Local government disaster management authority
- Major local hospitals / emergency services
- Reputable NGOs/Red Cross/relief organizations active in the area
- UN/local humanitarian coordination (if relevant)

REQUIREMENTS:
- Prioritize organizations that are local/regional to {location}
- Include official contact channels (email preferred). If email is not available, provide a contact form URL and organization website.
- Avoid random individuals; prefer official org contacts.
- Return STRICT JSON ONLY (no markdown) with this schema:
{{
  "contacts": [
    {{
      "name": "string or null",
      "role": "string (e.g., Emergency Operations Center, Relief Coordinator)",
      "organization": "string",
      "location_scope": "string (city/region/country)",
      "email": "string or null",
      "contact_form_url": "string or null",
      "website": "string or null",
      "phone": "string or null",
      "why_relevant": "string (1-2 sentences)",
      "evidence_url": "string (URL proving the contact/channel)"
    }}
  ]
}}"""

        try:
            # Get model and parameters override if specified (for evaluation variants)
            model = getattr(self, '_llm_params_override', {}).get('model', self.search_model)
            if 'search_model' in getattr(self, '_llm_params_override', {}):
                model = getattr(self, '_llm_params_override', {})['search_model']
            
            # Build LLM call parameters
            llm_params = {
                "model": model,
                "web_search_options": {},
                "messages": [{"role": "user", "content": query}],
            }
            
            # Override with variant_config if present
            override_params = getattr(self, '_llm_params_override', {})
            if 'temperature' in override_params:
                llm_params["temperature"] = override_params['temperature']
            for param in ['max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
                if param in override_params:
                    llm_params[param] = override_params[param]
            
            completion = self.search_client.chat.completions.create(**llm_params)
            
            msg = completion.choices[0].message
            raw = msg.content or ""

            parsed = _extract_json(raw)
            contacts_all: List[Dict[str, Any]] = []
            if isinstance(parsed, dict) and isinstance(parsed.get("contacts"), list):
                contacts_all = [c for c in parsed["contacts"] if isinstance(c, dict)]

            # Filter: only contacts with an email (we will send emails)
            contacts: List[Dict[str, Any]] = []
            for c in contacts_all:
                email = (c.get("email") or "").strip()
                if email and "@" in email:
                    c["email"] = email
                    contacts.append(c)

            print(f"[COORD SEARCH] ✅ Found {len(contacts_all)} contacts (parsed)")
            print(f"[COORD SEARCH] ✅ Email-capable contacts (filtered): {len(contacts)}")
            if contacts:
                print(f"\n[COORD SEARCH OUTPUT] Contact details (email-only; will send to these):")
                print(f"{'-'*80}")
                for i, c in enumerate(contacts, 1):
                    org = c.get("organization") or "Unknown organization"
                    name = c.get("name") or "N/A"
                    role = c.get("role") or "N/A"
                    scope = c.get("location_scope") or "N/A"
                    email = c.get("email") or "N/A"
                    phone = c.get("phone") or "N/A"
                    website = c.get("website") or "N/A"
                    contact_form = c.get("contact_form_url") or "N/A"
                    evidence = c.get("evidence_url") or "N/A"
                    why = c.get("why_relevant") or ""

                    print(f"\n{i}. {org}")
                    print(f"   Contact: {name} ({role})")
                    print(f"   Scope: {scope}")
                    print(f"   Email: {email}")
                    print(f"   Phone: {phone}")
                    print(f"   Website: {website}")
                    print(f"   Contact form: {contact_form}")
                    print(f"   Evidence: {evidence}")
                    if why:
                        print(f"   Why relevant: {why}")
                print(f"{'-'*80}")
            
            return {"contacts_raw": raw, "contacts": contacts}

        except Exception as e:
            print(f"[COORD SEARCH] ❌ Error: {str(e)}")
            return {"contacts_raw": {"error": str(e)}, "contacts": []}

    def _draft_emails(self, state: CoordinatorState):
        print(f"\n{'='*80}")
        print(f"[COORD STEP 2: DRAFT EMAILS] Drafting personalized outreach...")
        print(f"{'='*80}")

        location = state["location"]
        summary = state["disaster_summary"]
        sources = state.get("sources", [])
        relief_usd = state.get("relief_amount_usd")
        relief_eth = state.get("relief_amount_eth")
        vault_address = state.get("vault_address")

        contacts = state.get("contacts", [])[:8]
        if not contacts:
            print("[COORD DRAFT] ⚠️ No email-capable contacts found; skipping drafting.")
            return {"email_drafts_raw": None, "email_drafts": []}

        sources_text = "\n".join([f"- {s.get('title','')}: {s.get('url','')}" for s in sources[:5]])
        contacts_text = json.dumps(contacts, ensure_ascii=False, indent=2)

        prompt = f"""Draft urgent outreach emails that reflect the SERIOUSNESS of the disaster and ask the recipient to HELP people affected.

DISASTER LOCATION: {location}
DISASTER SUMMARY:
{summary}

SOURCES:
{sources_text if sources_text else "- (none provided)"}

OPTIONAL RELIEF CONTEXT:
- Relief estimate (USD): {relief_usd if relief_usd is not None else "N/A"}
- Relief estimate (ETH): {relief_eth if relief_eth is not None else "N/A"}
- Donation vault address: {vault_address if vault_address else "N/A"}

CONTACTS (JSON):
{contacts_text}

RULES:
- Produce one email draft per contact.
- Keep each email concise, specific, and respectful.
- All contacts provided have an email. Set "to_email" to the contact email EXACTLY.
- Do NOT frame it as "we offer our support" or "potential collaboration". This is an ACTION request: ask them to respond/assist people in the affected areas.
- Each email MUST:
  - Convey urgency/seriousness (use concrete facts from the DISASTER SUMMARY: affected areas, casualties, evacuations, scale, urgency).
  - Explain EXACTLY how THAT organization can help, based on the contact's "role", "organization", and "why_relevant" fields in CONTACTS JSON.
  - Include 3-6 tailored ACTION ITEMS (bullet list) that the org should do now (imperative statements), e.g. deploy teams, coordinate evacuations, open shelters, medical triage, incident coordination, logistics routing, public safety messaging, etc.
  - Include a short "Immediate actions" line with clear directives (no questions).
- Do NOT ask questions. Do NOT include question marks.
- Do NOT claim you verified anything on the ground.
- SIGNATURE MUST be exactly:
Sincerely,
The Hand
- Do NOT include placeholders like "[Your Name]".
- Include the source URL(s) as references in the email.
- Output STRICT JSON ONLY (no markdown) with schema:
{{
  "emails": [
    {{
      "to_name": "string or null",
      "to_role": "string",
      "to_org": "string",
      "to_email": "string or null",
      "contact_form_url": "string or null",
      "subject": "string",
      "body": "string (plain text email body)",
      "notes": "string (short internal note why this email is tailored)"
    }}
  ]
}}"""

        try:
            # Get model and parameters override if specified
            model = getattr(self, '_llm_params_override', {}).get('model', self.reasoning_model)
            
            # Build LLM call parameters
            llm_params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.4
            }
            
            # Override with variant_config if present
            override_params = getattr(self, '_llm_params_override', {})
            if 'temperature' in override_params:
                llm_params["temperature"] = override_params['temperature']
            for param in ['max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
                if param in override_params:
                    llm_params[param] = override_params[param]
            
            resp = self.reasoning_client.chat.completions.create(**llm_params)
            
            raw = resp.choices[0].message.content or ""
            print(f"[COORD DRAFT DEBUG] Raw LLM response length: {len(raw)} chars")
            if len(raw) < 500:
                print(f"[COORD DRAFT DEBUG] Full raw response: {raw}")
            else:
                print(f"[COORD DRAFT DEBUG] Raw response preview: {raw[:500]}...")
            
            parsed = _extract_json(raw)
            print(f"[COORD DRAFT DEBUG] Parsed JSON type: {type(parsed)}")
            print(f"[COORD DRAFT DEBUG] Parsed JSON keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'N/A'}")
            
            emails: List[Dict[str, Any]] = []
            if isinstance(parsed, dict) and isinstance(parsed.get("emails"), list):
                emails = [e for e in parsed["emails"] if isinstance(e, dict)]
                print(f"[COORD DRAFT DEBUG] Extracted {len(emails)} emails from parsed JSON")
            else:
                print(f"[COORD DRAFT DEBUG] No emails array found in parsed JSON")
                # If parsing failed, return the raw response for evaluation
                if raw:
                    print(f"[COORD DRAFT DEBUG] Returning raw response for evaluation (parsing failed)")
                    # Create a single "email" with the raw response as the body
                    # This lets the LLM judge evaluate the actual truncated/malformed output
                    emails = [{
                        "to_email": contacts[0].get("email", "") if contacts else "",
                        "to_org": contacts[0].get("organization", "Unknown") if contacts else "Unknown",
                        "to_role": contacts[0].get("role", "Unknown") if contacts else "Unknown",
                        "subject": "(Failed to parse - see raw output below)",
                        "body": raw,
                        "notes": "PARSE FAILED: Raw LLM output returned as-is for evaluation"
                    }]
                    print(f"[COORD DRAFT DEBUG] Created 1 email with raw response for evaluation")

            # Safety: only keep drafts whose to_email matches one of the contact emails
            allowed_emails = {(c.get("email") or "").strip() for c in contacts if (c.get("email") or "").strip()}
            print(f"[COORD DRAFT DEBUG] Allowed emails: {allowed_emails}")
            
            filtered_emails: List[Dict[str, Any]] = []
            for e in emails:
                to_email = (e.get("to_email") or "").strip()
                print(f"[COORD DRAFT DEBUG] Checking email with to_email='{to_email}' against allowed={allowed_emails}")
                if to_email in allowed_emails:
                    filtered_emails.append(e)
                    print(f"[COORD DRAFT DEBUG]   ✓ Email accepted")
                else:
                    print(f"[COORD DRAFT DEBUG]   ✗ Email filtered out (to_email not in allowed set)")

            print(f"[COORD DRAFT] ✅ Drafted {len(filtered_emails)} emails (parsed, email-only)")
            if emails:
                print(f"\n[COORD DRAFT OUTPUT] Drafted emails (full text):")
                print(f"{'-'*80}")
                
                for i, e in enumerate(filtered_emails, 1):
                    to_org = e.get("to_org") or "Unknown org"
                    to_name = e.get("to_name") or "N/A"
                    to_role = e.get("to_role") or "N/A"
                    to_email = e.get("to_email") or "N/A"
                    contact_form = e.get("contact_form_url") or "N/A"
                    subject = e.get("subject") or "(no subject)"
                    body = e.get("body") or ""
                    notes = e.get("notes") or ""

                    print(f"\n{i}. To: {to_org} — {to_name} ({to_role})")
                    print(f"   Email: {to_email}")
                    print(f"   Contact form: {contact_form}")
                    print(f"   Subject: {subject}")
                    print(f"   {'-'*76}")
                    print(body)
                    print(f"   {'-'*76}")
                    if notes:
                        print(f"   Notes: {notes}")
                print(f"{'-'*80}")
            
            return {"email_drafts_raw": raw, "email_drafts": filtered_emails}

        except Exception as e:
            print(f"[COORD DRAFT] ❌ Error: {str(e)}")
            return {"email_drafts_raw": {"error": str(e)}, "email_drafts": []}

    def _send_emails(self, state: CoordinatorState):
        print(f"\n{'='*80}")
        print(f"[COORD STEP 3: SEND EMAILS] Sending drafted emails...")
        print(f"{'='*80}")

        drafts = state.get("email_drafts", []) or []
        if not drafts:
            print("[COORD SEND] ⚠️ No drafted emails to send.")
            return {"send_results": []}

        results: List[Dict[str, Any]] = []
        for i, d in enumerate(drafts, 1):
            to_email = (d.get("to_email") or "").strip()
            subject = d.get("subject") or "(no subject)"
            body = d.get("body") or ""
            to_org = d.get("to_org") or "Unknown org"
            to_name = d.get("to_name") or "N/A"

            print(f"\n[COORD SEND] {i}/{len(drafts)} → {to_org} ({to_name}) <{to_email}>")
            try:
                _send_email_smtp(to_email=to_email, subject=subject, body_text=body)
                print("[COORD SEND] ✅ Sent")
                results.append({"to_email": to_email, "to_org": to_org, "status": "sent"})
            except Exception as e:
                print(f"[COORD SEND] ❌ Failed: {str(e)}")
                results.append({"to_email": to_email, "to_org": to_org, "status": "failed", "error": str(e)})

        return {"send_results": results}

    def _build_graph(self):
        workflow = StateGraph(CoordinatorState)
        workflow.add_node("search_contacts", self._search_contacts)
        workflow.add_node("draft_emails", self._draft_emails)
        workflow.add_node("send_emails", self._send_emails)

        workflow.set_entry_point("search_contacts")
        workflow.add_edge("search_contacts", "draft_emails")
        workflow.add_edge("draft_emails", "send_emails")
        workflow.add_edge("send_emails", END)

        return workflow.compile()

    def run(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        state: CoordinatorState = {
            "location": packet.get("location", ""),
            "disaster_summary": packet.get("disaster_summary", ""),
            "sources": packet.get("sources", []) or [],
            "relief_amount_usd": packet.get("relief_amount_usd"),
            "relief_amount_eth": packet.get("relief_amount_eth"),
            "vault_address": packet.get("vault_address"),
            "contacts_raw": None,
            "contacts": [],
            "email_drafts_raw": None,
            "email_drafts": [],
            "send_results": [],
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        return self.graph.invoke(state)

