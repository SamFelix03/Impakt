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

# Opik integration - import with fallback
try:
    from opik_integration import (
        OpikTracer, OpikSpan, trace_agent_execution,
        llm_judge, OpikAgentOptimizer, OPIK_AVAILABLE, OPIK_PROJECT
    )
    from opik_langgraph_helper import update_node_span_with_io, update_trace_with_feedback
except ImportError:
    OpikTracer = None
    OpikSpan = None
    llm_judge = None
    OpikAgentOptimizer = None
    OPIK_AVAILABLE = False
    OPIK_PROJECT = os.getenv("OPIK_PROJECT", "disaster-monitoring")
    update_node_span_with_io = None
    update_trace_with_feedback = None

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
        
        # Initialize Opik components
        self.opik_optimizer = OpikAgentOptimizer("response_coordinator") if (OPIK_AVAILABLE and OpikAgentOptimizer) else None
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

    def _search_contacts(self, state: CoordinatorState):
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("SEARCH CONTACTS")
        
        # Update LangGraph node span with input data
        input_data = {
            "location": state.get("location", ""),
            "disaster_summary_length": len(state.get("disaster_summary", "")),
            "disaster_summary_preview": state.get("disaster_summary", "")[:200]
        }
        update_node_span_with_io(input_data=input_data, metadata={"node": "search_contacts"})
        
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
            # Add input/output tracking
            with OpikSpan("contact_search_llm",
                         input_data={"query_preview": query[:150], "location": location},
                         output_data={}):
                completion = self.search_client.chat.completions.create(
                    model=self.search_model,
                    web_search_options={},
                    messages=[{"role": "user", "content": query}],
                )
                
                # Update span with output
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_span
                        raw = completion.choices[0].message.content or ""
                        update_current_span(metadata={
                            "output": {
                                "response_length": len(raw),
                                "model": self.search_model
                            }
                        })
                    except:
                        pass
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
            
            # Prepare output data
            output_data = {
                "contacts_found": len(contacts),
                "contacts_total": len(contacts_all),
                "valid_emails": len([c for c in contacts if c.get("email")]),
                "contact_organizations": [c.get("organization", "Unknown") for c in contacts[:5]]
            }
            
            # Evaluate contact search quality
            if llm_judge and contacts:
                with OpikSpan("evaluate_contact_search",
                             input_data={"contacts_count": len(contacts)},
                             output_data={}):
                    # Evaluate first few contacts
                    contact_quality_score = len([c for c in contacts if c.get("email") and "@" in c.get("email", "")]) / max(len(contacts), 1)
                    print(f"\n[OPIK EVALUATION] Contact Search Quality: {contact_quality_score:.2%} valid emails")
                    
                    # Send feedback score to Opik (format: list of dicts)
                    if OPIK_AVAILABLE:
                        try:
                            from opik.opik_context import update_current_trace
                            feedback_scores_list = [{
                                "name": "contact_search_quality",
                                "value": float(contact_quality_score * 10),
                                "reason": f"Contact search quality: {contact_quality_score:.2%} valid emails"
                            }]
                            update_current_trace(feedback_scores=feedback_scores_list)
                        except Exception as e:
                            print(f"[OPIK] Failed to send contact search feedback: {e}")
                    
                    self.evaluation_results.append({
                        "step": "contact_search",
                        "contacts_found": len(contacts),
                        "valid_emails": len([c for c in contacts if c.get("email")]),
                        "quality_score": contact_quality_score,
                        "timestamp": datetime.now().isoformat()
                    })
                    output_data["quality_score"] = contact_quality_score
            
            # Update LangGraph node span with output data
            update_node_span_with_io(
                output_data={
                    **output_data,
                    "contacts_found": len(contacts),
                    "contact_organizations": [c.get("organization", "Unknown") for c in contacts[:5]]
                },
                metadata={"node": "search_contacts"}
            )
            
            return {"contacts_raw": raw, "contacts": contacts}

        except Exception as e:
            print(f"[COORD SEARCH] ❌ Error: {str(e)}")
            return {"contacts_raw": {"error": str(e)}, "contacts": []}

    def _draft_emails(self, state: CoordinatorState):
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("DRAFT EMAILS")
        # Prepare input data
        input_data = {
            "location": state.get("location", ""),
            "contact_count": len(state.get("contacts", [])),
            "relief_amount_usd": state.get("relief_amount_usd"),
            "vault_address": state.get("vault_address")
        }
        
        # Update current span (created by LangGraph) with input data
        if OPIK_AVAILABLE:
            try:
                from opik.opik_context import update_current_span
                from opik_integration import OPIK_PROJECT
                update_current_span(metadata={
                    "input": input_data,
                    "project": OPIK_PROJECT,
                    "tags": [OPIK_PROJECT, "response_coordinator", "email_drafting"]
                })
            except:
                pass
        
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
            # Add input/output tracking
            with OpikSpan("email_drafting_llm",
                         input_data={"contacts_count": len(contacts), "prompt_preview": prompt[:150]},
                         output_data={}):
                resp = self.reasoning_client.chat.completions.create(
                    model=self.reasoning_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    max_tokens=350,
                )
                
                # Update span with output
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_span
                        raw = resp.choices[0].message.content or ""
                        update_current_span(metadata={
                            "output": {
                                "response_length": len(raw),
                                "model": self.reasoning_model
                            }
                        })
                    except:
                        pass
            raw = resp.choices[0].message.content or ""
            parsed = _extract_json(raw)
            emails: List[Dict[str, Any]] = []
            if isinstance(parsed, dict) and isinstance(parsed.get("emails"), list):
                emails = [e for e in parsed["emails"] if isinstance(e, dict)]

            # Safety: only keep drafts whose to_email matches one of the contact emails
            allowed_emails = {(c.get("email") or "").strip() for c in contacts if (c.get("email") or "").strip()}
            filtered_emails: List[Dict[str, Any]] = []
            for e in emails:
                to_email = (e.get("to_email") or "").strip()
                if to_email in allowed_emails:
                    filtered_emails.append(e)

            print(f"[COORD DRAFT] ✅ Drafted {len(filtered_emails)} emails (parsed, email-only)")
            if emails:
                print(f"\n[COORD DRAFT OUTPUT] Drafted emails (full text):")
                print(f"{'-'*80}")
                
                # Prepare output data
                output_data = {
                    "emails_drafted": len(filtered_emails),
                    "email_recipients": [e.get("to_org", "Unknown") for e in filtered_emails[:5]],
                    "email_subjects": [e.get("subject", "")[:50] for e in filtered_emails[:3]]
                }
                
                # Evaluate email quality
                email_evaluations = []
                if llm_judge:
                    with OpikSpan("evaluate_email_quality",
                                 input_data={"emails_to_evaluate": len(filtered_emails[:3])},
                                 output_data={}):
                        for e in filtered_emails[:3]:  # Evaluate first 3 emails
                            # Find matching contact
                            matching_contact = next(
                                (c for c in contacts if c.get("email") == e.get("to_email")),
                                {}
                            )
                            evaluation = llm_judge.evaluate_email_quality(
                                email_draft=e,
                                disaster_info=summary,
                                contact_info=matching_contact,
                                send_to_opik=True
                            )
                            email_evaluations.append(evaluation)
                            print(f"\n[OPIK EVALUATION] Email to {e.get('to_org', 'N/A')}: Score {evaluation.get('overall_score', 0):.2f}/10")
                
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
                
                if email_evaluations:
                    avg_score = sum(e.get('overall_score', 0) for e in email_evaluations) / len(email_evaluations)
                    print(f"\n[OPIK EVALUATION] Average Email Quality Score: {avg_score:.2f}/10")
                    self.evaluation_results.append({
                        "step": "email_drafting",
                        "emails_drafted": len(filtered_emails),
                        "evaluations": email_evaluations,
                        "average_score": avg_score,
                        "timestamp": datetime.now().isoformat()
                    })
                    output_data["average_email_quality_score"] = avg_score
                    output_data["individual_scores"] = [e.get('overall_score', 0) for e in email_evaluations]
                
                # Update LangGraph node span with output data
                update_node_span_with_io(
                    output_data={
                        **output_data,
                        "emails_drafted": len(filtered_emails),
                        "email_subjects": [e.get("subject", "")[:50] for e in filtered_emails[:3]]
                    },
                    metadata={"node": "draft_emails"}
                )
            
            return {"email_drafts_raw": raw, "email_drafts": filtered_emails}

        except Exception as e:
            print(f"[COORD DRAFT] ❌ Error: {str(e)}")
            return {"email_drafts_raw": {"error": str(e)}, "email_drafts": []}

    def _send_emails(self, state: CoordinatorState):
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("SEND EMAILS")
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
        # Set project name before trace creation
        from opik_integration import OPIK_AVAILABLE, OPIK_PROJECT
        if OPIK_AVAILABLE:
            import os
            os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT
        
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

