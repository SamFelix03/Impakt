"""
OptimizableAgent Wrappers for Opik Agent Optimization
Wraps LangGraph agents to enable prompt optimization
"""

from typing import Any, Dict, TYPE_CHECKING
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from opik_optimizer import OptimizableAgent, ChatPrompt
    OPIK_OPTIMIZER_AVAILABLE = True
except ImportError:
    OPIK_OPTIMIZER_AVAILABLE = False
    print("[WARNING] opik-optimizer not installed. Install with: pip install opik-optimizer")
    OptimizableAgent = None
    ChatPrompt = None

if TYPE_CHECKING:
    from opik_optimizer.api_objects import chat_prompt


class DisasterMonitoringOptimizableAgent(OptimizableAgent):
    """Optimizable wrapper for DisasterMonitoringAgent"""
    
    project_name = "disaster-monitoring"
    
    def __init__(self, base_agent=None):
        super().__init__()
        if base_agent is None:
            from disasterAgent import DisasterMonitoringAgent
            self.base_agent = DisasterMonitoringAgent()
        else:
            self.base_agent = base_agent
    
    def invoke_agent(
        self,
        prompts: Dict[str, "chat_prompt.ChatPrompt"],
        dataset_item: Dict[str, Any],
        allow_tool_use: bool = False,
        seed: int | None = None,
    ) -> str:
        """Invoke agent with optimized prompts"""
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Extract prompts
        search_prompt = prompts.get("search_disasters")
        calculate_relief_prompt = prompts.get("calculate_relief")
        
        # Prepare dataset item with defaults
        formatted_item = {
            "exclude_list": dataset_item.get("exclude_list", "none"),
            "disaster_query": dataset_item.get("disaster_query", ""),
            "location": dataset_item.get("location", ""),
            "weather_summary": dataset_item.get("weather_summary", "No weather data available"),
        }
        
        # Get messages from prompts
        disaster_info = ""
        if search_prompt:
            try:
                search_messages = search_prompt.get_messages(formatted_item)
                response = client.chat.completions.create(
                    model=self.base_agent.search_model,
                    messages=search_messages,
                    web_search_options={},
                    seed=seed
                )
                disaster_info = response.choices[0].message.content
            except Exception as e:
                disaster_info = f"Error: {str(e)}"
        else:
            disaster_info = formatted_item.get("disaster_query", "")
        
        # Calculate relief
        relief_output = "5000000"
        if calculate_relief_prompt:
            relief_item = {
                **formatted_item,
                "disaster_content": disaster_info,
                "weather_summary": formatted_item.get("weather_summary", "No weather data available")
            }
            try:
                relief_messages = calculate_relief_prompt.get_messages(relief_item)
                response = client.chat.completions.create(
                    model=self.base_agent.reasoning_model,
                    messages=relief_messages,
                    temperature=0.3,
                    seed=seed
                )
                relief_output = response.choices[0].message.content
            except Exception as e:
                relief_output = "5000000"  # Default
        
        # Return combined output for evaluation
        location = formatted_item.get("location", "Unknown")
        return f"Location: {location}\nDisaster Info: {disaster_info}\nRelief Amount: {relief_output}"


class ResponseCoordinatorOptimizableAgent(OptimizableAgent):
    """Optimizable wrapper for ResponseCoordinatorAgent"""
    
    project_name = "response-coordinator"
    
    def __init__(self, base_agent=None):
        super().__init__()
        if base_agent is None:
            from responseCoordinatorAgent import ResponseCoordinatorAgent
            self.base_agent = ResponseCoordinatorAgent()
        else:
            self.base_agent = base_agent
    
    def invoke_agent(
        self,
        prompts: Dict[str, "chat_prompt.ChatPrompt"],
        dataset_item: Dict[str, Any],
        allow_tool_use: bool = False,
        seed: int | None = None,
    ) -> str:
        """Invoke agent with optimized prompts"""
        from openai import OpenAI
        import json
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        search_contacts_prompt = prompts.get("search_contacts")
        draft_emails_prompt = prompts.get("draft_emails")
        
        # Prepare dataset item
        sources = dataset_item.get("sources", [])
        sources_text = "\n".join([f"- {s.get('title','')}: {s.get('url','')}" for s in sources[:5]])
        
        formatted_item = {
            "location": dataset_item.get("location", ""),
            "disaster_summary": dataset_item.get("disaster_summary", ""),
            "sources_text": sources_text if sources_text else "- (none provided)",
            "relief_usd": dataset_item.get("relief_amount_usd"),
            "relief_eth": dataset_item.get("relief_amount_eth"),
            "vault_address": dataset_item.get("vault_address"),
        }
        
        # Search contacts
        contacts_output = '{"contacts": []}'
        if search_contacts_prompt:
            try:
                search_messages = search_contacts_prompt.get_messages(formatted_item)
                response = client.chat.completions.create(
                    model=self.base_agent.search_model,
                    messages=search_messages,
                    web_search_options={},
                    seed=seed
                )
                contacts_output = response.choices[0].message.content
            except Exception as e:
                contacts_output = f'{{"contacts": []}}'
        
        # Draft emails
        emails_output = '{"emails": []}'
        if draft_emails_prompt:
            # Parse contacts from previous step
            try:
                contacts_data = json.loads(contacts_output)
                contacts = contacts_data.get("contacts", [])[:8]
            except:
                contacts = []
            
            email_item = {
                **formatted_item,
                "contacts_text": json.dumps(contacts, ensure_ascii=False, indent=2)
            }
            try:
                email_messages = draft_emails_prompt.get_messages(email_item)
                response = client.chat.completions.create(
                    model=self.base_agent.reasoning_model,
                    messages=email_messages,
                    temperature=0.4,
                    seed=seed
                )
                emails_output = response.choices[0].message.content
            except Exception as e:
                emails_output = '{"emails": []}'
        
        return f"Contacts: {contacts_output}\nEmails: {emails_output}"


class VerificationOptimizableAgent(OptimizableAgent):
    """Optimizable wrapper for VerificationAgent"""
    
    project_name = "claim-verification"
    
    def __init__(self, base_agent=None):
        super().__init__()
        if base_agent is None:
            from verificationAgent import NGOClaimVerifierAgent
            self.base_agent = NGOClaimVerifierAgent()
        else:
            self.base_agent = base_agent
    
    def invoke_agent(
        self,
        prompts: Dict[str, "chat_prompt.ChatPrompt"],
        dataset_item: Dict[str, Any],
        allow_tool_use: bool = False,
        seed: int | None = None,
    ) -> str:
        """Invoke agent with optimized prompts"""
        from openai import OpenAI
        import json
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        analyze_claim_prompt = prompts.get("analyze_claim")
        recommend_amount_prompt = prompts.get("recommend_amount")
        
        # Prepare dataset item
        formatted_item = {
            "content": dataset_item.get("claim_text", ""),
            "vault_usd": dataset_item.get("vault_balance_usd", 0.0),
            "url_texts": dataset_item.get("url_texts", []),
        }
        
        # Analyze claim
        claim_analysis = '{"claim_amount_usd": null, "claim_summary": "", "evidence_summary": ""}'
        if analyze_claim_prompt:
            try:
                claim_messages = analyze_claim_prompt.get_messages(formatted_item)
                response = client.chat.completions.create(
                    model=self.base_agent.reasoning_model,
                    messages=claim_messages,
                    temperature=0.2,
                    seed=seed
                )
                claim_analysis = response.choices[0].message.content
            except Exception as e:
                claim_analysis = '{"claim_amount_usd": null, "claim_summary": "", "evidence_summary": ""}'
        
        # Recommend amount
        recommendation_output = '{"recommended_amount_usd": 0.0}'
        if recommend_amount_prompt:
            try:
                analysis_data = json.loads(claim_analysis)
                claim_summary = analysis_data.get("claim_summary", "")
                evidence_summary = analysis_data.get("evidence_summary", "")
                claim_amount = analysis_data.get("claim_amount_usd")
            except:
                claim_summary = ""
                evidence_summary = ""
                claim_amount = None
            
            recommend_item = {
                "vault_usd": formatted_item.get("vault_usd", 0.0),
                "claim_amount": claim_amount,
                "claim_summary": claim_summary,
                "evidence_summary": evidence_summary
            }
            try:
                recommend_messages = recommend_amount_prompt.get_messages(recommend_item)
                response = client.chat.completions.create(
                    model=self.base_agent.reasoning_model,
                    messages=recommend_messages,
                    temperature=0.1,
                    seed=seed
                )
                recommendation_output = response.choices[0].message.content
            except Exception as e:
                recommendation_output = '{"recommended_amount_usd": 0.0}'
        
        return f"Claim Analysis: {claim_analysis}\nRecommendation: {recommendation_output}"


def extract_prompts_from_disaster_agent():
    """Extract current prompts from DisasterMonitoringAgent"""
    if not OPIK_OPTIMIZER_AVAILABLE:
        return {}
    
    prompts = {
        "search_disasters": ChatPrompt(
            messages=[
                {"role": "system", "content": "You are a disaster monitoring assistant."},
                {"role": "user", "content": """Search for ONE natural disaster or emergency that happened recently worldwide (could be from today, yesterday, last week, or even a few weeks ago).

DO NOT report on: {exclude_list}

Find ONE different disaster that hasn't been mentioned yet - it could be:
- Earthquake
- Flood
- Wildfire
- Storm/Hurricane/Cyclone
- Tsunami
- Volcanic eruption
- Landslide
- Any other major disaster

The disaster doesn't need to be from today - just find ANY disaster that happened in recent times that we haven't covered yet.
Provide complete information including SPECIFIC LOCATION (city/region/country), impact, casualties, and current status.
Include at least one reliable news source URL."""}
            ],
            model="openai/gpt-4o-search-preview"
        ),
        "extract_location": ChatPrompt(
            messages=[
                {"role": "system", "content": "You are a location extraction assistant."},
                {"role": "user", "content": """Extract the PRIMARY location (city, region, or country) affected by this disaster.

DISASTER INFO:
{disaster_content}

Return ONLY the location name, nothing else. Examples:
- "Madagascar"
- "Chile"
- "Tokyo, Japan"
- "California, USA"

Location:"""}
            ],
            model="openai/gpt-4o"
        ),
        "calculate_relief": ChatPrompt(
            messages=[
                {"role": "system", "content": "You are a disaster relief calculator."},
                {"role": "user", "content": """You are a disaster relief calculator. Based on the disaster information and current weather conditions, estimate the relief amount needed in USD.

DISASTER INFO:
{disaster_content}

CURRENT WEATHER CONDITIONS:
{weather_summary}

Consider:
1. Scale of disaster (casualties, displaced people, damage)
2. Current weather (harsh conditions = more relief needed)
3. Geographic area affected
4. Infrastructure damage
5. Population density

Provide a realistic relief amount estimate in USD. Output format:
ONLY output a number (e.g., 5000000 for $5M, 250000 for $250K)

Amount in USD:"""}
            ],
            model="openai/gpt-4o"
        )
    }
    return prompts


def extract_prompts_from_coordinator_agent():
    """Extract current prompts from ResponseCoordinatorAgent"""
    if not OPIK_OPTIMIZER_AVAILABLE:
        return {}
    
    prompts = {
        "search_contacts": ChatPrompt(
            messages=[
                {"role": "system", "content": "You are helping coordinate disaster response outreach."},
                {"role": "user", "content": """You are helping coordinate disaster response outreach.

DISASTER LOCATION (focus area): {location}

DISASTER SUMMARY:
{disaster_summary}

SOURCES (for context):
{sources_text}

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
}}"""}
            ],
            model="openai/gpt-4o-search-preview"
        ),
        "draft_emails": ChatPrompt(
            messages=[
                {"role": "system", "content": "You are drafting urgent disaster outreach emails."},
                {"role": "user", "content": """Draft urgent outreach emails that reflect the SERIOUSNESS of the disaster and ask the recipient to HELP people affected.

DISASTER LOCATION: {location}
DISASTER SUMMARY:
{disaster_summary}

SOURCES:
{sources_text}

OPTIONAL RELIEF CONTEXT:
- Relief estimate (USD): {relief_usd}
- Relief estimate (ETH): {relief_eth}
- Donation vault address: {vault_address}

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
}}"""}
            ],
            model="openai/gpt-4o"
        )
    }
    return prompts


def extract_prompts_from_verification_agent():
    """Extract current prompts from VerificationAgent"""
    if not OPIK_OPTIMIZER_AVAILABLE:
        return {}
    
    prompts = {
        "analyze_claim": ChatPrompt(
            messages=[
                {"role": "system", "content": "You are verifying an NGO reimbursement claim."},
                {"role": "user", "content": """You are verifying an NGO reimbursement claim.

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
}}"""}
            ],
            model="openai/gpt-4o"
        ),
        "recommend_amount": ChatPrompt(
            messages=[
                {"role": "system", "content": "You recommend how much USD to release from a disaster relief vault to an NGO."},
                {"role": "user", "content": """You recommend how much USD to release from a disaster relief vault to an NGO.

VAULT_BALANCE_USD: {vault_usd}
CLAIM_AMOUNT_USD: {claim_amount}

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
{{ "recommended_amount_usd": 1234.56 }}"""}
            ],
            model="openai/gpt-4o"
        )
    }
    return prompts
