# impakt

## Important Links

- **Pitch Deck**: Coming Soon
- **Demo Video**: Coming Soon
- **Live App**: [View Here](https://impakt-iota.vercel.app)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Why It's Needed / The Problem](#why-its-needed--the-problem)
3. [What We Do to Make a Change](#what-we-do-to-make-a-change)
4. [How AI Has Enabled Us](#how-ai-has-enabled-us)
5. [How Everything Works](#how-everything-works)
   - [Disaster Detection and Analysis Pipeline](#disaster-detection-and-analysis-pipeline)
   - [Donation and Claim System](#donation-and-claim-system)
6. [Agents](#agents)
   - [Disaster Monitoring Agent](#disaster-monitoring-agent)
   - [Response Coordinator Agent](#response-coordinator-agent)
   - [Verification Agent](#verification-agent)
   - [Telegram Agent](#telegram-agent)
   - [Summary](#summary)
7. [Opik](#opik)
   - [run_opik_optimizations.py — what it does and why it matters](#run_opik_optimizationspy--what-it-does-and-why-it-matters)
   - [run_opik_tests.py — datasets, experiments, and how results are used](#run_opik_testspy--datasets-experiments-and-how-results-are-used)
   - [Traces and LLM-as-Judge scoring in all four agents](#traces-and-llm-as-judge-scoring-in-all-four-agents)

---

## Introduction

**impakt** is a completely AI-powered natural disaster response and relief platform that revolutionizes how we respond to global crises. Built on **LangGraph** workflows and powered by advanced AI agents, our platform automatically identifies the latest disaster news from across the internet, raises awareness on Twitter/X, notifies related organizations via email, and empowers people to make a real difference through donations and community engagement.

At its core, impakt leverages cutting-edge AI agents implemented as **LangGraph** workflows to:
- **Detect disasters in real-time** using OpenAI web search to scan global news sources, automatically extracting location data and filtering out already-reported incidents
- **Gather comprehensive disaster intelligence** by integrating with **Geoapify** for precise geocoding, **WeatherXM** for real-time weather data from up to 5 active weather stations, and **CoinGecko** for cryptocurrency pricing
- **Create immediate awareness** through automated Twitter/X posts (via Tweepy) and personalized email outreach to 6-10 relevant organizations in disaster locations (via SMTP)
- **Enable transparent donations** through **Ethereum smart contracts** that create on-chain vaults for each disaster, with all donations recorded publicly on the blockchain
- **Foster active communities** via automated Telegram group creation, where AI agents provide real-time assistance using disaster data from **Supabase**

Our system is continuously improved through **Opik** observability and **LLM-as-Judge** evaluation, ensuring that disaster detection, relief estimation, email quality, and claim verification improve over time. By combining artificial intelligence with decentralized technology and community-driven action, impakt ensures that help reaches those in need faster, more efficiently, and with greater transparency than ever before.

---

## Why It's Needed / The Problem

Natural disasters strike without warning, leaving communities devastated and in urgent need of assistance. Traditional disaster response systems face critical challenges that impakt addresses:

### The Critical Problems We Solve

1. **Delayed Awareness and Response**
   - Disasters often go unnoticed or underreported, especially in remote or less-covered regions
   - By the time traditional media picks up a story, critical hours or days have passed
   - **Our Solution**: Our **Disaster Monitoring Agent** (implemented as a LangGraph workflow) continuously monitors global news sources using OpenAI web search, identifying disasters within minutes. It automatically extracts locations, fetches real-time weather data from WeatherXM stations, estimates relief needs, creates Ethereum vaults, and posts alerts to Twitter/X—all in a single automated pipeline

2. **Fragmented Information and Coordination**
   - Relief efforts are often disorganized, with multiple organizations working in isolation
   - Communities lack centralized platforms to discuss needs and coordinate responses
   - **Our Solution**: Our **Telegram Agent** automatically creates dedicated Telegram groups for each disaster, pulling real-time data from Supabase. The agent maintains conversation history per topic and provides instant, empathetic responses about disaster details, weather conditions, donation status, and ways to help—all users need to do is tag the agent in the group

3. **Lack of Transparency in Donations**
   - Donors often don't know where their money goes or if it reaches those in need
   - Organizations struggle to prove their legitimacy and need for funds
   - **Our Solution**: Every disaster automatically triggers the creation of an **Ethereum smart contract vault** (via our VaultCreationTool), with all donations recorded publicly on-chain. Our **Verification Agent** uses AI-powered fact-checking with vision models to analyze claim evidence (including uploaded receipts and bills), then recommends payout amounts that go through community voting before funds are released

4. **Inefficient Fund Distribution**
   - Traditional donation systems have high overhead costs and slow distribution
   - Organizations must navigate complex application processes to receive funding
   - **Our Solution**: Our **Verification Agent** (implemented as a LangGraph workflow) streamlines the entire process: organizations submit claims with text, URLs, and images; the agent fetches and analyzes evidence, uses vision models to verify receipts/bills, recommends USD payout amounts, and provides human-readable justifications. The system includes a voting API for community-driven adjustments and automated on-chain fund release

5. **Knowledge Gaps in Disaster Response**
   - People want to help but lack information about the disaster, affected areas, and how to contribute effectively
   - **Our Solution**: Our **Telegram Agent** maintains per-topic conversation history and provides personalized responses based on real-time disaster data from Supabase. The agent can refetch the latest disaster information (including donation progress and vault addresses) on-demand, ensuring users always have current, accurate information about how to help

By using AI agents to quickly identify and create awareness about disasters around the world, enabling transparent donations that organizations can claim for financial support, and creating Telegram Communities where people can gather and discuss events with AI-guided assistance, impakt transforms disaster response from reactive to proactive, from fragmented to coordinated, and from opaque to transparent.

---

## What We Do to Make a Change

impakt creates impact through three interconnected pillars:

### 1. Community Formation

When a disaster is detected, impakt automatically creates a dedicated Telegram community for that specific event. These communities serve as:

- **Information Hubs**: Real-time updates about the disaster, affected areas, and relief efforts
- **Coordination Centers**: Where volunteers, donors, and organizations can connect and collaborate
- **AI-Guided Support**: An AI agent is always available in the community, ready to answer questions, provide detailed information about the incident, weather conditions, and suggest ways to help—simply by tagging it

This community-driven approach ensures that help is organized, informed, and effective.

### 2. Awareness Creation

Speed matters in disaster response. impakt's AI agents work in concert to ensure maximum visibility:

- **Automatically post to Twitter/X** via our Disaster Monitoring Agent using Tweepy, generating tweets (≤280 chars) that include disaster info, relief amounts (USD + ETH), vault address, and source URLs. The agent validates tweet content before posting to ensure quality.

- **Send targeted emails to related organizations** through our **Response Coordinator Agent** (a LangGraph sub-agent). This agent uses OpenAI web search to identify 6-10 reputable organizations in the disaster location (emergency management, local government, hospitals, NGOs, Red Cross, etc.), then drafts personalized, action-oriented emails (3-6 action items) for up to 8 contacts and sends them via SMTP.

- **Include comprehensive information** in both channels: tweets and emails contain disaster location, summary, weather conditions from WeatherXM stations, relief amounts, and vault addresses for direct donation links.

- **Maintain quality through evaluation**: All email drafts are evaluated by LLM-as-Judge for urgency, clarity, relevance, professionalism, and action items, with scores tracked via Opik for continuous improvement.

By automating awareness creation through integrated LangGraph workflows, we ensure that critical information reaches the right people at the right time—both publicly through social media and directly to organizations that can provide immediate relief—mobilizing support when it matters most.

### 3. Financial Support

impakt enables transparent, efficient financial support through a fully automated, AI-verified system:

- **Blockchain-Based Donations**: When a disaster is detected, our Disaster Monitoring Agent automatically creates an **Ethereum smart contract vault** (via VaultCreationTool), converting estimated relief amounts from USD to ETH using real-time CoinGecko prices. All donations are recorded publicly on-chain with transaction hashes and block numbers.

- **Smart Contract Vaults**: Funds are securely held in on-chain vaults until approved for distribution. The vault address is included in all awareness materials (tweets, emails) and stored in Supabase for easy access.

- **AI-Powered Claim Verification**: Our **Verification Agent** (LangGraph workflow) processes organization claims through a comprehensive pipeline: it fetches vault balance from Web3, analyzes up to 3 URLs from claim text, uses vision-capable models to verify uploaded receipts/bills, and recommends USD payout amounts based on evidence strength and vault capacity.

- **Democratic Fund Allocation**: The Verification Agent provides human-readable justifications for recommended amounts. Our voting API allows community members to adjust recommendations (higher/lower, capped at 30% of relief fund) or vote to release funds, ensuring community consensus before distribution.

- **Automated Distribution**: Once approved, the system validates vault balance, converts USD→ETH→Wei, and calls the vault's `withdraw()` function to transfer funds directly to verified organization wallets, with all transactions recorded on-chain.

This system ensures that donations reach legitimate organizations efficiently, with full transparency, AI-powered verification, and community oversight—all automated through LangGraph workflows.

---

## How AI Has Enabled Us

Artificial Intelligence is the backbone of impakt, enabling capabilities that would be impossible at scale through manual processes:

### AI-Powered Disaster Detection
- **Continuous Monitoring**: Our Disaster Monitoring Agent (LangGraph workflow) uses **OpenAI web search** (e.g. `gpt-4o-search-preview`) to scan global news sources 24/7, identifying disaster-related content in real-time while excluding already-reported disasters
- **Intelligent Filtering**: Advanced natural language processing distinguishes between actual disasters and false alarms, with LLM-as-Judge evaluation scoring disaster detection quality (accuracy, relevance, completeness, source quality, location precision) via Opik
- **Location Extraction**: A reasoning model automatically extracts the primary affected location (city/region/country) from disaster text, then **Geoapify** geocoding converts location names to precise bounding boxes (min_lat, max_lat, min_lon, max_lon) for weather data retrieval

### Automated Analysis and Evaluation
- **Weather Data Integration**: Our agents integrate with **WeatherXM API** to find active weather stations (lastDayQod > 0) within disaster bounding boxes, fetching latest readings from up to 5 stations. Weather data (temperature, humidity, precipitation, wind) is summarized and used for relief calculation
- **Funding Assessment**: A reasoning model analyzes disaster text and weather summaries to estimate relief amounts in USD, considering scale, weather conditions, area, infrastructure, and population density. Relief calculations are evaluated by LLM-as-Judge for appropriateness and reasoning quality
- **Cryptocurrency Integration**: **CoinGecko API** provides real-time ETH/USD prices, enabling automatic conversion of USD relief estimates to ETH for on-chain vault creation
- **Risk Evaluation**: The system evaluates multiple factors (vault balance, evidence strength, claim amount) to ensure responsible fund allocation, with recommendations clamped to safe ranges

### Intelligent Community Management
- **Automated Community Creation**: Our Telegram Agent automatically creates and configures Telegram groups for each disaster, pulling real-time data from **Supabase** `disaster_events` table (title, location, description, target_amount, total_donations, vault_address, etc.)
- **Real-Time Assistance**: The Telegram Agent maintains per-topic conversation history, uses personalized system prompts (with optional UserBehaviorTracker integration), and provides instant, empathetic responses using OpenAI chat completion. The agent can refetch latest disaster data on-demand when users ask for current information
- **Information Synthesis**: The agent compiles information from Supabase disaster records, presenting it in an accessible, actionable format. All responses are evaluated by LLM-as-Judge for relevance, accuracy, helpfulness, tone, and completeness, with scores tracked via Opik

### Fact-Checking and Verification
- **Claim Validation**: Our Verification Agent (LangGraph workflow) fact-checks organization claims through a multi-step pipeline: it fetches and summarizes up to 3 URLs from claim text, analyzes claim content to extract requested amounts and evidence, and uses **vision-capable models** to verify uploaded images (receipts, bills) for authenticity
- **Legitimacy Verification**: The agent validates vault addresses via Web3, reads on-chain balances, and ensures recommended amounts don't exceed vault capacity or claimed amounts. Evidence summaries are generated from both text and image analysis
- **Pattern Recognition**: LLM-as-Judge evaluation scores claim verification quality, identifying weak evidence and recommending conservative amounts when claims lack substantiation. The system clamps recommendations to safe ranges to protect vault integrity

### Social Media and Communication Automation
- **Content Generation**: Our Disaster Monitoring Agent uses a reasoning model to generate compelling tweets (≤280 chars) including disaster info, relief amounts (USD + ETH), vault address, and source URLs. The agent validates tweet content (length, required elements) before posting via **Tweepy**
- **Email Notifications**: Our Response Coordinator Agent (LangGraph sub-agent) uses OpenAI web search to identify 6-10 reputable organizations in disaster locations, then drafts personalized emails (3-6 action items, urgent tone) for up to 8 contacts. Emails are sent via SMTP with per-recipient tracking. Email quality is evaluated by LLM-as-Judge (urgency, clarity, relevance, professionalism, action items) with scores tracked via Opik
- **Multi-Platform Coordination**: The Disaster Monitoring Agent orchestrates the entire awareness pipeline: it invokes the Response Coordinator Agent for email outreach, generates and posts tweets, and stores all results (contacts, email drafts, tweet content) in state for reporting

Without AI and LangGraph workflows, impakt would require hundreds of human operators working around the clock. With our automated agent pipelines, continuous Opik observability, and LLM-as-Judge evaluation, we can respond to disasters globally, instantly, and at scale—while continuously improving through optimization and experimentation. This ensures that help reaches those in need faster than ever before, with quality that improves over time.

---

## How Everything Works

impakt operates through two main systems that work together to create a complete disaster response ecosystem.

### Disaster Detection and Analysis Pipeline

<img width="1467" height="772" alt="Screenshot 2026-02-01 at 12 52 40 AM" src="https://github.com/user-attachments/assets/93799a84-7552-415a-9ca4-d7f836fbca70" />

This pipeline automatically detects disasters, gathers comprehensive information, and prepares them for community response and funding.

#### Phase 1: Get Disaster Details, Location & Weather

1. **Search Disasters** (`search`)
   - Uses **OpenAI web search** (e.g. `gpt-4o-search-preview`) to find one recent natural disaster that has not been reported yet
   - Excludes already-reported disasters from previous runs
   - Collects disaster description and source URLs
   - **Opik tracing** and optional **LLM-as-Judge** evaluation scores disaster detection quality

2. **Extract Location** (`extract_location`)
   - Single LLM call (reasoning model) extracts the primary affected location (city/region/country) from disaster text
   - Output: short location string (e.g. "Chile", "Tokyo, Japan")

3. **Get Bounding Box** (`get_bbox`)
   - **Geoapify** geocoding converts location name to a bounding box (min_lat, max_lat, min_lon, max_lon) for the area

4. **Get Weather** (`get_weather`)
   - **WeatherXM API**: finds stations inside the bbox, fetches latest weather from up to 5 active stations (lastDayQod > 0)
   - For each selected station, calls `GET /stations/{station_id}/latest` to get latest readings (temperature, humidity, precipitation, wind, etc.)
   - Returns weather summary from successful station data for relief calculation

#### Phase 2: Analyse & Evaluate Funding Value

1. **Calculate Relief** (`calculate_relief`)
   - LLM call: given disaster text and weather summary, estimates relief amount in USD (single number)
   - Considers scale, weather, area, infrastructure, population density
   - **Opik tracing** and optional **LLM-as-Judge** evaluation scores relief appropriateness and reasoning

2. **Get ETH Price** (`get_eth_price`)
   - **CoinGecko API**: fetches current ETH/USD price (used to convert USD relief to ETH)

3. **Convert to ETH** (`convert_to_eth`)
   - Pure logic: `relief_usd / eth_price` → `relief_amount_eth`

4. **Create Vault** (`create_vault`)
   - **VaultCreationTool**: connects to Ethereum (e.g. Sepolia), calls factory `createVault(disaster_name, relief_amount_wei)`
   - Returns vault address, transaction hash, block number

5. **Save to Supabase** (`save_to_supabase`)
   - Extracts title, description, occurred_at from disaster text (via LLM) and inserts a row into `disaster_events` table
   - Stores: title, description, location, occurred_at, vault_address, target_amount, read_more_link, etc.

#### Phase 3: Create Awareness

1. **Coordinate Outreach** (`coordinate_outreach`)
   - Invokes the **Response Coordinator Agent** (LangGraph sub-agent) with a packet: location, disaster summary, sources, relief amounts, vault address
   - Response Coordinator Agent:
     - **Search Contacts** (`search_contacts`): Uses OpenAI web search to find 6-10 reputable organizations/people in the disaster location (emergency management, local government, hospitals, NGOs, Red Cross, etc.). Returns strict JSON with contacts array (name, role, organization, email, website, why_relevant, etc.). Filters to only contacts with valid emails.
     - **Draft Emails** (`draft_emails`): For each contact (up to 8), reasoning model produces personalized email draft. Rules: concise, urgent, action-oriented (3-6 action items), signature "Sincerely, The Hand". LLM-as-Judge evaluates email quality (urgency, clarity, relevance, professionalism, action items).
     - **Send Emails** (`send_emails`): Sends each drafted email via SMTP (config from env), records per-recipient result (sent/failed).

2. **Generate Tweet** (`generate`)
   - LLM generates a single tweet (≤280 chars) including disaster info, relief amount (USD + ETH), vault address, and source URL

3. **Validate Tweet** (`validate`)
   - Checks length ≤280, presence of source URL, relief amount, and vault address
   - If invalid and attempts < 3, routing goes back to `generate`; otherwise proceeds to post

4. **Post to Twitter** (`post`)
   - **TwitterTool** (Tweepy): posts the validated tweet to Twitter/X

5. **Display Result** (`display`)
   - Prints final report (location, relief, vault, weather stations, disaster info, outreach contacts/drafts, tweet)
   - Appends the reported location to `reported_disasters` so the next run skips it

**Result**: A complete disaster response setup—from detection to awareness (both public Twitter posts and direct organizational email outreach) to community formation—all automated through LangGraph workflows and ready for action.

### Donation and Claim System

<img width="1467" height="772" alt="Screenshot 2026-02-01 at 12 52 50 AM" src="https://github.com/user-attachments/assets/ad49fcc2-edee-41a3-8003-1d19bc6f727b" />

This system manages the flow of donations and ensures transparent, democratic fund distribution to organizations.

#### When You Want to Help! (Donation Process)

1. **Donor Contribution**
   - Donors contribute funds to the Vault smart contract
   - All donations are recorded publicly on the blockchain
   - Donors can optionally join an AI-assisted Telegram community dedicated to helping the cause

2. **Vault Smart Contract**
   - Securely holds all donated funds
   - Maintains a public, transparent record of all donations
   - Ensures funds cannot be accessed without proper approval

#### When You Want Help (Organization Claim Process)

1. **Organization Submission**
   - Organizations (Orgs) submit claims for funds
   - Claims include details about their relief efforts and funding needs

2. **Verification Agent** (LangGraph workflow via POST `/verify`)
   - **Prepare** (`prepare`): Extracts up to 3 URLs from claim text
   - **Fetch Vault Balance** (`vault`): Validates vault_address, reads balance from Web3 (Ethereum RPC), fetches CoinGecko ETH price, computes vault balance in USD
   - **Fetch Links** (`links`): HTTP GET each URL, stores snippet (first 5000 chars) per URL
   - **Summarize Links** (`links_summary`): LLM summarizes URL content relevant to claim verification
   - **Analyze Claim** (`claim`): LLM extracts requested amount (if any), claim summary (impact/actions), and evidence summary from claim text and link snippets
   - **Analyze Images** (`images`): If images uploaded (bills/receipts), vision-capable model summarizes what they show and whether they support the claim
   - **Recommend Amount** (`recommend`): LLM recommends single USD number based on vault_balance_usd, claim_amount_usd, claim_summary, evidence_summary. Rules: recommendation ≤ vault balance; if evidence weak, conservative; if claim amount given, recommendation ≤ claim amount. Clamped to safe ranges. LLM-as-Judge evaluates claim verification quality.
   - **Explain Decision** (`explain`): LLM produces human-readable justification (bullet points) for recommended amount

3. **Community Voting**
   - Claim becomes open for voting
   - Voters (community members) review and vote on the claim
   - Democratic process ensures community consensus

3. **Voting API** (POST `/vote`)
   - **Adjustment Mode**: `vote: "higher" | "lower"` with claim_submitted, relief_fund, disaster_details
     - LLM suggests delta_usd (positive for higher, negative for lower), capped at 30% of relief_fund
     - Returns updated_relief_fund, delta, reasoning_summary
     - Optional LLM-as-Judge evaluates vote adjustment
   - **Release Mode**: `vote: "release"` with wallet_address, amount, vault_address
     - Validates vault and balance, converts amount USD→ETH→Wei
     - Checks that server is vault creator, then calls vault `withdraw(to_wallet, amount_wei)`
     - Returns transaction details

4. **Fund Distribution**
   - **IF ACCEPT/RELEASE**: Voting API approves and executes on-chain transfer
   - `withdraw()` function transfers funds from Vault smart contract to organization wallet
   - Funds are sent directly to the verified organization with transaction hash recorded
   - **IF REJECT**: No funds are sent, and the claim is closed

**Result**: A transparent, democratic system where donations are securely held and distributed only to verified, community-approved organizations, ensuring that funds reach legitimate relief efforts.

---

# Agents

This document describes the agent pipelines in the IMPAKT disaster relief system in detail.

---

## Disaster Monitoring Agent

<img width="758" height="444" alt="image" src="https://github.com/user-attachments/assets/2688191d-8bef-4ed3-95c1-fa2e264c133d" />

**File:** `disasterAgent.py`

The Disaster Monitoring Agent discovers recent disasters, estimates relief needs, creates on-chain vaults, coordinates outreach, and posts alerts. It is implemented as a **LangGraph** workflow with multiple tools and a sub-agent (Response Coordinator).

### Pipeline (step-by-step)

1. **Search disasters** (`search`)  
   - Uses **OpenAI web search** (e.g. `gpt-4o-search-preview`) to find **one** recent natural disaster in the world (earthquake, flood, storm, etc.) that has not been reported yet.  
   - Excludes already-reported disasters.  
   - Collects disaster description and source URLs.  
   - **Opik:** `OpikTracer` around `disaster_monitoring.search_disasters`; optional **LLM-as-Judge** evaluates disaster detection (accuracy, relevance, completeness, source quality, location precision).

2. **Extract location** (`extract_location`)  
   - Single LLM call (reasoning model) to extract the **primary affected location** (city/region/country) from the disaster text.  
   - Output is a short location string (e.g. `"Chile"`, `"Tokyo, Japan"`).

3. **Get bounding box** (`get_bbox`)  
   - **Geoapify** geocoding: converts location name to a bounding box (`min_lat`, `max_lat`, `min_lon`, `max_lon`) for the area.

4. **Get weather** (`get_weather`)  
   - **WeatherXM** API: finds stations inside the bbox, fetches latest weather from up to 5 stations.  
   - **Stations:** Calls `GET /stations/bounds` with the bbox (min_lat, max_lat, min_lon, max_lon); returns a list of stations (each has `id`, `name`, `lastDayQod`). Only stations with `lastDayQod` > 0 (active/recent data) are kept; up to 5 are chosen at random.  
   - **Latest reading:** For each selected station, calls `GET /stations/{station_id}/latest`. The response is stored as the station’s `data` (WeatherXM’s latest reading; typically includes temperature, humidity, precipitation, wind, and other metrics as returned by their API).  
   - **Return shape:** `{ "bbox": bbox, "successful": [ { "stationId", "stationName", "data": <latest JSON> }, ... ], "failed": [ { "stationId", "error" }, ... ] }`.  
   - Builds a weather summary from the `successful` entries (e.g. first 3 stations’ `data`) used later for relief calculation.

5. **Calculate relief** (`calculate_relief`)  
   - LLM call: given disaster text and weather summary, estimates **relief amount in USD** (single number).  
   - Considers scale, weather, area, infrastructure, population density.  
   - **Opik:** `OpikTracer` around `disaster_monitoring.calculate_relief`; optional **LLM-as-Judge** evaluates relief appropriateness and reasoning.

6. **Get ETH price** (`get_eth_price`)  
   - **CoinGecko** API: fetches current ETH/USD price (used to convert USD relief to ETH).

7. **Convert to ETH** (`convert_to_eth`)  
   - Pure logic: `relief_usd / eth_price` → `relief_amount_eth`.

8. **Create vault** (`create_vault`)  
   - **VaultCreationTool**: connects to Ethereum (e.g. Sepolia), calls factory `createVault(disaster_name, relief_amount_wei)`.  
   - Returns vault address, tx hash, block number.

9. **Save to Supabase** (`save_to_supabase`)  
   - Extracts title, description, occurred_at from disaster text (via LLM) and inserts a row into `disaster_events` (title, description, location, occurred_at, vault_address, target_amount, read_more_link, etc.).

10. **Coordinate outreach** (`coordinate_outreach`)  
    - Invokes the **Response Coordinator Agent** with a packet: location, disaster summary, sources, relief amounts, vault address.  
    - That agent finds contacts and drafts/sends emails; results (contacts + email drafts) are stored in state as `outreach`.

11. **Generate tweet** (`generate`)  
    - LLM generates a single tweet (≤280 chars) including disaster info, relief amount (USD + ETH), vault address, and source URL.

12. **Validate tweet** (`validate`)  
    - Checks length ≤280, presence of source URL, relief amount, and vault address.  
    - If invalid and attempts < 3, routing goes back to `generate`; otherwise proceeds to post.

13. **Post to Twitter** (`post`)  
    - **TwitterTool** (Tweepy): posts the tweet.

14. **Display result** (`display`)  
    - Prints final report (location, relief, vault, weather stations, disaster info, outreach contacts/drafts, tweet).  
    - Appends the reported location to `reported_disasters` so the next run skips it.

### Tools and dependencies

- **OpenAI** (search + reasoning), **LangGraph**, **Geoapify**, **WeatherXM**, **CoinGecko**, **Tweepy**, **Web3/eth_account** (Ethereum), **Supabase**, **ResponseCoordinatorAgent**.  
- Opik: `opik_integration`, `opik_langgraph_helper` (tracing, spans, feedback).

### Entrypoint and execution

- **Entry:** `run_single_check(state)` for one cycle, or `run_continuous_monitoring(interval_seconds)` for a loop.  
- State is a TypedDict (`AgentState`) carrying iteration, reported_disasters, current_result, disaster_info, location, bbox, weather_data, relief amounts, vault_info, outreach, tweet fields, etc.  
- When Opik is enabled, the compiled graph is invoked with `config={"callbacks": [opik_tracer]}` so the whole run is traced as a LangGraph trace.

---

## Response Coordinator Agent

<img width="1284" height="506" alt="543179367-ab5d928c-1952-4320-b632-b59b68373480" src="https://github.com/user-attachments/assets/94f437e6-a72e-48ae-9b42-e0f2c49d3f24" />

**File:** `responseCoordinatorAgent.py`

The Response Coordinator Agent finds relevant responders (NGOs, agencies, hospitals) for a disaster location and drafts/sends personalized outreach emails. It is used as a **sub-agent** by the Disaster Monitoring Agent but can also be run standalone with a packet (location, disaster summary, sources, relief amounts, vault address). NOTE: The email used for the demo was a personal email address.

### Pipeline (step-by-step)

1. **Search contacts** (`search_contacts`)  
   - **OpenAI web search**: given disaster location and summary (and optional sources), asks for 6–10 reputable organizations/people in or responsible for that location (emergency management, local government, hospitals, NGOs, Red Cross, etc.).  
   - Response must be **strict JSON** with a `contacts` array; each contact has name, role, organization, location_scope, email, contact_form_url, website, phone, why_relevant, evidence_url.  
   - Parsing uses a best-effort JSON extractor (direct parse, ```json blocks, or first `{` to last `}`).  
   - **Filter:** only contacts with a valid email are kept (rest are dropped for the email step).  
   - **Opik:** span for contact search LLM; optional feedback (e.g. contact_search_quality) can be sent to the current trace.

2. **Draft emails** (`draft_emails`)  
   - For each contact (up to 8), the **reasoning model** produces one email draft.  
   - Prompt includes: disaster location, summary, sources, optional relief/vault context, and the contacts JSON.  
   - Rules: concise, urgent, action-oriented (3–6 action items), no “we offer support” framing; signature must be “Sincerely, The Hand”.  
   - Output is **strict JSON**: `emails` array with to_name, to_role, to_org, to_email, contact_form_url, subject, body, notes.  
   - Only drafts whose `to_email` matches a contact’s email are kept.  
   - **Opik:** span for email-drafting LLM; optional **LLM-as-Judge** evaluates email quality (urgency, clarity, relevance, professionalism, action items) and scores can be sent to the trace.

3. **Send emails** (`send_emails`)  
   - For each drafted email: sends via **SMTP** (config from env: SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL).  
   - Records per-recipient result (sent / failed with error).

### State and graph

- **CoordinatorState** TypedDict: location, disaster_summary, sources, relief_amount_usd, relief_amount_eth, vault_address; contacts_raw, contacts; email_drafts_raw, email_drafts; send_results; created_at.  
- **Graph:** `search_contacts` → `draft_emails` → `send_emails` → END.  
- **Entry:** `run(packet)` where `packet` is a dict with the input fields above; state is built from the packet and the graph is invoked (with Opik tracer in config when available).

### Dependencies

- **OpenAI**, **LangGraph**, **python-dotenv**, SMTP (smtplib, email).  
- Opik: same integration and helper as Disaster Monitoring; LangGraph tracer created after compile with `get_graph(xray=True)`.

---

## Verification Agent

**File:** `verificationAgent.py`

The Verification Agent evaluates NGO reimbursement claims against a disaster relief vault and recommends a USD payout amount. It does **not** perform on-chain transfers; it only outputs a **recommended_amount_usd**. The same codebase also exposes a **vote** API for payout adjustment (higher/lower) and for consensus-based release from a vault.

### Pipeline (LangGraph, used by `/verify`)

1. **Prepare** (`prepare`)  
   - Extracts up to 3 URLs from the claim text; initializes url_texts, links_summary, images_summary, decision_summary.

2. **Fetch vault balance** (`vault`)  
   - Validates `vault_address`, reads balance from **Web3** (Ethereum RPC), fetches **CoinGecko** ETH price, computes vault balance in USD.  
   - State: vault_balance_eth, eth_price_usd, vault_balance_usd.

3. **Fetch links** (`links`)  
   - HTTP GET each URL; stores snippet (e.g. first 5000 chars) per URL in `url_texts` (or error message).

4. **Summarize links** (`links_summary`)  
   - LLM summarizes what the URL snippets say that is relevant to verifying the claim (bullet points).

5. **Analyze claim** (`claim`)  
   - LLM: given vault balance USD, claim text, and link snippets, extracts requested amount (if any), claim summary (impact/actions), and evidence summary.  
   - Output: claim_amount_usd (or null), claim_summary, evidence_summary.

6. **Analyze images** (`images`)  
   - If the user uploaded images (bills/receipts): **vision-capable** model summarizes what they show and whether they support the claim.  
   - Result is merged into evidence_summary (and images_summary).

7. **Recommend amount** (`recommend`)  
   - LLM: given vault_balance_usd, claim_amount_usd, claim_summary, evidence_summary, recommends a single USD number.  
   - Rules: recommendation ≤ vault balance; if evidence weak, conservative; if claim amount given, recommendation ≤ claim amount.  
   - Output is parsed as JSON `recommended_amount_usd`; then **clamped** to [0, vault_usd] and to ≤ claim_amount if present.  
   - **Opik:** span for recommend LLM; optional **LLM-as-Judge** evaluates claim verification quality and sends feedback scores to the trace.

8. **Explain decision** (`explain`)  
   - LLM produces a short, human-readable justification (bullet points) for the recommended amount—no chain-of-thought, facts only.

### API

- **POST /verify** (FastAPI): `content` (claim text), `vault_address`, optional `images` (multipart).  
  - Runs the graph above; returns JSON: `recommended_amount_usd`, `decision_summary`, `debug`.  
- **POST /vote** (JSON body):  
  - **Adjustment mode:** `claim_submitted`, `relief_fund`, `disaster_details`, `vote: "higher" | "lower"`.  
    - LLM suggests a delta_usd (positive for higher, negative for lower), capped at 30% of relief_fund; returns updated_relief_fund, delta, reasoning_summary.  
    - Optional LLM-as-Judge evaluates vote adjustment; feedback can be sent to Opik.  
  - **Release mode:** `wallet_address`, `amount`, `vault_address`, `vote: "release"` (or similar).  
    - Validates vault and balance, converts amount USD→ETH→Wei, checks that the server is the vault creator, then calls vault `withdraw(to_wallet, amount_wei)` and returns tx details.

### State and dependencies

- **VerifyState** TypedDict: content, vault_address, images, verbose, debug; urls, url_texts, links_summary; vault/balance fields; claim/evidence/images summaries; decision_summary; recommended_amount_usd.  
- **Dependencies:** FastAPI, OpenAI, LangGraph, Web3, requests, python-dotenv; Opik integration and helper.

---

## Telegram Agent

<img width="1551" height="936" alt="543180327-9a6397fa-c8c8-4e02-bf0d-d18e57d8548a" src="https://github.com/user-attachments/assets/7410925f-8431-4da9-a909-5ea3522ca142" />

**File:** `group/agent.py` (outside `agents/`)

The Telegram Agent is a **Telegram bot** that answers user questions about a specific disaster event. It uses disaster data (e.g. from Supabase), keeps conversation history per topic, and can optionally use Opik tracing and LLM-as-Judge to score response quality and feed a behavior tracker.

### Pipeline (per user message)

1. **Trace creation**  
   - If Opik is available: `start_as_current_trace(...)` with name like `telegram_bot_response_{topic_id}`, input (user_message, topic_id, user_id, disaster_id), tags, metadata, `flush=True`.

2. **Fetch updated disaster info**  
   - If the user message suggests they want “latest”/“current”/“donation”/etc., the agent can refetch the disaster row from **Supabase** `disaster_events` by id and refresh `disaster_data`.

3. **System prompt**  
   - Builds a system prompt from `disaster_data`: title, location, occurred_at, description, target_amount, total_donations, vault_address, read_more_link, tg_group_link.  
   - Instructs the model to answer only about this disaster, be empathetic and factual, and not invent information.

4. **Personalization**  
   - If a **UserBehaviorTracker** is available, `get_personalized_prompt_adjustments(user_id, base_prompt)` can modify the system prompt (e.g. from learned preferences).

5. **Conversation history**  
   - Per `topic_id`: list of messages starting with system (and possibly personalized) prompt, then alternating user/assistant.  
   - Current user message is appended.

6. **LLM call**  
   - **OpenAI** (e.g. `gpt-4o-mini`): chat completion with conversation history, temperature and max_tokens set.  
   - Optional Opik span: `start_as_current_span("telegram_bot_llm_call", ...)` with input metadata; after completion, `update_current_span` with output metadata.

7. **Trace output**  
   - If tracing: `update_current_trace(output={ bot_response, response_length }, metadata=...)`.

8. **LLM-as-Judge (optional)**  
   - If Opik and `llm_judge` are available: `evaluate_telegram_response(user_message, bot_response, disaster_context)`.  
   - Criteria: relevance, accuracy, helpfulness, tone, completeness; returns overall_score and reasoning.  
   - Score can be stored and passed to `behavior_tracker.learn_from_feedback(...)`.  
   - Feedback scores can be sent to the current trace via `update_current_trace(feedback_scores=...)`.

9. **Behavior tracking**  
   - `behavior_tracker.record_interaction(...)` with user_id, interaction_type (e.g. telegram_message), agent_name, input/output, optional satisfaction_score and metadata.

### Context and configuration

- **DisasterAgent** is instantiated per disaster (e.g. per Supabase row or topic).  
- Bot token, supergroup chat id, Supabase URL/keys, OpenAI API key come from env (or constants).  
- The bot runs in a group; messages are associated with a topic and optionally user_id.  
- **Dependencies:** python-telegram-bot, OpenAI (async), Supabase; optional Opik, UserBehaviorTracker from `user_behavior_tracker` (when run from a context that can import it).

---

## Summary

| Agent                    | Role                                                                 | Entry / API                          | Main external services                          |
|--------------------------|----------------------------------------------------------------------|--------------------------------------|-------------------------------------------------|
| Disaster Monitoring       | Find disasters, estimate relief, create vault, outreach, tweet       | `run_single_check` / continuous loop | OpenAI, Geoapify, WeatherXM, CoinGecko, Web3, Twitter, Supabase, Response Coordinator |
| Response Coordinator     | Find contacts, draft and send emails                                 | `run(packet)`                        | OpenAI (search + chat), SMTP                    |
| Verification             | Recommend USD payout for NGO claim; vote adjustment/release          | POST `/verify`, POST `/vote`         | OpenAI, Web3, CoinGecko, FastAPI                 |
| Telegram                 | Answer questions about a disaster in a Telegram group               | Telegram message handler             | OpenAI, Supabase, optional Opik & behavior tracker |

All four agents can integrate with **Opik** for tracing and, where implemented, **LLM-as-Judge** feedback scores on the same project (e.g. `disaster-monitoring`).

---

## Opik

Opik provides **observability** (traces), **evaluation** (LLM-as-Judge scoring), **experiments** (offline evaluations), and **optimization** (prompt/parameter tuning) for all four agents. This section describes how tracing and scoring work in each agent, and how the optimization and test runners fit into the system.

### run_opik_optimizations.py — what it does and why it matters

This script runs **Opik Agent Optimizer** algorithms (from `opik-optimizer`) to improve **prompts** (and optionally parameters) for the **Disaster Monitoring**, **Response Coordinator**, and **Verification** agents.

<img width="1647" height="845" alt="543177965-c8b05dec-1bbc-429a-9ac0-cdb30e29f80d" src="https://github.com/user-attachments/assets/35d81586-2fd9-4c4d-af86-4c0530bcb3d6" />

<img width="1637" height="832" alt="543177918-021646a2-7db9-471f-b933-e93622cd62e6" src="https://github.com/user-attachments/assets/f899aa33-f9e9-4a9d-8bd8-3ec3c7d693db" />

**What it does**

1. **Datasets**  
   Uses `opik_datasets.create_all_datasets()` to create or get Opik datasets:
   - **disaster-monitoring-optimization:** disaster scenarios (location, disaster type, expected relief range, etc.).
   - **response-coordinator-optimization:** location, disaster summary, sources, relief amounts, vault address, expected contact/email quality.
   - **verification-optimization:** claim text, vault balance, evidence, expected recommendation behavior.

2. **Metrics**  
   Each agent has a metric (from `opik_metrics`) that scores outputs, often using the same **LLM-as-Judge** logic as in production (e.g. disaster detection quality, relief appropriateness, email quality, claim verification quality). Metrics return `ScoreResult` (name, value, reason) so the optimizer can compare candidates.

3. **Prompt extraction**  
   `opik_optimizable_agents` extracts prompt definitions (e.g. system/user templates) from the three agents into **ChatPrompt**-style objects for the optimizer (search_disasters, calculate_relief, draft_emails, search_contacts, recommend_amount, analyze_claim).

4. **Optimization algorithms**  
   For each prompt, the script can run:
   - **MetaPrompt:** LLM critiques and rewrites the prompt; improves wording and structure.
   - **HRPO (Hierarchical Reflective Prompt Optimizer):** Batches failures, finds root causes, proposes targeted fixes.
   - **Evolutionary:** Genetic-style search over prompt variants (mutation/crossover).
   - **Few-Shot Bayesian:** Optuna-driven search over few-shot example sets (used for draft_emails).
   - **GEPA:** External GEPA-based prompt search (if installed).
   - **Parameter:** Optuna over temperature, top_p, max_tokens (used for calculate_relief).

   Environment variables control cost: `OPIK_MAX_TRIALS`, `OPIK_N_SAMPLES`, `OPIK_N_THREADS`, `OPIK_QUICK_MODE` (MetaPrompt only).

5. **Outputs**  
   - Console: initial score, final score, improvement, best-prompt preview per run.
   - **optimization_results_YYYYMMDD_HHMMSS.json:** serialized results (initial_score, final_score, improvement, best_prompt_preview) per agent/prompt/optimizer.
   - **opik_optimization_log_YYYYMMDD_HHMMSS.txt:** full log (stdout/stderr teed to file).
   - When `OPIK_API_KEY` is set, trials and traces are sent to the Opik project; you can inspect **Evaluation → Optimization runs** in the dashboard.

**Improvements and why it’s crucial**

- The optimizer **finds better prompts** (and sometimes parameters) than hand-tuning by repeatedly evaluating candidates on the same datasets and metrics. That directly improves **disaster detection**, **relief estimation**, **email quality**, and **claim verification** without changing code logic.
- The **best prompts** from the JSON and dashboard are the recommended system improvements. They are **not** auto-applied to the repo; they are **incorporated** by copying the best prompt text (or parameter set) into the corresponding agent (e.g. disasterAgent.py, responseCoordinatorAgent.py, verificationAgent.py) or by using them in A/B tests. Doing this in a controlled way avoids regressions and keeps a clear audit trail.
- Running optimizations periodically (e.g. when adding new dataset rows or after product changes) keeps agent quality aligned with metrics and reduces reliance on ad-hoc prompt edits.

**Example: Disaster relief calculator optimization (Optimization Studio)**

<img width="1661" height="895" alt="543177997-3e390517-55c5-4762-880a-119202407a5a" src="https://github.com/user-attachments/assets/3d2fe736-2850-4fc8-aaec-96978e4a79b8" />

We ran **6 optimizations across 3 datasets** (disaster monitoring, response coordinator, verification). One representative run is the **disaster relief calculator** optimization (Optimization Studio run **agricultural_lemur_2708**, completed).

- **What was optimized:** The **calculate_relief** prompt for the Disaster Monitoring Agent—the prompt that asks the LLM to estimate relief amount in USD from disaster info and weather.
- **Metric:** **disaster_monitoring_metric** (scores relief appropriateness and reasoning on the optimization dataset).
- **Trials:** The optimizer ran two prompt variants:
  - **Trial 1 — greasy_contract_1101 (baseline):** Original prompt. **System:** “You are a disaster relief calculator.” **User:** instructions to estimate relief in USD from `DISASTER INFO:` and `CURRENT WEATHER CONDITIONS:`, with the usual considerations (scale, weather, area, infrastructure, population density) and output format (single number). **Score:** **0.258** (0% change).
  - **Trial 2 — fantastic_goat_6556 (best prompt):** Same user instruction and placeholders. The **only change** was in the **System** block: in addition to “You are a disaster relief calculator.” the optimizer **added** “You are equipped to evaluate disaster relief estimates.” So the model is explicitly told it can *evaluate* its own estimates, which encourages more careful reasoning. **Score:** **0.274** (**↑ 6%** vs baseline).
- **Optimization progress:** In the Optimization Studio, the **Optimization progress** graph shows the two trials: score on the Y-axis (about 0.25–0.27), with the line going from 0.258 (greasy_contract_1101) to 0.274 (fantastic_goat_6556), and **fantastic_goat_6556** labeled as **Best prompt**.
- **Trials table:** Lists each trial with **Trial**, **Prompt** (preview), **disaster_monitoring_metric**, **Change**, and **Created**. The best trial is highlighted (e.g. green) with **↑ 6%**.
- **Compare prompts:** The **Compare prompts** modal shows **Baseline** vs **Current** (best). The **diff** highlights the system change: the added line “You are equipped to evaluate disaster relief estimates.” (e.g. in green) and the original line (e.g. in red). The metric comparison is shown as **0.258 → 0.274 +6.164%**.
- **Takeaway:** A small, targeted change to the **system** instruction (adding an “evaluate” framing) gave a **+6.164%** gain on the disaster_monitoring_metric. That best prompt is what we use when incorporating optimization results into the agent (e.g. updating the calculate_relief prompt in `disasterAgent.py`). The same flow—trials, progress graph, best prompt, Compare prompts—applies to the other 5 optimizations across the other two datasets (response coordinator, verification); each has its own run name, metric, and best prompt in the Optimization Studio.

---

### run_opik_tests.py — datasets, experiments, and how results are used

This script runs **offline evaluations** using Opik’s **evaluate()** API. Each run creates an **experiment** in the Opik project and populates the **Experiments** page so you can compare variants (e.g. baseline vs. temperature 0.7 vs. post-optimization) on the same datasets.

**What it does**

1. **Datasets (from opik_evaluations)**  
   In-memory datasets are converted to Opik format and inserted via `client.get_or_create_dataset(...)`:
   - **Disaster Detection:** e.g. 3 items (earthquake, flood in Asia, wildfire with exclusion). Each item has `input` (query, exclude_list) and `expected` (has_location, has_sources, disaster_type, etc.).
   - **Relief Calculation:** e.g. 2 items (major earthquake, localized flood) with disaster_info, weather_data, location and expected min/max amount and reasoning checks.
   - **Email Quality:** e.g. 1 item (Chile earthquake, Red Cross contact, relief amount) with expected urgency, action items, professional tone, etc.
   - **Claim Verification:** e.g. 2 items (valid claim with receipts, unsubstantiated claim) with claim text, vault balance, evidence and expected recommendation range/conservativeness.
   - **Telegram Response:** e.g. 2 items (disaster query, donation query) with user_message, disaster_context and expected relevance/helpfulness/tone.
   <img width="1666" height="876" alt="543178080-9e791ba7-4d38-4822-af74-5fcf1c8a5ad5" src="https://github.com/user-attachments/assets/aacef61d-2a72-41f4-9a6d-94f56cd3f000" />
   The script builds **experiment-specific** dataset names (e.g. including evaluation_type and test_functionality) so different runs can share or isolate data as intended.

2. **Task functions**  
   For each evaluation type, a **task** function is defined. It receives a dataset item, optionally a **variant_config** (temperature, model_name, max_tokens, etc.), and an optional **test_functionality** (e.g. only `search_disasters` or only `draft_email`). The task:
   - Disables or avoids internal Opik tracing so it does not conflict with the trace context that `evaluate()` creates.
   - Instantiates or reuses the right agent (disaster, coordinator, verification), applies variant_config (e.g. temperature override, model name), runs only the requested functionality if `test_functionality` is set, and returns a single string (or structured output) that the scoring function can consume.

3. **Scoring functions**  
   Scorers map `(dataset_item, task_output) -> score or ScoreResult`. They often use the same **LLM-as-Judge** APIs as in production (e.g. disaster detection, email quality, claim verification) so that experiment scores are comparable to trace feedback scores.

4. **evaluate() and experiment names**  
   The script calls **opik.evaluation.evaluate(** dataset=..., task=..., scoring_metrics=[...], experiment_name=..., project_name=..., experiment_config=... **)**. That creates one **experiment** per evaluation run. Experiment names include the evaluation type and variant (e.g. `"Disaster Detection - search_disasters - temp_0.2"`, `"Email Quality - draft_email - max_tokens_300"`). So the **Experiments** page in the Opik dashboard lists all runs and lets you **compare** two or more experiments (e.g. baseline vs. max_tokens_100 vs. max_tokens_300) on the same dataset.

5. **Outputs**  
   - Console and **opik_evaluation_log_YYYYMMDD_HHMMSS.txt:** progress, experiment names, experiment IDs, aggregate scores, errors.
   - **evaluation_results/opik_evaluation_report.json** and **evaluation_results/final_evaluation_report_YYYYMMDD_HHMMSS.json:** experiment IDs, scores, and high-level summary so you can tie dashboard experiments to local reports.

**Experiments page and comparison**

- In the Opik dashboard, **Evaluation → Experiments** shows every run as an experiment. Each experiment has **experiment items** (one per dataset row): input, LLM output, feedback scores, and **trace** link. So you can see exactly which trace and which scores correspond to each row.

**Example: Eight experiments we ran**

We ran **8 experiments** via `run_opik_tests.py`:

<img width="1650" height="882" alt="543178114-66992cac-9de7-49db-be10-c4749bcd9034" src="https://github.com/user-attachments/assets/d2ce4f4b-4d61-46bd-9072-5303cd685ced" />

| Experiment name | Dataset |
|-----------------|---------|
| Disaster Detection - calculate_relief - switched model | disaster-detection-evaluation-calculate_relief (2 runs) |
| Disaster Detection - extract_location - switched temperature | disaster-detection-evaluation-extract_location (2 runs) |
| Email Quality - draft_email - max_tokens_100 | email-quality-evaluation-draft_email (2 runs)  |
| Relief Calculation - calculate_relief - model_gpt4o | relief-calculation-evaluation-calculate_relief (2 runs) |

Example: **Email quality improvement: max_tokens 100 → 300 (0.68 → 0.84)**

<img width="1658" height="893" alt="543178170-389963d4-16f7-4084-a049-8d9d1d3deaa2" src="https://github.com/user-attachments/assets/11a17c66-08ff-4381-9fea-ec754f9fd4a0" />

One representative experiment is the improvement in email quality:

We compared **Email Quality - draft_email - max_tokens_100** (baseline) and **Email Quality - draft_email - max_tokens_300** in the Opik **Compare (2)** view. With **max_tokens=100**, the coordinator agent produced shorter drafts and the **email_quality (avg)** feedback score was **0.68**. With **max_tokens=300**, the score rose to **0.84** (+0.16, about +24% relative to baseline). The **Feedback scores distribution** chart in Compare (2) shows the two bars (0.68 vs 0.84). We use the higher limit when incorporating experiment results into the Response Coordinator Agent.

**How experiment results are incorporated and why it’s crucial**

- **Experiment results** (scores and comparisons) are the evidence we used to **choose** which configuration or prompt to use in production. 
- Running **run_opik_tests.py** regularly (e.g. for every candidate prompt or after run_opik_optimizations.py) gives a **regression and comparison** layer: the same datasets and metrics ensure that new variants are measured the same way and that the system keeps improving instead of drifting.

### Traces and LLM-as-Judge scoring in all four agents

<img width="1667" height="898" alt="543178233-49d5ac0d-3a17-4c2e-9df1-5aed210decfd" src="https://github.com/user-attachments/assets/16b4ef1f-fb57-4521-88d5-cf52a716610f" />

**Traces** record the flow of each run (steps, inputs/outputs, timing). **LLM-as-Judge** uses an LLM to score outputs against criteria and sends those scores to Opik as **feedback scores** on the trace, so the impakt team can see quality per run in the dashboard.

---

#### 1. Disaster Monitoring Agent

<img width="1662" height="908" alt="543178448-9c3947e2-47f6-45fa-9595-12e4c3be0a74" src="https://github.com/user-attachments/assets/59fa922f-6144-442a-b3ce-f17048e1ed6a" />

- **Traces**
  - The full LangGraph run is traced via **OpikTracer** (LangChain/LangGraph integration): `config={"callbacks": [opik_tracer]}` on `graph.invoke(...)`. The tracer is created from the compiled graph with `get_graph(xray=True)`.
  - Inside key nodes, **OpikTracer** and **OpikSpan** wrap specific operations:
    - `disaster_monitoring.search_disasters`: trace around the search step; span for `openai_web_search`.
    - `disaster_monitoring.calculate_relief`: trace around relief calculation; span for `relief_calculation_llm`.
  - **opik_langgraph_helper** is used to update node spans with input/output (`update_node_span_with_io`) and to log the current trace ID for debugging.

- **LLM-as-Judge and scoring**
  <img width="1425" height="645" alt="543178586-9e12b78d-52a4-472e-aebd-07de9bd8c90c" src="https://github.com/user-attachments/assets/1eb103b7-c332-45fc-8ca9-3825e6e7d456" />
  - **Search (disaster detection):** After the web-search response, `llm_judge.evaluate_disaster_detection(disaster_info, location, sources, ...)` runs. It returns an overall score and per-criterion scores (accuracy, relevance, completeness, source quality, location precision). These are sent to the current trace via `update_current_trace(feedback_scores=...)` (when `send_to_opik=True`), so the `disaster_monitoring.search_disasters` span shows **Trace scores** in the Opik UI.
  <img width="1427" height="545" alt="543178622-b5a67346-cba7-41d0-be2c-790691aabe0b" src="https://github.com/user-attachments/assets/4f82156d-0f77-41c6-a0f4-c653ee941d5d" />
  - **Relief calculation:** After the relief-amount LLM call, `llm_judge.evaluate_relief_calculation(disaster_info, weather_data, calculated_amount, location, ...)` runs. Again, scores are attached to the trace so the relief step has feedback in the dashboard.

---

#### 2. Response Coordinator Agent

- **Traces**
  - Same pattern: LangGraph is invoked with `config={"callbacks": [opik_tracer]}`. The graph has three nodes: search_contacts → draft_emails → send_emails.
  - **OpikSpan** is used around the contact-search LLM call and the email-drafting LLM call. **opik_langgraph_helper** updates node spans with input/output for search_contacts and draft_emails.

- **LLM-as-Judge and scoring**
  <img width="847" height="451" alt="543178677-5e1bb2f5-c713-4093-a02e-35606eeb5a12" src="https://github.com/user-attachments/assets/8b46d982-2a2b-4b31-aba3-0f49b2383ce8" />
  - **Contact search:** After parsing contacts, a simple quality score (e.g. fraction of contacts with valid email) can be computed and sent to the trace via `update_current_trace(feedback_scores=[...])`.
  - **Draft emails:** For each drafted email (e.g. first 3), `llm_judge.evaluate_email_quality(email_draft, disaster_info, contact_info, ...)` runs. Criteria include urgency, clarity, relevance, professionalism, action items. The scores are attached to the current trace so the **response_coordinator.draft_emails** span shows feedback. If draft_emails does not show scores in the UI, it is usually because the trace context (e.g. from the LangGraph tracer) is not the same one that `update_current_trace` uses—ensuring feedback is sent on the same trace that LangGraph creates fixes this.

---

#### 3. Verification Agent

- **Traces**
  - The verification graph is invoked with `config={"callbacks": [opik_tracer]}`. Nodes: prepare → vault → links → links_summary → claim → images → recommend → explain.
  - **OpikSpan** wraps the recommend-amount LLM call (`recommend_amount_llm`). **opik_langgraph_helper** is used to set input/output on the recommend node span.

- **LLM-as-Judge and scoring**
  <img width="1416" height="896" alt="543178711-e51c820d-f0b1-44ce-a7b9-4df77bf09ed0" src="https://github.com/user-attachments/assets/5428a598-dbae-4f2d-b19b-5238c21ac97c" />
  - **Recommend amount:** After computing the recommended USD amount, `llm_judge.evaluate_claim_verification(claim_text, recommended_amount, vault_balance, evidence_summary, ...)` runs. The returned overall score and per-criterion scores are sent to the current trace via `update_current_trace(feedback_scores=...)`, so the verification run shows **Trace scores** in the Opik UI.
  <img width="1423" height="654" alt="543178797-da9acfef-66cb-4fee-a7df-e1991b674342" src="https://github.com/user-attachments/assets/32431c5a-1cdc-4c1a-813c-6efac0291b90" />
  - **Vote endpoint:** For adjustment (higher/lower) and release, the vote handler is wrapped in **OpikTracer**. The adjustment path uses **OpikSpan** around the adjustment LLM call and `llm_judge.evaluate_vote_adjustment(...)`; again, feedback scores are sent to the trace so vote runs are scored in the dashboard.

---

#### 4. Telegram Agent

- **Traces**
  - Each user message is wrapped in a trace via **opik.start_as_current_trace(...)** (e.g. `telegram_bot_response_{topic_id}`), with input (user_message, topic_id, user_id, disaster_id), tags, and metadata. A span is created for the LLM call (`telegram_bot_llm_call`) with input/output metadata. The trace is updated with the bot response as output.

- **LLM-as-Judge and scoring**
  <img width="1430" height="581" alt="543178838-efadfda2-fbf6-4fb4-8c48-47e31399fb94" src="https://github.com/user-attachments/assets/31781536-8175-4243-95cd-92217712339a" />
  - After the bot replies, `llm_judge.evaluate_telegram_response(user_message, bot_response, disaster_context)` runs. It returns overall_score and criteria (relevance, accuracy, helpfulness, tone, completeness). These are sent to the current trace via `update_current_trace(feedback_scores=...)`. The score can also be passed to the behavior tracker (`learn_from_feedback`) for personalization. So each Telegram interaction can show feedback scores on its trace in the Opik dashboard.

---

**impakt** - Making disaster response faster, smarter, and more transparent through the power of AI and community.
