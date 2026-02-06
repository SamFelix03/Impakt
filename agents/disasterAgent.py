"""
Intelligent Disaster Monitoring Agent with Relief Amount Calculation
Uses LangGraph + OpenAI Web Search + Geoapify + WeatherXM + Twitter
Requires: pip install openai langgraph langchain-openai tweepy requests python-dotenv
"""

import os
import time
import re
import random
import json
import requests
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any
from openai import OpenAI
from langgraph.graph import StateGraph, END
import tweepy
from pathlib import Path
from dotenv import load_dotenv
from web3 import Web3
from eth_account import Account
from supabase import create_client, Client
from responseCoordinatorAgent import ResponseCoordinatorAgent
from opik_integration import (
    OpikTracer, OpikSpan, trace_agent_execution,
    llm_judge, OpikAgentOptimizer, OPIK_AVAILABLE, OPIK_PROJECT
)
from opik_langgraph_helper import update_node_span_with_io, update_trace_with_feedback

load_dotenv()

# Initialize OpenAI clients
search_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
reasoning_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# API Keys
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
WEATHER_XM_API_KEY = os.getenv("WEATHER_XM_API_KEY")

# Blockchain Configuration
RPC_URL = 'https://ethereum-sepolia-rpc.publicnode.com'
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
FACTORY_ADDRESS = os.getenv('FACTORY_ADDRESS')

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

FACTORY_ABI = [
    {
        "inputs": [
            {"internalType": "string", "name": "_disasterName", "type": "string"},
            {"internalType": "uint256", "name": "_reliefAmount", "type": "uint256"}
        ],
        "name": "createVault",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "string", "name": "_disasterName", "type": "string"}],
        "name": "getVaultByDisaster",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    }
]


class GeoapifyTool:
    """Tool to get bounding box coordinates from location name"""
    
    @staticmethod
    def get_bbox(location: str):
        """Get bbox coordinates for a location"""
        print(f"\n[GEOAPIFY TOOL] Getting bbox for: {location}")
        
        url = "https://api.geoapify.com/v1/geocode/search"
        params = {
            "text": location,
            "limit": 1,
            "apiKey": GEOAPIFY_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            features = data.get("features", [])
            
            if not features:
                print(f"[GEOAPIFY TOOL] ❌ Location not found")
                return None
            
            bbox = features[0].get("bbox")
            if not bbox or len(bbox) != 4:
                print(f"[GEOAPIFY TOOL] ❌ Bounding box not available")
                return None
            
            result = {
                "min_lat": bbox[1],
                "max_lat": bbox[3],
                "min_lon": bbox[0],
                "max_lon": bbox[2]
            }
            
            print(f"[GEOAPIFY TOOL] ✅ Bbox found: {result}")
            print(f"\n[GEOAPIFY OUTPUT] Bounding Box Coordinates:")
            print(f"  Min Latitude:  {result['min_lat']}")
            print(f"  Max Latitude:  {result['max_lat']}")
            print(f"  Min Longitude: {result['min_lon']}")
            print(f"  Max Longitude: {result['max_lon']}")
            print(f"  Coverage Area: {abs(result['max_lat'] - result['min_lat']):.2f}° × {abs(result['max_lon'] - result['min_lon']):.2f}°")
            
            return result
            
        except Exception as e:
            print(f"[GEOAPIFY TOOL] ❌ Error: {str(e)}")
            return None


class WeatherXMTool:
    """Tool to get weather conditions from WeatherXM stations"""
    
    BASE_URL = "https://pro.weatherxm.com/api/v1"
    
    @staticmethod
    def get_weather_data(bbox: dict):
        """Get weather data from stations in the bounding box"""
        print(f"\n[WEATHERXM TOOL] Fetching weather data for bbox: {bbox}")
        
        headers = {"X-API-KEY": WEATHER_XM_API_KEY}
        
        # Fetch stations in bbox
        bbox_url = (
            f"{WeatherXMTool.BASE_URL}/stations/bounds"
            f"?min_lat={bbox['min_lat']}&max_lat={bbox['max_lat']}"
            f"&min_lon={bbox['min_lon']}&max_lon={bbox['max_lon']}"
        )
        
        try:
            print(f"[WEATHERXM TOOL] Querying stations...")
            res = requests.get(bbox_url, headers=headers)
            res.raise_for_status()
            data = res.json()
            stations = data.get("stations", [])
            
            if not stations:
                print(f"[WEATHERXM TOOL] ⚠️ No stations found in bbox")
                return {"successful": [], "failed": [], "bbox": bbox}
            
            print(f"[WEATHERXM TOOL] Found {len(stations)} stations")
            
            # Filter active stations
            active_stations = [s for s in stations if s.get("lastDayQod", 0) > 0]
            print(f"[WEATHERXM TOOL] Active stations: {len(active_stations)}")
            
            if not active_stations:
                return {"successful": [], "failed": [], "bbox": bbox}
            
            # Select up to 5 random stations
            selected = random.sample(active_stations, min(5, len(active_stations)))
            print(f"[WEATHERXM TOOL] Selected {len(selected)} stations for data fetch")
            
            successful = []
            failed = []
            
            # Fetch latest data
            for station in selected:
                station_id = station["id"]
                station_name = station.get("name")
                
                try:
                    latest_url = f"{WeatherXMTool.BASE_URL}/stations/{station_id}/latest"
                    r = requests.get(latest_url, headers=headers)
                    r.raise_for_status()
                    
                    successful.append({
                        "stationId": station_id,
                        "stationName": station_name,
                        "data": r.json()
                    })
                    print(f"[WEATHERXM TOOL]   ✅ {station_name}")
                    
                except Exception as e:
                    failed.append({"stationId": station_id, "error": str(e)})
                    print(f"[WEATHERXM TOOL]   ❌ {station_name}: {str(e)}")
            
            result = {
                "bbox": bbox,
                "successful": successful,
                "failed": failed
            }
            
            print(f"[WEATHERXM TOOL] ✅ Completed: {len(successful)} successful, {len(failed)} failed")
            
            # Print detailed weather output
            print(f"\n[WEATHERXM OUTPUT] Weather Data Summary:")
            print(f"  Total Stations Found: {len(stations)}")
            print(f"  Active Stations: {len(active_stations)}")
            print(f"  Data Retrieved From: {len(successful)} stations")
            
            if successful:
                print(f"\n[WEATHERXM OUTPUT] Station Details:")
                for i, station in enumerate(successful, 1):
                    print(f"\n  Station {i}: {station['stationName']}")
                    print(f"    ID: {station['stationId']}")
                    data = station.get('data', {})
                    if data:
                        print(f"    Latest Data:")
                        # Print key weather metrics
                        for key, value in list(data.items())[:10]:  # First 10 fields
                            print(f"      {key}: {value}")
            
            if failed:
                print(f"\n[WEATHERXM OUTPUT] Failed Stations: {len(failed)}")
                for fail in failed:
                    print(f"    {fail['stationId']}: {fail['error']}")
            
            result = {
                "bbox": bbox,
                "successful": successful,
                "failed": failed
            }
            
            return result
            
        except Exception as e:
            print(f"[WEATHERXM TOOL] ❌ Error: {str(e)}")
            return {"successful": [], "failed": [], "bbox": bbox, "error": str(e)}


class TwitterTool:
    """Twitter posting tool using Tweepy"""
    
    def __init__(self):
        self.client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_APP_KEY"),
            consumer_secret=os.getenv("TWITTER_APP_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_SECRET")
        )
    
    def post_tweet(self, tweet_text: str):
        """Post a tweet"""
        try:
            print(f"[TWITTER TOOL] Posting tweet ({len(tweet_text)} chars)...")
            response = self.client.create_tweet(text=tweet_text)
            print(f"[TWITTER TOOL] ✅ Success! ID: {response.data['id']}")
            
            print(f"\n[TWITTER OUTPUT]")
            print(f"  Tweet ID: {response.data['id']}")
            print(f"  Tweet URL: https://twitter.com/user/status/{response.data['id']}")
            print(f"  Status: Successfully posted")
            
            return {
                "success": True,
                "tweet_id": response.data['id'],
                "tweet_text": tweet_text
            }
            
        except Exception as e:
            print(f"[TWITTER TOOL] ❌ Error: {str(e)}")
            print(f"\n[TWITTER OUTPUT]")
            print(f"  Status: Failed")
            print(f"  Error: {str(e)}")
            return {"success": False, "error": str(e)}


class CoinGeckoTool:
    """Tool to get ETH price from CoinGecko"""
    
    @staticmethod
    def get_eth_price_usd():
        """Get current ETH price in USD"""
        print(f"\n[COINGECKO TOOL] Fetching ETH price...")
        
        url = "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            eth_price = data["ethereum"]["usd"]
            print(f"[COINGECKO TOOL] ✅ ETH Price: ${eth_price:,.2f} USD")
            
            print(f"\n[COINGECKO OUTPUT]")
            print(f"  Current ETH/USD: ${eth_price:,.2f}")
            print(f"  Source: CoinGecko API")
            
            return eth_price
            
        except Exception as e:
            print(f"[COINGECKO TOOL] ❌ Error: {str(e)}")
            return None


class VaultCreationTool:
    """Tool to create disaster relief vaults on blockchain"""
    
    def __init__(self):
        """Initialize Web3 connection"""
        self.w3 = Web3(Web3.HTTPProvider(RPC_URL))
        
        if not self.w3.is_connected():
            raise Exception("Failed to connect to Ethereum node")
        
        print(f"[VAULT TOOL] ✅ Connected to Ethereum Sepolia")
        
        # Setup account
        if not PRIVATE_KEY:
            raise Exception("PRIVATE_KEY not found in .env file")
        
        private_key = PRIVATE_KEY if not PRIVATE_KEY.startswith('0x') else PRIVATE_KEY[2:]
        self.account = Account.from_key(private_key)
        
        print(f"[VAULT TOOL] Wallet: {self.account.address}")
        
        # Setup factory contract
        if not FACTORY_ADDRESS:
            raise Exception("FACTORY_ADDRESS not found in .env file")
        
        self.factory = self.w3.eth.contract(
            address=Web3.to_checksum_address(FACTORY_ADDRESS),
            abi=FACTORY_ABI
        )
    
    def create_vault(self, disaster_name: str, relief_amount_eth: float):
        """Create a new disaster vault"""
        print(f"\n[VAULT TOOL] Creating vault...")
        print(f"[VAULT TOOL] Disaster: {disaster_name}")
        print(f"[VAULT TOOL] Target: {relief_amount_eth} ETH")
        
        try:
            # Convert ETH to Wei
            relief_amount_wei = self.w3.to_wei(relief_amount_eth, 'ether')
            
            # Build transaction
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Estimate gas
            gas_estimate = self.factory.functions.createVault(
                disaster_name,
                relief_amount_wei
            ).estimate_gas({'from': self.account.address})
            
            gas_limit = int(gas_estimate * 1.2)
            
            # Get gas prices
            base_fee = self.w3.eth.gas_price
            max_priority_fee = self.w3.to_wei(2, 'gwei')
            max_fee = base_fee + max_priority_fee
            
            print(f"[VAULT TOOL] Gas estimate: {gas_estimate}")
            print(f"[VAULT TOOL] Building transaction...")
            
            tx = self.factory.functions.createVault(
                disaster_name,
                relief_amount_wei
            ).build_transaction({
                'from': self.account.address,
                'nonce': nonce,
                'gas': gas_limit,
                'maxFeePerGas': max_fee,
                'maxPriorityFeePerGas': max_priority_fee,
                'chainId': self.w3.eth.chain_id
            })
            
            # Sign and send
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            print(f"[VAULT TOOL] Transaction: {tx_hash.hex()}")
            print(f"[VAULT TOOL] Waiting for confirmation...")
            
            # Wait for receipt
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print(f"[VAULT TOOL] ✅ Confirmed in block {tx_receipt['blockNumber']}")
            
            # Get vault address from logs
            vault_address = None
            if tx_receipt['logs']:
                # The vault address is typically in the logs or can be queried
                # For now, we'll call the getter function
                vault_address = self.factory.functions.getVaultByDisaster(disaster_name).call()
            
            print(f"[VAULT TOOL] ✅ Vault created: {vault_address}")
            
            print(f"\n[VAULT CREATION OUTPUT]")
            print(f"  Disaster Name: {disaster_name}")
            print(f"  Relief Target: {relief_amount_eth} ETH")
            print(f"  Vault Address: {vault_address}")
            print(f"  Transaction Hash: {tx_hash.hex()}")
            print(f"  Block Number: {tx_receipt['blockNumber']}")
            print(f"  Gas Used: {tx_receipt['gasUsed']}")
            
            return {
                "success": True,
                "vault_address": vault_address,
                "tx_hash": tx_hash.hex(),
                "block_number": tx_receipt['blockNumber']
            }
            
        except Exception as e:
            print(f"[VAULT TOOL] ❌ Error: {str(e)}")
            print(f"\n[VAULT CREATION OUTPUT]")
            print(f"  Status: Failed")
            print(f"  Error: {str(e)}")
            
            return {
                "success": False,
                "error": str(e)
            }


# Define agent state
class AgentState(TypedDict):
    iteration: int
    reported_disasters: List[str]
    current_result: str
    disaster_info: dict
    location: str
    bbox: Optional[dict]
    weather_data: Optional[dict]
    relief_amount_usd: Optional[int]
    eth_price: Optional[float]
    relief_amount_eth: Optional[float]
    vault_info: Optional[dict]
    outreach: Optional[dict]
    tweet_draft: str
    tweet_is_valid: bool
    tweet_iterations: int
    tweet_result: dict
    all_disasters_found: List[dict]


class DisasterMonitoringAgent:
    def __init__(self, search_model="gpt-4o-search-preview", reasoning_model="gpt-4o"):
        self.search_model = search_model
        self.reasoning_model = reasoning_model
        self.geoapify_tool = GeoapifyTool()
        self.weatherxm_tool = WeatherXMTool()
        self.coingecko_tool = CoinGeckoTool()
        self.vault_tool = VaultCreationTool()
        self.twitter_tool = TwitterTool()
        self.response_coordinator = ResponseCoordinatorAgent(
            search_model=self.search_model,
            reasoning_model=self.reasoning_model
        )
        
        # Initialize Supabase client
        self.supabase: Optional[Client] = None
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                self.supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
                print(f"[INIT] ✅ Supabase client initialized")
            except Exception as e:
                print(f"[INIT] ⚠️ Failed to initialize Supabase: {e}")
        
        # Initialize Opik components
        self.opik_optimizer = OpikAgentOptimizer("disaster_monitoring") if (OPIK_AVAILABLE and OpikAgentOptimizer) else None
        self.evaluation_results = []  # Store evaluation results
        
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
    
    def _search_disasters(self, state: AgentState):
        """Search for recent disasters"""
        # Log trace ID for this step
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("SEARCH DISASTERS")
        
        # Update LangGraph node span with input data
        input_data = {
            "iteration": state.get("iteration", 0),
            "reported_disasters": state.get("reported_disasters", []),
            "reported_count": len(state.get("reported_disasters", []))
        }
        update_node_span_with_io(input_data=input_data, metadata={"node": "search_disasters"})
        
        with OpikTracer("disaster_monitoring.search_disasters", {
            "project": OPIK_PROJECT,
            "tags": [OPIK_PROJECT, "disaster_monitoring", "search"],
            **input_data
        }, project_tag=OPIK_PROJECT):
            print(f"\n{'='*80}")
            print(f"[STEP 1: SEARCH] Starting disaster search...")
            print(f"{'='*80}")
            
            reported = state["reported_disasters"]
            exclude_list = ", ".join(reported) if reported else "none"
            
            query = f"""Search for ONE natural disaster or emergency that happened recently in the world (could be from today, yesterday, last week, or even a few weeks ago).

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
Include at least one reliable news source URL."""
            
            print(f"[SEARCH] Calling OpenAI web search API...")
            
            try:
                # Add input/output to span
                with OpikSpan("openai_web_search",
                             input_data={"query_preview": query[:100], "model": self.search_model},
                             output_data={}):
                    completion = search_client.chat.completions.create(
                        model=self.search_model,
                        web_search_options={},
                        messages=[{"role": "user", "content": query}]
                    )
                    
                    # Update span with output after completion
                    if OPIK_AVAILABLE:
                        try:
                            from opik.opik_context import update_current_span
                            update_current_span(metadata={
                                "output": {
                                    "response_length": len(completion.choices[0].message.content) if completion.choices else 0,
                                    "model": self.search_model
                                }
                            })
                        except:
                            pass
                
                response = completion.choices[0].message
                sources = []
                
                if hasattr(response, 'annotations') and response.annotations:
                    for annotation in response.annotations:
                        if annotation.type == "url_citation":
                            sources.append({
                                "title": annotation.url_citation.title,
                                "url": annotation.url_citation.url
                            })
                
                print(f"[SEARCH] ✅ Found disaster with {len(sources)} sources")
                print(f"\n[SEARCH OUTPUT] Disaster Information:")
                print(f"{'-'*80}")
                print(response.content)
                print(f"{'-'*80}")
                
                if sources:
                    print(f"\n[SEARCH OUTPUT] Sources:")
                    for i, source in enumerate(sources, 1):
                        print(f"  {i}. {source['title']}")
                        print(f"     URL: {source['url']}")
                
                # Prepare output data for Opik
                output_data = {
                    "disaster_found": True,
                    "disaster_info_length": len(response.content),
                    "sources_count": len(sources),
                    "sources": [{"title": s.get("title", ""), "url": s.get("url", "")} for s in sources[:3]]
                }
                
                # Evaluate disaster detection quality
                evaluation = None
                if llm_judge:
                    with OpikSpan("evaluate_disaster_detection", 
                                 input_data={"disaster_info_preview": response.content[:200]},
                                 output_data={}):
                        evaluation = llm_judge.evaluate_disaster_detection(
                            disaster_info=response.content,
                            location="",  # Will be extracted later
                            sources=sources,
                            expected_criteria={},
                            send_to_opik=True
                        )
                        score = evaluation.get('overall_score', 0)
                        if 'error' in evaluation:
                            print(f"\n[OPIK EVALUATION] ❌ ERROR: {evaluation.get('error', 'Unknown error')}")
                            print(f"[OPIK EVALUATION] Disaster Detection Score: {score:.2f}/10 (ERROR - check logs)")
                        else:
                            print(f"\n[OPIK EVALUATION] Disaster Detection Score: {score:.2f}/10")
                        self.evaluation_results.append({
                            "step": "disaster_detection",
                            "evaluation": evaluation,
                            "timestamp": datetime.now().isoformat()
                        })
                        output_data["evaluation_score"] = evaluation.get('overall_score', 0)
                
                # Update LangGraph node span with output data
                update_node_span_with_io(
                    output_data={
                        **output_data,
                        "disaster_info_preview": response.content[:300],
                        "sources": [{"title": s.get("title", ""), "url": s.get("url", "")} for s in sources[:3]]
                    },
                    metadata={"node": "search_disasters"}
                )
                
                return {
                    "current_result": response.content,
                    "disaster_info": {"raw_response": response.content, "sources": sources},
                    "tweet_iterations": 0,
                    "tweet_is_valid": False
                }
                
            except Exception as e:
                print(f"[SEARCH] ❌ Error: {str(e)}")
                return {
                    "current_result": f"Error: {e}",
                    "disaster_info": {"error": str(e)},
                    "tweet_iterations": 0,
                    "tweet_is_valid": False
                }
    
    def _extract_location(self, state: AgentState):
        """Extract location from disaster info using AI"""
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("EXTRACT LOCATION")
        
        print(f"\n{'='*80}")
        print(f"[STEP 2: EXTRACT LOCATION] Using AI to extract location...")
        print(f"{'='*80}")
        
        disaster_content = state["current_result"]
        
        prompt = f"""Extract the PRIMARY location (city, region, or country) affected by this disaster.

DISASTER INFO:
{disaster_content}

Return ONLY the location name, nothing else. Examples:
- "Madagascar"
- "Chile"
- "Tokyo, Japan"
- "California, USA"

Location:"""
        
        try:
            # Check for temperature override (used during evaluation)
            temp = getattr(self, '_temp_override', 0)
            response = reasoning_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp
            )
            
            location = response.choices[0].message.content.strip()
            print(f"[EXTRACT LOCATION] ✅ Extracted: {location}")
            print(f"\n[EXTRACT LOCATION OUTPUT]")
            print(f"  Primary Location: {location}")
            
            return {"location": location}
            
        except Exception as e:
            print(f"[EXTRACT LOCATION] ❌ Error: {str(e)}")
            return {"location": "Unknown"}
    
    def _get_bbox(self, state: AgentState):
        """Get bounding box for the location"""
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("GET BBOX")
        
        print(f"\n{'='*80}")
        print(f"[STEP 3: GET BBOX] Fetching bounding box...")
        print(f"{'='*80}")
        
        location = state["location"]
        bbox = self.geoapify_tool.get_bbox(location)
        
        return {"bbox": bbox}
    
    def _get_weather(self, state: AgentState):
        """Get weather data for the location"""
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("GET WEATHER")
        
        print(f"\n{'='*80}")
        print(f"[STEP 4: GET WEATHER] Fetching weather data...")
        print(f"{'='*80}")
        
        bbox = state["bbox"]
        
        if not bbox:
            print(f"[GET WEATHER] ⚠️ No bbox available, skipping weather fetch")
            return {"weather_data": None}
        
        weather_data = self.weatherxm_tool.get_weather_data(bbox)
        
        return {"weather_data": weather_data}
    
    def _calculate_relief(self, state: AgentState):
        """Calculate relief amount using AI based on disaster severity and weather"""
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("CALCULATE RELIEF")
        
        # Update LangGraph node span with input data
        input_data = {
            "location": state.get("location", ""),
            "disaster_info_preview": state.get("current_result", "")[:200],
            "has_weather_data": bool(state.get("weather_data")),
            "weather_stations_count": len(state.get("weather_data", {}).get("successful", [])) if state.get("weather_data") else 0
        }
        update_node_span_with_io(input_data=input_data, metadata={"node": "calculate_relief"})
        
        with OpikTracer("disaster_monitoring.calculate_relief", {
            "project": OPIK_PROJECT,
            "tags": [OPIK_PROJECT, "disaster_monitoring", "relief_calculation"],
            **input_data
        }, project_tag=OPIK_PROJECT):
            print(f"\n{'='*80}")
            print(f"[STEP 5: CALCULATE RELIEF] Calculating relief amount...")
            print(f"{'='*80}")
            
            disaster_content = state["current_result"]
            weather_data = state["weather_data"]
            
            # Prepare weather summary
            weather_summary = "No weather data available"
            if weather_data and weather_data.get("successful"):
                stations = weather_data["successful"]
                weather_summary = f"Weather data from {len(stations)} stations:\n"
                for s in stations[:3]:  # Use first 3 stations
                    data = s.get("data", {})
                    weather_summary += f"- {s['stationName']}: {data}\n"
            
            prompt = f"""You are a disaster relief calculator. Based on the disaster information and current weather conditions, estimate the relief amount needed in USD.

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
Provide a realistic relief amount estimate within 200000 to 1000000 USD.

Amount in USD:"""
            
            try:
                # Add input/output to span
                with OpikSpan("relief_calculation_llm",
                             input_data={"prompt_preview": prompt[:150], "model": self.reasoning_model},
                             output_data={}):
                    # Check for temperature override (used during evaluation)
                    temp = getattr(self, '_temp_override', 0.3)
                    response = reasoning_client.chat.completions.create(
                        model=self.reasoning_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temp
                    )
                    
                    # Update span with output
                    if OPIK_AVAILABLE:
                        try:
                            from opik.opik_context import update_current_span
                            amount_str = response.choices[0].message.content.strip()
                            update_current_span(metadata={
                                "output": {
                                    "raw_response": amount_str,
                                    "model": self.reasoning_model
                                }
                            })
                        except:
                            pass
                
                amount_str = response.choices[0].message.content.strip()
                # Extract number from response
                amount = int(re.sub(r'[^\d]', '', amount_str))
                
                print(f"[CALCULATE RELIEF] ✅ Estimated relief: ${amount:,} USD")
                
                print(f"\n[RELIEF CALCULATION OUTPUT]")
                print(f"  Raw AI Response: {amount_str}")
                print(f"  Parsed Amount: ${amount:,} USD")
                print(f"  Breakdown:")
                print(f"    - Disaster Severity: Analyzed from casualty/damage reports")
                print(f"    - Weather Impact: Based on {len(weather_data.get('successful', [])) if weather_data else 0} weather stations")
                print(f"    - Geographic Scope: Derived from affected area")
                
                # Prepare output data
                output_data = {
                    "relief_amount_usd": amount,
                    "relief_amount_formatted": f"${amount:,}",
                    "calculation_method": "ai_based",
                    "weather_data_used": bool(weather_data)
                }
                
                # Evaluate relief calculation quality
                evaluation = None
                if llm_judge:
                    with OpikSpan("evaluate_relief_calculation",
                                 input_data={"calculated_amount": amount, "location": state.get("location", "")},
                                 output_data={}):
                        evaluation = llm_judge.evaluate_relief_calculation(
                            disaster_info=disaster_content,
                            weather_data=weather_data or {},
                            calculated_amount=amount,
                            location=state.get("location", ""),
                            send_to_opik=True
                        )
                        score = evaluation.get('overall_score', 0)
                        if 'error' in evaluation:
                            print(f"\n[OPIK EVALUATION] ❌ ERROR: {evaluation.get('error', 'Unknown error')}")
                            print(f"[OPIK EVALUATION] Relief Calculation Score: {score:.2f}/10 (ERROR - check logs)")
                        else:
                            print(f"\n[OPIK EVALUATION] Relief Calculation Score: {score:.2f}/10")
                        if evaluation.get('suggested_range'):
                            print(f"  Suggested Range: ${evaluation['suggested_range'].get('min', 0):,} - ${evaluation['suggested_range'].get('max', 0):,}")
                        self.evaluation_results.append({
                            "step": "relief_calculation",
                            "evaluation": evaluation,
                            "calculated_amount": amount,
                            "timestamp": datetime.now().isoformat()
                        })
                        output_data["evaluation_score"] = evaluation.get('overall_score', 0)
                        if evaluation.get('suggested_range'):
                            output_data["suggested_range"] = evaluation['suggested_range']
                
                # Update LangGraph node span with output data
                if OPIK_AVAILABLE:
                    try:
                        from opik.opik_context import update_current_span
                        from opik.exceptions import OpikException
                        update_current_span(metadata={
                            "output": output_data,
                            "relief_amount_usd": amount,
                            "relief_amount_formatted": f"${amount:,}"
                        })
                    except OpikException as e:
                        # Silently skip if no span context (expected when called directly)
                        if "no span in the context" not in str(e).lower():
                            print(f"[OPIK] Failed to update span: {e}")
                    except Exception as e:
                        # Only print if not about missing span context
                        if "no span in the context" not in str(e).lower():
                            print(f"[OPIK] Failed to update span: {e}")
                
                return {"relief_amount_usd": amount}
                
            except Exception as e:
                print(f"[CALCULATE RELIEF] ❌ Error: {str(e)}, using default")
                return {"relief_amount_usd": 1000000}  # Default $1M
    
    def _get_eth_price(self, state: AgentState):
        """Get current ETH price from CoinGecko"""
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("GET ETH PRICE")
        
        print(f"\n{'='*80}")
        print(f"[STEP 6: GET ETH PRICE] Fetching ETH/USD rate...")
        print(f"{'='*80}")
        
        eth_price = self.coingecko_tool.get_eth_price_usd()
        
        if not eth_price:
            print(f"[GET ETH PRICE] ⚠️ Using default price: $3000")
            eth_price = 3000  # Fallback price
        
        return {"eth_price": eth_price}
    
    def _convert_to_eth(self, state: AgentState):
        """Convert USD relief amount to ETH"""
        print(f"\n{'='*80}")
        print(f"[STEP 7: CONVERT TO ETH] Converting USD to ETH...")
        print(f"{'='*80}")
        
        relief_usd = state["relief_amount_usd"]
        eth_price = state["eth_price"]
        
        relief_eth = relief_usd / eth_price
        
        print(f"[CONVERT] Relief in USD: ${relief_usd:,}")
        print(f"[CONVERT] ETH Price: ${eth_price:,.2f}")
        print(f"[CONVERT] Relief in ETH: {relief_eth:.6f} ETH")
        
        print(f"\n[CONVERSION OUTPUT]")
        print(f"  Amount in USD: ${relief_usd:,}")
        print(f"  ETH/USD Rate: ${eth_price:,.2f}")
        print(f"  Amount in ETH: {relief_eth:.6f} ETH")
        print(f"  Conversion: ${relief_usd:,} ÷ ${eth_price:,.2f} = {relief_eth:.6f} ETH")
        
        return {"relief_amount_eth": relief_eth}
    
    def _create_vault(self, state: AgentState):
        """Create blockchain vault for donations"""
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("CREATE VAULT")
        
        print(f"\n{'='*80}")
        print(f"[STEP 8: CREATE VAULT] Creating blockchain vault...")
        print(f"{'='*80}")
        
        location = state["location"]
        relief_eth = state["relief_amount_eth"]
        
        # Create disaster name (truncate if too long)
        disaster_name = f"{location} Relief {datetime.now().strftime('%Y')}"[:50]
        
        vault_result = self.vault_tool.create_vault(disaster_name, relief_eth)
        
        return {"vault_info": vault_result}
    
    def _generate_tweet(self, state: AgentState):
        """Generate tweet with relief amount and vault address"""
        print(f"\n{'='*80}")
        print(f"[STEP 9: GENERATE TWEET] Creating tweet...")
        print(f"{'='*80}")
        
        disaster_content = state["current_result"]
        sources = state["disaster_info"].get("sources", [])
        relief_usd = state["relief_amount_usd"]
        relief_eth = state["relief_amount_eth"]
        location = state["location"]
        vault_info = state.get("vault_info", {})
        vault_address = vault_info.get("vault_address", "N/A") if vault_info.get("success") else "N/A"
        
        source_url = sources[0]["url"] if sources else "https://example.com"
        
        # Calculate available space
        # We need to fit: disaster info + relief amount + vault address + source URL
        url_part = f"\n\n🔗 {source_url}"
        vault_part = f"\n💰 Vault: {vault_address}" if vault_address != "N/A" else ""
        
        fixed_parts_length = len(url_part) + len(vault_part)
        available_chars = 280 - fixed_parts_length
        
        print(f"[GENERATE] Relief: ${relief_usd:,} USD ({relief_eth:.6f} ETH)")
        print(f"[GENERATE] Vault: {vault_address}")
        print(f"[GENERATE] Available chars: {available_chars}")
        
        prompt = f"""Create a disaster alert tweet.

DISASTER: {disaster_content}
LOCATION: {location}
RELIEF NEEDED: ${relief_usd:,} USD ({relief_eth:.6f} ETH)
VAULT ADDRESS: {vault_address}
SOURCE URL: {source_url}

REQUIREMENTS:
- Include: 🚨 emoji, disaster type, location
- MUST include relief amount in both USD and ETH format: "Relief: $X (Y ETH)"
- MUST include vault address line: "\\n💰 Vault: {vault_address}"
- MUST include source URL line: "\\n\\n🔗 {source_url}"
- Total ≤280 chars
- Available for main content: {available_chars} chars
- Be urgent and impactful

The vault address and URL parts are FIXED and must be included exactly as shown.

Output ONLY the complete tweet, nothing else."""
        
        try:
            response = reasoning_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            tweet = response.choices[0].message.content.strip()
            print(f"[GENERATE] ✅ Tweet: {len(tweet)} chars")
            
            print(f"\n[TWEET GENERATION OUTPUT]")
            print(f"  Iteration: {state['tweet_iterations'] + 1}")
            print(f"  Tweet Length: {len(tweet)} characters")
            print(f"  Tweet Content:")
            print(f"  {'-'*76}")
            print(f"  {tweet}")
            print(f"  {'-'*76}")
            
            return {
                "tweet_draft": tweet,
                "tweet_iterations": state["tweet_iterations"] + 1
            }
            
        except Exception as e:
            print(f"[GENERATE] ❌ Error: {str(e)}")
            fallback = f"🚨 Disaster in {location}\nRelief: ${relief_usd:,} ({relief_eth:.4f} ETH)\n💰 Vault: {vault_address}\n\n🔗 {source_url}"
            return {
                "tweet_draft": fallback,
                "tweet_iterations": state["tweet_iterations"] + 1
            }
    
    def _validate_tweet(self, state: AgentState):
        """Validate tweet"""
        print(f"\n{'='*80}")
        print(f"[STEP 10: VALIDATE] Checking tweet...")
        print(f"{'='*80}")
        
        tweet = state["tweet_draft"]
        sources = state["disaster_info"].get("sources", [])
        source_url = sources[0]["url"] if sources else ""
        relief_usd = state["relief_amount_usd"]
        relief_eth = state["relief_amount_eth"]
        vault_info = state.get("vault_info", {})
        vault_address = vault_info.get("vault_address", "") if vault_info.get("success") else ""
        
        length_ok = len(tweet) <= 280
        has_url = source_url in tweet if source_url else False
        has_relief = (str(relief_usd) in tweet or f"${relief_usd:,}" in tweet) and (str(relief_eth)[:6] in tweet or "ETH" in tweet)
        has_vault = vault_address in tweet if vault_address else True  # Skip if no vault
        
        is_valid = length_ok and has_url and has_relief and has_vault
        
        print(f"[VALIDATE] Length: {len(tweet)}/280 ✅" if length_ok else f"[VALIDATE] Length: {len(tweet)}/280 ❌")
        print(f"[VALIDATE] Has URL: {'✅' if has_url else '❌'}")
        print(f"[VALIDATE] Has relief amount: {'✅' if has_relief else '❌'}")
        print(f"[VALIDATE] Has vault address: {'✅' if has_vault else '❌'}")
        print(f"[VALIDATE] Valid: {'✅' if is_valid else '❌'}")
        
        print(f"\n[VALIDATION OUTPUT]")
        print(f"  Checks:")
        print(f"    ✓ Length check: {len(tweet)}/280 chars - {'PASS' if length_ok else 'FAIL'}")
        print(f"    ✓ URL present: {'PASS' if has_url else 'FAIL'}")
        if not has_url and source_url:
            print(f"      Expected URL: {source_url}")
        print(f"    ✓ Relief amount: {'PASS' if has_relief else 'FAIL'}")
        if not has_relief:
            print(f"      Expected: ${relief_usd:,} and ETH amount")
        print(f"    ✓ Vault address: {'PASS' if has_vault else 'FAIL'}")
        if not has_vault and vault_address:
            print(f"      Expected vault: {vault_address}")
        print(f"  Overall: {'✅ VALID - Ready to post' if is_valid else '❌ INVALID - Needs regeneration'}")
        
        return {"tweet_is_valid": is_valid}
    
    def _post_to_twitter(self, state: AgentState):
        """Post tweet"""
        print(f"\n{'='*80}")
        print(f"[STEP 11: POST] Posting to Twitter...")
        print(f"{'='*80}")
        
        result = self.twitter_tool.post_tweet(state["tweet_draft"])
        return {"tweet_result": result}

    def _save_to_supabase(self, state: AgentState):
        """Save disaster information to Supabase database"""
        print(f"\n{'='*80}")
        print(f"[STEP 8.5: SAVE TO DB] Saving disaster to Supabase...")
        print(f"{'='*80}")
        
        if not self.supabase:
            print(f"[SAVE TO DB] ⚠️ Supabase not initialized, skipping")
            return {}
        
        try:
            # Extract disaster information
            disaster_content = state.get("current_result", "")
            location = state.get("location", "")
            vault_info = state.get("vault_info", {})
            vault_address = vault_info.get("vault_address") if vault_info.get("success") else None
            relief_amount_usd = state.get("relief_amount_usd")
            sources = state.get("disaster_info", {}).get("sources", [])
            read_more_link = sources[0].get("url") if sources else None
            
            # Use AI to extract structured information
            extraction_prompt = f"""Extract structured information from this disaster report:

DISASTER REPORT:
{disaster_content}

LOCATION: {location}

Extract and return ONLY a JSON object with these exact fields:
{{
    "title": "VERY SHORT disaster title (max 40 characters, be concise - e.g., 'Earthquake in Chile' or 'Flood in Madagascar')",
    "description": "Full disaster description",
    "occurred_at": "ISO 8601 timestamp (YYYY-MM-DDTHH:MM:SSZ) or approximate date if exact time unknown"
}}

IMPORTANT: The title must be VERY SHORT - maximum 40 characters. Use format like "DisasterType in Location" (e.g., "Earthquake in Tokyo", "Flood in California").

Return ONLY valid JSON, nothing else."""

            try:
                response = reasoning_client.chat.completions.create(
                    model=self.reasoning_model,
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                
                extracted_data = response.choices[0].message.content
                disaster_data = json.loads(extracted_data)
                
                title = disaster_data.get("title", f"Disaster in {location}")
                # Ensure title is very short (max 40 chars)
                title = title[:40].strip()
                if len(title) > 40:
                    # If still too long, try to truncate intelligently
                    title = title[:37] + "..."
                
                description = disaster_data.get("description", disaster_content[:500])
                occurred_at_str = disaster_data.get("occurred_at", datetime.now().isoformat() + "Z")
                
                # Parse occurred_at to ensure proper format
                try:
                    # Try to parse the timestamp
                    if 'T' in occurred_at_str:
                        occurred_at = occurred_at_str
                    else:
                        # If it's just a date, add time
                        occurred_at = occurred_at_str + "T00:00:00Z"
                except:
                    occurred_at = datetime.now().isoformat() + "Z"
                
            except Exception as e:
                print(f"[SAVE TO DB] ⚠️ Failed to extract structured data: {e}, using defaults")
                # Fallback to simple extraction - generate very short title
                if disaster_content:
                    # Try to extract disaster type and location for short title
                    first_line = disaster_content.split('\n')[0]
                    # Look for common disaster keywords
                    disaster_types = ['earthquake', 'flood', 'fire', 'hurricane', 'cyclone', 'tsunami', 'volcano', 'landslide', 'storm', 'tornado']
                    disaster_type = None
                    for dt in disaster_types:
                        if dt.lower() in first_line.lower():
                            disaster_type = dt.capitalize()
                            break
                    
                    if disaster_type:
                        title = f"{disaster_type} in {location}"[:40]
                    else:
                        title = f"Disaster in {location}"[:40]
                else:
                    title = f"Disaster in {location}"[:40]
                
                description = disaster_content[:1000] if disaster_content else "Disaster information"
                occurred_at = datetime.now().isoformat() + "Z"
            
            # Prepare data for Supabase
            # Note: location is extracted separately and stored in state['location']
            disaster_record = {
                "title": title,
                "description": description,
                "location": location,  # Location extracted separately in _extract_location step
                "occurred_at": occurred_at,
                "vault_address": vault_address,
                "target_amount": float(relief_amount_usd) if relief_amount_usd else None,
                "read_more_link": read_more_link,
                "total_donations": 0
                # tg_group_link is intentionally omitted (will be NULL/empty in DB)
            }
            
            # Remove None values to avoid issues (except for tg_group_link which we want to leave empty)
            disaster_record = {k: v for k, v in disaster_record.items() if v is not None}
            
            print(f"[SAVE TO DB] Inserting disaster record:")
            print(f"  Title: {title[:50]}...")
            print(f"  Location: {location}")
            print(f"  Vault: {vault_address or 'None'}")
            print(f"  Target Amount: ${relief_amount_usd:,}" if relief_amount_usd else "  Target Amount: None")
            
            # Insert into Supabase
            result = self.supabase.table('disaster_events').insert(disaster_record).execute()
            
            if result.data:
                disaster_id = result.data[0].get('id')
                print(f"[SAVE TO DB] ✅ Successfully saved disaster with ID: {disaster_id}")
                return {"supabase_disaster_id": disaster_id}
            else:
                print(f"[SAVE TO DB] ⚠️ Insert completed but no data returned")
                return {}
                
        except Exception as e:
            print(f"[SAVE TO DB] ❌ Error saving to Supabase: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _coordinate_outreach(self, state: AgentState):
        """Invoke separate response coordination agent to draft outreach emails"""
        from opik_langgraph_helper import log_current_trace_id
        log_current_trace_id("COORDINATE OUTREACH")
        
        print(f"\n{'='*80}")
        print(f"[STEP 9: COORDINATE] Finding responders & drafting emails...")
        print(f"{'='*80}")

        sources = state.get("disaster_info", {}).get("sources", []) if state.get("disaster_info") else []
        vault_info = state.get("vault_info", {}) or {}
        vault_address = vault_info.get("vault_address") if vault_info.get("success") else None

        packet: Dict[str, Any] = {
            "location": state.get("location", ""),
            "disaster_summary": state.get("current_result", ""),
            "sources": sources,
            "relief_amount_usd": state.get("relief_amount_usd"),
            "relief_amount_eth": state.get("relief_amount_eth"),
            "vault_address": vault_address,
        }

        try:
            outreach = self.response_coordinator.run(packet)
            email_count = len(outreach.get("email_drafts", []) or [])
            contact_count = len(outreach.get("contacts", []) or [])
            print(f"[COORDINATE] ✅ Contacts: {contact_count}, Email drafts: {email_count}")
            return {"outreach": outreach}
        except Exception as e:
            print(f"[COORDINATE] ❌ Error: {str(e)}")
            return {"outreach": {"success": False, "error": str(e), "contacts": [], "email_drafts": []}}
    
    def _display_result(self, state: AgentState):
        """Display final report"""
        print(f"\n{'='*80}")
        print(f"[STEP 12: DISPLAY] Final Report")
        print(f"{'='*80}")
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n🚨 DISASTER ALERT #{state['iteration']} - {current_time}")
        print(f"{'='*80}")
        print(f"📍 Location: {state['location']}")
        print(f"💰 Relief Amount: ${state['relief_amount_usd']:,} USD ({state['relief_amount_eth']:.6f} ETH)")
        print(f"💱 ETH Price: ${state['eth_price']:,.2f} USD")
        
        vault_info = state.get("vault_info", {})
        if vault_info.get("success"):
            print(f"🏦 Vault Address: {vault_info['vault_address']}")
            print(f"📜 Transaction: {vault_info['tx_hash']}")
        else:
            print(f"🏦 Vault Creation: Failed - {vault_info.get('error', 'Unknown error')}")
        
        print(f"🌦️  Weather Stations: {len(state.get('weather_data', {}).get('successful', []))}")
        print(f"\n📰 DISASTER INFO:\n{state['current_result']}")

        outreach = state.get("outreach") or {}
        drafts = outreach.get("email_drafts", []) or []
        contacts = outreach.get("contacts", []) or []
        if drafts or contacts:
            print(f"\n📬 OUTREACH (CONTACTS + DRAFTED EMAILS)")
            print(f"{'-'*80}")
            print(f"Contacts found: {len(contacts)}")
            print(f"Emails drafted: {len(drafts)}")

            if contacts:
                print(f"\n[OUTREACH CONTACTS]")
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

            if drafts:
                print(f"\n[OUTREACH EMAIL DRAFTS]")
                for i, d in enumerate(drafts, 1):
                    to_org = d.get("to_org") or "Unknown org"
                    to_name = d.get("to_name") or "N/A"
                    to_role = d.get("to_role") or "N/A"
                    to_email = d.get("to_email") or "N/A"
                    contact_form = d.get("contact_form_url") or "N/A"
                    subject = d.get("subject") or "(no subject)"
                    body = d.get("body") or ""
                    notes = d.get("notes") or ""

                    print(f"\n{i}. To: {to_org} — {to_name} ({to_role})")
                    print(f"   Email: {to_email}")
                    print(f"   Contact form: {contact_form}")
                    print(f"   Subject: {subject}")
                    print(f"   {'-'*76}")
                    print(body)
                    print(f"   {'-'*76}")
                    if notes:
                        print(f"   Notes: {notes}")
        
        tweet = state.get("tweet_result", {})
        if tweet.get("success"):
            print(f"\n🐦 TWEET POSTED:")
            print(f"   ID: {tweet['tweet_id']}")
            print(f"   URL: https://twitter.com/user/status/{tweet['tweet_id']}")
            print(f"   Iterations: {state['tweet_iterations']}")
            print(f"\n   {tweet['tweet_text']}")
        
        print(f"\n{'='*80}\n")
        
        disaster_id = state["location"][:50]
        return {"reported_disasters": state["reported_disasters"] + [disaster_id]}
    
    def _should_regenerate(self, state: AgentState):
        """Routing logic"""
        print(f"\n[ROUTING DECISION]")
        
        if state["tweet_is_valid"]:
            print(f"  Decision: PROCEED TO POST")
            print(f"  Reason: Tweet passed all validation checks")
            return "post"
        elif state["tweet_iterations"] >= 3:
            print(f"  Decision: PROCEED TO POST (Max iterations reached)")
            print(f"  Reason: Attempted {state['tweet_iterations']} times, posting best attempt")
            return "post"
        else:
            print(f"  Decision: REGENERATE TWEET")
            print(f"  Reason: Tweet failed validation, attempting regeneration")
            print(f"  Attempt: {state['tweet_iterations'] + 1}/3")
            return "regenerate"
    
    def _build_graph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("search", self._search_disasters)
        workflow.add_node("extract_location", self._extract_location)
        workflow.add_node("get_bbox", self._get_bbox)
        workflow.add_node("get_weather", self._get_weather)
        workflow.add_node("calculate_relief", self._calculate_relief)
        workflow.add_node("get_eth_price", self._get_eth_price)
        workflow.add_node("convert_to_eth", self._convert_to_eth)
        workflow.add_node("create_vault", self._create_vault)
        workflow.add_node("save_to_supabase", self._save_to_supabase)
        workflow.add_node("coordinate_outreach", self._coordinate_outreach)
        workflow.add_node("generate", self._generate_tweet)
        workflow.add_node("validate", self._validate_tweet)
        workflow.add_node("post", self._post_to_twitter)
        workflow.add_node("display", self._display_result)
        
        # Define flow
        workflow.set_entry_point("search")
        workflow.add_edge("search", "extract_location")
        workflow.add_edge("extract_location", "get_bbox")
        workflow.add_edge("get_bbox", "get_weather")
        workflow.add_edge("get_weather", "calculate_relief")
        workflow.add_edge("calculate_relief", "get_eth_price")
        workflow.add_edge("get_eth_price", "convert_to_eth")
        workflow.add_edge("convert_to_eth", "create_vault")
        workflow.add_edge("create_vault", "save_to_supabase")
        workflow.add_edge("save_to_supabase", "coordinate_outreach")
        workflow.add_edge("coordinate_outreach", "generate")
        workflow.add_edge("generate", "validate")
        
        workflow.add_conditional_edges(
            "validate",
            self._should_regenerate,
            {"regenerate": "generate", "post": "post"}
        )
        
        workflow.add_edge("post", "display")
        workflow.add_edge("display", END)
        
        return workflow.compile()
    
    def run_single_check(self, state: AgentState):
        """Run single check with Opik tracing"""
        # Set project name before trace creation
        from opik_integration import OPIK_AVAILABLE, OPIK_PROJECT
        if OPIK_AVAILABLE:
            import os
            os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT
        
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
    
    def run_continuous_monitoring(self, interval_seconds=300):
        """Run continuous monitoring"""
        print(f"🌍 Disaster Monitoring with Blockchain Vault Creation")
        print(f"⏱️  Interval: {interval_seconds}s")
        print(f"🔧 Tools: Geoapify, WeatherXM, CoinGecko, Ethereum, Twitter")
        print(f"🏦 Network: Sepolia Testnet")
        print(f"{'='*80}\n")
        
        state = {
            "iteration": 0,
            "reported_disasters": [],
            "current_result": "",
            "disaster_info": {},
            "location": "",
            "bbox": None,
            "weather_data": None,
            "relief_amount_usd": None,
            "eth_price": None,
            "relief_amount_eth": None,
            "vault_info": None,
            "outreach": None,
            "tweet_draft": "",
            "tweet_is_valid": False,
            "tweet_iterations": 0,
            "tweet_result": {},
            "all_disasters_found": []
        }
        
        try:
            while True:
                state["iteration"] += 1
                state = self.run_single_check(state)
                
                print(f"⏳ Waiting {interval_seconds} seconds...\n")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopped. Total: {state['iteration']}\n")


def main():
    agent = DisasterMonitoringAgent(
        search_model="gpt-4o-search-preview",
        reasoning_model="gpt-4o"
    )
    agent.run_continuous_monitoring(interval_seconds=300)


if __name__ == "__main__":
    main()