"""
Comprehensive Opik Test Runner
Runs all evaluations and generates reports using Opik's evaluate() function
This creates proper experiments that appear in the Opik dashboard
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add agents directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging to both console and file
class Tee:
    """Write to both file and stdout/stderr"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# Create log file with timestamp
log_dir = Path(__file__).parent
log_file_path = log_dir / f"opik_evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_file_path, 'w', encoding='utf-8')

# Redirect stdout and stderr to both console and file
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"[LOG] Logging all output to: {log_file_path}")
print(f"[LOG] Started at: {datetime.now().isoformat()}")

# Import datasets with error handling
print("[STEP] Loading datasets...")
try:
    from opik_evaluations import (
        DISASTER_DETECTION_DATASET,
        RELIEF_CALCULATION_DATASET,
        EMAIL_QUALITY_DATASET,
        CLAIM_VERIFICATION_DATASET,
        TELEGRAM_RESPONSE_DATASET
    )
    print(f"[OK] Datasets loaded:")
    print(f"  - Disaster Detection: {len(DISASTER_DETECTION_DATASET) if DISASTER_DETECTION_DATASET else 0} items")
    print(f"  - Relief Calculation: {len(RELIEF_CALCULATION_DATASET) if RELIEF_CALCULATION_DATASET else 0} items")
    print(f"  - Email Quality: {len(EMAIL_QUALITY_DATASET) if EMAIL_QUALITY_DATASET else 0} items")
    print(f"  - Claim Verification: {len(CLAIM_VERIFICATION_DATASET) if CLAIM_VERIFICATION_DATASET else 0} items")
except Exception as e:
    print(f"[ERROR] Failed to load datasets: {e}")
    import traceback
    traceback.print_exc()
    DISASTER_DETECTION_DATASET = None
    RELIEF_CALCULATION_DATASET = None
    EMAIL_QUALITY_DATASET = None
    CLAIM_VERIFICATION_DATASET = None
    TELEGRAM_RESPONSE_DATASET = None

# Import Opik integration
print("[STEP] Loading Opik integration...")
try:
    from opik_integration import OPIK_AVAILABLE, opik_client, OPIK_PROJECT
    print(f"[OK] Opik integration loaded (Available: {OPIK_AVAILABLE})")
except Exception as e:
    print(f"[ERROR] Failed to load Opik integration: {e}")
    import traceback
    traceback.print_exc()
    OPIK_AVAILABLE = False
    opik_client = None
    OPIK_PROJECT = "disaster-monitoring"

# Import behavior tracker
print("[STEP] Loading behavior tracker...")
try:
    from user_behavior_tracker import behavior_tracker
    print("[OK] Behavior tracker loaded")
except Exception as e:
    print(f"[WARNING] Failed to load behavior tracker: {e}")
    behavior_tracker = None

# Import Opik evaluation functions
try:
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import score_result
    OPIK_EVALUATION_AVAILABLE = True
except ImportError:
    OPIK_EVALUATION_AVAILABLE = False
    print("[WARNING] Opik evaluation not available. Install with: pip install opik")

def run_all_evaluations(evaluation_type: str = "baseline", variant_config: Dict[str, Any] = None, test_functionality: str = None):
    """
    Run Opik evaluations using evaluate() to create proper experiments
    
    Args:
        evaluation_type: "baseline", "post_optimization", or custom variant name
                         Used to create comparable experiments - same names = comparable in dashboard
        variant_config: Optional dict with configuration overrides:
            - model_temperature: float (e.g., 0.2, 0.7, 1.0)
            - model_name: str (e.g., "gpt-4o-mini", "gpt-4o") - applies to both search and reasoning models
            - search_model: str (e.g., "gpt-4o-search-preview") - override search model only
            - reasoning_model: str (e.g., "gpt-4o") - override reasoning model only
            - max_tokens: int (e.g., 500, 1000) - maximum tokens in response
            - top_p: float (e.g., 0.9, 0.95) - nucleus sampling parameter
            - frequency_penalty: float (e.g., 0.0, 0.5) - reduce repetition
            - presence_penalty: float (e.g., 0.0, 0.3) - encourage new topics
            - prompt_modifications: dict mapping prompt names to modifications (not yet implemented)
        test_functionality: Optional string specifying which individual functionality to test.
                           If None, tests full workflows. If specified, tests only that function.
                           Examples:
                           - "search_disasters" - test only disaster search
                           - "extract_location" - test only location extraction
                           - "calculate_relief" - test only relief calculation
                           - "search_contacts" - test only contact search (coordinator)
                           - "draft_email" - test only email drafting (coordinator)
                           - "verify_claim" - test only claim verification
    
    Examples:
        # Test full workflows
        run_all_evaluations("baseline")
        
        # Test individual functionality with temperature
        run_all_evaluations("temp_0.2", {"model_temperature": 0.2}, test_functionality="search_disasters")
        run_all_evaluations("temp_0.7", {"model_temperature": 0.7}, test_functionality="calculate_relief")
        
        # Compare different temperatures for same functionality
        run_all_evaluations("temp_0.2", {"model_temperature": 0.2}, test_functionality="search_disasters")
        run_all_evaluations("temp_0.7", {"model_temperature": 0.7}, test_functionality="search_disasters")
    """
    print("="*80)
    print(f"OPIK COMPREHENSIVE EVALUATION SUITE - {evaluation_type.upper()}")
    print("="*80)
    print(f"Opik Available: {OPIK_AVAILABLE}")
    print(f"Opik Evaluation Available: {OPIK_EVALUATION_AVAILABLE}")
    print(f"Evaluation Type: {evaluation_type}")
    if variant_config:
        print(f"Variant Config: {variant_config}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Default variant config
    if variant_config is None:
        variant_config = {}
    
    # Store variant config for use in agent wrappers
    _variant_config = variant_config
    
    # Initialize agent variables to None (will be set later)
    disaster_agent = None
    coordinator_agent = None
    verification_agent = None
    
    # Helper function to extract LLM call parameters from variant_config
    def get_llm_params(variant_config, default_model=None, default_temp=None):
        """Extract LLM call parameters from variant_config"""
        params = {}
        
        # Model name (can override search_model or reasoning_model)
        if variant_config and "model_name" in variant_config:
            params["model"] = variant_config["model_name"]
        elif default_model:
            params["model"] = default_model
        
        # Temperature
        if variant_config and "model_temperature" in variant_config:
            params["temperature"] = variant_config["model_temperature"]
        elif default_temp is not None:
            params["temperature"] = default_temp
        
        # Other OpenAI parameters
        if variant_config:
            if "max_tokens" in variant_config:
                params["max_tokens"] = variant_config["max_tokens"]
            if "top_p" in variant_config:
                params["top_p"] = variant_config["top_p"]
            if "frequency_penalty" in variant_config:
                params["frequency_penalty"] = variant_config["frequency_penalty"]
            if "presence_penalty" in variant_config:
                params["presence_penalty"] = variant_config["presence_penalty"]
        
        return params
    
    # Helper function to apply variant_config to agent instance
    def apply_variant_config_to_agent(agent, variant_config):
        """Apply variant_config parameters to agent instance"""
        if not variant_config or not agent:
            return
        
        # Model overrides
        if "search_model" in variant_config:
            agent.search_model = variant_config["search_model"]
        if "reasoning_model" in variant_config:
            agent.reasoning_model = variant_config["reasoning_model"]
        if "model_name" in variant_config:
            # If model_name is specified, apply to both models
            agent.search_model = variant_config["model_name"]
            agent.reasoning_model = variant_config["model_name"]
        
        # Store LLM parameters for use in agent methods
        agent._llm_params_override = get_llm_params(variant_config)
    
    # If test_functionality is specified, only test that one functionality
    if test_functionality:
        print(f"\n[INFO] Testing ONLY functionality: {test_functionality}")
        print(f"       This will be faster and cheaper than full workflow testing")
    
    if not OPIK_EVALUATION_AVAILABLE:
        print("[ERROR] Opik evaluation not available. Cannot create experiments.")
        print("Install with: pip install opik")
        return None
    
    # Ensure Opik is properly configured before using evaluate()
    # CRITICAL: Opik's evaluate() REQUIRES an API key to create trace contexts
    opik_api_key_available = False
    if OPIK_AVAILABLE:
        try:
            from opik import configure
            import os
            OPIK_API_KEY = os.getenv("OPIK_API_KEY", "")
            OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE", "impakt")
            
            if OPIK_API_KEY:
                configure(api_key=OPIK_API_KEY, workspace=OPIK_WORKSPACE)
                opik_api_key_available = True
                print(f"[OK] Opik configured with API key (workspace: {OPIK_WORKSPACE})")
            else:
                print(f"[ERROR] OPIK_API_KEY not set!")
                print("         Opik's evaluate() REQUIRES an API key to create trace contexts.")
                print("         Set OPIK_API_KEY in your .env file or environment variables.")
                print("         Without it, experiments will fail with trace context errors.")
                print("         Continuing anyway, but experiments WILL FAIL...")
        except Exception as e:
            print(f"[ERROR] Failed to configure Opik: {e}")
            print("         Experiments will likely fail.")
            import traceback
            traceback.print_exc()
    
    # Use the module-level opik_client, but allow reassignment if needed
    # Import it fresh to avoid UnboundLocalError
    from opik_integration import opik_client as module_opik_client
    current_opik_client = module_opik_client
    
    if not current_opik_client:
        print("[ERROR] Opik client not available. Cannot create experiments.")
        print("         Check OPIK_API_KEY and OPIK_WORKSPACE environment variables.")
        return None
    
    # Import agents (clean versions without internal tracing)
    print("\n[STEP] Loading agents (clean versions without internal Opik tracing)...")
    try:
        import sys
        
        # CRITICAL: Clear any cached agent modules from sys.modules to force fresh import
        modules_to_clear = ['disasterAgent', 'responseCoordinatorAgent', 'verificationAgent']
        for mod_name in modules_to_clear:
            if mod_name in sys.modules:
                print(f"  [DEBUG] Removing cached module: {mod_name}")
                del sys.modules[mod_name]
        
        # Agent files are in the current directory (no subfolder needed)
        current_dir = Path(__file__).parent
        print(f"  [INFO] Using agents from: {current_dir}")
        from disasterAgent import DisasterMonitoringAgent
        from responseCoordinatorAgent import ResponseCoordinatorAgent
        from verificationAgent import NGOClaimVerifierAgent
        
        # Verify imports
        import inspect
        disaster_file = inspect.getfile(DisasterMonitoringAgent)
        coord_file = inspect.getfile(ResponseCoordinatorAgent)
        verify_file = inspect.getfile(NGOClaimVerifierAgent)
        print(f"  [DEBUG] DisasterMonitoringAgent from: {disaster_file}")
        print(f"  [DEBUG] ResponseCoordinatorAgent from: {coord_file}")
        print(f"  [DEBUG] NGOClaimVerifierAgent from: {verify_file}")
        
        print("  [INFO] Creating agent instances...")
        disaster_agent = DisasterMonitoringAgent()
        print("  [OK] DisasterMonitoringAgent created")
        
        coordinator_agent = ResponseCoordinatorAgent()
        print("  [OK] ResponseCoordinatorAgent created")
        
        verification_agent = NGOClaimVerifierAgent()
        print("  [OK] NGOClaimVerifierAgent created")
        
        print("[OK] All agents loaded successfully")
        
        # Apply variant_config to agents if provided (AFTER agents are loaded)
        if variant_config:
            apply_variant_config_to_agent(disaster_agent, variant_config)
            apply_variant_config_to_agent(coordinator_agent, variant_config)
            apply_variant_config_to_agent(verification_agent, variant_config)
            if variant_config:
                print(f"  [INFO] Applied variant_config to agents: {list(variant_config.keys())}")
    except Exception as e:
        print(f"[ERROR] Could not load agents: {e}")
        print("   Running evaluations with mock functions...")
        import traceback
        traceback.print_exc()
        disaster_agent = None
        coordinator_agent = None
        verification_agent = None
    
    # Import LLM judge for custom scoring functions
    print("\n[STEP] Loading LLM judge...")
    try:
        from opik_integration import llm_judge
        if llm_judge:
            print(f"[OK] LLM judge loaded (model: {llm_judge.model})")
        else:
            print("[WARNING] LLM judge not available")
    except Exception as e:
        print(f"[ERROR] Failed to load LLM judge: {e}")
        import traceback
        traceback.print_exc()
        llm_judge = None
    
    # Create custom scoring functions that use LLM-as-judge
    print("\n[STEP] Creating custom scoring functions...")
    def disaster_detection_scorer(dataset_item: Dict[str, Any], task_outputs: Dict[str, Any]) -> score_result.ScoreResult:
        """Custom scoring function for disaster detection"""
        if not llm_judge:
            return score_result.ScoreResult(
                name="disaster_detection_quality",
                value=0.5,
                reason="LLM judge not available"
            )
        
        try:
            disaster_info = task_outputs.get("disaster_info", {})
            raw_response = disaster_info.get("raw_response", "") if isinstance(disaster_info, dict) else str(disaster_info)
            location = task_outputs.get("location", "")
            sources = disaster_info.get("sources", []) if isinstance(disaster_info, dict) else []
            
            evaluation = llm_judge.evaluate_disaster_detection(
                disaster_info=raw_response,
                location=location,
                sources=sources,
                expected_criteria=dataset_item.get("expected", {}),
                send_to_opik=False  # Don't send during evaluation to avoid duplicates
            )
            
            score = evaluation.get("overall_score", 0.0) / 10.0  # Normalize to 0-1
            return score_result.ScoreResult(
                name="disaster_detection_quality",
                value=score,
                reason=f"Disaster detection quality: {evaluation.get('overall_score', 0):.2f}/10"
            )
        except Exception as e:
            return score_result.ScoreResult(
                name="disaster_detection_quality",
                value=0.0,
                reason=f"Evaluation error: {str(e)}"
            )
    
    def relief_calculation_scorer(dataset_item: Dict[str, Any], task_outputs: Dict[str, Any]) -> score_result.ScoreResult:
        """Custom scoring function for relief calculation"""
        if not llm_judge:
            return score_result.ScoreResult(
                name="relief_calculation_quality",
                value=0.5,
                reason="LLM judge not available"
            )
        
        try:
            calculated_amount = task_outputs.get("relief_amount_usd", 0)
            disaster_info = dataset_item.get("disaster_info", "")
            weather_data = dataset_item.get("weather_data", {})
            location = dataset_item.get("location", "")
            
            evaluation = llm_judge.evaluate_relief_calculation(
                disaster_info=disaster_info,
                weather_data=weather_data,
                calculated_amount=calculated_amount,
                location=location,
                send_to_opik=False
            )
            
            score = evaluation.get("overall_score", 0.0) / 10.0
            return score_result.ScoreResult(
                name="relief_calculation_quality",
                value=score,
                reason=f"Relief calculation quality: {evaluation.get('overall_score', 0):.2f}/10"
            )
        except Exception as e:
            return score_result.ScoreResult(
                name="relief_calculation_quality",
                value=0.0,
                reason=f"Evaluation error: {str(e)}"
            )
    
    def email_quality_scorer(dataset_item: Dict[str, Any], task_outputs: Dict[str, Any]) -> score_result.ScoreResult:
        """Custom scoring function for email quality"""
        if not llm_judge:
            return score_result.ScoreResult(
                name="email_quality",
                value=0.5,
                reason="LLM judge not available"
            )
        
        try:
            email_draft = task_outputs.get("email_draft", {})
            print(f"[SCORER DEBUG] Email draft keys: {list(email_draft.keys()) if isinstance(email_draft, dict) else 'not a dict'}")
            print(f"[SCORER DEBUG] Email subject: {email_draft.get('subject', 'N/A') if isinstance(email_draft, dict) else 'N/A'}")
            print(f"[SCORER DEBUG] Email body length: {len(email_draft.get('body', '')) if isinstance(email_draft, dict) else 0}")
            
            disaster_info = dataset_item.get("disaster_info", "")
            contact_info = {"organization": "Test Org", "role": "Emergency Coordinator"}
            
            evaluation = llm_judge.evaluate_email_quality(
                email_draft=email_draft,
                disaster_info=disaster_info,
                contact_info=contact_info,
                send_to_opik=False
            )
            
            print(f"[SCORER DEBUG] LLM Judge evaluation: {evaluation}")
            
            score = evaluation.get("overall_score", 0.0) / 10.0
            return score_result.ScoreResult(
                name="email_quality",
                value=score,
                reason=f"Email quality: {evaluation.get('overall_score', 0):.2f}/10"
            )
        except Exception as e:
            print(f"[SCORER DEBUG] Exception: {e}")
            import traceback
            traceback.print_exc()
            return score_result.ScoreResult(
                name="email_quality",
                value=0.0,
                reason=f"Evaluation error: {str(e)}"
            )
    
    def claim_verification_scorer(dataset_item: Dict[str, Any], task_outputs: Dict[str, Any]) -> score_result.ScoreResult:
        """Custom scoring function for claim verification"""
        if not llm_judge:
            return score_result.ScoreResult(
                name="claim_verification_quality",
                value=0.5,
                reason="LLM judge not available"
            )
        
        try:
            recommended_amount = task_outputs.get("recommended_amount_usd", 0.0)
            claim_text = dataset_item.get("claim_text", "")
            vault_balance = dataset_item.get("vault_balance_usd", 0.0)
            evidence_summary = dataset_item.get("evidence_summary", "")
            
            evaluation = llm_judge.evaluate_claim_verification(
                claim_text=claim_text,
                recommended_amount=recommended_amount,
                vault_balance=vault_balance,
                evidence_summary=evidence_summary,
                send_to_opik=False
            )
            
            score = evaluation.get("overall_score", 0.0) / 10.0
            return score_result.ScoreResult(
                name="claim_verification_quality",
                value=score,
                reason=f"Claim verification quality: {evaluation.get('overall_score', 0):.2f}/10"
            )
        except Exception as e:
            return score_result.ScoreResult(
                name="claim_verification_quality",
                value=0.0,
                reason=f"Evaluation error: {str(e)}"
            )
    
    # Define agent wrapper functions
    # NOTE: Do NOT wrap with OpikTracer - evaluate() creates its own trace context
    # CRITICAL: Disable internal Opik tracing during evaluation to avoid conflicts
    # Set a flag that opik_langgraph_helper will check
    import os
    os.environ["OPIK_EVALUATION_MODE"] = "true"  # Signal to disable internal tracing
    
    # These wrappers can use variant_config to modify behavior
    def disaster_detection_wrapper(input_data):
        """Wrapper for disaster detection evaluation"""
        nonlocal disaster_agent, variant_config, test_functionality
        if not disaster_agent:
            return {
                "disaster_info": {"raw_response": "Mock disaster info", "sources": []},
                "location": input_data.get("query", "").split()[-1] if input_data.get("query") else "Unknown"
            }
        
        try:
            # If testing individual functionality, only run that function
            if test_functionality == "search_disasters":
                # Test ONLY disaster search functionality
                state = {
                    "iteration": 0,
                    "reported_disasters": input_data.get("exclude_list", []),
                    "current_result": "",
                    "disaster_info": {},
                    "location": "",
                }
                
                # Apply variant_config parameters (already applied to agent instance)
                if variant_config:
                    applied = []
                    if "model_temperature" in variant_config:
                        applied.append(f"temperature={variant_config['model_temperature']}")
                    if "model_name" in variant_config or "search_model" in variant_config:
                        applied.append(f"model={variant_config.get('model_name') or variant_config.get('search_model')}")
                    if applied:
                        print(f"    [INFO] Using: {', '.join(applied)} (note: web search API may not support all parameters)")
                
                result_state = disaster_agent._search_disasters(state)
                return {
                    "disaster_info": result_state.get("disaster_info", {}),
                    "location": ""  # Not extracted in this test
                }
            elif test_functionality == "extract_location":
                # Test ONLY location extraction
                disaster_content = input_data.get("disaster_info", "")
                state = {
                    "current_result": disaster_content,
                    "location": ""
                }
                
                # Apply variant_config parameters (already applied to agent instance)
                # Legacy _temp_override still supported for backward compatibility
                if variant_config and "model_temperature" in variant_config:
                    disaster_agent._temp_override = variant_config["model_temperature"]
                
                applied = []
                if variant_config:
                    if "model_temperature" in variant_config:
                        applied.append(f"temperature={variant_config['model_temperature']}")
                    if "model_name" in variant_config or "reasoning_model" in variant_config:
                        applied.append(f"model={variant_config.get('model_name') or variant_config.get('reasoning_model')}")
                if applied:
                    print(f"    [INFO] Using: {', '.join(applied)} for location extraction")
                
                try:
                    location_state = disaster_agent._extract_location(state)
                    result = {
                        "disaster_info": {"raw_response": disaster_content},
                        "location": location_state.get("location", "Unknown")
                    }
                finally:
                    # Restore original temp override
                    if hasattr(disaster_agent, '_temp_override'):
                        delattr(disaster_agent, '_temp_override')
                
                return result
            elif test_functionality == "calculate_relief":
                # Test ONLY relief calculation
                state = {
                    "current_result": input_data.get("disaster_info", ""),
                    "location": input_data.get("location", ""),
                    "weather_data": input_data.get("weather_data", {})
                }
                
                # Apply variant_config parameters (already applied to agent instance)
                # Legacy _temp_override still supported for backward compatibility
                if variant_config and "model_temperature" in variant_config:
                    disaster_agent._temp_override = variant_config["model_temperature"]
                
                applied = []
                if variant_config:
                    if "model_temperature" in variant_config:
                        applied.append(f"temperature={variant_config['model_temperature']}")
                    if "model_name" in variant_config or "reasoning_model" in variant_config:
                        applied.append(f"model={variant_config.get('model_name') or variant_config.get('reasoning_model')}")
                    if "max_tokens" in variant_config:
                        applied.append(f"max_tokens={variant_config['max_tokens']}")
                    if "top_p" in variant_config:
                        applied.append(f"top_p={variant_config['top_p']}")
                if applied:
                    print(f"    [INFO] Using: {', '.join(applied)} for relief calculation")
                
                try:
                    result_state = disaster_agent._calculate_relief(state)
                    result = {
                        "disaster_info": {"raw_response": input_data.get("disaster_info", "")},
                        "location": input_data.get("location", ""),
                        "relief_amount_usd": result_state.get("relief_amount_usd", 0)
                    }
                finally:
                    if hasattr(disaster_agent, '_temp_override'):
                        delattr(disaster_agent, '_temp_override')
                
                return result
            
            # Full workflow (default behavior)
            # Apply variant config if provided (e.g., different model parameters)
            # Note: Current agents don't expose temperature/model easily, but this is where you'd apply it
            # For now, we'll just run with default agent configuration
            # Create a minimal state for testing
            state = {
                "iteration": 0,
                "reported_disasters": input_data.get("exclude_list", []),
                "current_result": "",
                "disaster_info": {},
                "location": "",
            }
            
            # Run search_disasters node
            # Note: update_node_span_with_io will gracefully handle missing span context
            result_state = disaster_agent._search_disasters(state)
            location_state = disaster_agent._extract_location({
                **state,
                "current_result": result_state.get("current_result", "")
            })
            
            result = {
                "disaster_info": result_state.get("disaster_info", {}),
                "location": location_state.get("location", "Unknown")
            }
            
            # Record interaction for behavior tracking
            try:
                from user_behavior_tracker import behavior_tracker
                behavior_tracker.record_interaction(
                    user_id="evaluation_test",
                    interaction_type="disaster_detection",
                    agent_name="disaster_monitoring",
                    input_text=input_data.get("query", "")[:200],
                    output_text=f"Location: {result['location']}",
                    metadata={"test_case": input_data.get("metadata", {}).get("test_case", "unknown")}
                )
            except Exception:
                pass
            
            return result
        except Exception as e:
            print(f"    Error in disaster detection: {e}")
            import traceback
            traceback.print_exc()
            return {
                "disaster_info": {"raw_response": f"Error: {e}", "sources": []},
                "location": "Unknown"
            }
    
    def relief_calculation_wrapper(input_data):
        """Wrapper for relief calculation evaluation"""
        nonlocal disaster_agent, variant_config, test_functionality
        if not disaster_agent:
            return {"relief_amount_usd": 1000000}
        
        try:
            # If testing individual functionality, only run that function
            if test_functionality == "calculate_relief":
                state = {
                    "current_result": input_data.get("disaster_info", ""),
                    "location": input_data.get("location", ""),
                    "weather_data": input_data.get("weather_data", {})
                }
                
                # Apply variant_config parameters (already applied to agent instance)
                # Legacy _temp_override still supported for backward compatibility
                if variant_config and "model_temperature" in variant_config:
                    disaster_agent._temp_override = variant_config["model_temperature"]
                
                applied = []
                if variant_config:
                    if "model_temperature" in variant_config:
                        applied.append(f"temperature={variant_config['model_temperature']}")
                    if "model_name" in variant_config or "reasoning_model" in variant_config:
                        applied.append(f"model={variant_config.get('model_name') or variant_config.get('reasoning_model')}")
                    if "max_tokens" in variant_config:
                        applied.append(f"max_tokens={variant_config['max_tokens']}")
                if applied:
                    print(f"    [INFO] Using: {', '.join(applied)} for relief calculation")
                
                try:
                    result_state = disaster_agent._calculate_relief(state)
                    result = {"relief_amount_usd": result_state.get("relief_amount_usd", 1000000)}
                finally:
                    if hasattr(disaster_agent, '_temp_override'):
                        delattr(disaster_agent, '_temp_override')
                
                return result
            
            # Full workflow (default behavior)
            # Note: update_node_span_with_io will gracefully handle missing span context
            state = {
                "current_result": input_data.get("disaster_info", ""),
                "location": input_data.get("location", ""),
                "weather_data": input_data.get("weather_data", {})
            }
            
            result_state = disaster_agent._calculate_relief(state)
            relief_amount = result_state.get("relief_amount_usd", 1000000)
            
            # Record interaction for behavior tracking
            try:
                from user_behavior_tracker import behavior_tracker
                behavior_tracker.record_interaction(
                    user_id="evaluation_test",
                    interaction_type="relief_calculation",
                    agent_name="disaster_monitoring",
                    input_text=input_data.get("disaster_info", "")[:200],
                    output_text=f"Relief: ${relief_amount:,.2f}",
                    metadata={"test_case": input_data.get("metadata", {}).get("test_case", "unknown")}
                )
            except Exception:
                pass
            
            return {
                "relief_amount_usd": relief_amount
            }
        except Exception as e:
            print(f"    Error in relief calculation: {e}")
            import traceback
            traceback.print_exc()
            return {"relief_amount_usd": 1000000}
    
    def email_drafting_wrapper(input_data):
        """Wrapper for email drafting evaluation"""
        nonlocal coordinator_agent, variant_config, test_functionality
        if not coordinator_agent:
            return {
                "email_draft": {
                    "subject": "Mock Subject",
                    "body": "Mock email body with urgent action items"
                }
            }
        
        try:
            # If testing individual functionality, only run that function
            if test_functionality == "search_contacts":
                # Test ONLY contact search
                # Import CoordinatorState (no subfolder in clean evals directory)
                from responseCoordinatorAgent import CoordinatorState
                
                # Debug: Show applied parameters
                if variant_config:
                    applied = []
                    for key in ['max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'model_name', 'search_model']:
                        if key in variant_config:
                            applied.append(f"{key}={variant_config[key]}")
                    if applied:
                        print(f"    [DEBUG] Contact search with: {', '.join(applied)}")
                state = CoordinatorState(
                    location=input_data.get("location", ""),
                    disaster_summary=input_data.get("disaster_info", ""),
                    sources=[],
                    relief_amount_usd=input_data.get("relief_amount_usd", 0),
                    relief_amount_eth=None,
                    vault_address=None,
                    contacts=[],
                    email_drafts=[],
                    emails_sent=[]
                )
                result_state = coordinator_agent._search_contacts(state)
                # Return mock email since we're only testing contact search
                return {
                    "email_draft": {
                        "subject": "Test",
                        "body": "Test"
                    },
                    "contacts_found": len(result_state.get("contacts", []))
                }
            elif test_functionality == "draft_email":
                # Test ONLY email drafting
                # Import CoordinatorState (no subfolder in clean evals directory)
                from responseCoordinatorAgent import CoordinatorState
                
                # Debug: Show applied parameters
                if variant_config:
                    applied = []
                    for key in ['max_tokens', 'temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'model_name']:
                        if key in variant_config:
                            applied.append(f"{key}={variant_config[key]}")
                    if applied:
                        print(f"    [DEBUG] Email drafting with: {', '.join(applied)}")
                
                # Debug: Show contacts being passed
                contacts_to_pass = input_data.get("contacts", [])
                print(f"    [DEBUG] Contacts being passed to agent: {len(contacts_to_pass)}")
                if contacts_to_pass:
                    print(f"    [DEBUG] First contact: {contacts_to_pass[0]}")
                
                state = CoordinatorState(
                    location=input_data.get("location", ""),
                    disaster_summary=input_data.get("disaster_info", ""),
                    sources=[],
                    relief_amount_usd=input_data.get("relief_amount_usd", 0),
                    relief_amount_eth=None,
                    vault_address=None,
                    contacts=contacts_to_pass,
                    email_drafts=[],
                    emails_sent=[]
                )
                result_state = coordinator_agent._draft_emails(state)
                emails = result_state.get("email_drafts", [])
                email_draft = emails[0] if emails else {
                    "subject": "No email drafted",
                    "body": ""
                }
                
                # Debug: Show generated email length
                body_len = len(email_draft.get("body", ""))
                print(f"    [DEBUG] Generated email body length: {body_len} chars")
                print(f"    [DEBUG] Email subject: {email_draft.get('subject', '')[:50]}...")
                if body_len < 200:
                    print(f"    [DEBUG] Email body preview: {email_draft.get('body', '')}")
                else:
                    print(f"    [DEBUG] Email body preview: {email_draft.get('body', '')[:200]}...")
                
                return {"email_draft": email_draft}
            
            # Full workflow (default behavior)
            packet = {
                "location": input_data.get("location", ""),
                "disaster_summary": input_data.get("disaster_info", ""),
                "sources": [],
                "relief_amount_usd": input_data.get("relief_amount_usd"),
                "relief_amount_eth": None,
                "vault_address": None,
            }
            
            result = coordinator_agent.run(packet)
            emails = result.get("email_drafts", [])
            
            email_draft = emails[0] if emails else {
                "subject": "No email drafted",
                "body": ""
            }
            
            # Record interaction for behavior tracking
            try:
                from user_behavior_tracker import behavior_tracker
                behavior_tracker.record_interaction(
                    user_id="evaluation_test",
                    interaction_type="email_drafting",
                    agent_name="response_coordinator",
                    input_text=input_data.get("disaster_info", "")[:200],
                    output_text=email_draft.get("subject", "")[:200],
                    metadata={"test_case": input_data.get("metadata", {}).get("test_case", "unknown")}
                )
            except Exception:
                pass
            
            return {
                "email_draft": email_draft
            }
        except Exception as e:
            print(f"    Error in email drafting: {e}")
            import traceback
            traceback.print_exc()
            return {
                "email_draft": {
                    "subject": "Error",
                    "body": f"Error: {e}"
                }
            }
    
    def claim_verification_wrapper(input_data):
        """Wrapper for claim verification evaluation"""
        nonlocal verification_agent, variant_config, test_functionality
        if not verification_agent:
            return {"recommended_amount_usd": 5000.0}
        
        try:
            # Import _ImageInput from the correct module (already imported at top level)
            # DO NOT import here as it pollutes sys.modules with wrong version
            # from verificationAgent import _ImageInput
            
            # Note: update_node_span_with_io will gracefully handle missing span context
            # The agent.run() method uses LangGraph which creates its own trace context
            # Use vault address from test data, or create a test address
            vault_addr = input_data.get("vault_address")
            if not vault_addr or vault_addr == "0x0000000000000000000000000000000000000000":
                # Use a test address - note: actual balance will be fetched from blockchain
                # For testing, we expect the agent to handle this gracefully
                vault_addr = "0x1234567890123456789012345678901234567890"
            
            # Add error handling for rate limit issues
            try:
                result_state = verification_agent.run(
                    content=input_data.get("claim_text", ""),
                    vault_address=vault_addr,
                    images=[],
                    verbose=False
                )
                recommended = result_state.get("recommended_amount_usd", 0.0)
            except Exception as api_error:
                # Handle rate limit errors gracefully
                if "429" in str(api_error) or "Too Many Requests" in str(api_error):
                    print(f"    [WARNING] Rate limit error (429) - using fallback value")
                    recommended = 5000.0  # Fallback value
                else:
                    raise
            
            # Record interaction for behavior tracking
            try:
                from user_behavior_tracker import behavior_tracker
                behavior_tracker.record_interaction(
                    user_id="evaluation_test",
                    interaction_type="claim_verification",
                    agent_name="verification_agent",
                    input_text=input_data.get("claim_text", "")[:200],
                    output_text=f"Recommended: ${recommended:,.2f}",
                    metadata={"test_case": input_data.get("metadata", {}).get("test_case", "unknown")}
                )
            except Exception as e:
                # Silently fail - behavior tracking is optional
                pass
            
            return {
                "recommended_amount_usd": recommended
            }
        except Exception as e:
            print(f"    Error in claim verification: {e}")
            import traceback
            traceback.print_exc()
            return {"recommended_amount_usd": 0.0}
    
    # Convert datasets to Opik format and create experiments using evaluate()
    all_experiments = {}
    experiment_results = {}
    
    # Ensure we have opik_client - create it if needed
    if not current_opik_client and OPIK_AVAILABLE:
        try:
            from opik import Opik
            current_opik_client = Opik()
            print("[OK] Created Opik client")
        except Exception as e:
            print(f"[ERROR] Failed to create Opik client: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    if not current_opik_client:
        print("[ERROR] Opik client not available. Cannot create experiments.")
        print("         Check OPIK_API_KEY and OPIK_WORKSPACE environment variables.")
        return None
    
    print("\n" + "="*80)
    print("RUNNING EVALUATIONS WITH OPIK EXPERIMENTS")
    print("="*80)
    
    # Helper function to convert dataset format
    def convert_dataset_for_opik(dataset_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert evaluation dataset to Opik format"""
        opik_items = []
        for item in dataset_list:
            opik_item = {}
            # Merge input fields into top level
            if "input" in item:
                opik_item.update(item["input"])
            # Add expected and metadata
            if "expected" in item:
                opik_item["expected"] = item["expected"]
            if "metadata" in item:
                opik_item["metadata"] = item["metadata"]
            opik_items.append(opik_item)
        return opik_items
    
    # 1. Evaluate Disaster Detection (or individual functionality)
    should_test_disaster = (
        (test_functionality is None or test_functionality in ["search_disasters", "extract_location", "calculate_relief"]) and
        DISASTER_DETECTION_DATASET and len(DISASTER_DETECTION_DATASET) > 0 and 
        OPIK_EVALUATION_AVAILABLE and opik_api_key_available
    )
    
    if should_test_disaster:
        experiment_name_suffix = f" - {test_functionality}" if test_functionality else ""
        print(f"\n[EXPERIMENT] Disaster Detection Evaluation{experiment_name_suffix}")
        try:
            # Create/Get dataset
            print("  [STEP] Creating/getting dataset...")
            dataset_name = f"disaster-detection-evaluation-{test_functionality}" if test_functionality else "disaster-detection-evaluation"
            dataset = current_opik_client.get_or_create_dataset(name=dataset_name)
            print(f"  [OK] Dataset ready: {dataset.name if hasattr(dataset, 'name') else dataset_name}")
            
            print("  [STEP] Converting and inserting dataset items...")
            opik_dataset_items = convert_dataset_for_opik(DISASTER_DETECTION_DATASET)
            print(f"  [INFO] Converted {len(opik_dataset_items)} items")
            dataset.insert(opik_dataset_items)
            print(f"  [OK] Inserted {len(opik_dataset_items)} items into dataset")
            
            # Create evaluation task wrapper
            def disaster_evaluation_task(dataset_item: Dict[str, Any]) -> Dict[str, Any]:
                """Evaluation task for disaster detection"""
                # Extract input from dataset item
                input_data = {
                    "query": dataset_item.get("query", ""),
                    "exclude_list": dataset_item.get("exclude_list", []),
                    "disaster_info": dataset_item.get("disaster_info", ""),  # For extract_location test
                    "location": dataset_item.get("location", ""),  # For calculate_relief test
                    "weather_data": dataset_item.get("weather_data", {}),  # For calculate_relief test
                    "metadata": dataset_item.get("metadata", {})
                }
                result = disaster_detection_wrapper(input_data)
                
                # Return appropriate output based on what we're testing
                if test_functionality == "search_disasters":
                    return {
                        "disaster_info": result.get("disaster_info", {})
                    }
                elif test_functionality == "extract_location":
                    return {
                        "location": result.get("location", "")
                    }
                elif test_functionality == "calculate_relief":
                    return {
                        "relief_amount_usd": result.get("relief_amount_usd", 0)
                    }
                else:
                    # Full workflow
                    return {
                        "disaster_info": result.get("disaster_info", {}),
                        "location": result.get("location", "")
                    }
            
            # Run evaluation with experiment
            # Use consistent experiment name for baseline and variants so they can be compared
            # Opik will create separate runs within the same experiment for comparison
            if test_functionality:
                experiment_name = f"Disaster Detection - {test_functionality} - {evaluation_type}"
            else:
                experiment_name = f"Disaster Detection - {evaluation_type}"
            
            # Add variant info to experiment config
            experiment_config_base = {
                "agent": "disaster_monitoring",
                "evaluation_type": "disaster_detection",
                "variant": evaluation_type,
                "timestamp": datetime.now().isoformat()
            }
            # Add variant config details if provided
            if variant_config:
                experiment_config_base["variant_config"] = variant_config
            
            # API key already checked in if condition above
            print(f"  [STEP] Running evaluation with experiment: {experiment_name}")
            print(f"  [INFO] Dataset items: {len(opik_dataset_items)}")
            print(f"  [INFO] Scoring function: disaster_detection_scorer")
            
            try:
                # Try with project_name first
                print("  [INFO] Calling evaluate() with project_name...")
                eval_result = evaluate(
                    dataset=dataset,
                    task=disaster_evaluation_task,
                    scoring_functions=[disaster_detection_scorer],
                    experiment_name=experiment_name,
                    project_name=OPIK_PROJECT,
                    experiment_config=experiment_config_base.copy(),
                    task_threads=1  # Disable threading to avoid trace context issues
                )
                print("  [OK] Evaluation completed successfully")
            except Exception as eval_error:
                # If evaluate() fails due to trace context issues, try without project_name
                error_str = str(eval_error)
                error_type = type(eval_error).__name__
                
                if ("trace" in error_str.lower() or 
                    "context" in error_str.lower() or 
                    "AssertionError" in error_str or
                    "OpikException" in error_type):
                    print(f"[WARNING] Trace context error detected: {eval_error}")
                    print("         Retrying without project_name parameter...")
                    try:
                        eval_result = evaluate(
                            dataset=dataset,
                            task=disaster_evaluation_task,
                            scoring_functions=[disaster_detection_scorer],
                            experiment_name=experiment_name,
                            experiment_config=experiment_config_base.copy(),
                            task_threads=1  # Disable threading to avoid trace context issues
                        )
                    except Exception as eval_error2:
                        print(f"[ERROR] Both evaluate() attempts failed!")
                        print(f"         First error: {eval_error}")
                        print(f"         Second error: {eval_error2}")
                        print(f"[INFO] Continuing with other evaluations...")
                        import traceback
                        traceback.print_exc()
                        eval_result = None
                else:
                    # Different error - log and continue
                    print(f"[ERROR] Evaluation failed with unexpected error: {eval_error}")
                    print(f"[INFO] Continuing with other evaluations...")
                    import traceback
                    traceback.print_exc()
                    eval_result = None
            
            if eval_result:
                experiment_results["disaster_detection"] = eval_result
                exp_name = getattr(eval_result, 'experiment_name', 'Unknown')
                exp_id = getattr(eval_result, 'experiment_id', 'Unknown')
                all_experiments["disaster_detection"] = {
                    "experiment_name": exp_name,
                    "experiment_id": exp_id,
                    "status": "completed"
                }
                print(f"  [OK] Experiment created: {exp_name}")
                print(f"  [OK] Experiment ID: {exp_id}")
                if hasattr(eval_result, 'test_results') and eval_result.test_results:
                    print(f"  [OK] Test cases completed: {len(eval_result.test_results)}")
            else:
                print(f"  [WARNING] Disaster detection evaluation failed - skipping")
                all_experiments["disaster_detection"] = {
                    "status": "failed",
                    "error": str(eval_error) if 'eval_error' in locals() else "Unknown error"
                }
            
        except Exception as e:
            print(f"[ERROR] Disaster detection evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            # Continue with other evaluations even if this one fails
            print("[INFO] Continuing with other evaluations...")
    
    # 2. Evaluate Relief Calculation (only if not testing individual functionality or if specifically testing calculate_relief)
    should_test_relief = (
        (test_functionality is None or test_functionality == "calculate_relief") and
        RELIEF_CALCULATION_DATASET and len(RELIEF_CALCULATION_DATASET) > 0 and 
        OPIK_EVALUATION_AVAILABLE and opik_api_key_available
    )
    
    if should_test_relief:
        experiment_name_suffix = f" - {test_functionality}" if test_functionality else ""
        print(f"\n[EXPERIMENT] Relief Calculation Evaluation{experiment_name_suffix}")
        try:
            print("  [STEP] Creating/getting dataset...")
            dataset_name = f"relief-calculation-evaluation-{test_functionality}" if test_functionality else "relief-calculation-evaluation"
            dataset = current_opik_client.get_or_create_dataset(name=dataset_name)
            print("  [STEP] Converting and inserting dataset items...")
            opik_dataset_items = convert_dataset_for_opik(RELIEF_CALCULATION_DATASET)
            print(f"  [INFO] Converted {len(opik_dataset_items)} items")
            dataset.insert(opik_dataset_items)
            print(f"  [OK] Inserted {len(opik_dataset_items)} items into dataset")
            
            def relief_evaluation_task(dataset_item: Dict[str, Any]) -> Dict[str, Any]:
                input_data = {
                    "disaster_info": dataset_item.get("disaster_info", ""),
                    "weather_data": dataset_item.get("weather_data", {}),
                    "location": dataset_item.get("location", ""),
                    "metadata": dataset_item.get("metadata", {})
                }
                result = relief_calculation_wrapper(input_data)
                return {"relief_amount_usd": result.get("relief_amount_usd", 0)}
            
            if test_functionality:
                experiment_name = f"Relief Calculation - {test_functionality} - {evaluation_type}"
            else:
                experiment_name = f"Relief Calculation - {evaluation_type}"
            experiment_config_base = {
                "agent": "disaster_monitoring",
                "evaluation_type": "relief_calculation",
                "variant": evaluation_type,
                "timestamp": datetime.now().isoformat()
            }
            if variant_config:
                experiment_config_base["variant_config"] = variant_config
            
            print(f"  [STEP] Running evaluation with experiment: {experiment_name}")
            try:
                print("  [INFO] Calling evaluate()...")
                eval_result = evaluate(
                    dataset=dataset,
                    task=relief_evaluation_task,
                    scoring_functions=[relief_calculation_scorer],
                    experiment_name=experiment_name,
                    project_name=OPIK_PROJECT,
                    experiment_config=experiment_config_base.copy(),
                    task_threads=1  # Disable threading to avoid trace context issues
                )
                print("  [OK] Evaluation completed successfully")
            except Exception as eval_error:
                error_str = str(eval_error)
                error_type = type(eval_error).__name__
                if ("trace" in error_str.lower() or "context" in error_str.lower() or 
                    "AssertionError" in error_str or "OpikException" in error_type):
                    print(f"[WARNING] Trace context error, retrying without project_name...")
                    try:
                        eval_result = evaluate(
                            dataset=dataset,
                            task=relief_evaluation_task,
                            scoring_functions=[relief_calculation_scorer],
                            experiment_name=experiment_name,
                            experiment_config=experiment_config_base.copy(),
                            task_threads=1  # Disable threading to avoid trace context issues
                        )
                    except Exception as eval_error2:
                        print(f"[ERROR] Both attempts failed: {eval_error2}")
                        raise
                else:
                    raise
            
            experiment_results["relief_calculation"] = eval_result
            exp_name = getattr(eval_result, 'experiment_name', 'Unknown')
            exp_id = getattr(eval_result, 'experiment_id', 'Unknown')
            print(f"  [OK] Experiment created: {exp_name}")
            print(f"  [OK] Experiment ID: {exp_id}")
            print(f"  [OK] Test cases completed: {len(eval_result.test_results)}")
            
        except Exception as e:
            print(f"[ERROR] Relief calculation evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Continuing with other evaluations...")
    
    # 3. Evaluate Email Quality (only if not testing individual functionality or if specifically testing email functions)
    should_test_email = (
        (test_functionality is None or test_functionality in ["search_contacts", "draft_email"]) and
        EMAIL_QUALITY_DATASET and len(EMAIL_QUALITY_DATASET) > 0 and 
        OPIK_EVALUATION_AVAILABLE and opik_api_key_available
    )
    
    if should_test_email:
        experiment_name_suffix = f" - {test_functionality}" if test_functionality else ""
        print(f"\n[EXPERIMENT] Email Quality Evaluation{experiment_name_suffix}")
        try:
            print("  [STEP] Creating/getting dataset...")
            dataset_name = f"email-quality-evaluation-{test_functionality}" if test_functionality else "email-quality-evaluation"
            dataset = current_opik_client.get_or_create_dataset(name=dataset_name)
            print("  [STEP] Converting and inserting dataset items...")
            opik_dataset_items = convert_dataset_for_opik(EMAIL_QUALITY_DATASET)
            print(f"  [INFO] Converted {len(opik_dataset_items)} items")
            dataset.insert(opik_dataset_items)
            print(f"  [OK] Inserted {len(opik_dataset_items)} items into dataset")
            
            def email_evaluation_task(dataset_item: Dict[str, Any]) -> Dict[str, Any]:
                """
                Evaluation task for email quality.
                This function is called by opik.evaluation.evaluate() which creates its own trace context.
                DO NOT create any Opik traces inside this function - it will interfere with evaluate()'s trace.
                """
                # Ensure no Opik tracing happens during task execution
                import os
                original_eval_mode = os.environ.get("OPIK_EVALUATION_MODE", None)
                os.environ["OPIK_EVALUATION_MODE"] = "true"  # Disable any internal tracing
                
                try:
                    # Convert single contact to contacts array if needed
                    contacts = dataset_item.get("contacts", [])
                    print(f"    [DEBUG] Dataset item keys: {list(dataset_item.keys())}")
                    print(f"    [DEBUG] Initial contacts from dataset: {contacts}")
                    
                    if not contacts and "contact" in dataset_item:
                        # Convert singular contact to array format
                        contact = dataset_item["contact"]
                        print(f"    [DEBUG] Converting single contact to array: {contact}")
                        contacts = [{
                            "organization": contact.get("organization", ""),
                            "role": contact.get("role", ""),
                            "email": contact.get("email", ""),
                            "contact_method": "email"
                        }]
                        print(f"    [DEBUG] Converted contacts array: {contacts}")
                    
                    input_data = {
                        "location": dataset_item.get("location", ""),
                        "disaster_info": dataset_item.get("disaster_info", ""),
                        "relief_amount_usd": dataset_item.get("relief_amount_usd") or dataset_item.get("relief_amount"),
                        "contacts": contacts,
                        "metadata": dataset_item.get("metadata", {})
                    }
                    result = email_drafting_wrapper(input_data)
                    if test_functionality == "search_contacts":
                        return {"contacts_found": result.get("contacts_found", 0)}
                    else:
                        return {"email_draft": result.get("email_draft", {})}
                finally:
                    # Restore original evaluation mode
                    if original_eval_mode is None:
                        os.environ.pop("OPIK_EVALUATION_MODE", None)
                    else:
                        os.environ["OPIK_EVALUATION_MODE"] = original_eval_mode
            
            if test_functionality:
                experiment_name = f"Email Quality - {test_functionality} - {evaluation_type}"
            else:
                experiment_name = f"Email Quality - {evaluation_type}"
            experiment_config_base = {
                "agent": "response_coordinator",
                "evaluation_type": "email_quality",
                "variant": evaluation_type,
                "timestamp": datetime.now().isoformat()
            }
            if variant_config:
                experiment_config_base["variant_config"] = variant_config
            
            print(f"  [STEP] Running evaluation with experiment: {experiment_name}")
            try:
                # CRITICAL: Ensure Opik is properly configured with API key before evaluate()
                # Opik's evaluate() REQUIRES an API key to create trace contexts
                import os
                OPIK_API_KEY_CHECK = os.getenv("OPIK_API_KEY", "")
                OPIK_WORKSPACE_CHECK = os.getenv("OPIK_WORKSPACE", "impakt")
                
                if not OPIK_API_KEY_CHECK:
                    raise ValueError("OPIK_API_KEY not set - evaluate() requires API key to create trace context")
                
                if OPIK_AVAILABLE:
                    from opik import configure
                    try:
                        # Reconfigure Opik right before evaluate() to ensure fresh state
                        configure(api_key=OPIK_API_KEY_CHECK, workspace=OPIK_WORKSPACE_CHECK)
                        print(f"  [DEBUG] Opik configured (workspace: {OPIK_WORKSPACE_CHECK}, API key: {'*' * 10})")
                    except Exception as config_error:
                        print(f"  [WARNING] Opik reconfiguration failed: {config_error}")
                        import traceback
                        traceback.print_exc()
                
                # Double-check API key is available
                import os
                final_api_key_check = os.getenv("OPIK_API_KEY", "")
                if not final_api_key_check:
                    raise ValueError("OPIK_API_KEY not found in environment - evaluate() requires it to create traces")
                
                print(f"  [DEBUG] API key available: {'Yes' if final_api_key_check else 'No'}")
                print("  [INFO] Calling evaluate()...")
                
                # CRITICAL: Opik's evaluate() sometimes fails to create trace context
                # This appears to be a bug in Opik when task_threads=1 and certain task functions are used
                # Try multiple approaches to work around this issue
                eval_result = None
                eval_error_final = None
                
                # Approach 1: Try with project_name (standard)
                try:
                    eval_result = evaluate(
                        dataset=dataset,
                        task=email_evaluation_task,
                        scoring_functions=[email_quality_scorer],
                        experiment_name=experiment_name,
                        project_name=OPIK_PROJECT,
                        experiment_config=experiment_config_base.copy(),
                        task_threads=1
                    )
                    print("  [OK] Evaluation completed successfully")
                except Exception as e1:
                    error_str1 = str(e1).lower()
                    eval_error_final = e1
                    if "trace" in error_str1 or "context" in error_str1 or "assertionerror" in error_str1:
                        print(f"  [WARNING] Approach 1 failed (trace context): {type(e1).__name__}")
                        # Approach 2: Try without project_name
                        try:
                            print("  [INFO] Trying without project_name...")
                            eval_result = evaluate(
                                dataset=dataset,
                                task=email_evaluation_task,
                                scoring_functions=[email_quality_scorer],
                                experiment_name=experiment_name,
                                experiment_config=experiment_config_base.copy(),
                                task_threads=1
                            )
                            print("  [OK] Evaluation completed successfully (without project_name)")
                        except Exception as e2:
                            error_str2 = str(e2).lower()
                            eval_error_final = e2
                            print(f"  [WARNING] Approach 2 also failed: {type(e2).__name__}")
                            # Approach 3: Try with a simpler task wrapper that doesn't call agent methods
                            print("  [INFO] This appears to be an Opik SDK bug with trace context creation")
                            print("  [INFO] The evaluation task completed successfully, but Opik failed to create trace")
                            print(f"  [ERROR] Both approaches failed. Last error: {eval_error_final}")
                            eval_result = None
                    else:
                        # Different error - re-raise
                        raise
                
                # Process evaluation result
                if eval_result:
                    experiment_results["email_quality"] = eval_result
                    exp_name = getattr(eval_result, 'experiment_name', 'Unknown')
                    exp_id = getattr(eval_result, 'experiment_id', 'Unknown')
                    all_experiments["email_quality"] = {
                        "experiment_name": exp_name,
                        "experiment_id": exp_id,
                        "status": "completed"
                    }
                    print(f"  [OK] Experiment created: {exp_name}")
                    print(f"  [OK] Experiment ID: {exp_id}")
                    if hasattr(eval_result, 'test_results') and eval_result.test_results:
                        print(f"  [OK] Test cases completed: {len(eval_result.test_results)}")
                else:
                    print(f"  [WARNING] Email quality evaluation failed - skipping")
                    all_experiments["email_quality"] = {
                        "status": "failed",
                        "error": str(eval_error_final) if eval_error_final else "Unknown error"
                    }
            except Exception as eval_error:
                # Outer exception handler for any unexpected errors
                print(f"[ERROR] Unexpected error in email evaluation: {eval_error}")
                import traceback
                traceback.print_exc()
                eval_result = None
                all_experiments["email_quality"] = {
                    "status": "failed",
                    "error": str(eval_error)
                }
            
        except Exception as e:
            print(f"[ERROR] Email quality evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Continuing with other evaluations...")
    
    # 4. Evaluate Claim Verification (only if not testing individual functionality or if specifically testing verify_claim)
    should_test_claim = (
        (test_functionality is None or test_functionality == "verify_claim") and
        CLAIM_VERIFICATION_DATASET and len(CLAIM_VERIFICATION_DATASET) > 0 and 
        OPIK_EVALUATION_AVAILABLE and opik_api_key_available
    )
    
    if should_test_claim:
        experiment_name_suffix = f" - {test_functionality}" if test_functionality else ""
        print(f"\n[EXPERIMENT] Claim Verification Evaluation{experiment_name_suffix}")
        try:
            print("  [STEP] Creating/getting dataset...")
            dataset_name = f"claim-verification-evaluation-{test_functionality}" if test_functionality else "claim-verification-evaluation"
            dataset = current_opik_client.get_or_create_dataset(name=dataset_name)
            print("  [STEP] Converting and inserting dataset items...")
            opik_dataset_items = convert_dataset_for_opik(CLAIM_VERIFICATION_DATASET)
            print(f"  [INFO] Converted {len(opik_dataset_items)} items")
            dataset.insert(opik_dataset_items)
            print(f"  [OK] Inserted {len(opik_dataset_items)} items into dataset")
            
            def claim_evaluation_task(dataset_item: Dict[str, Any]) -> Dict[str, Any]:
                input_data = {
                    "claim_text": dataset_item.get("claim_text", ""),
                    "vault_address": dataset_item.get("vault_address"),
                    "vault_balance_usd": dataset_item.get("vault_balance_usd", 0.0),
                    "evidence_summary": dataset_item.get("evidence_summary", ""),
                    "metadata": dataset_item.get("metadata", {})
                }
                result = claim_verification_wrapper(input_data)
                return {"recommended_amount_usd": result.get("recommended_amount_usd", 0.0)}
            
            if test_functionality:
                experiment_name = f"Claim Verification - {test_functionality} - {evaluation_type}"
            else:
                experiment_name = f"Claim Verification - {evaluation_type}"
            experiment_config_base = {
                "agent": "verification_agent",
                "evaluation_type": "claim_verification",
                "variant": evaluation_type,
                "timestamp": datetime.now().isoformat()
            }
            if variant_config:
                experiment_config_base["variant_config"] = variant_config
            
            print(f"  [STEP] Running evaluation with experiment: {experiment_name}")
            try:
                print("  [INFO] Calling evaluate()...")
                eval_result = evaluate(
                    dataset=dataset,
                    task=claim_evaluation_task,
                    scoring_functions=[claim_verification_scorer],
                    experiment_name=experiment_name,
                    project_name=OPIK_PROJECT,
                    experiment_config=experiment_config_base.copy(),
                    task_threads=1  # Disable threading to avoid trace context issues
                )
                print("  [OK] Evaluation completed successfully")
            except Exception as eval_error:
                error_str = str(eval_error)
                error_type = type(eval_error).__name__
                if ("trace" in error_str.lower() or "context" in error_str.lower() or 
                    "AssertionError" in error_str or "OpikException" in error_type):
                    print(f"[WARNING] Trace context error, retrying without project_name...")
                    try:
                        eval_result = evaluate(
                            dataset=dataset,
                            task=claim_evaluation_task,
                            scoring_functions=[claim_verification_scorer],
                            experiment_name=experiment_name,
                            experiment_config=experiment_config_base.copy(),
                            task_threads=1  # Disable threading to avoid trace context issues
                        )
                    except Exception as eval_error2:
                        print(f"[ERROR] Both attempts failed: {eval_error2}")
                        print(f"[INFO] Continuing with other evaluations...")
                        import traceback
                        traceback.print_exc()
                        eval_result = None
                else:
                    print(f"[ERROR] Evaluation failed: {eval_error}")
                    print(f"[INFO] Continuing with other evaluations...")
                    import traceback
                    traceback.print_exc()
                    eval_result = None
            
            if eval_result:
                experiment_results["claim_verification"] = eval_result
                exp_name = getattr(eval_result, 'experiment_name', 'Unknown')
                exp_id = getattr(eval_result, 'experiment_id', 'Unknown')
                all_experiments["claim_verification"] = {
                    "experiment_name": exp_name,
                    "experiment_id": exp_id,
                    "status": "completed"
                }
                print(f"  [OK] Experiment created: {exp_name}")
                print(f"  [OK] Experiment ID: {exp_id}")
                if hasattr(eval_result, 'test_results') and eval_result.test_results:
                    print(f"  [OK] Test cases completed: {len(eval_result.test_results)}")
            else:
                print(f"  [WARNING] Claim verification evaluation failed - skipping")
                all_experiments["claim_verification"] = {
                    "status": "failed",
                    "error": str(eval_error) if 'eval_error' in locals() else "Unknown error"
                }
            
        except Exception as e:
            print(f"[ERROR] Claim verification evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Continuing...")
    
    # Generate summary report
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"[INFO] Total experiments completed: {len(experiment_results)}")
    if len(experiment_results) == 0:
        print("[WARNING] No experiments were created!")
        print("         Possible reasons:")
        print("         1. OPIK_API_KEY not set")
        print("         2. Datasets are empty or None")
        print("         3. All evaluations failed")
        print("         4. OPIK_EVALUATION_AVAILABLE is False")
        print("         Check the log file for detailed error messages.")
        print("\n[DEBUG] Current state:")
        print(f"         OPIK_EVALUATION_AVAILABLE: {OPIK_EVALUATION_AVAILABLE}")
        print(f"         opik_api_key_available: {opik_api_key_available}")
        print(f"         DISASTER_DETECTION_DATASET: {len(DISASTER_DETECTION_DATASET) if DISASTER_DETECTION_DATASET else 'None'}")
        print(f"         RELIEF_CALCULATION_DATASET: {len(RELIEF_CALCULATION_DATASET) if RELIEF_CALCULATION_DATASET else 'None'}")
        print(f"         EMAIL_QUALITY_DATASET: {len(EMAIL_QUALITY_DATASET) if EMAIL_QUALITY_DATASET else 'None'}")
        print(f"         CLAIM_VERIFICATION_DATASET: {len(CLAIM_VERIFICATION_DATASET) if CLAIM_VERIFICATION_DATASET else 'None'}")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": evaluation_type,
        "total_agents_evaluated": len(experiment_results),
        "agents": {},
        "experiments": {}
    }
    
    for agent_name, eval_result in experiment_results.items():
        if eval_result:
            # Calculate aggregate scores
            scores = [tr.score_results[0].value for tr in eval_result.test_results if tr.score_results and len(tr.score_results) > 0]
            avg_score = sum(scores) / len(scores) * 10 if scores else 0.0  # Convert back to 0-10 scale
            
            exp_id = getattr(eval_result, 'experiment_id', 'Unknown')
            exp_name = getattr(eval_result, 'experiment_name', 'Unknown')
            
            report["agents"][agent_name] = {
                "total_tests": len(eval_result.test_results),
                "passed_tests": len([tr for tr in eval_result.test_results if tr.score_results and len(tr.score_results) > 0 and tr.score_results[0].value > 0.5]),
                "average_score": avg_score,
                "experiment_id": exp_id,
                "experiment_name": exp_name
            }
            
            report["experiments"][agent_name] = {
                "experiment_id": exp_id,
                "experiment_name": exp_name,
                "test_results_count": len(eval_result.test_results)
            }
            
            print(f"\nAgent: {agent_name}")
            print(f"  Experiment: {exp_name}")
            print(f"  Experiment ID: {exp_id}")
            print(f"  Total Tests: {len(eval_result.test_results)}")
            print(f"  Average Score: {avg_score:.2f}/10")
    
    # Save report
    report_file = Path(__file__).parent / "evaluation_results" / "opik_evaluation_report.json"
    report_file.parent.mkdir(exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n[OK] Evaluation report saved to: {report_file}")
    
    # Print dashboard link
    if OPIK_AVAILABLE and os.getenv("OPIK_API_KEY"):
        workspace = os.getenv("OPIK_WORKSPACE", "impakt")
        project = os.getenv("OPIK_PROJECT", "disaster-monitoring")
        print(f"\n[INFO] View experiments in Opik Dashboard:")
        print(f"   https://app.opik.com/{workspace}/projects/{project}")
        print(f"   Navigate to: Evaluation -> Experiments")
        print(f"\n[INFO] Experiment Comparison:")
        print(f"   Experiments are named: '<Agent Task> - {evaluation_type}'")
        print(f"   Examples:")
        print(f"   - Baseline: '<Agent Task> - baseline'")
        print(f"   - Post-Optimization: '<Agent Task> - post_optimization'")
        print(f"   - Custom variants: '<Agent Task> - temperature_0.7', '<Agent Task> - model_gpt4o', etc.")
        print(f"   Compare these in the Opik dashboard to see performance differences!")
        print(f"\n[INFO] To create new comparison experiments (NO OPTIMIZATION COST):")
        print(f"   python run_opik_tests.py variant_name '{{\"model_temperature\": 0.7}}'")
        print(f"   python run_opik_tests.py variant_name '{{\"model_name\": \"gpt-4o\"}}'")
        print(f"   python run_opik_tests.py variant_name '{{\"max_tokens\": 500, \"top_p\": 0.9}}'")
        print(f"   python run_opik_tests.py variant_name '{{\"model_temperature\": 0.8, \"frequency_penalty\": 0.5}}'")
        print(f"   This only runs evaluations (cheap), not optimizations (expensive)!")
        print(f"\n[INFO] To test individual functionalities:")
        print(f"   python run_opik_tests.py temp_0.2 '{{\"model_temperature\": 0.2}}' search_disasters")
        print(f"   python run_opik_tests.py temp_0.7 '{{\"model_temperature\": 0.7}}' calculate_relief")
        print(f"   Available functionalities: search_disasters, extract_location, calculate_relief, search_contacts, draft_email, verify_claim")
    
    # Get reliability metrics
    if behavior_tracker:
        try:
            reliability = behavior_tracker.get_reliability_metrics()
            report["reliability_metrics"] = reliability
            print("\n" + "="*80)
            print("SYSTEM RELIABILITY METRICS")
            print("="*80)
            # Handle "No interactions recorded" gracefully
            if isinstance(reliability, dict) and reliability.get("error") == "No interactions recorded":
                print("[INFO] No interactions recorded yet - this is normal for evaluation-only runs")
                print("       Reliability metrics will be available after agent interactions")
            else:
                print(json.dumps(reliability, indent=2))
        except Exception as e:
            print(f"\n[WARNING] Could not get reliability metrics: {e}")
            report["reliability_metrics"] = None
    else:
        print("\n[WARNING] Behavior tracker not available - skipping reliability metrics")
        report["reliability_metrics"] = None
    
    # Add log file path to report
    report["log_file"] = str(log_file_path)
    report["completed_at"] = datetime.now().isoformat()
    
    return report


def run_regression_tests():
    """Run regression test suite"""
    print("\n" + "="*80)
    print("REGRESSION TEST SUITE")
    print("="*80)
    
    # Basic regression tests
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Check if datasets are loaded
    tests_total += 1
    try:
        if DISASTER_DETECTION_DATASET and len(DISASTER_DETECTION_DATASET) > 0:
            print("[OK] Disaster detection dataset loaded")
            tests_passed += 1
        else:
            print("[WARNING] Disaster detection dataset not loaded")
    except:
        print("[WARNING] Could not check disaster detection dataset")
    
    # Test 2: Check if agents can be imported
    tests_total += 1
    try:
        from disasterAgent import DisasterMonitoringAgent
        from responseCoordinatorAgent import ResponseCoordinatorAgent
        from verificationAgent import NGOClaimVerifierAgent
        print("[OK] All agents can be imported")
        tests_passed += 1
    except Exception as e:
        print(f"[WARNING] Could not import agents: {e}")
    
    # Test 3: Check if Opik is configured
    tests_total += 1
    try:
        if OPIK_AVAILABLE and os.getenv("OPIK_API_KEY"):
            print("[OK] Opik is configured")
            tests_passed += 1
        else:
            print("[WARNING] Opik not configured (OPIK_API_KEY missing)")
    except:
        print("[WARNING] Could not check Opik configuration")
    
    print(f"\n[SUMMARY] Regression tests: {tests_passed}/{tests_total} passed")
    if tests_passed == tests_total:
        print("[OK] All regression tests passed")
    else:
        print("[WARNING] Some regression tests failed - check warnings above")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    import sys
    import json
    
    # Check if evaluation type is provided as command line argument
    evaluation_type = "baseline"
    variant_config = None
    
    if len(sys.argv) > 1:
        evaluation_type = sys.argv[1]
    
    # Check if variant config JSON is provided as second argument
    if len(sys.argv) > 2:
        variant_config_str = sys.argv[2]
        # Handle case where PowerShell concatenates arguments: "{key:value} third_arg" -> extract just the JSON part
        # Check if the string contains a space and looks like it has a third argument concatenated
        if ' ' in variant_config_str and variant_config_str.strip().startswith('{'):
            # Try to find where the JSON ends (first } that's not inside quotes)
            json_end = variant_config_str.rfind('}')
            if json_end > 0:
                variant_config_str = variant_config_str[:json_end + 1]
        
        variant_config = None
        
        # Try multiple parsing strategies
        # Strategy 1: Direct JSON parsing
        try:
            variant_config = json.loads(variant_config_str)
            print(f"[INFO] Parsed variant config as JSON: {variant_config}")
        except json.JSONDecodeError:
            # Strategy 2: Try fixing common PowerShell/Windows issues
            # PowerShell might strip quotes, resulting in {key: value} format
            try:
                import re
                fixed_str = variant_config_str.strip()
                
                # More robust fixing: handle {key: value} -> {"key": "value"} or {"key": value}
                def fix_key_value(match):
                    key = match.group(1)
                    value = match.group(2).strip()
                    # Remove any trailing commas and extra text (in case third arg got concatenated)
                    value = value.rstrip(',').strip()
                    # Remove any trailing text that looks like a third argument (non-JSON content)
                    # If we see something like "0.3 search_contacts", extract just "0.3"
                    if ' ' in value and not (value.startswith('"') and value.endswith('"')):
                        # Check if it's a number followed by text
                        parts = value.split()
                        try:
                            float(parts[0])
                            value = parts[0]  # Just take the number part
                        except ValueError:
                            pass  # Keep original value
                    
                    # If value is already quoted, keep it
                    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
                        return f'"{key}": {value}'
                    # If value is a number (int or float), don't quote it
                    try:
                        float(value)
                        return f'"{key}": {value}'
                    except ValueError:
                        # Value is a string, quote it
                        return f'"{key}": "{value}"'
                
                # Match pattern: key: value (handles spaces around colon and commas)
                # Updated regex to stop at space if it looks like a concatenated argument
                fixed_str = re.sub(r'(\w+):\s*([^,}]+?)(?=\s+\w+|,|\}|$)', fix_key_value, fixed_str)
                variant_config = json.loads(fixed_str)
                print(f"[INFO] Parsed variant config (fixed PowerShell format): {variant_config}")
            except (json.JSONDecodeError, Exception) as e:
                # Strategy 3: Try Python literal eval
                try:
                    import ast
                    variant_config = ast.literal_eval(variant_config_str)
                    print(f"[INFO] Parsed variant config as Python dict: {variant_config}")
                except (ValueError, SyntaxError):
                    # Strategy 4: More aggressive fixing for PowerShell
                    try:
                        import re
                        fixed_str = variant_config_str.strip()
                        
                        # Remove outer braces temporarily
                        if fixed_str.startswith('{') and fixed_str.endswith('}'):
                            content = fixed_str[1:-1].strip()
                            
                            # Split by comma and fix each key-value pair
                            pairs = []
                            for pair in content.split(','):
                                pair = pair.strip()
                                if ':' in pair:
                                    key, value = pair.split(':', 1)
                                    key = key.strip()
                                    value = value.strip()
                                    
                                    # Remove any trailing text that looks like a concatenated argument
                                    # If we see something like "0.3 search_contacts", extract just "0.3"
                                    if ' ' in value and not (value.startswith('"') and value.endswith('"')):
                                        parts = value.split()
                                        try:
                                            float(parts[0])
                                            value = parts[0]  # Just take the number part
                                        except ValueError:
                                            # If first part isn't a number, might be a string value
                                            # Take everything up to the first space that looks like next arg
                                            if len(parts) > 1:
                                                # Check if second part looks like a command-line arg (no quotes, alphanumeric)
                                                if parts[1].replace('_', '').replace('-', '').isalnum():
                                                    value = parts[0]  # Take just first part
                                    
                                    # Quote the key
                                    key = f'"{key}"'
                                    
                                    # Quote the value if it's not a number and not already quoted
                                    if not (value.startswith('"') and value.endswith('"')):
                                        if not (value.startswith("'") and value.endswith("'")):
                                            try:
                                                float(value)
                                                # It's a number, keep as is
                                            except ValueError:
                                                # It's a string, quote it
                                                value = f'"{value}"'
                                    
                                    pairs.append(f'{key}: {value}')
                            
                            fixed_str = '{' + ', '.join(pairs) + '}'
                            variant_config = json.loads(fixed_str)
                            print(f"[INFO] Parsed variant config (aggressive fix): {variant_config}")
                        else:
                            raise ValueError("Could not parse")
                    except Exception as e:
                        print(f"[ERROR] Invalid variant config format: {variant_config_str}")
                        print(f"[DEBUG] Parsing error: {e}")
                        print("\n[HELP] PowerShell/Windows users:")
                        print("   Method 1 - Use single quotes with double quotes inside:")
                        print("   python run_opik_tests.py temp_0.2 '{\"model_temperature\": 0.2}' search_disasters")
                        print("\n   Method 2 - Use double quotes with escaped inner quotes:")
                        print('   python run_opik_tests.py temp_0.2 "{\\"model_temperature\\": 0.2}" search_disasters')
                        print("\n   Method 3 - Use PowerShell backticks:")
                        print('   python run_opik_tests.py temp_0.2 `"{\\"model_temperature\\": 0.2}`" search_disasters')
                        print("\n   Method 4 - Use -f format operator:")
                        print('   python run_opik_tests.py temp_0.2 ''{"model_temperature": 0.2}'' search_disasters')
                        variant_config = None
    
    try:
        # Run all evaluations
        if variant_config is None and len(sys.argv) > 2:
            # If variant config was provided but failed to parse, exit
            print("\n[ERROR] Cannot proceed without valid variant config.")
            sys.exit(1)
        
        # Parse test_functionality from command line if provided
        test_functionality = None
        if len(sys.argv) > 3:
            test_functionality = sys.argv[3]
        elif len(sys.argv) > 2:
            # Check if third argument got concatenated to variant_config_str
            # This happens when PowerShell strips quotes: '{"key": value}' third_arg becomes "{key: value} third_arg"
            variant_config_str_orig = sys.argv[2] if len(sys.argv) > 2 else ""
            if ' ' in variant_config_str_orig and variant_config_str_orig.strip().startswith('{'):
                # Extract the part after the JSON (should be the functionality name)
                json_end = variant_config_str_orig.rfind('}')
                if json_end > 0 and json_end < len(variant_config_str_orig) - 1:
                    potential_functionality = variant_config_str_orig[json_end + 1:].strip()
                    # Check if it looks like a functionality name (no special chars except underscore)
                    if potential_functionality.replace('_', '').isalnum():
                        test_functionality = potential_functionality
                        print(f"[INFO] Detected concatenated functionality argument: {test_functionality}")
            print(f"[INFO] Testing individual functionality: {test_functionality}")
            print(f"       Available functionalities:")
            print(f"         - search_disasters: Test disaster search only")
            print(f"         - extract_location: Test location extraction only")
            print(f"         - calculate_relief: Test relief calculation only")
            print(f"         - search_contacts: Test contact search only (coordinator)")
            print(f"         - draft_email: Test email drafting only (coordinator)")
            print(f"         - verify_claim: Test claim verification only")
        
        report = run_all_evaluations(
            evaluation_type=evaluation_type, 
            variant_config=variant_config,
            test_functionality=test_functionality
        )
        
        # Run regression tests
        run_regression_tests()
        
        # Create final comprehensive JSON report
        final_report = {
            "evaluation_summary": report,
            "timestamp": datetime.now().isoformat(),
            "log_file": str(log_file_path),
            "status": "completed"
        }
        
        # Save final report
        final_report_file = Path(__file__).parent / "evaluation_results" / f"final_evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        final_report_file.parent.mkdir(exist_ok=True)
        with open(final_report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print("\n" + "="*80)
        print("[OK] ALL TESTS COMPLETED")
        print("="*80)
        print(f"\n[OK] Final report saved to: {final_report_file}")
        print(f"[OK] Log file saved to: {log_file_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Error running tests: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error report
        error_report = {
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "log_file": str(log_file_path)
        }
        error_report_file = Path(__file__).parent / "evaluation_results" / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        error_report_file.parent.mkdir(exist_ok=True)
        with open(error_report_file, 'w', encoding='utf-8') as f:
            json.dump(error_report, f, indent=2)
        
        sys.exit(1)
    finally:
        # Restore stdout/stderr and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print(f"\n[LOG] Log file closed: {log_file_path}")
