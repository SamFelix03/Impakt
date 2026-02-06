"""
Main Opik Agent Optimization Test Script
Runs all optimization algorithms on all 3 agents
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

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
log_file_path = log_dir / f"opik_optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
log_file = open(log_file_path, 'w', encoding='utf-8')

# Redirect stdout and stderr to both console and file
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"[LOG] Logging all output to: {log_file_path}")
print(f"[LOG] Started at: {datetime.now().isoformat()}")
print("="*80)

# Initialize Opik FIRST before importing optimizer
try:
    import opik
    from opik import configure
    OPIK_AVAILABLE = True
    
    # Configure Opik with API key and workspace
    OPIK_API_KEY = os.getenv("OPIK_API_KEY", "")
    OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE", "handz-disaster-relief")
    OPIK_PROJECT = os.getenv("OPIK_PROJECT", "disaster-monitoring")
    
    if OPIK_API_KEY:
        configure(api_key=OPIK_API_KEY, workspace=OPIK_WORKSPACE)
        print(f"[OPIK] [OK] Configured with API key (workspace: {OPIK_WORKSPACE})")
    else:
        configure(workspace=OPIK_WORKSPACE)
        print(f"[OPIK] [WARNING] Configured without API key (workspace: {OPIK_WORKSPACE})")
        print(f"[OPIK]    Set OPIK_API_KEY environment variable to enable dashboard logging")
    
    # Set project name for all Opik operations
    os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT
    print(f"[OPIK] Project: {OPIK_PROJECT}")
    
except ImportError:
    OPIK_AVAILABLE = False
    print("[WARNING] Opik SDK not installed. Install with: pip install opik")

# Check dependencies
try:
    from opik_optimizer import (
        ChatPrompt, MetaPromptOptimizer, HRPO, EvolutionaryOptimizer,
        FewShotBayesianOptimizer, ParameterOptimizer
    )
    try:
        from opik_optimizer import GEPAOptimizer
        GEPA_AVAILABLE = True
    except ImportError:
        GEPA_AVAILABLE = False
        print("[INFO] GEPA optimizer not available. Install with: pip install gepa")
    
    OPIK_OPTIMIZER_AVAILABLE = True
    print("[OK] opik-optimizer imported successfully")
except ImportError as e:
    print(f"[ERROR] opik-optimizer not installed: {e}")
    print("Install with: pip install opik-optimizer")
    OPIK_OPTIMIZER_AVAILABLE = False

if not OPIK_OPTIMIZER_AVAILABLE:
    exit(1)

# Import our modules
from opik_datasets import create_all_datasets
from opik_metrics import (
    get_disaster_monitoring_metric,
    get_response_coordinator_metric,
    get_verification_metric
)
from opik_optimizable_agents import (
    DisasterMonitoringOptimizableAgent,
    ResponseCoordinatorOptimizableAgent,
    VerificationOptimizableAgent,
    extract_prompts_from_disaster_agent,
    extract_prompts_from_coordinator_agent,
    extract_prompts_from_verification_agent
)

# Configuration
OPTIMIZER_MODEL = os.getenv("OPIK_OPTIMIZER_MODEL", "openai/gpt-4o-mini")
CHATPROMPT_MODEL_DISASTER_SEARCH = "openai/gpt-4o-search-preview"
CHATPROMPT_MODEL_DISASTER_REASONING = "openai/gpt-4o-mini"
CHATPROMPT_MODEL_COORDINATOR_SEARCH = "openai/gpt-4o-search-preview"
CHATPROMPT_MODEL_COORDINATOR_REASONING = "openai/gpt-4o-mini"
CHATPROMPT_MODEL_VERIFICATION = "openai/gpt-4o-mini"

# Optimization parameters - REDUCED DEFAULTS TO SAVE COST
# Set environment variables to override:
#   OPIK_MAX_TRIALS=2 (default, was 5)
#   OPIK_N_SAMPLES=all (default, uses all dataset items instead of 50)
#   OPIK_N_THREADS=1 (default, was 4 - sequential to reduce parallel API calls)
#   OPIK_QUICK_MODE=true (only run MetaPrompt optimizer, skip others)
MAX_TRIALS = int(os.getenv("OPIK_MAX_TRIALS", "2"))  # Reduced from 5 to 2
N_SAMPLES_STR = os.getenv("OPIK_N_SAMPLES", "all")  # Use "all" to match dataset size
N_SAMPLES = None if N_SAMPLES_STR.lower() == "all" else int(N_SAMPLES_STR)
N_THREADS = int(os.getenv("OPIK_N_THREADS", "1"))  # Reduced from 4 to 1 (sequential)
QUICK_MODE = os.getenv("OPIK_QUICK_MODE", "false").lower() == "true"  # Only MetaPrompt if True


def run_metaprompt_optimization(prompt, dataset, metric, agent_class, prompt_name: str):
    """Run MetaPrompt optimization"""
    print(f"\n{'='*80}")
    print(f"[METAPROMPT] Optimizing: {prompt_name}")
    print(f"{'='*80}")
    
    # Configure optimizer (project_name is set via OPIK_PROJECT_NAME env var)
    optimizer = MetaPromptOptimizer(
        model=OPTIMIZER_MODEL,
        n_threads=N_THREADS
    )
    
    try:
        # Validate metric is callable before optimization
        if not callable(metric):
            raise ValueError(f"Metric must be callable, got {type(metric)}")
        
        # For single-prompt optimization, we can pass agent_class if needed
        # Otherwise, optimizer uses default LiteLLM execution
        optimize_kwargs = {
            "prompt": prompt,
            "dataset": dataset,
            "metric": metric,
            "max_trials": MAX_TRIALS,
        }
        # Only add n_samples if specified (None means use all dataset items)
        if N_SAMPLES is not None:
            optimize_kwargs["n_samples"] = N_SAMPLES
        
        # Add agent_class if provided (for custom execution logic)
        if agent_class:
            optimize_kwargs["agent_class"] = agent_class
        
        result = optimizer.optimize_prompt(**optimize_kwargs)
        
        print(f"\n[METAPROMPT] [OK] Completed: {prompt_name}")
        print(f"  Initial Score: {result.initial_score:.4f}")
        print(f"  Final Score: {result.score:.4f}")
        print(f"  Improvement: {result.score - result.initial_score:.4f}")
        print(f"  Best Prompt Preview: {str(result.prompt)[:200]}...")
        
        return result
    except Exception as e:
        print(f"[METAPROMPT] [ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_hrpo_optimization(prompt, dataset, metric, agent_class, prompt_name: str):
    """Run HRPO optimization"""
    print(f"\n{'='*80}")
    print(f"[HRPO] Optimizing: {prompt_name}")
    print(f"{'='*80}")
    
    optimizer = HRPO(
        model=OPTIMIZER_MODEL,
        n_threads=N_THREADS,
        max_parallel_batches=4
    )
    
    try:
        optimize_kwargs = {
            "prompt": prompt,
            "dataset": dataset,
            "metric": metric,
            "max_trials": MAX_TRIALS,
        }
        if N_SAMPLES is not None:
            optimize_kwargs["n_samples"] = N_SAMPLES
        if agent_class:
            optimize_kwargs["agent_class"] = agent_class
        
        result = optimizer.optimize_prompt(**optimize_kwargs)
        
        print(f"\n[HRPO] [OK] Completed: {prompt_name}")
        print(f"  Initial Score: {result.initial_score:.4f}")
        print(f"  Final Score: {result.score:.4f}")
        print(f"  Improvement: {result.score - result.initial_score:.4f}")
        
        return result
    except Exception as e:
        print(f"[HRPO] [ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_evolutionary_optimization(prompt, dataset, metric, agent_class, prompt_name: str):
    """Run Evolutionary optimization"""
    print(f"\n{'='*80}")
    print(f"[EVOLUTIONARY] Optimizing: {prompt_name}")
    print(f"{'='*80}")
    
    optimizer = EvolutionaryOptimizer(
        model=OPTIMIZER_MODEL,
        population_size=5,
        num_generations=3,
        n_threads=N_THREADS
    )
    
    try:
        optimize_kwargs = {
            "prompt": prompt,
            "dataset": dataset,
            "metric": metric,
        }
        if N_SAMPLES is not None:
            optimize_kwargs["n_samples"] = N_SAMPLES
        if agent_class:
            optimize_kwargs["agent_class"] = agent_class
        
        result = optimizer.optimize_prompt(**optimize_kwargs)
        
        print(f"\n[EVOLUTIONARY] [OK] Completed: {prompt_name}")
        print(f"  Initial Score: {result.initial_score:.4f}")
        print(f"  Final Score: {result.score:.4f}")
        print(f"  Improvement: {result.score - result.initial_score:.4f}")
        
        return result
    except Exception as e:
        print(f"[EVOLUTIONARY] [ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_fewshot_bayesian_optimization(prompt, dataset, metric, agent_class, prompt_name: str):
    """Run Few-Shot Bayesian optimization"""
    print(f"\n{'='*80}")
    print(f"[FEW-SHOT BAYESIAN] Optimizing: {prompt_name}")
    print(f"{'='*80}")
    
    optimizer = FewShotBayesianOptimizer(
        model=OPTIMIZER_MODEL,
        n_threads=N_THREADS
    )
    
    try:
        optimize_kwargs = {
            "prompt": prompt,
            "dataset": dataset,
            "metric": metric,
            "max_trials": MAX_TRIALS,
        }
        if N_SAMPLES is not None:
            optimize_kwargs["n_samples"] = N_SAMPLES
        if agent_class:
            optimize_kwargs["agent_class"] = agent_class
        
        result = optimizer.optimize_prompt(**optimize_kwargs)
        
        print(f"\n[FEW-SHOT BAYESIAN] [OK] Completed: {prompt_name}")
        print(f"  Initial Score: {result.initial_score:.4f}")
        print(f"  Final Score: {result.score:.4f}")
        print(f"  Improvement: {result.score - result.initial_score:.4f}")
        
        return result
    except Exception as e:
        print(f"[FEW-SHOT BAYESIAN] [ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_gepa_optimization(prompt, dataset, metric, agent_class, prompt_name: str):
    """Run GEPA optimization"""
    if not GEPA_AVAILABLE:
        print(f"[GEPA] [WARNING] Skipping {prompt_name} - GEPA not available")
        return None
    
    print(f"\n{'='*80}")
    print(f"[GEPA] Optimizing: {prompt_name}")
    print(f"{'='*80}")
    
    optimizer = GEPAOptimizer(
        model=OPTIMIZER_MODEL,
        n_threads=N_THREADS
    )
    
    try:
        optimize_kwargs = {
            "prompt": prompt,
            "dataset": dataset,
            "metric": metric,
            "max_trials": MAX_TRIALS,
        }
        if N_SAMPLES is not None:
            optimize_kwargs["n_samples"] = N_SAMPLES
        if agent_class:
            optimize_kwargs["agent_class"] = agent_class
        
        result = optimizer.optimize_prompt(**optimize_kwargs)
        
        print(f"\n[GEPA] [OK] Completed: {prompt_name}")
        print(f"  Initial Score: {result.initial_score:.4f}")
        print(f"  Final Score: {result.score:.4f}")
        print(f"  Improvement: {result.score - result.initial_score:.4f}")
        
        return result
    except Exception as e:
        print(f"[GEPA] [ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def run_parameter_optimization(prompt, dataset, metric, agent_class, prompt_name: str):
    """Run Parameter optimization"""
    print(f"\n{'='*80}")
    print(f"[PARAMETER] Optimizing: {prompt_name}")
    print(f"{'='*80}")
    
    optimizer = ParameterOptimizer(
        model=OPTIMIZER_MODEL,
        n_threads=N_THREADS
    )
    
    try:
        # ParameterOptimizer uses optimize_parameter instead of optimize_prompt
        from opik_optimizer import ParameterSearchSpace
        
        # Define parameter search space (temperature, top_p, etc.)
        parameter_space = ParameterSearchSpace(
            temperature={"min": 0.0, "max": 1.0},
            top_p={"min": 0.1, "max": 1.0},
            max_tokens={"min": 100, "max": 2000}
        )
        
        optimize_param_kwargs = {
            "prompt": prompt,
            "dataset": dataset,
            "metric": metric,
            "parameter_space": parameter_space,
            "max_trials": MAX_TRIALS,
        }
        if N_SAMPLES is not None:
            optimize_param_kwargs["n_samples"] = N_SAMPLES
        
        result = optimizer.optimize_parameter(**optimize_param_kwargs)
        
        print(f"\n[PARAMETER] [OK] Completed: {prompt_name}")
        print(f"  Initial Score: {result.initial_score:.4f}")
        print(f"  Final Score: {result.score:.4f}")
        print(f"  Improvement: {result.score - result.initial_score:.4f}")
        
        return result
    except Exception as e:
        print(f"[PARAMETER] [ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def optimize_disaster_monitoring_agent(datasets: Dict[str, Any]):
    """Optimize DisasterMonitoringAgent prompts"""
    print(f"\n{'#'*80}")
    print(f"# DISASTER MONITORING AGENT OPTIMIZATION")
    print(f"{'#'*80}")
    
    dataset = datasets["disaster_monitoring"]
    metric = get_disaster_monitoring_metric()
    if not metric:
        print("[ERROR] Failed to get disaster monitoring metric")
        return {}
    prompts = extract_prompts_from_disaster_agent()
    
    results = {}
    
    # Optimize search_disasters prompt (single prompt optimization)
    search_prompt = prompts["search_disasters"]
    results["search_disasters"] = {
        "metaprompt": run_metaprompt_optimization(
            search_prompt, dataset, metric, None, "search_disasters"
        ),
    }
    if not QUICK_MODE:
        results["search_disasters"].update({
            "hrpo": run_hrpo_optimization(
                search_prompt, dataset, metric, None, "search_disasters"
            ),
            "evolutionary": run_evolutionary_optimization(
                search_prompt, dataset, metric, None, "search_disasters"
            ),
        })
    
    # Optimize calculate_relief prompt (single prompt optimization)
    relief_prompt = prompts["calculate_relief"]
    results["calculate_relief"] = {
        "metaprompt": run_metaprompt_optimization(
            relief_prompt, dataset, metric, None, "calculate_relief"
        ),
    }
    if not QUICK_MODE:
        results["calculate_relief"].update({
            "hrpo": run_hrpo_optimization(
                relief_prompt, dataset, metric, None, "calculate_relief"
            ),
            "parameter": run_parameter_optimization(
                relief_prompt, dataset, metric, None, "calculate_relief"
            ),
        })
    
    return results


def optimize_response_coordinator_agent(datasets: Dict[str, Any]):
    """Optimize ResponseCoordinatorAgent prompts"""
    print(f"\n{'#'*80}")
    print(f"# RESPONSE COORDINATOR AGENT OPTIMIZATION")
    print(f"{'#'*80}")
    
    dataset = datasets["response_coordinator"]
    metric = get_response_coordinator_metric()
    if not metric:
        print("[ERROR] Failed to get response coordinator metric")
        return {}
    prompts = extract_prompts_from_coordinator_agent()
    
    results = {}
    
    # Optimize draft_emails prompt (most important) - single prompt optimization
    email_prompt = prompts["draft_emails"]
    results["draft_emails"] = {
        "metaprompt": run_metaprompt_optimization(
            email_prompt, dataset, metric, None, "draft_emails"
        ),
    }
    if not QUICK_MODE:
        results["draft_emails"].update({
            "fewshot_bayesian": run_fewshot_bayesian_optimization(
                email_prompt, dataset, metric, None, "draft_emails"
            ),
            "evolutionary": run_evolutionary_optimization(
                email_prompt, dataset, metric, None, "draft_emails"
            ),
        })
    
    # Optimize search_contacts prompt - single prompt optimization
    contacts_prompt = prompts["search_contacts"]
    results["search_contacts"] = {
        "metaprompt": run_metaprompt_optimization(
            contacts_prompt, dataset, metric, None, "search_contacts"
        ),
    }
    
    return results


def optimize_verification_agent(datasets: Dict[str, Any]):
    """Optimize VerificationAgent prompts"""
    print(f"\n{'#'*80}")
    print(f"# VERIFICATION AGENT OPTIMIZATION")
    print(f"{'#'*80}")
    
    dataset = datasets["verification"]
    metric = get_verification_metric()
    if not metric:
        print("[ERROR] Failed to get verification metric")
        return {}
    prompts = extract_prompts_from_verification_agent()
    
    results = {}
    
    # Optimize recommend_amount prompt (most critical) - single prompt optimization
    recommend_prompt = prompts["recommend_amount"]
    results["recommend_amount"] = {
        "metaprompt": run_metaprompt_optimization(
            recommend_prompt, dataset, metric, None, "recommend_amount"
        ),
    }
    if not QUICK_MODE:
        results["recommend_amount"].update({
            "hrpo": run_hrpo_optimization(
                recommend_prompt, dataset, metric, None, "recommend_amount"
            ),
            "gepa": run_gepa_optimization(
                recommend_prompt, dataset, metric, None, "recommend_amount"
            ),
        })
    
    # Optimize analyze_claim prompt - single prompt optimization
    analyze_prompt = prompts["analyze_claim"]
    results["analyze_claim"] = {
        "metaprompt": run_metaprompt_optimization(
            analyze_prompt, dataset, metric, None, "analyze_claim"
        ),
    }
    if not QUICK_MODE:
        results["analyze_claim"].update({
            "hrpo": run_hrpo_optimization(
                analyze_prompt, dataset, metric, None, "analyze_claim"
            ),
        })
    
    return results


def main():
    """Main optimization execution"""
    print(f"\n{'='*80}")
    print(f"OPIK AGENT OPTIMIZATION TEST SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*80}")
    
    # Cost warning
    print(f"\n[COST WARNING] This script makes many API calls and can be expensive!")
    print(f"  Current settings:")
    print(f"    Max Trials: {MAX_TRIALS} (reduced from default 5)")
    print(f"    N Samples: {N_SAMPLES if N_SAMPLES else 'all (matches dataset size)'} (reduced from default 50)")
    print(f"    N Threads: {N_THREADS} (reduced from default 4 - sequential)")
    print(f"    Quick Mode: {QUICK_MODE} (only MetaPrompt optimizer if True)")
    
    if QUICK_MODE:
        num_optimizations = 6  # 6 prompts × 1 optimizer (MetaPrompt only)
        print(f"\n  Quick mode enabled: Only MetaPrompt optimizer will run")
        print(f"  Estimated optimizations: ~{num_optimizations} runs")
    else:
        num_optimizations = 17  # Multiple optimizers per prompt
        print(f"\n  Full mode: All optimizers will run")
        print(f"  Estimated optimizations: ~{num_optimizations} runs")
        print(f"  Set OPIK_QUICK_MODE=true to reduce to only MetaPrompt optimizer")
    
    print(f"\n  To further reduce costs:")
    print(f"    - Set OPIK_MAX_TRIALS=1 (minimum)")
    print(f"    - Set OPIK_N_SAMPLES=3 (use fewer dataset items)")
    print(f"    - Set OPIK_QUICK_MODE=true (only MetaPrompt)")
    print(f"{'='*80}")
    
    print(f"\nConfiguration:")
    print(f"  Optimizer Model: {OPTIMIZER_MODEL}")
    print(f"  Max Trials: {MAX_TRIALS}")
    print(f"  N Samples: {N_SAMPLES if N_SAMPLES else 'all'}")
    print(f"  N Threads: {N_THREADS}")
    print(f"  Quick Mode: {QUICK_MODE}")
    
    # Create datasets
    print(f"\n{'='*80}")
    print("STEP 1: Creating datasets...")
    print(f"{'='*80}")
    datasets = create_all_datasets()
    
    # Run optimizations
    print(f"\n{'='*80}")
    print("STEP 2: Running optimizations...")
    print(f"{'='*80}")
    
    all_results = {
        "disaster_monitoring": optimize_disaster_monitoring_agent(datasets),
        "response_coordinator": optimize_response_coordinator_agent(datasets),
        "verification": optimize_verification_agent(datasets),
    }
    
    # Summary
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    
    for agent_name, agent_results in all_results.items():
        print(f"\n{agent_name.upper()}:")
        for prompt_name, prompt_results in agent_results.items():
            print(f"  {prompt_name}:")
            for optimizer_name, result in prompt_results.items():
                if result:
                    improvement = result.score - result.initial_score
                    print(f"    {optimizer_name}: {result.initial_score:.4f} → {result.score:.4f} ({improvement:+.4f})")
                else:
                    print(f"    {optimizer_name}: Failed")
    
    # Save results
    results_file = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        # Convert results to serializable format
        serializable_results = {}
        for agent_name, agent_results in all_results.items():
            serializable_results[agent_name] = {}
            for prompt_name, prompt_results in agent_results.items():
                serializable_results[agent_name][prompt_name] = {}
                for optimizer_name, result in prompt_results.items():
                    if result:
                        serializable_results[agent_name][prompt_name][optimizer_name] = {
                            "initial_score": result.initial_score,
                            "final_score": result.score,
                            "improvement": result.score - result.initial_score,
                            "best_prompt_preview": str(result.prompt)[:500]
                        }
                    else:
                        serializable_results[agent_name][prompt_name][optimizer_name] = None
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"Log file saved to: {log_file_path}")
    print(f"Completed: {datetime.now().isoformat()}")
    print(f"{'='*80}")
    
    # Dashboard information
    if OPIK_AVAILABLE and os.getenv("OPIK_API_KEY"):
        workspace = os.getenv("OPIK_WORKSPACE", "handz-disaster-relief")
        project = os.getenv("OPIK_PROJECT", "disaster-monitoring")
        print(f"\n{'='*80}")
        print("📊 VIEW RESULTS IN OPIK DASHBOARD:")
        print(f"{'='*80}")
        print(f"  Workspace: {workspace}")
        print(f"  Project: {project}")
        print(f"  Dashboard URL: https://app.opik.com/{workspace}/projects/{project}")
        print(f"\n  Navigate to: Evaluation → Optimization runs")
        print(f"  You'll see:")
        print(f"    - All optimization trials with scores")
        print(f"    - Best prompts found by each optimizer")
        print(f"    - Comparison of initial vs optimized scores")
        print(f"    - Detailed trace-level evidence for each trial")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("[WARNING] DASHBOARD LOGGING DISABLED")
        print(f"{'='*80}")
        print("  Set OPIK_API_KEY environment variable to enable dashboard logging")
        print("  Results are still saved locally in the JSON file")
        print(f"{'='*80}")
    
    # Print log file location at the end
    print(f"\n{'='*80}")
    print(f"[LOG] All output logged to: {log_file_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # Restore original stdout/stderr and close log file
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()
        print(f"\n[LOG] Log file saved to: {log_file_path}")
        print(f"[LOG] Completed at: {datetime.now().isoformat()}")
