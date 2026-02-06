"""
Opik Integration Module for Disaster Relief System
Provides comprehensive observability, evaluation, and optimization capabilities
"""

import os
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

load_dotenv()

try:
    from opik import Opik, configure, start_as_current_trace, start_as_current_span
    from opik.evaluation import evaluate, evaluate_prompt
    # Dataset might not be available in all versions - import separately
    try:
        from opik.evaluation import Dataset
    except ImportError:
        Dataset = None  # Will handle gracefully
    try:
        from opik.optimization import AgentOptimizer, MetaPromptOptimizer, EvolutionaryOptimizer
    except ImportError:
        # opik-optimizer might not be installed
        AgentOptimizer = None
        MetaPromptOptimizer = None
        EvolutionaryOptimizer = None
    from opik.integrations.langchain import OpikTracer as LangGraphOpikTracer
    try:
        from opik.integrations.openai import trace_openai
    except ImportError:
        trace_openai = None
    OPIK_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Opik not installed or import error: {e}")
    print("Install with: pip install opik opik-optimizer")
    OPIK_AVAILABLE = False
    # Create mock classes for development
    class Opik:
        def __init__(self, *args, **kwargs):
            pass
    class Dataset:
        pass
    class LangGraphOpikTracer:
        def __init__(self, *args, **kwargs):
            pass
    def evaluate(*args, **kwargs):
        return {}
    def start_as_current_trace(*args, **kwargs):
        return lambda x: x
    def start_as_current_span(*args, **kwargs):
        return lambda x: x

# Opik Configuration
OPIK_API_KEY = os.getenv("OPIK_API_KEY", "")
OPIK_WORKSPACE = os.getenv("OPIK_WORKSPACE", "handz-disaster-relief")
OPIK_PROJECT = os.getenv("OPIK_PROJECT", "disaster-monitoring")

# Initialize Opik if available
if OPIK_AVAILABLE:
    try:
        # Set project name as environment variable for Opik to pick up
        os.environ["OPIK_PROJECT_NAME"] = OPIK_PROJECT
        
        if OPIK_API_KEY:
            configure(api_key=OPIK_API_KEY, workspace=OPIK_WORKSPACE)
        else:
            configure(workspace=OPIK_WORKSPACE)
        opik_client = Opik()
        print(f"[OK] Opik initialized successfully (workspace: {OPIK_WORKSPACE}, project: {OPIK_PROJECT})")
    except Exception as e:
        print(f"[WARNING] Opik initialization failed: {e}")
        opik_client = None
else:
    opik_client = None
    print("[WARNING] Opik SDK not installed")


@dataclass
class AgentMetrics:
    """Metrics for agent performance tracking"""
    agent_name: str
    timestamp: str
    execution_time: float
    success: bool
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    custom_metrics: Dict[str, Any] = None
    
    def to_dict(self):
        return asdict(self)


class OpikTracer:
    """Context manager for Opik tracing with rich metadata"""
    
    def __init__(self, trace_name: str, metadata: Optional[Dict[str, Any]] = None, project_tag: Optional[str] = None):
        self.trace_name = trace_name
        self.metadata = metadata or {}
        self.project_name = project_tag or OPIK_PROJECT
        # Add project as tag/metadata
        if project_tag:
            self.metadata["project"] = project_tag
            if "tags" not in self.metadata:
                self.metadata["tags"] = []
            if project_tag not in self.metadata["tags"]:
                self.metadata["tags"].append(project_tag)
        self.trace_context = None
        
    def __enter__(self):
        if OPIK_AVAILABLE and opik_client:
            self.trace_context = start_as_current_trace(
                name=self.trace_name,
                project_name=self.project_name,
                metadata=self.metadata,
                tags=self.metadata.get("tags", [])
            )
            return self.trace_context.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.trace_context:
            return self.trace_context.__exit__(exc_type, exc_val, exc_tb)
        return False


class OpikSpan:
    """Context manager for Opik spans with rich I/O data"""
    
    def __init__(self, span_name: str, metadata: Optional[Dict[str, Any]] = None, 
                 input_data: Optional[Dict[str, Any]] = None, 
                 output_data: Optional[Dict[str, Any]] = None):
        self.span_name = span_name
        self.metadata = metadata or {}
        self.input_data = input_data
        self.output_data = output_data
        self.span_context = None
        
    def __enter__(self):
        if OPIK_AVAILABLE and opik_client:
            # Add input/output to metadata for visibility
            if self.input_data:
                self.metadata["input"] = self.input_data
            if self.output_data:
                self.metadata["output"] = self.output_data
            
            self.span_context = start_as_current_span(
                name=self.span_name,
                metadata=self.metadata
            )
            return self.span_context.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span_context:
            # Update span with output data if available
            if self.output_data and OPIK_AVAILABLE:
                try:
                    from opik.opik_context import update_current_span
                    update_current_span(metadata={"output": self.output_data})
                except:
                    pass
            return self.span_context.__exit__(exc_type, exc_val, exc_tb)
        return False


def trace_agent_execution(agent_name: str, operation: str):
    """Decorator for tracing agent operations"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            with OpikTracer(f"{agent_name}.{operation}", {
                "agent": agent_name,
                "operation": operation,
                "timestamp": datetime.now().isoformat()
            }):
                with OpikSpan(f"{operation}_execution"):
                    start_time = datetime.now()
                    try:
                        result = func(*args, **kwargs)
                        execution_time = (datetime.now() - start_time).total_seconds()
                        
                        # Log success metrics
                        if OPIK_AVAILABLE and opik_client:
                            metrics = AgentMetrics(
                                agent_name=agent_name,
                                timestamp=datetime.now().isoformat(),
                                execution_time=execution_time,
                                success=True
                            )
                            # Store metrics (would integrate with Opik metrics API)
                        
                        return result
                    except Exception as e:
                        execution_time = (datetime.now() - start_time).total_seconds()
                        
                        # Log error metrics
                        if OPIK_AVAILABLE and opik_client:
                            metrics = AgentMetrics(
                                agent_name=agent_name,
                                timestamp=datetime.now().isoformat(),
                                execution_time=execution_time,
                                success=False,
                                error=str(e)
                            )
                        
                        raise
        return wrapper
    return decorator


class LLMJudgeEvaluator:
    """LLM-as-Judge evaluation system with Opik feedback score integration"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for LLM judge")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        print(f"[LLMJudgeEvaluator] Initialized with model: {model}")
    
    def _send_feedback_to_opik(self, trace_name: str, evaluation: Dict[str, Any], metric_name: str = "quality_score"):
        """Send evaluation results as feedback scores to Opik"""
        if not OPIK_AVAILABLE:
            return
        
        try:
            from opik.opik_context import update_current_trace
            from opik.exceptions import OpikException
            
            # Opik expects feedback_scores as a list of dictionaries
            feedback_scores = []
            
            # Extract overall score
            overall_score = evaluation.get('overall_score', 0)
            if overall_score:
                feedback_scores.append({
                    "name": metric_name,
                    "value": float(overall_score),
                    "reason": evaluation.get('reasoning', f"Overall {metric_name} score")
                })
            
            # Add individual criterion scores if available
            for key, value in evaluation.items():
                if key not in ['overall_score', 'reasoning', 'suggested_range'] and isinstance(value, (int, float)):
                    feedback_scores.append({
                        "name": f"{metric_name}_{key}",
                        "value": float(value),
                        "reason": f"{key.replace('_', ' ').title()} criterion score"
                    })
            
            # Update current trace with feedback scores
            if feedback_scores:
                update_current_trace(feedback_scores=feedback_scores)
                print(f"[OPIK] Sent feedback scores to Opik: {len(feedback_scores)} scores")
        except OpikException as e:
            # If there's no trace in context (e.g., called outside LangGraph), silently skip
            # This is expected when calling agent methods directly for evaluation
            if "no trace in the context" in str(e).lower():
                # Silently skip - this is normal when not in LangGraph context
                pass
            else:
                # Re-raise other Opik exceptions
                print(f"[OPIK] Failed to send feedback scores: {e}")
        except Exception as e:
            # Only print error if it's not about missing trace context
            error_msg = str(e).lower()
            if "no trace in the context" not in error_msg:
                print(f"[OPIK] Failed to send feedback scores: {e}")
                import traceback
                traceback.print_exc()
    
    def evaluate_disaster_detection(
        self,
        disaster_info: str,
        location: str,
        sources: List[Dict[str, str]],
        expected_criteria: Dict[str, Any],
        send_to_opik: bool = True
    ) -> Dict[str, Any]:
        """Evaluate disaster detection quality"""
        prompt = f"""You are evaluating a disaster detection system. Rate the quality of this disaster detection.

DISASTER INFORMATION:
{disaster_info}

LOCATION EXTRACTED: {location}

SOURCES:
{json.dumps(sources, indent=2)}

EVALUATION CRITERIA:
1. Accuracy: Is the disaster information accurate and verifiable?
2. Relevance: Is this a real, recent disaster?
3. Completeness: Does it include location, impact, and sources?
4. Source Quality: Are sources reliable and recent?
5. Location Precision: Is the location correctly extracted?

Rate each criterion 1-10 and provide overall score (1-10).
Return JSON:
{{
    "accuracy": 8,
    "relevance": 9,
    "completeness": 7,
    "source_quality": 8,
    "location_precision": 9,
    "overall_score": 8.2,
    "reasoning": "Brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            evaluation = json.loads(response.choices[0].message.content)
            
            # Send feedback scores to Opik
            if send_to_opik:
                self._send_feedback_to_opik("disaster_detection", evaluation, "disaster_detection_quality")
            
            return evaluation
        except Exception as e:
            return {"error": str(e), "overall_score": 0}
    
    def evaluate_relief_calculation(
        self,
        disaster_info: str,
        weather_data: Dict[str, Any],
        calculated_amount: float,
        location: str,
        send_to_opik: bool = True
    ) -> Dict[str, Any]:
        """Evaluate relief amount calculation quality"""
        prompt = f"""You are evaluating a disaster relief calculation system. Assess if the calculated relief amount is appropriate.

DISASTER INFORMATION:
{disaster_info}

LOCATION: {location}

WEATHER DATA:
{json.dumps(weather_data, indent=2) if weather_data else "No weather data"}

CALCULATED RELIEF AMOUNT: ${calculated_amount:,.2f} USD

EVALUATION CRITERIA:
1. Appropriateness: Is the amount appropriate for the disaster scale?
2. Reasoning: Does it consider casualties, damage, weather conditions?
3. Realism: Is it realistic compared to similar disasters?
4. Completeness: Does it account for all relevant factors?

Rate each criterion 1-10 and provide overall score.
Return JSON:
{{
    "appropriateness": 8,
    "reasoning": 7,
    "realism": 9,
    "completeness": 8,
    "overall_score": 8.0,
    "suggested_range": {{"min": 100000, "max": 5000000}},
    "reasoning": "Brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            evaluation = json.loads(response.choices[0].message.content)
            
            # Send feedback scores to Opik
            if send_to_opik:
                self._send_feedback_to_opik("relief_calculation", evaluation, "relief_calculation_quality")
            
            return evaluation
        except Exception as e:
            return {"error": str(e), "overall_score": 0}
    
    def evaluate_email_quality(
        self,
        email_draft: Dict[str, str],
        disaster_info: str,
        contact_info: Dict[str, Any],
        send_to_opik: bool = True
    ) -> Dict[str, Any]:
        """Evaluate outreach email quality"""
        print(f"[LLM JUDGE] Evaluating email...")
        print(f"[LLM JUDGE] Email subject: {email_draft.get('subject', 'N/A')}")
        print(f"[LLM JUDGE] Email body length: {len(email_draft.get('body', ''))}")
        print(f"[LLM JUDGE] Email body preview: {email_draft.get('body', '')[:200] if email_draft.get('body') else 'EMPTY'}")
        
        prompt = f"""Evaluate the quality of this disaster outreach email.

EMAIL DRAFT:
To: {email_draft.get('to_email', 'N/A')}
Subject: {email_draft.get('subject', 'N/A')}
Body:
{email_draft.get('body', '')}

DISASTER CONTEXT:
{disaster_info}

CONTACT INFO:
{json.dumps(contact_info, indent=2)}

EVALUATION CRITERIA:
1. Urgency: Does it convey appropriate urgency?
2. Clarity: Is it clear and actionable?
3. Relevance: Is it tailored to the recipient?
4. Professionalism: Is it professional and respectful?
5. Action Items: Are action items clear and specific?

IMPORTANT: If the email body is empty, malformed, truncated, or just raw JSON, give it a very low score (1-3/10).
If the email is incomplete or cut off mid-sentence, penalize it heavily.

Rate each criterion 1-10.
Return JSON:
{{
    "urgency": 9,
    "clarity": 8,
    "relevance": 8,
    "professionalism": 9,
    "action_items": 8,
    "overall_score": 8.4,
    "reasoning": "Brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            evaluation = json.loads(response.choices[0].message.content)
            print(f"[LLM JUDGE] Evaluation result: overall_score={evaluation.get('overall_score', 0)}/10")
            
            # Send feedback scores to Opik
            if send_to_opik:
                self._send_feedback_to_opik("email_quality", evaluation, "email_quality_score")
            
            return evaluation
        except Exception as e:
            return {"error": str(e), "overall_score": 0}
    
    def evaluate_claim_verification(
        self,
        claim_text: str,
        recommended_amount: float,
        vault_balance: float,
        evidence_summary: str,
        send_to_opik: bool = True
    ) -> Dict[str, Any]:
        """Evaluate NGO claim verification quality"""
        prompt = f"""Evaluate the quality of this claim verification decision.

CLAIM TEXT:
{claim_text}

RECOMMENDED AMOUNT: ${recommended_amount:,.2f}
VAULT BALANCE: ${vault_balance:,.2f}

EVIDENCE SUMMARY:
{evidence_summary}

EVALUATION CRITERIA:
1. Accuracy: Is the recommended amount appropriate?
2. Evidence Review: Was evidence properly analyzed?
3. Safety: Is the recommendation conservative enough?
4. Reasoning: Is the reasoning sound?
5. Constraints: Are constraints (vault balance, claim amount) respected?

Rate each criterion 1-10.
Return JSON:
{{
    "accuracy": 8,
    "evidence_review": 7,
    "safety": 9,
    "reasoning": 8,
    "constraints": 10,
    "overall_score": 8.4,
    "reasoning": "Brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            evaluation = json.loads(response.choices[0].message.content)
            
            # Send feedback scores to Opik
            if send_to_opik:
                self._send_feedback_to_opik("claim_verification", evaluation, "claim_verification_quality")
            
            return evaluation
        except Exception as e:
            return {"error": str(e), "overall_score": 0}
    
    def evaluate_vote_adjustment(
        self,
        vote_direction: str,
        original_relief_fund: float,
        delta_usd: float,
        updated_relief_fund: float,
        claim_submitted: str,
        disaster_details: str,
        reasoning_summary: str,
        send_to_opik: bool = True
    ) -> Dict[str, Any]:
        """Evaluate vote adjustment quality"""
        prompt = f"""Evaluate the quality of this vote-based relief fund adjustment.

VOTE DIRECTION: {vote_direction}
ORIGINAL RELIEF FUND: ${original_relief_fund:,.2f}
DELTA ADJUSTMENT: ${delta_usd:,.2f}
UPDATED RELIEF FUND: ${updated_relief_fund:,.2f}

CLAIM SUBMITTED:
{claim_submitted[:500]}

DISASTER DETAILS:
{disaster_details[:500]}

REASONING SUMMARY:
{reasoning_summary}

EVALUATION CRITERIA:
1. Vote Alignment: Does the delta match the vote direction (higher = positive, lower = negative)?
2. Magnitude Appropriateness: Is the adjustment magnitude reasonable (5-25% typical, max 30%)?
3. Reasoning Quality: Is the reasoning clear and justified?
4. Constraint Compliance: Are constraints properly enforced (30% cap, sign enforcement)?
5. Context Awareness: Does the adjustment consider the disaster context and claim?

Rate each criterion 1-10.
Return JSON:
{{
    "vote_alignment": 9,
    "magnitude_appropriateness": 8,
    "reasoning_quality": 7,
    "constraint_compliance": 10,
    "context_awareness": 8,
    "overall_score": 8.4,
    "reasoning": "Brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            evaluation = json.loads(response.choices[0].message.content)
            
            # Send feedback scores to Opik
            if send_to_opik:
                self._send_feedback_to_opik("vote_adjustment", evaluation, "vote_adjustment_quality")
            
            return evaluation
        except Exception as e:
            return {"error": str(e), "overall_score": 0}
    
    def evaluate_telegram_response(
        self,
        user_message: str,
        bot_response: str,
        disaster_context: Dict[str, Any],
        send_to_opik: bool = True
    ) -> Dict[str, Any]:
        """Evaluate Telegram bot response quality"""
        prompt = f"""Evaluate the quality of this Telegram bot response.

USER MESSAGE:
{user_message}

BOT RESPONSE:
{bot_response}

DISASTER CONTEXT:
{json.dumps(disaster_context, indent=2)}

EVALUATION CRITERIA:
1. Relevance: Does it answer the user's question?
2. Accuracy: Is the information accurate?
3. Helpfulness: Is it helpful and actionable?
4. Tone: Is the tone appropriate (empathetic, professional)?
5. Completeness: Does it provide sufficient information?

Rate each criterion 1-10.
Return JSON:
{{
    "relevance": 9,
    "accuracy": 8,
    "helpfulness": 8,
    "tone": 9,
    "completeness": 7,
    "overall_score": 8.2,
    "reasoning": "Brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            evaluation = json.loads(response.choices[0].message.content)
            return evaluation
        except Exception as e:
            return {"error": str(e), "overall_score": 0}


class OpikAgentOptimizer:
    """Agent Optimizer for prompt tuning"""
    
    def __init__(self, agent_name: str):
        if not OPIK_AVAILABLE:
            self.optimizer = None
            return
        
        try:
            # Check if optimization modules are available
            if AgentOptimizer is None or MetaPromptOptimizer is None or EvolutionaryOptimizer is None:
                self.optimizer = None
                return
            
            # Initialize optimizer with multiple strategies
            self.optimizer = AgentOptimizer(
                agent_name=agent_name,
                optimizers=[
                    MetaPromptOptimizer(),
                    EvolutionaryOptimizer(),
                ]
            )
        except Exception as e:
            print(f"[WARNING] Failed to initialize Agent Optimizer: {e}")
            self.optimizer = None
    
    def optimize_prompt(
        self,
        base_prompt: str,
        evaluation_function: Callable,
        dataset: List[Dict[str, Any]],
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Optimize a prompt using Opik Agent Optimizer"""
        if not self.optimizer:
            return {"optimized_prompt": base_prompt, "improvement": 0}
        
        try:
            # Create evaluation dataset using Opik client API
            if not OPIK_AVAILABLE or not opik_client:
                return {"optimized_prompt": base_prompt, "error": "Opik client not available"}
            
            # Create a temporary dataset for optimization
            temp_dataset = opik_client.get_or_create_dataset(name=f"temp-optimization-{datetime.now().timestamp()}")
            temp_dataset.insert(dataset)
            eval_dataset = temp_dataset
            
            # Run optimization
            result = self.optimizer.optimize(
                prompt=base_prompt,
                evaluation_fn=evaluation_function,
                dataset=eval_dataset,
                max_iterations=max_iterations
            )
            
            return {
                "optimized_prompt": result.best_prompt,
                "improvement": result.improvement_score,
                "iterations": result.iterations,
                "metrics": result.metrics
            }
        except Exception as e:
            print(f"⚠️ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return {"optimized_prompt": base_prompt, "error": str(e)}


# Global evaluator instance - ALWAYS create LLM judge (it only needs OpenAI, not Opik)
try:
    # Check if OpenAI API key is available
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        # Use gpt-4o-mini for cost efficiency (can be overridden via env var)
        judge_model = os.getenv("OPIK_LLM_JUDGE_MODEL", "gpt-4o-mini")
        llm_judge = LLMJudgeEvaluator(model=judge_model)
        print(f"[LLM JUDGE] Initialized LLM-as-Judge evaluator (model: {llm_judge.model})")
    else:
        llm_judge = None
        print("[LLM JUDGE] WARNING: OPENAI_API_KEY not set. LLM judge not available.")
except Exception as e:
    llm_judge = None
    print(f"[LLM JUDGE] ERROR: Failed to initialize LLM judge: {str(e)}")
