"""
Metric Definitions for Opik Agent Optimization
Composite metrics combining quality and efficiency for all 3 agents
"""

from typing import Dict, Any, Optional
from opik.evaluation.metrics.score_result import ScoreResult

# Import opik-optimizer components
try:
    from opik_optimizer import MultiMetricObjective
    OPIK_OPTIMIZER_AVAILABLE = True
except ImportError as e:
    OPIK_OPTIMIZER_AVAILABLE = False
    MultiMetricObjective = None
    print(f"[WARNING] opik-optimizer MultiMetricObjective not available: {e}")

# Import opik evaluation metrics
try:
    from opik.evaluation.metrics import TotalSpanCost, SpanDuration
except ImportError:
    TotalSpanCost = None
    SpanDuration = None

# Import llm_judge separately to avoid circular imports
# Use lazy import to ensure it's initialized when needed
_llm_judge_cache = None

def get_llm_judge():
    """Get LLM judge instance, initializing if needed"""
    global _llm_judge_cache
    if _llm_judge_cache is None:
        try:
            from opik_integration import llm_judge as imported_judge
            _llm_judge_cache = imported_judge
            if _llm_judge_cache is None:
                # Try to initialize if None
                from opik_integration import LLMJudgeEvaluator
                import os
                if os.getenv("OPENAI_API_KEY"):
                    _llm_judge_cache = LLMJudgeEvaluator()
                    print("[METRICS] Successfully initialized llm_judge")
        except ImportError as e:
            print(f"[METRICS] ERROR: Failed to import llm_judge: {str(e)}")
            _llm_judge_cache = False  # Use False to indicate not available
    return _llm_judge_cache if _llm_judge_cache is not False else None

# Import OPIK_AVAILABLE
try:
    from opik_integration import OPIK_AVAILABLE
except ImportError:
    OPIK_AVAILABLE = False


def disaster_detection_quality(dataset_item: Dict[str, Any], llm_output: str, task_span=None) -> ScoreResult:
    """Evaluate disaster detection quality using LLM-as-judge"""
    llm_judge = get_llm_judge()
    if not llm_judge:
        # Return a default score instead of 0.0 to indicate metric is working
        # Check if output contains expected location as a simple heuristic
        expected_location = dataset_item.get("expected_location", "").lower()
        output_lower = llm_output.lower()
        if expected_location and expected_location in output_lower:
            return ScoreResult(name="disaster_detection_quality", value=0.5, reason="LLM judge not available, using location match heuristic")
        return ScoreResult(name="disaster_detection_quality", value=0.3, reason="LLM judge not available, basic check")
    
    try:
        # Extract location from output (simplified - in real scenario, parse from structured output)
        location = dataset_item.get("expected_location", "")
        sources = dataset_item.get("sources", [])
        
        evaluation = llm_judge.evaluate_disaster_detection(
            disaster_info=llm_output,
            location=location,
            sources=sources,
            expected_criteria={},
            send_to_opik=False  # Don't send during optimization to avoid duplicates
        )
        
        score = evaluation.get("overall_score", 0.0) / 10.0  # Normalize to 0-1
        return ScoreResult(
            name="disaster_detection_quality",
            value=score,
            reason=f"Disaster detection quality: {evaluation.get('overall_score', 0):.2f}/10"
        )
    except Exception as e:
        # Return a non-zero default score to indicate metric ran but had an error
        print(f"[METRIC ERROR] disaster_detection_quality: {str(e)}")
        return ScoreResult(name="disaster_detection_quality", value=0.2, reason=f"Evaluation error: {str(e)}")


def location_extraction_accuracy(dataset_item: Dict[str, Any], llm_output: str, task_span=None) -> ScoreResult:
    """Evaluate location extraction accuracy"""
    """Check if location was correctly extracted"""
    expected_location = dataset_item.get("expected_location", "").lower().strip()
    
    # Simple check: see if expected location appears in output
    output_lower = llm_output.lower()
    is_correct = expected_location in output_lower if expected_location else False
    
    return ScoreResult(
        name="location_extraction_accuracy",
        value=1.0 if is_correct else 0.0,
        reason=f"Location match: {expected_location} {'found' if is_correct else 'not found'}"
    )


def relief_calculation_appropriateness(dataset_item: Dict[str, Any], llm_output: str, task_span=None) -> ScoreResult:
    """Evaluate relief calculation using LLM-as-judge"""
    llm_judge = get_llm_judge()
    try:
        import re
        # Extract numeric amount from output
        numbers = re.findall(r'\d+', llm_output.replace(',', ''))
        calculated_amount = float(numbers[0]) if numbers else 0.0
        
        # Fallback: Check if amount is in expected range
        expected_range = dataset_item.get("expected_relief_range_usd", {})
        if expected_range and calculated_amount > 0:
            min_amount = expected_range.get("min", 0)
            max_amount = expected_range.get("max", float('inf'))
            if min_amount <= calculated_amount <= max_amount:
                fallback_score = 0.7  # Good if in range
            else:
                fallback_score = 0.3  # Partial credit if out of range
        else:
            fallback_score = 0.5 if calculated_amount > 0 else 0.2  # Some credit if number found
        
        if not llm_judge:
            return ScoreResult(
                name="relief_calculation_appropriateness", 
                value=fallback_score, 
                reason=f"LLM judge not available, using range check. Amount: ${calculated_amount:,.0f}"
            )
        
        disaster_info = llm_output[:500]  # Use output as disaster info context
        
        evaluation = llm_judge.evaluate_relief_calculation(
            disaster_info=disaster_info,
            weather_data={},
            calculated_amount=calculated_amount,
            location=dataset_item.get("location", ""),
            send_to_opik=False
        )
        
        score = evaluation.get("overall_score", 0.0) / 10.0
        return ScoreResult(
            name="relief_calculation_appropriateness",
            value=score,
            reason=f"Relief calculation quality: {evaluation.get('overall_score', 0):.2f}/10"
        )
    except Exception as e:
        print(f"[METRIC ERROR] relief_calculation_appropriateness: {str(e)}")
        return ScoreResult(name="relief_calculation_appropriateness", value=0.3, reason=f"Evaluation error: {str(e)}")


def contact_search_quality(dataset_item: Dict[str, Any], llm_output: str, task_span=None) -> ScoreResult:
    """Evaluate contact search quality"""
    """Evaluate contact search quality - check if valid emails found"""
    try:
        import json
        import re
        
        # Try to extract JSON from output
        json_match = re.search(r'\{[^{}]*"contacts"[^{}]*\}', llm_output, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            contacts = parsed.get("contacts", [])
        else:
            # Fallback: count email-like patterns
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', llm_output)
            contacts = [{"email": email} for email in emails]
        
        valid_emails = [c for c in contacts if c.get("email") and "@" in c.get("email", "")]
        expected_min = dataset_item.get("expected_contacts_min", 1)
        
        # Score based on finding at least expected minimum
        score = min(1.0, len(valid_emails) / max(expected_min, 1))
        
        return ScoreResult(
            name="contact_search_quality",
            value=score,
            reason=f"Found {len(valid_emails)} valid contacts (expected min: {expected_min})"
        )
    except Exception as e:
        print(f"[METRIC ERROR] contact_search_quality: {str(e)}")
        # Fallback: check if output contains email-like patterns
        import re
        emails_found = len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', llm_output))
        fallback_score = min(0.6, emails_found * 0.2) if emails_found > 0 else 0.2
        return ScoreResult(name="contact_search_quality", value=fallback_score, reason=f"Evaluation error: {str(e)}, found {emails_found} emails")


def email_quality_score(dataset_item: Dict[str, Any], llm_output: str, task_span=None) -> ScoreResult:
    """Evaluate email quality using LLM-as-judge"""
    llm_judge = get_llm_judge()
    try:
        import json
        import re
        
        # Extract email drafts from JSON output
        json_match = re.search(r'\{[^{}]*"emails"[^{}]*\}', llm_output, re.DOTALL)
        if not json_match:
            # Fallback: check if output looks like email content
            has_subject = "subject" in llm_output.lower() or "to:" in llm_output.lower()
            has_body = len(llm_output) > 100
            fallback_score = 0.4 if (has_subject and has_body) else 0.2
            return ScoreResult(name="email_quality_score", value=fallback_score, reason="No email JSON found, using format check")
        
        parsed = json.loads(json_match.group(0))
        emails = parsed.get("emails", [])
        
        if not emails:
            return ScoreResult(name="email_quality_score", value=0.3, reason="No emails found in JSON")
        
        # Fallback score based on email structure
        email_draft = emails[0]
        has_subject = bool(email_draft.get("subject"))
        has_body = bool(email_draft.get("body"))
        has_to = bool(email_draft.get("to_email"))
        fallback_score = 0.5 + (0.1 if has_subject else 0) + (0.1 if has_body else 0) + (0.1 if has_to else 0)
        
        if not llm_judge:
            return ScoreResult(
                name="email_quality_score", 
                value=fallback_score, 
                reason=f"LLM judge not available, using structure check. Found {len(emails)} email(s)"
            )
        
        # Evaluate first email
        disaster_info = dataset_item.get("disaster_summary", "")
        contact_info = {"organization": "Test Org", "role": "Emergency Coordinator"}
        
        evaluation = llm_judge.evaluate_email_quality(
            email_draft=email_draft,
            disaster_info=disaster_info,
            contact_info=contact_info,
            send_to_opik=False
        )
        
        score = evaluation.get("overall_score", 0.0) / 10.0
        return ScoreResult(
            name="email_quality_score",
            value=score,
            reason=f"Email quality: {evaluation.get('overall_score', 0):.2f}/10"
        )
    except Exception as e:
        print(f"[METRIC ERROR] email_quality_score: {str(e)}")
        return ScoreResult(name="email_quality_score", value=0.3, reason=f"Evaluation error: {str(e)}")


def claim_verification_accuracy(dataset_item: Dict[str, Any], llm_output: str, task_span=None) -> ScoreResult:
    """Evaluate claim verification quality using LLM-as-judge"""
    llm_judge = get_llm_judge()
    try:
        import json
        import re
        
        # Extract recommended amount from JSON output
        json_match = re.search(r'\{[^{}]*"recommended_amount_usd"[^{}]*\}', llm_output, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            recommended_amount = parsed.get("recommended_amount_usd", 0.0)
        else:
            # Fallback: extract number
            numbers = re.findall(r'\d+\.?\d*', llm_output)
            recommended_amount = float(numbers[0]) if numbers else 0.0
        
        # Fallback score based on whether amount was found
        fallback_score = 0.5 if recommended_amount > 0 else 0.3
        
        # Check if output contains verification keywords
        verification_keywords = ["verified", "approved", "recommend", "amount", "claim"]
        has_keywords = any(keyword in llm_output.lower() for keyword in verification_keywords)
        if has_keywords:
            fallback_score = max(fallback_score, 0.6)
        
        if not llm_judge:
            return ScoreResult(
                name="claim_verification_accuracy", 
                value=fallback_score, 
                reason=f"LLM judge not available, using keyword/amount check. Amount: ${recommended_amount:,.0f}"
            )
        
        claim_text = dataset_item.get("claim_text", "")
        vault_balance = dataset_item.get("vault_balance_usd", 0.0)
        evidence_summary = dataset_item.get("evidence_summary", llm_output[:500])
        
        evaluation = llm_judge.evaluate_claim_verification(
            claim_text=claim_text,
            recommended_amount=recommended_amount,
            vault_balance=vault_balance,
            evidence_summary=evidence_summary,
            send_to_opik=False
        )
        
        score = evaluation.get("overall_score", 0.0) / 10.0
        return ScoreResult(
            name="claim_verification_accuracy",
            value=score,
            reason=f"Claim verification quality: {evaluation.get('overall_score', 0):.2f}/10"
        )
    except Exception as e:
        print(f"[METRIC ERROR] claim_verification_accuracy: {str(e)}")
        return ScoreResult(name="claim_verification_accuracy", value=0.3, reason=f"Evaluation error: {str(e)}")


def constraint_compliance(dataset_item: Dict[str, Any], llm_output: str, task_span=None) -> ScoreResult:
    """Check if recommendation respects constraints (vault balance, claim amount)"""
    try:
        import json
        import re
        
        # Extract recommended amount
        json_match = re.search(r'\{[^{}]*"recommended_amount_usd"[^{}]*\}', llm_output, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            recommended_amount = parsed.get("recommended_amount_usd", 0.0)
        else:
            numbers = re.findall(r'\d+\.?\d*', llm_output)
            recommended_amount = float(numbers[0]) if numbers else 0.0
        
        vault_balance = dataset_item.get("vault_balance_usd", float('inf'))
        claim_amount = dataset_item.get("claim_amount_usd")
        
        # Check constraints
        within_vault = recommended_amount <= vault_balance
        within_claim = recommended_amount <= claim_amount if claim_amount else True
        
        score = 1.0 if (within_vault and within_claim) else 0.0
        return ScoreResult(
            name="constraint_compliance",
            value=score,
            reason=f"Constraints: vault={within_vault}, claim={within_claim}"
        )
    except Exception as e:
        print(f"[METRIC ERROR] constraint_compliance: {str(e)}")
        # Fallback: assume partial compliance
        return ScoreResult(name="constraint_compliance", value=0.5, reason=f"Evaluation error: {str(e)}")


def cost_efficiency(dataset_item: Dict[str, Any], llm_output: str, task_span=None) -> ScoreResult:
    """Calculate cost efficiency using span cost"""
    # If task_span is not available, return a neutral score (0.5) instead of 0.0
    # This prevents the entire metric from being 0.0 when span is missing
    if not OPIK_AVAILABLE or not task_span:
        # Return neutral score - cost efficiency can't be measured without span
        return ScoreResult(name="cost_efficiency", value=0.5, reason="Task span not available, using neutral score")
    
    try:
        if TotalSpanCost is None:
            return ScoreResult(name="cost_efficiency", value=0.5, reason="TotalSpanCost not available, using neutral score")
        
        cost_metric = TotalSpanCost()
        cost_result = cost_metric.score(task_span=task_span)
        cost_usd = cost_result.value
        
        # Normalize: lower cost = higher score (inverted)
        # Assume baseline cost of $1.00, score = 1.0 - min(1.0, cost / 1.0)
        normalized_score = max(0.0, 1.0 - min(1.0, cost_usd / 1.0))
        
        return ScoreResult(
            name="cost_efficiency",
            value=normalized_score,
            reason=f"Cost: ${cost_usd:.4f} USD"
        )
    except Exception as e:
        # Return neutral score instead of 0.0
        print(f"[METRIC ERROR] cost_efficiency: {str(e)}")
        return ScoreResult(name="cost_efficiency", value=0.5, reason=f"Evaluation error: {str(e)}")


# Composite metrics for each agent - return callable functions
def get_disaster_monitoring_metric():
    """Composite metric for DisasterMonitoringAgent - returns callable function"""
    if not OPIK_OPTIMIZER_AVAILABLE:
        return None
    
    objective = MultiMetricObjective(
        metrics=[
            lambda item, output: disaster_detection_quality(item, output, None),
            lambda item, output: location_extraction_accuracy(item, output, None),
            lambda item, output: relief_calculation_appropriateness(item, output, None),
            lambda item, output: cost_efficiency(item, output, None),
        ],
        weights=[0.4, 0.2, 0.3, -0.1],  # Negative weight for cost (minimize)
        name="disaster_monitoring_composite"
    )
    
    # Return a callable wrapper function that properly handles the MultiMetricObjective
    def metric_function(dataset_item, llm_output, task_span=None):
        try:
            # MultiMetricObjective only accepts dataset_item and llm_output (no task_span)
            # The individual metrics will receive task_span=None from MultiMetricObjective
            result = objective(dataset_item, llm_output)
            
            # Debug: Print result if score is suspiciously low
            if hasattr(result, 'value') and result.value == 0.0:
                print(f"[METRIC WARNING] disaster_monitoring_metric returned 0.0. Reason: {getattr(result, 'reason', 'unknown')}")
            
            # MultiMetricObjective returns ScoreResult or float - return as-is
            return result
        except Exception as e:
            # Print error for debugging
            print(f"[METRIC ERROR] disaster_monitoring_metric failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return a default score on error (non-zero to indicate metric ran)
            from opik.evaluation.metrics.score_result import ScoreResult
            return ScoreResult(name="error", value=0.1, reason=f"Metric error: {str(e)}")
    
    # Ensure it's recognized as callable with proper signature
    metric_function.__name__ = "disaster_monitoring_metric"
    metric_function.__qualname__ = "disaster_monitoring_metric"
    return metric_function


def get_response_coordinator_metric():
    """Composite metric for ResponseCoordinatorAgent - returns callable function"""
    if not OPIK_OPTIMIZER_AVAILABLE:
        return None
    
    objective = MultiMetricObjective(
        metrics=[
            lambda item, output: contact_search_quality(item, output, None),
            lambda item, output: email_quality_score(item, output, None),
            lambda item, output: cost_efficiency(item, output, None),
        ],
        weights=[0.3, 0.6, -0.1],  # Email quality is most important
        name="response_coordinator_composite"
    )
    
    # Return a callable wrapper function that properly handles the MultiMetricObjective
    def metric_function(dataset_item, llm_output, task_span=None):
        try:
            # MultiMetricObjective only accepts dataset_item and llm_output (no task_span)
            result = objective(dataset_item, llm_output)
            return result
        except Exception as e:
            print(f"[METRIC ERROR] response_coordinator_metric failed: {str(e)}")
            import traceback
            traceback.print_exc()
            from opik.evaluation.metrics.score_result import ScoreResult
            return ScoreResult(name="error", value=0.1, reason=f"Metric error: {str(e)}")
    
    metric_function.__name__ = "response_coordinator_metric"
    metric_function.__qualname__ = "response_coordinator_metric"
    return metric_function


def get_verification_metric():
    """Composite metric for VerificationAgent - returns callable function"""
    if not OPIK_OPTIMIZER_AVAILABLE:
        return None
    
    objective = MultiMetricObjective(
        metrics=[
            lambda item, output: claim_verification_accuracy(item, output, None),
            lambda item, output: constraint_compliance(item, output, None),
            lambda item, output: cost_efficiency(item, output, None),
        ],
        weights=[0.5, 0.4, -0.1],
        name="verification_composite"
    )
    
    # Return a callable wrapper function that properly handles the MultiMetricObjective
    def metric_function(dataset_item, llm_output, task_span=None):
        try:
            # MultiMetricObjective only accepts dataset_item and llm_output (no task_span)
            result = objective(dataset_item, llm_output)
            return result
        except Exception as e:
            print(f"[METRIC ERROR] verification_metric failed: {str(e)}")
            import traceback
            traceback.print_exc()
            from opik.evaluation.metrics.score_result import ScoreResult
            return ScoreResult(name="error", value=0.1, reason=f"Metric error: {str(e)}")
    
    metric_function.__name__ = "verification_metric"
    metric_function.__qualname__ = "verification_metric"
    return metric_function
