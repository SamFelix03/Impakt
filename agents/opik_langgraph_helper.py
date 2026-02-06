"""
Helper functions to enhance LangGraph nodes with Opik I/O capture
"""

from typing import Dict, Any, Optional
from opik_integration import OPIK_AVAILABLE, OPIK_PROJECT

def log_current_trace_id(step_name: str = ""):
    """Log the current trace ID for a step"""
    if not OPIK_AVAILABLE:
        print(f"[OPIK DEBUG] [STEP: {step_name}] OPIK_AVAILABLE is False")
        return
    
    try:
        from opik.opik_context import get_current_trace_data
        trace_data = get_current_trace_data()
        
        # Extract trace ID from trace_data
        trace_id = None
        if trace_data:
            if hasattr(trace_data, 'trace_id'):
                trace_id = trace_data.trace_id
            elif isinstance(trace_data, dict) and 'trace_id' in trace_data:
                trace_id = trace_data['trace_id']
            elif hasattr(trace_data, 'id'):
                trace_id = trace_data.id
        
        if trace_id:
            prefix = f"[OPIK] [STEP: {step_name}]" if step_name else "[OPIK]"
            print(f"{prefix} Trace ID: {trace_id}")
            print(f"{prefix} Trace URL: https://app.opik.com/impakt/projects/{OPIK_PROJECT}/traces/{trace_id}")
        else:
            print(f"[OPIK DEBUG] [STEP: {step_name}] Trace ID not found in trace_data")
    except Exception as e:
        # Log the error instead of silently failing
        error_msg = str(e).lower()
        if "no trace" in error_msg or "no span" in error_msg or "cannot import" in error_msg:
            print(f"[OPIK DEBUG] [STEP: {step_name}] ⚠️ No trace context available (normal during LangGraph execution)")
        else:
            print(f"[OPIK DEBUG] [STEP: {step_name}] ❌ Error getting trace ID: {e}")

def update_node_span_with_io(input_data: Optional[Dict[str, Any]] = None, 
                              output_data: Optional[Dict[str, Any]] = None,
                              metadata: Optional[Dict[str, Any]] = None):
    """Update the current LangGraph node span with input/output data"""
    if not OPIK_AVAILABLE:
        return
    
    try:
        from opik.opik_context import update_current_span
        from opik.exceptions import OpikException
        
        span_metadata = {}
        if input_data:
            span_metadata["input"] = input_data
        if output_data:
            span_metadata["output"] = output_data
        if metadata:
            span_metadata.update(metadata)
        
        # Always add project tag
        span_metadata["project"] = OPIK_PROJECT
        if "tags" not in span_metadata:
            span_metadata["tags"] = [OPIK_PROJECT]
        
        update_current_span(metadata=span_metadata)
    except OpikException as e:
        # If there's no span in context (e.g., called outside LangGraph), silently skip
        # This is expected when calling agent methods directly for evaluation
        if "no span in the context" in str(e).lower() or "no trace in the context" in str(e).lower():
            # Silently skip - this is normal when not in LangGraph context
            pass
        else:
            # Re-raise other Opik exceptions
            raise
    except Exception as e:
        # Only print error if it's not about missing span context
        error_msg = str(e).lower()
        if "no span in the context" not in error_msg and "no trace in the context" not in error_msg:
            print(f"[OPIK] Failed to update node span: {e}")


def update_trace_with_feedback(feedback_scores: Dict[str, float],
                               metadata: Optional[Dict[str, Any]] = None):
    """Update the current trace with feedback scores and metadata"""
    if not OPIK_AVAILABLE:
        return
    
    try:
        from opik.opik_context import update_current_trace
        from opik.exceptions import OpikException
        
        trace_metadata = metadata or {}
        trace_metadata["project"] = OPIK_PROJECT
        if "tags" not in trace_metadata:
            trace_metadata["tags"] = [OPIK_PROJECT]
        
        # Convert dictionary to list of feedback score dictionaries (Opik format)
        feedback_scores_list = []
        for name, value in feedback_scores.items():
            feedback_scores_list.append({
                "name": name,
                "value": float(value),
                "reason": f"{name.replace('_', ' ').title()} evaluation score"
            })
        
        # Update trace (project_name is not a valid parameter)
        if feedback_scores_list:
            update_current_trace(
                feedback_scores=feedback_scores_list,
                metadata=trace_metadata
            )
            print(f"[OPIK] Updated trace with {len(feedback_scores_list)} feedback scores")
    except OpikException as e:
        # Trace context not available (trace already closed or not started)
        # This is normal after LangGraph execution completes - the trace is automatically closed
        # Feedback scores should ideally be sent during node execution, not after graph completion
        if "no trace in the context" in str(e).lower():
            # Silently skip - this is expected behavior after LangGraph completes
            pass
        else:
            print(f"[OPIK] Failed to update trace: {e}")
    except Exception as e:
        print(f"[OPIK] Failed to update trace: {e}")
        import traceback
        traceback.print_exc()
