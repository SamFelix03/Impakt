"""
Opik Evaluation Datasets and Test Suites
Comprehensive evaluation infrastructure for all agents
"""

import json
from typing import List, Dict, Any
from datetime import datetime
from opik_integration import llm_judge, OPIK_AVAILABLE
import opik

# Evaluation Datasets

DISASTER_DETECTION_DATASET = [
    {
        "input": {
            "query": "Find recent earthquake disasters",
            "exclude_list": []
        },
        "expected": {
            "has_location": True,
            "has_sources": True,
            "min_sources": 1,
            "disaster_type": "earthquake"
        },
        "metadata": {
            "test_case": "earthquake_detection",
            "difficulty": "medium"
        }
    },
    {
        "input": {
            "query": "Find recent flood disasters in Asia",
            "exclude_list": []
        },
        "expected": {
            "has_location": True,
            "location_contains": ["Asia", "China", "India", "Bangladesh"],
            "has_sources": True,
            "disaster_type": "flood"
        },
        "metadata": {
            "test_case": "flood_detection_asia",
            "difficulty": "medium"
        }
    },
    {
        "input": {
            "query": "Find wildfire disasters",
            "exclude_list": ["California Wildfire 2024"]
        },
        "expected": {
            "has_location": True,
            "has_sources": True,
            "excludes": ["California Wildfire 2024"],
            "disaster_type": "wildfire"
        },
        "metadata": {
            "test_case": "wildfire_exclusion",
            "difficulty": "hard"
        }
    }
]

RELIEF_CALCULATION_DATASET = [
    {
        "input": {
            "disaster_info": "Major earthquake in Chile, magnitude 7.5, affecting 500,000 people, 200 casualties, widespread infrastructure damage",
            "weather_data": {
                "successful": [
                    {"stationName": "Santiago Station", "data": {"temperature": 15, "humidity": 60}}
                ]
            },
            "location": "Chile"
        },
        "expected": {
            "min_amount": 1000000,
            "max_amount": 50000000,
            "reasoning_checks": ["casualties", "affected_population", "infrastructure"]
        },
        "metadata": {
            "test_case": "major_earthquake",
            "difficulty": "medium"
        }
    },
    {
        "input": {
            "disaster_info": "Localized flood in small town, 50 people affected, minor property damage",
            "weather_data": {
                "successful": [
                    {"stationName": "Local Station", "data": {"precipitation": 150}}
                ]
            },
            "location": "Small Town"
        },
        "expected": {
            "min_amount": 10000,
            "max_amount": 500000,
            "reasoning_checks": ["scale", "affected_population"]
        },
        "metadata": {
            "test_case": "localized_flood",
            "difficulty": "easy"
        }
    }
]

EMAIL_QUALITY_DATASET = [
    {
        "input": {
            "disaster_info": "Major earthquake in Chile affecting 500,000 people",
            "contact": {
                "organization": "Red Cross Chile",
                "role": "Emergency Response Coordinator",
                "email": "emergency@redcross.cl"
            },
            "relief_amount": 5000000
        },
        "expected": {
            "has_urgency": True,
            "has_action_items": True,
            "min_action_items": 3,
            "professional_tone": True,
            "tailored_to_org": True
        },
        "metadata": {
            "test_case": "red_cross_email",
            "difficulty": "medium"
        }
    }
]

CLAIM_VERIFICATION_DATASET = [
    {
        "input": {
            "claim_text": "We provided food and medical supplies to 500 affected families. Total cost: $25,000. Receipts attached.",
            "vault_address": "0x1234567890123456789012345678901234567890",  # Test vault address
            "vault_balance": 100000,  # Expected balance for test
            "evidence_summary": "Receipts for food purchases and medical supplies visible in images"
        },
        "expected": {
            "recommended_amount_range": [20000, 30000],
            "conservative": True,
            "respects_vault_balance": True
        },
        "metadata": {
            "test_case": "valid_claim_with_receipts",
            "difficulty": "medium"
        }
    },
    {
        "input": {
            "claim_text": "We need $100,000 for disaster relief",
            "vault_address": "0x1234567890123456789012345678901234567890",  # Test vault address
            "vault_balance": 50000,  # Expected balance for test
            "evidence_summary": "No receipts or evidence provided"
        },
        "expected": {
            "recommended_amount_range": [0, 10000],
            "conservative": True,
            "respects_vault_balance": True
        },
        "metadata": {
            "test_case": "unsubstantiated_claim",
            "difficulty": "hard"
        }
    }
]

TELEGRAM_RESPONSE_DATASET = [
    {
        "input": {
            "user_message": "What happened in the disaster?",
            "disaster_context": {
                "title": "Earthquake in Chile",
                "location": "Chile",
                "description": "Major earthquake affecting 500,000 people"
            }
        },
        "expected": {
            "mentions_disaster": True,
            "provides_location": True,
            "helpful": True,
            "appropriate_tone": True
        },
        "metadata": {
            "test_case": "basic_disaster_query",
            "difficulty": "easy"
        }
    },
    {
        "input": {
            "user_message": "How can I donate?",
            "disaster_context": {
                "title": "Earthquake in Chile",
                "vault_address": "0x123...",
                "target_amount": 5000000
            }
        },
        "expected": {
            "provides_vault_address": True,
            "provides_donation_info": True,
            "helpful": True
        },
        "metadata": {
            "test_case": "donation_query",
            "difficulty": "easy"
        }
    }
]


class EvaluationRunner:
    """Run comprehensive evaluations using Opik"""
    
    def __init__(self):
        self.results = []
    
    def evaluate_disaster_detection(self, agent_function, dataset: List[Dict[str, Any]]):
        """Evaluate disaster detection agent"""
        print("\n" + "="*80)
        print("EVALUATING DISASTER DETECTION AGENT")
        print("="*80)
        
        results = []
        for test_case in dataset:
            print(f"\n[TEST] {test_case['metadata']['test_case']}")
            try:
                # Run agent
                result = agent_function(test_case['input'])
                
                # Evaluate with LLM-as-judge
                if llm_judge:
                    evaluation = llm_judge.evaluate_disaster_detection(
                        disaster_info=result.get('disaster_info', {}).get('raw_response', ''),
                        location=result.get('location', ''),
                        sources=result.get('disaster_info', {}).get('sources', []),
                        expected_criteria=test_case['expected']
                    )
                    
                    # Check expected criteria
                    passed = True
                    checks = []
                    
                    if test_case['expected'].get('has_location'):
                        checks.append(("has_location", bool(result.get('location'))))
                    
                    if test_case['expected'].get('has_sources'):
                        sources = result.get('disaster_info', {}).get('sources', [])
                        checks.append(("has_sources", len(sources) > 0))
                    
                    test_result = {
                        "test_case": test_case['metadata']['test_case'],
                        "passed": passed and all(c[1] for c in checks),
                        "evaluation": evaluation,
                        "checks": checks,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(test_result)
                    print(f"  Score: {evaluation.get('overall_score', 0):.2f}/10")
                    print(f"  Passed: {test_result['passed']}")
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "test_case": test_case['metadata']['test_case'],
                    "passed": False,
                    "error": str(e)
                })
        
        self.results.append({
            "agent": "disaster_detection",
            "results": results,
            "pass_rate": sum(r.get('passed', False) for r in results) / len(results) if results else 0
        })
        return results
    
    def evaluate_relief_calculation(self, agent_function, dataset: List[Dict[str, Any]]):
        """Evaluate relief calculation"""
        print("\n" + "="*80)
        print("EVALUATING RELIEF CALCULATION")
        print("="*80)
        
        results = []
        for test_case in dataset:
            print(f"\n[TEST] {test_case['metadata']['test_case']}")
            try:
                # Run agent
                result = agent_function(test_case['input'])
                calculated_amount = result.get('relief_amount_usd', 0)
                
                # Evaluate with LLM-as-judge
                if llm_judge:
                    evaluation = llm_judge.evaluate_relief_calculation(
                        disaster_info=test_case['input']['disaster_info'],
                        weather_data=test_case['input'].get('weather_data', {}),
                        calculated_amount=calculated_amount,
                        location=test_case['input'].get('location', '')
                    )
                    
                    # Check expected range
                    expected = test_case['expected']
                    in_range = (
                        expected.get('min_amount', 0) <= calculated_amount <= expected.get('max_amount', float('inf'))
                    )
                    
                    test_result = {
                        "test_case": test_case['metadata']['test_case'],
                        "calculated_amount": calculated_amount,
                        "in_expected_range": in_range,
                        "evaluation": evaluation,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(test_result)
                    print(f"  Calculated: ${calculated_amount:,}")
                    print(f"  Expected Range: ${expected.get('min_amount', 0):,} - ${expected.get('max_amount', 0):,}")
                    print(f"  Score: {evaluation.get('overall_score', 0):.2f}/10")
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "test_case": test_case['metadata']['test_case'],
                    "error": str(e)
                })
        
        self.results.append({
            "agent": "relief_calculation",
            "results": results
        })
        return results
    
    def evaluate_email_quality(self, agent_function, dataset: List[Dict[str, Any]]):
        """Evaluate email drafting quality"""
        print("\n" + "="*80)
        print("EVALUATING EMAIL QUALITY")
        print("="*80)
        
        results = []
        for test_case in dataset:
            print(f"\n[TEST] {test_case['metadata']['test_case']}")
            try:
                # Run agent
                result = agent_function(test_case['input'])
                email_draft = result.get('email_draft', {})
                
                # Evaluate with LLM-as-judge
                if llm_judge:
                    evaluation = llm_judge.evaluate_email_quality(
                        email_draft=email_draft,
                        disaster_info=test_case['input']['disaster_info'],
                        contact_info=test_case['input']['contact']
                    )
                    
                    # Check expected criteria
                    expected = test_case['expected']
                    checks = []
                    
                    if expected.get('has_urgency'):
                        body = email_draft.get('body', '').lower()
                        urgency_words = ['urgent', 'immediate', 'critical', 'emergency', 'asap']
                        checks.append(("has_urgency", any(word in body for word in urgency_words)))
                    
                    if expected.get('has_action_items'):
                        body = email_draft.get('body', '')
                        action_items = body.count('-') + body.count('•') + body.count('*')
                        checks.append(("has_action_items", action_items >= expected.get('min_action_items', 3)))
                    
                    test_result = {
                        "test_case": test_case['metadata']['test_case'],
                        "evaluation": evaluation,
                        "checks": checks,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(test_result)
                    print(f"  Score: {evaluation.get('overall_score', 0):.2f}/10")
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "test_case": test_case['metadata']['test_case'],
                    "error": str(e)
                })
        
        self.results.append({
            "agent": "email_quality",
            "results": results
        })
        return results
    
    def evaluate_claim_verification(self, agent_function, dataset: List[Dict[str, Any]]):
        """Evaluate claim verification"""
        print("\n" + "="*80)
        print("EVALUATING CLAIM VERIFICATION")
        print("="*80)
        
        results = []
        for test_case in dataset:
            print(f"\n[TEST] {test_case['metadata']['test_case']}")
            try:
                # Run agent
                result = agent_function(test_case['input'])
                recommended_amount = result.get('recommended_amount_usd', 0)
                
                # Evaluate with LLM-as-judge
                if llm_judge:
                    evaluation = llm_judge.evaluate_claim_verification(
                        claim_text=test_case['input']['claim_text'],
                        recommended_amount=recommended_amount,
                        vault_balance=test_case['input']['vault_balance'],
                        evidence_summary=test_case['input'].get('evidence_summary', '')
                    )
                    
                    # Check expected range
                    expected = test_case['expected']
                    in_range = (
                        expected.get('recommended_amount_range', [0, 0])[0] <= recommended_amount <= 
                        expected.get('recommended_amount_range', [0, float('inf')])[1]
                    )
                    
                    test_result = {
                        "test_case": test_case['metadata']['test_case'],
                        "recommended_amount": recommended_amount,
                        "in_expected_range": in_range,
                        "evaluation": evaluation,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(test_result)
                    print(f"  Recommended: ${recommended_amount:,.2f}")
                    print(f"  Expected Range: ${expected.get('recommended_amount_range', [0, 0])[0]:,.2f} - ${expected.get('recommended_amount_range', [0, 0])[1]:,.2f}")
                    print(f"  Score: {evaluation.get('overall_score', 0):.2f}/10")
            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    "test_case": test_case['metadata']['test_case'],
                    "error": str(e)
                })
        
        self.results.append({
            "agent": "claim_verification",
            "results": results
        })
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_agents_evaluated": len(self.results),
            "agents": {}
        }
        
        for agent_result in self.results:
            agent_name = agent_result['agent']
            results = agent_result['results']
            
            report["agents"][agent_name] = {
                "total_tests": len(results),
                "passed_tests": sum(1 for r in results if r.get('passed', False) or r.get('in_expected_range', False)),
                "average_score": sum(
                    r.get('evaluation', {}).get('overall_score', 0) 
                    for r in results 
                    if r.get('evaluation')
                ) / len([r for r in results if r.get('evaluation')]) if results else 0,
                "results": results
            }
        
        return report


# Create Opik datasets if available
if OPIK_AVAILABLE:
    try:
        client = opik.Opik()
        
        # Convert datasets to Opik format (list of dicts)
        def convert_to_opik_format(dataset_list):
            """Convert evaluation dataset format to Opik dataset format"""
            opik_items = []
            for item in dataset_list:
                # Merge input, expected, and metadata into a single dict
                opik_item = {}
                if "input" in item:
                    opik_item.update(item["input"])
                if "expected" in item:
                    opik_item.update({f"expected_{k}": v for k, v in item["expected"].items()})
                if "metadata" in item:
                    opik_item["metadata"] = item["metadata"]
                opik_items.append(opik_item)
            return opik_items
        
        # Create datasets using Opik client API
        disaster_detection_ds = client.get_or_create_dataset(name="disaster-detection-evaluation")
        disaster_detection_ds.insert(convert_to_opik_format(DISASTER_DETECTION_DATASET))
        disaster_detection_dataset = disaster_detection_ds
        
        relief_calculation_ds = client.get_or_create_dataset(name="relief-calculation-evaluation")
        relief_calculation_ds.insert(convert_to_opik_format(RELIEF_CALCULATION_DATASET))
        relief_calculation_dataset = relief_calculation_ds
        
        email_quality_ds = client.get_or_create_dataset(name="email-quality-evaluation")
        email_quality_ds.insert(convert_to_opik_format(EMAIL_QUALITY_DATASET))
        email_quality_dataset = email_quality_ds
        
        claim_verification_ds = client.get_or_create_dataset(name="claim-verification-evaluation")
        claim_verification_ds.insert(convert_to_opik_format(CLAIM_VERIFICATION_DATASET))
        claim_verification_dataset = claim_verification_ds
        
        telegram_response_ds = client.get_or_create_dataset(name="telegram-response-evaluation")
        telegram_response_ds.insert(convert_to_opik_format(TELEGRAM_RESPONSE_DATASET))
        telegram_response_dataset = telegram_response_ds
        
        print("✅ Opik datasets created successfully")
    except Exception as e:
        print(f"⚠️ Failed to create Opik datasets: {e}")
        import traceback
        traceback.print_exc()
        # Set to None so code doesn't break
        disaster_detection_dataset = None
        relief_calculation_dataset = None
        email_quality_dataset = None
        claim_verification_dataset = None
        telegram_response_dataset = None
else:
    disaster_detection_dataset = None
    relief_calculation_dataset = None
    email_quality_dataset = None
    claim_verification_dataset = None
    telegram_response_dataset = None
