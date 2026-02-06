"""
Dataset Generation for Opik Agent Optimization
Creates training and validation datasets for all 3 agents
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

try:
    import opik
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    print("[WARNING] Opik not installed. Install with: pip install opik")


def create_disaster_monitoring_dataset():
    """Create dataset for DisasterMonitoringAgent optimization"""
    if not OPIK_AVAILABLE:
        print("[ERROR] Opik not available. Cannot create datasets.")
        return None
    
    client = opik.Opik()
    dataset = client.get_or_create_dataset(name="disaster-monitoring-optimization")
    
    # Sample disaster scenarios for optimization
    examples = [
        {
            "disaster_query": "Search for ONE natural disaster that happened recently worldwide. Find earthquake, flood, wildfire, storm, or other major disaster with specific location, impact, casualties, and current status.",
            "location": "Chile",
            "disaster_type": "earthquake",
            "expected_location": "Chile",
            "expected_relief_range_usd": {"min": 1000000, "max": 10000000},
            "metadata": {"split": "train", "disaster_type": "earthquake", "severity": "high"}
        },
        {
            "disaster_query": "Search for ONE natural disaster that happened recently worldwide. Find earthquake, flood, wildfire, storm, or other major disaster with specific location, impact, casualties, and current status.",
            "location": "Madagascar",
            "disaster_type": "flood",
            "expected_location": "Madagascar",
            "expected_relief_range_usd": {"min": 500000, "max": 5000000},
            "metadata": {"split": "train", "disaster_type": "flood", "severity": "medium"}
        },
        {
            "disaster_query": "Search for ONE natural disaster that happened recently worldwide. Find earthquake, flood, wildfire, storm, or other major disaster with specific location, impact, casualties, and current status.",
            "location": "California, USA",
            "disaster_type": "wildfire",
            "expected_location": "California",
            "expected_relief_range_usd": {"min": 2000000, "max": 20000000},
            "metadata": {"split": "train", "disaster_type": "wildfire", "severity": "high"}
        },
        {
            "disaster_query": "Search for ONE natural disaster that happened recently worldwide. Find earthquake, flood, wildfire, storm, or other major disaster with specific location, impact, casualties, and current status.",
            "location": "Philippines",
            "disaster_type": "typhoon",
            "expected_location": "Philippines",
            "expected_relief_range_usd": {"min": 1000000, "max": 10000000},
            "metadata": {"split": "train", "disaster_type": "typhoon", "severity": "high"}
        },
        {
            "disaster_query": "Search for ONE natural disaster that happened recently worldwide. Find earthquake, flood, wildfire, storm, or other major disaster with specific location, impact, casualties, and current status.",
            "location": "Japan",
            "disaster_type": "tsunami",
            "expected_location": "Japan",
            "expected_relief_range_usd": {"min": 5000000, "max": 50000000},
            "metadata": {"split": "validation", "disaster_type": "tsunami", "severity": "high"}
        },
        {
            "disaster_query": "Search for ONE natural disaster that happened recently worldwide. Find earthquake, flood, wildfire, storm, or other major disaster with specific location, impact, casualties, and current status.",
            "location": "Bangladesh",
            "disaster_type": "flood",
            "expected_location": "Bangladesh",
            "expected_relief_range_usd": {"min": 500000, "max": 5000000},
            "metadata": {"split": "validation", "disaster_type": "flood", "severity": "medium"}
        },
        {
            "disaster_query": "Search for ONE natural disaster that happened recently worldwide. Find earthquake, flood, wildfire, storm, or other major disaster with specific location, impact, casualties, and current status.",
            "location": "Indonesia",
            "disaster_type": "volcanic_eruption",
            "expected_location": "Indonesia",
            "expected_relief_range_usd": {"min": 1000000, "max": 10000000},
            "metadata": {"split": "validation", "disaster_type": "volcanic", "severity": "high"}
        },
    ]
    
    # Insert examples
    dataset.insert(examples)
    print(f"[DATASET] Created disaster-monitoring-optimization with {len(examples)} examples")
    return dataset


def create_response_coordinator_dataset():
    """Create dataset for ResponseCoordinatorAgent optimization"""
    if not OPIK_AVAILABLE:
        print("[ERROR] Opik not available. Cannot create datasets.")
        return None
    
    client = opik.Opik()
    dataset = client.get_or_create_dataset(name="response-coordinator-optimization")
    
    examples = [
        {
            "location": "Chile",
            "disaster_summary": "A magnitude 7.5 earthquake struck central Chile, causing widespread damage to infrastructure. Over 500 people injured, 50 confirmed dead. Multiple buildings collapsed in Santiago and surrounding areas. Emergency services are overwhelmed.",
            "sources": [{"title": "Chile Earthquake Report", "url": "https://example.com/chile-earthquake"}],
            "relief_amount_usd": 5000000,
            "relief_amount_eth": 1.5,
            "vault_address": "0x1234567890123456789012345678901234567890",
            "expected_contacts_min": 3,
            "expected_email_quality_min": 7.0,
            "metadata": {"split": "train", "disaster_type": "earthquake", "location_type": "country"}
        },
        {
            "location": "Madagascar",
            "disaster_summary": "Severe flooding in northern Madagascar has displaced over 10,000 people. Heavy rains continue, making rescue operations difficult. Food and medical supplies urgently needed.",
            "sources": [{"title": "Madagascar Flood Update", "url": "https://example.com/madagascar-flood"}],
            "relief_amount_usd": 2000000,
            "relief_amount_eth": 0.6,
            "vault_address": "0x1234567890123456789012345678901234567890",
            "expected_contacts_min": 2,
            "expected_email_quality_min": 7.0,
            "metadata": {"split": "train", "disaster_type": "flood", "location_type": "country"}
        },
        {
            "location": "California, USA",
            "disaster_summary": "Wildfire spreads across 50,000 acres in Northern California. Evacuation orders issued for multiple communities. Air quality severely degraded. Firefighters struggling to contain blaze.",
            "sources": [{"title": "California Wildfire Alert", "url": "https://example.com/california-fire"}],
            "relief_amount_usd": 10000000,
            "relief_amount_eth": 3.0,
            "vault_address": "0x1234567890123456789012345678901234567890",
            "expected_contacts_min": 4,
            "expected_email_quality_min": 8.0,
            "metadata": {"split": "train", "disaster_type": "wildfire", "location_type": "state"}
        },
        {
            "location": "Philippines",
            "disaster_summary": "Typhoon makes landfall in the Philippines, bringing torrential rains and winds up to 150 mph. Coastal areas flooded, power outages widespread. Thousands evacuated.",
            "sources": [{"title": "Philippines Typhoon Report", "url": "https://example.com/philippines-typhoon"}],
            "relief_amount_usd": 8000000,
            "relief_amount_eth": 2.4,
            "vault_address": "0x1234567890123456789012345678901234567890",
            "expected_contacts_min": 3,
            "expected_email_quality_min": 7.5,
            "metadata": {"split": "validation", "disaster_type": "typhoon", "location_type": "country"}
        },
        {
            "location": "Bangladesh",
            "disaster_summary": "Monsoon flooding affects millions in Bangladesh. River levels at record highs. Agricultural land destroyed. Disease outbreaks feared.",
            "sources": [{"title": "Bangladesh Flood Crisis", "url": "https://example.com/bangladesh-flood"}],
            "relief_amount_usd": 3000000,
            "relief_amount_eth": 0.9,
            "vault_address": "0x1234567890123456789012345678901234567890",
            "expected_contacts_min": 2,
            "expected_email_quality_min": 7.0,
            "metadata": {"split": "validation", "disaster_type": "flood", "location_type": "country"}
        },
    ]
    
    dataset.insert(examples)
    print(f"[DATASET] Created response-coordinator-optimization with {len(examples)} examples")
    return dataset


def create_verification_dataset():
    """Create dataset for VerificationAgent optimization"""
    if not OPIK_AVAILABLE:
        print("[ERROR] Opik not available. Cannot create datasets.")
        return None
    
    client = opik.Opik()
    dataset = client.get_or_create_dataset(name="verification-optimization")
    
    examples = [
        {
            "claim_text": "We provided emergency medical supplies to 500 families affected by the earthquake in Chile. Purchased antibiotics, painkillers, bandages, and basic medical equipment. Total cost: $45,000 USD. Receipts attached.",
            "vault_address": "0x1234567890123456789012345678901234567890",
            "vault_balance_usd": 5000000.0,
            "claim_amount_usd": 45000.0,
            "expected_recommended_range": {"min": 40000.0, "max": 50000.0},
            "evidence_quality": "high",
            "metadata": {"split": "train", "claim_type": "medical_supplies", "evidence_type": "receipts"}
        },
        {
            "claim_text": "Distributed food packages to 1000 displaced families in Madagascar flood zone. Rice, beans, cooking oil, and clean water. Cost breakdown: $25,000 for food, $5,000 for transportation. See attached invoices.",
            "vault_address": "0x1234567890123456789012345678901234567890",
            "vault_balance_usd": 2000000.0,
            "claim_amount_usd": 30000.0,
            "expected_recommended_range": {"min": 25000.0, "max": 35000.0},
            "evidence_quality": "high",
            "metadata": {"split": "train", "claim_type": "food_distribution", "evidence_type": "invoices"}
        },
        {
            "claim_text": "Emergency shelter setup for wildfire evacuees in California. Provided tents, blankets, and basic amenities for 200 people for 2 weeks. Total expenses: $60,000. Documentation available at https://example.com/shelter-docs",
            "vault_address": "0x1234567890123456789012345678901234567890",
            "vault_balance_usd": 10000000.0,
            "claim_amount_usd": 60000.0,
            "expected_recommended_range": {"min": 50000.0, "max": 65000.0},
            "evidence_quality": "medium",
            "metadata": {"split": "train", "claim_type": "shelter", "evidence_type": "web_docs"}
        },
        {
            "claim_text": "We need $100,000 for disaster relief but don't have receipts yet. Will provide documentation later.",
            "vault_address": "0x1234567890123456789012345678901234567890",
            "vault_balance_usd": 5000000.0,
            "claim_amount_usd": 100000.0,
            "expected_recommended_range": {"min": 0.0, "max": 20000.0},
            "evidence_quality": "low",
            "metadata": {"split": "validation", "claim_type": "unspecified", "evidence_type": "none"}
        },
        {
            "claim_text": "Medical team deployed to Philippines typhoon area. Treated 300 patients, provided vaccinations, and set up temporary clinic. Costs: $35,000 for medical supplies, $10,000 for team travel, $5,000 for clinic setup. All receipts provided.",
            "vault_address": "0x1234567890123456789012345678901234567890",
            "vault_balance_usd": 8000000.0,
            "claim_amount_usd": 50000.0,
            "expected_recommended_range": {"min": 45000.0, "max": 55000.0},
            "evidence_quality": "high",
            "metadata": {"split": "validation", "claim_type": "medical_services", "evidence_type": "receipts"}
        },
        {
            "claim_text": "Distributed clean water and sanitation kits to 2000 families in Bangladesh flood area. Water purification tablets, soap, buckets. Total: $40,000. See evidence: https://example.com/water-distribution",
            "vault_address": "0x1234567890123456789012345678901234567890",
            "vault_balance_usd": 3000000.0,
            "claim_amount_usd": 40000.0,
            "expected_recommended_range": {"min": 35000.0, "max": 45000.0},
            "evidence_quality": "medium",
            "metadata": {"split": "validation", "claim_type": "water_sanitation", "evidence_type": "web_docs"}
        },
    ]
    
    dataset.insert(examples)
    print(f"[DATASET] Created verification-optimization with {len(examples)} examples")
    return dataset


def create_all_datasets():
    """Create all datasets for optimization"""
    print("[DATASET] Creating datasets for Opik Agent Optimization...")
    
    disaster_dataset = create_disaster_monitoring_dataset()
    coordinator_dataset = create_response_coordinator_dataset()
    verification_dataset = create_verification_dataset()
    
    print("[DATASET] All datasets created successfully!")
    return {
        "disaster_monitoring": disaster_dataset,
        "response_coordinator": coordinator_dataset,
        "verification": verification_dataset
    }


if __name__ == "__main__":
    create_all_datasets()
