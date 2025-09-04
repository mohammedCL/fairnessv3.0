#!/usr/bin/env python3
"""
Test script to verify the integration of comprehensive fairness metrics
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from app.services.analysis_service import FairnessMetrics

def test_fairness_metrics():
    """Test the comprehensive fairness metrics"""
    print("Testing Comprehensive Fairness Metrics Integration...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create sensitive attribute (binary)
    sensitive_attr = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    
    # Create biased predictions (group 1 has lower approval rate)
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    
    # Create biased predictions
    bias_factor = 0.3
    y_pred = y_true.copy()
    
    # Introduce bias: reduce positive predictions for group 1
    group_1_mask = sensitive_attr == 1
    group_1_positive = (y_pred == 1) & group_1_mask
    flip_indices = np.random.choice(
        np.where(group_1_positive)[0], 
        size=int(np.sum(group_1_positive) * bias_factor), 
        replace=False
    )
    y_pred[flip_indices] = 0
    
    # Create prediction probabilities
    y_prob = y_pred + np.random.normal(0, 0.1, size=n_samples)
    y_prob = np.clip(y_prob, 0, 1)
    
    print(f"Sample size: {n_samples}")
    print(f"Sensitive attribute distribution: {np.bincount(sensitive_attr)}")
    print(f"True labels distribution: {np.bincount(y_true)}")
    print(f"Predicted labels distribution: {np.bincount(y_pred)}")
    
    # Test fairness metrics
    fairness_calc = FairnessMetrics()
    
    print("\n=== Testing Individual Metrics ===")
    
    # Test statistical parity
    stat_parity = fairness_calc.statistical_parity(y_pred, sensitive_attr)
    print(f"Statistical Parity: {stat_parity:.4f}")
    
    # Test disparate impact
    disp_impact = fairness_calc.disparate_impact(y_pred, sensitive_attr)
    print(f"Disparate Impact: {disp_impact:.4f}")
    
    # Test equal opportunity
    eq_opp = fairness_calc.equal_opportunity(y_true, y_pred, sensitive_attr)
    print(f"Equal Opportunity: {eq_opp:.4f}")
    
    # Test equalized odds
    eq_odds = fairness_calc.equalized_odds(y_true, y_pred, sensitive_attr)
    print(f"Equalized Odds: {eq_odds:.4f}")
    
    # Test calibration
    calibration = fairness_calc.calibration(y_true, y_prob, sensitive_attr)
    print(f"Calibration: {calibration:.4f}")
    
    # Test generalized entropy index
    gei = fairness_calc.generalized_entropy_index(y_true, y_pred)
    print(f"Generalized Entropy Index: {gei:.4f}")
    
    print("\n=== Testing Comprehensive Metrics ===")
    
    # Test compute_all_metrics
    all_metrics = fairness_calc.compute_all_metrics(y_true, y_pred, y_prob, sensitive_attr)
    
    print("All computed metrics:")
    for metric_name, value in all_metrics.items():
        if isinstance(value, dict):
            print(f"  {metric_name}: {value}")
        else:
            print(f"  {metric_name}: {value:.4f}")
    
    print("\n=== Testing Multi-class Sensitive Attribute ===")
    
    # Test with multi-class sensitive attribute
    sensitive_attr_multi = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])
    
    stat_parity_multi = fairness_calc.statistical_parity(y_pred, sensitive_attr_multi)
    disp_impact_multi = fairness_calc.disparate_impact(y_pred, sensitive_attr_multi)
    
    print(f"Multi-class Statistical Parity: {stat_parity_multi:.4f}")
    print(f"Multi-class Disparate Impact: {disp_impact_multi:.4f}")
    
    all_metrics_multi = fairness_calc.compute_all_metrics(y_true, y_pred, y_prob, sensitive_attr_multi)
    print(f"Multi-class metrics computed: {len(all_metrics_multi)} metrics")
    print(f"Number of sensitive groups: {all_metrics_multi.get('num_sensitive_groups', 'N/A')}")
    print(f"Is multi-class sensitive: {all_metrics_multi.get('is_multiclass_sensitive', 'N/A')}")
    
    print("\n✅ All fairness metrics integration tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_fairness_metrics()
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
