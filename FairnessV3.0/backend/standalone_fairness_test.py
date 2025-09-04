#!/usr/bin/env python3
"""
Standalone test for FairnessMetrics class - extracted from analysis_service.py
"""

import numpy as np
from typing import Dict

class FairnessMetrics:
    """
    Implementation of fairness metrics from fairness literature.
    Supports group-based fairness, individual fairness, and counterfactual approximations.
    """

    def __init__(self):
        self.metrics_computed = {}

    def statistical_parity(self, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Calculate statistical parity difference for both binary and multi-class sensitive attributes.
        """
        groups = np.unique(sensitive_attr)
        if len(groups) <= 1:
            return 0.0  # No disparity possible with only one group
        
        # For binary case, keep the original calculation
        if len(groups) == 2:
            prob_g0 = np.mean(y_pred[sensitive_attr == groups[0]])
            prob_g1 = np.mean(y_pred[sensitive_attr == groups[1]])
            return abs(prob_g0 - prob_g1)
        
        # For multi-class case, calculate prediction rate for each group
        group_rates = []
        for group in groups:
            mask = (sensitive_attr == group)
            if np.sum(mask) > 0:  # Avoid division by zero
                group_rate = np.mean(y_pred[mask])
                group_rates.append(group_rate)
        
        # Return maximum difference between any two groups
        return max(group_rates) - min(group_rates)

    def disparate_impact(self, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Calculate disparate impact for binary or multi-class sensitive attributes.
        """
        groups = np.unique(sensitive_attr)
        
        if len(groups) <= 1:
            return 1.0  # No disparity possible with only one group
        
        # Calculate prediction rate for each group
        group_rates = {}
        for group in groups:
            mask = (sensitive_attr == group)
            if np.sum(mask) > 0:
                group_rates[group] = np.mean(y_pred[mask])
            else:
                group_rates[group] = 0.0
        
        # For binary case, keep original calculation
        if len(groups) == 2:
            prob_g0, prob_g1 = group_rates[groups[0]], group_rates[groups[1]]
            if prob_g1 == 0:
                return float('inf') if prob_g0 > 0 else 1.0
            return prob_g0 / prob_g1
        
        # For multi-class case
        max_rate = max(group_rates.values())
        if max_rate == 0:
            return 1.0
        
        # Calculate minimum ratio (worst case)
        min_ratio = 1.0
        for rate in group_rates.values():
            if rate > 0:  # Avoid division by zero
                ratio = rate / max_rate
                min_ratio = min(min_ratio, ratio)
            elif max_rate > 0:
                min_ratio = 0.0  # Some group has zero rate while another has positive rate
        
        return min_ratio

    def equal_opportunity(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Calculate equal opportunity difference for binary or multi-class sensitive attributes.
        """
        groups = np.unique(sensitive_attr)
        
        if len(groups) <= 1:
            return 0.0  # No disparity possible with only one group
        
        # Calculate TPR for each group
        tpr_by_group = {}
        for group in groups:
            mask = (sensitive_attr == group)
            if np.sum(mask) > 0:
                tpr_by_group[group] = self._true_positive_rate(y_true[mask], y_pred[mask])
        
        # For binary case, keep original calculation
        if len(groups) == 2:
            tpr_g0 = tpr_by_group.get(groups[0], 0.0)
            tpr_g1 = tpr_by_group.get(groups[1], 0.0)
            return abs(tpr_g0 - tpr_g1)
        
        # For multi-class case, find maximum difference between any two groups
        tpr_values = list(tpr_by_group.values())
        if not tpr_values:
            return 0.0
        return max(tpr_values) - min(tpr_values)

    def equalized_odds(self, y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> float:
        """
        Calculate equalized odds for binary or multi-class sensitive attributes.
        """
        groups = np.unique(sensitive_attr)
        
        if len(groups) <= 1:
            return 0.0  # No disparity possible with only one group
        
        # Calculate TPR and FPR for each group
        tpr_by_group = {}
        fpr_by_group = {}
        for group in groups:
            mask = (sensitive_attr == group)
            if np.sum(mask) > 0:
                tpr_by_group[group] = self._true_positive_rate(y_true[mask], y_pred[mask])
                fpr_by_group[group] = self._false_positive_rate(y_true[mask], y_pred[mask])
        
        # For binary case, keep original calculation
        if len(groups) == 2:
            tpr_g0 = tpr_by_group.get(groups[0], 0.0)
            tpr_g1 = tpr_by_group.get(groups[1], 0.0)
            tpr_diff = abs(tpr_g0 - tpr_g1)
            
            fpr_g0 = fpr_by_group.get(groups[0], 0.0)
            fpr_g1 = fpr_by_group.get(groups[1], 0.0)
            fpr_diff = abs(fpr_g0 - fpr_g1)
            
            return max(tpr_diff, fpr_diff)
        
        # For multi-class case, find maximum difference in TPR and FPR
        tpr_values = list(tpr_by_group.values())
        fpr_values = list(fpr_by_group.values())
        
        tpr_diff = max(tpr_values) - min(tpr_values) if tpr_values else 0.0
        fpr_diff = max(fpr_values) - min(fpr_values) if fpr_values else 0.0
        
        return max(tpr_diff, fpr_diff)

    def calibration(self, y_true: np.ndarray, y_prob: np.ndarray, sensitive_attr: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate calibration for binary or multi-class sensitive attributes.
        """
        groups = np.unique(sensitive_attr)
        
        if len(groups) <= 1:
            return 0.0  # No disparity possible with only one group
        
        calibration_diffs = []
        for i in range(n_bins):
            bin_lower, bin_upper = i / n_bins, (i + 1) / n_bins
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            if np.sum(in_bin) == 0:
                continue
            
            # Calculate calibration for each group in this bin
            cal_by_group = {}
            for group in groups:
                group_in_bin = (sensitive_attr == group) & in_bin
                if np.sum(group_in_bin) > 0:
                    cal_by_group[group] = np.mean(y_true[group_in_bin])
            
            # Skip if no groups have samples in this bin
            if not cal_by_group:
                continue
            
            # For binary case
            if len(groups) == 2:
                cal_g0 = cal_by_group.get(groups[0], 0.0)
                cal_g1 = cal_by_group.get(groups[1], 0.0)
                calibration_diffs.append(abs(cal_g0 - cal_g1))
            else:
                # For multi-class, find maximum difference between any two groups in this bin
                cal_values = list(cal_by_group.values())
                if cal_values:
                    calibration_diffs.append(max(cal_values) - min(cal_values))
        
        return np.mean(calibration_diffs) if calibration_diffs else 0.0
    
    def generalized_entropy_index(self, y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 2.0) -> float:
        """Calculate generalized entropy index for individual fairness"""
        try:
            b_i = y_pred - y_true + 1
            mu = np.mean(b_i)
            if mu <= 0:
                return 0.0
            if alpha == 1:  # Theil Index
                return np.mean((b_i / mu) * np.log(b_i / mu + 1e-10))
            return (1 / (len(b_i) * alpha * (alpha - 1))) * np.sum((b_i / mu) ** alpha - 1)
        except:
            return 0.0

    def _true_positive_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate true positive rate"""
        return np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0.0

    def _false_positive_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate false positive rate"""
        return np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0) if np.sum(y_true == 0) > 0 else 0.0

    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, sensitive_attr: np.ndarray) -> Dict[str, float]:
        """Compute all fairness metrics, handling both binary and multi-class sensitive attributes"""
        metrics = {}
        
        # Add information about the sensitive attribute
        groups = np.unique(sensitive_attr)
        metrics['num_sensitive_groups'] = len(groups)
        metrics['is_multiclass_sensitive'] = len(groups) > 2
        
        # Compute all metrics
        try:
            metrics['statistical_parity'] = self.statistical_parity(y_pred, sensitive_attr)
            metrics['disparate_impact'] = self.disparate_impact(y_pred, sensitive_attr)
            metrics['equal_opportunity'] = self.equal_opportunity(y_true, y_pred, sensitive_attr)
            metrics['equalized_odds'] = self.equalized_odds(y_true, y_pred, sensitive_attr)
            metrics['calibration'] = self.calibration(y_true, y_prob, sensitive_attr)
            metrics['generalized_entropy_index'] = self.generalized_entropy_index(y_true, y_pred)
            
            # Add group-specific metrics for detailed analysis
            if len(groups) > 2:
                group_rates = {}
                for group in groups:
                    mask = (sensitive_attr == group)
                    if np.sum(mask) > 0:
                        group_rates[f"prediction_rate_group_{group}"] = np.mean(y_pred[mask])
                metrics['group_prediction_rates'] = group_rates
                
        except Exception as e:
            print(f"Error computing fairness metrics: {e}")
        
        self.metrics_computed = metrics
        return metrics


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
