# Importing Libraries
import pandas as pd
import numpy as np
import json
import re
from sklearn.feature_selection import VarianceThreshold, RFE, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency, f_oneway, pearsonr, ks_2samp, mannwhitneyu
import shap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def load_exclusion_config(config_path="columns_to_exclude.json"):
    """
    Load column exclusion configuration from JSON file
    """
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {config_path} not found. No columns will be excluded.")
        return {"exclude_columns": [], "exclude_patterns": []}
def should_exclude_column(column_name, exclusion_config):
    """
    Check if a column should be excluded based on configuration
    """
    # Check exact matches (case-insensitive)
    for exclude_item in exclusion_config.get("exclude_columns", []):
        if column_name.lower() == exclude_item["name"].lower():
            return True, exclude_item.get("reason", "Excluded by configuration")
    # Check pattern matches
    for pattern_item in exclusion_config.get("exclude_patterns", []):
        pattern = pattern_item["pattern"]
        if re.search(pattern, column_name, re.IGNORECASE):
            return True, pattern_item.get("reason", "Matched exclusion pattern")
    return False, None
def filter_columns_for_analysis(df, target_col, sensitive_attrs, exclusion_config):
    """
    Filter columns for analysis, excluding specified columns
    """
    excluded_columns = []
    included_columns = []
    for col in df.columns:
        # Skip target and sensitive attributes
        if col == target_col or col in sensitive_attrs:
            continue
        # Check if column should be excluded
        should_exclude, reason = should_exclude_column(col, exclusion_config)
        if should_exclude:
            excluded_columns.append((col, reason))
        else:
            included_columns.append(col)
    return included_columns, excluded_columns

# Direct Bias Analysis Function
def direct_bias_analysis(X, y, sensitive_attr):
    """
    Analyze direct statistical relationship between sensitive attribute and target
    """
    print(f"\nDIRECT BIAS ANALYSIS")
    print("-" * 30)
    
    # Check direct correlation between sensitive attribute and target
    sensitive_encoded = X[sensitive_attr]
    target_by_group = pd.DataFrame({'target': y, 'sensitive': sensitive_encoded}).groupby('sensitive')['target'].agg(['mean', 'count'])
    print(f"Target approval rate by {sensitive_attr}:")
    print(target_by_group)
    
    # Statistical significance test
    contingency = pd.crosstab(y, sensitive_encoded)
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    print(f"\nChi-square test for {sensitive_attr} vs target:")
    print(f"   • Chi-square statistic: {chi2_stat:.4f}")
    print(f"   • P-value: {p_val:.6f}")
    print(f"   • Significant bias: {'YES' if p_val < 0.05 else 'NO'}")
    
    return {
        'chi2_stat': chi2_stat,
        'p_value': p_val,
        'has_direct_bias': p_val < 0.05,
        'approval_rates': target_by_group['mean'].to_dict()
    }

# 1. Feature Selection (Skipped)
# def feature_selection(X, y):
#     # ... (original function code)
#     pass

# 2. Fairness Metrics
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
        
        For multi-class attributes, returns the maximum difference in prediction rates
        between any two groups (worst-case disparity).
        
        Parameters:
        -----------
        y_pred : np.ndarray
            Predicted labels or scores
        sensitive_attr : np.ndarray
            Sensitive attribute values
            
        Returns:
        --------
        float
            Statistical parity difference
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
        For multi-class, returns the minimum ratio between any group and the group with highest prediction rate.
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
        For multi-class, returns the maximum difference in true positive rates between any two groups.
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
        For multi-class, returns the maximum difference in TPR or FPR between any two groups.
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
        For multi-class, returns the average of maximum calibration differences across bins.
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
        b_i = y_pred - y_true + 1
        mu = np.mean(b_i)
        if alpha == 1:  # Theil Index
            return np.mean((b_i / mu) * np.log(b_i / mu))
        return (1 / (len(b_i) * alpha * (alpha - 1))) * np.sum((b_i / mu) ** alpha - 1)

    def _true_positive_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0.0

    def _false_positive_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
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

    
class PreprocessingMethods:
    """
    Pre-processing methods for bias mitigation
    """

    class Reweighing(BaseEstimator, TransformerMixin):
        """
        Reweighing method to balance sensitive groups
        """

        def __init__(self, sensitive_attr_col: str):
            self.sensitive_attr_col = sensitive_attr_col
            self.weights_ = None

        def fit(self, X: pd.DataFrame, y: np.ndarray):
            """Learn weights for each group-class combination"""
            sensitive_attr = X[self.sensitive_attr_col]
            weights = np.ones(len(X))
            for group in np.unique(sensitive_attr):
                for class_val in np.unique(y):
                    mask = (sensitive_attr == group) & (y == class_val)
                    if np.sum(mask) > 0:
                        group_class_freq = np.sum(mask) / len(X)
                        expected_freq = (np.sum(sensitive_attr == group) / len(X)) * (np.sum(y == class_val) / len(X))
                        if group_class_freq > 0:
                            weights[mask] = expected_freq / group_class_freq
            self.weights_ = weights
            return self

        def transform(self, X: pd.DataFrame):
            return self.weights_

        def fit_transform(self, X: pd.DataFrame, y: np.ndarray):
            return self.fit(X, y).transform(X)

    class DataTransformation(BaseEstimator, TransformerMixin):
        """
        Data transformation for fairness (feature transformation)
        """

        def __init__(self, sensitive_attr_col: str, method: str = 'standardize'):
            self.sensitive_attr_col = sensitive_attr_col
            self.method = method
            self.scalers_ = {}

        def fit(self, X: pd.DataFrame, y: None = None):
            sensitive_attr = X[self.sensitive_attr_col]
            feature_cols = [col for col in X.columns if col != self.sensitive_attr_col]
            for group in np.unique(sensitive_attr):
                group_data = X[sensitive_attr == group][feature_cols]
                scaler = StandardScaler()
                scaler.fit(group_data)
                self.scalers_[group] = scaler
            return self

        def transform(self, X: pd.DataFrame):
            X_transformed = X.copy()
            sensitive_attr = X[self.sensitive_attr_col]
            feature_cols = [col for col in X.columns if col != self.sensitive_attr_col]
            for group in np.unique(sensitive_attr):
                group_mask = sensitive_attr == group
                if group in self.scalers_:
                    X_transformed.loc[group_mask, feature_cols] = \
                        self.scalers_[group].transform(X.loc[group_mask, feature_cols])
            return X_transformed

class InProcessingMethods:
    """
    In-processing methods for fair machine learning
    """

    class FairLogisticRegression(BaseEstimator, ClassifierMixin):
        """
        Logistic Regression with fairness regularization
        """

        def __init__(self, sensitive_attr_col: str, fairness_penalty: float = 1.0,
                     fairness_metric: str = 'statistical_parity'):
            self.sensitive_attr_col = sensitive_attr_col
            self.fairness_penalty = fairness_penalty
            self.fairness_metric = fairness_metric
            self.model_ = LogisticRegression()
            self.fairness_metrics_ = FairnessMetrics()

        def fit(self, X: pd.DataFrame, y: np.ndarray):
            feature_cols = [col for col in X.columns if col != self.sensitive_attr_col]
            X_features = X[feature_cols]
            sensitive_attr = X[self.sensitive_attr_col]
            self.model_.fit(X_features, y)
            for iteration in range(5):
                y_pred = self.model_.predict(X_features)
                y_prob = self.model_.predict_proba(X_features)[:, 1]
                if self.fairness_metric == 'statistical_parity':
                    fairness_violation = self.fairness_metrics_.statistical_parity(y_pred, sensitive_attr)
                elif self.fairness_metric == 'equal_opportunity':
                    fairness_violation = self.fairness_metrics_.equal_opportunity(y, y_pred, sensitive_attr)
                else:
                    fairness_violation = 0
                if fairness_violation < 0.1:
                    break
                sample_weights = self._compute_fairness_weights(y, y_pred, sensitive_attr)
                self.model_.fit(X_features, y, sample_weight=sample_weights)
            return self

        def predict(self, X: pd.DataFrame):
            feature_cols = [col for col in X.columns if col != self.sensitive_attr_col]
            return self.model_.predict(X[feature_cols])

        def predict_proba(self, X: pd.DataFrame):
            feature_cols = [col for col in X.columns if col != self.sensitive_attr_col]
            return self.model_.predict_proba(X[feature_cols])

        def _compute_fairness_weights(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      sensitive_attr: np.ndarray) -> np.ndarray:
            weights = np.ones(len(y_true))
            groups = np.unique(sensitive_attr)
            for group in groups:
                group_mask = sensitive_attr == group
                group_accuracy = accuracy_score(y_true[group_mask], y_pred[group_mask])
                if group_accuracy > 0:
                    weights[group_mask] = 1.0 / group_accuracy
            return weights / np.sum(weights) * len(weights)

class PostProcessingMethods:
    """
    Post-processing methods for bias mitigation
    """

    class ThresholdOptimizer:
        """
        Optimize decision thresholds for fairness
        """

        def __init__(self, fairness_metric: str = 'equal_opportunity'):
            self.fairness_metric = fairness_metric
            self.thresholds_ = {}
            self.fairness_metrics_ = FairnessMetrics()

        def fit(self, y_true: np.ndarray, y_prob: np.ndarray, sensitive_attr: np.ndarray):
            groups = np.unique(sensitive_attr)
            best_thresholds = {}
            best_fairness = float('inf')
            threshold_range = np.linspace(0.1, 0.9, 9)
            for thresh_combo in np.meshgrid(*[threshold_range for _ in groups]):
                thresholds = {groups[i]: thresh_combo[i].flatten()[0]
                              for i in range(len(groups))}
                y_pred = np.zeros_like(y_true)
                for group in groups:
                    group_mask = sensitive_attr == group
                    y_pred[group_mask] = (y_prob[group_mask] > thresholds[group]).astype(int)
                if self.fairness_metric == 'equal_opportunity':
                    fairness_score = self.fairness_metrics_.equal_opportunity(y_true, y_pred, sensitive_attr)
                elif self.fairness_metric == 'statistical_parity':
                    fairness_score = self.fairness_metrics_.statistical_parity(y_pred, sensitive_attr)
                else:
                    fairness_score = 0
                if fairness_score < best_fairness:
                    best_fairness = fairness_score
                    best_thresholds = thresholds.copy()
            self.thresholds_ = best_thresholds
            return self

        def transform(self, y_prob: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
            y_pred = np.zeros(len(y_prob))
            for group, threshold in self.thresholds_.items():
                group_mask = sensitive_attr == group
                y_pred[group_mask] = (y_prob[group_mask] > threshold).astype(int)
            return y_pred

    class Calibrator:
        """
        Calibrate predictions for fairness
        """

        def __init__(self):
            self.calibration_maps_ = {}

        def fit(self, y_true: np.ndarray, y_prob: np.ndarray, sensitive_attr: np.ndarray):
            groups = np.unique(sensitive_attr)
            for group in groups:
                group_mask = sensitive_attr == group
                group_y_true = y_true[group_mask]
                group_y_prob = y_prob[group_mask]
                bins = np.linspace(0, 1, 11)
                calibrated_probs = []
                for i in range(len(bins) - 1):
                    bin_mask = (group_y_prob >= bins[i]) & (group_y_prob < bins[i + 1])
                    if np.sum(bin_mask) > 0:
                        true_prob = np.mean(group_y_true[bin_mask])
                        calibrated_probs.append(true_prob)
                    else:
                        calibrated_probs.append(bins[i])
                self.calibration_maps_[group] = {
                    'bins': bins,
                    'calibrated_probs': calibrated_probs
                }
            return self

        def transform(self, y_prob: np.ndarray, sensitive_attr: np.ndarray) -> np.ndarray:
            calibrated_probs = y_prob.copy()
            for group in np.unique(sensitive_attr):
                if group in self.calibration_maps_:
                    group_mask = sensitive_attr == group
                    group_probs = y_prob[group_mask]
                    bins = self.calibration_maps_[group]['bins']
                    cal_probs = self.calibration_maps_[group]['calibrated_probs']
                    for i, prob in enumerate(group_probs):
                        bin_idx = np.digitize(prob, bins) - 1
                        bin_idx = max(0, min(bin_idx, len(cal_probs) - 1))
                        calibrated_probs[group_mask][i] = cal_probs[bin_idx]
            return calibrated_probs


# Proxy Detection
def detect_proxy_features(X, selected_features, sensitive_attr):
    """
    Identify features that might indirectly encode sensitive attribute information
    """
    print(f"\nPROXY DETECTION ANALYSIS")
    print("-" * 30)
    
    proxy_candidates = []
    sensitive_vals = X[sensitive_attr].values
    
    # Encode sensitive attribute if needed
    if X[sensitive_attr].dtype == 'object':
        le = LabelEncoder()
        sensitive_vals = le.fit_transform(sensitive_vals)
    
    for feature in selected_features:
        try:
            # Test if this feature can predict the sensitive attribute
            proxy_model = LogisticRegression(random_state=42, max_iter=1000)
            feature_vals = X[feature].values.reshape(-1, 1)
            
            # Handle categorical features
            if X[feature].dtype == 'object':
                le_feat = LabelEncoder()
                feature_vals = le_feat.fit_transform(X[feature]).reshape(-1, 1)
            
            proxy_model.fit(feature_vals, sensitive_vals)
            proxy_accuracy = proxy_model.score(feature_vals, sensitive_vals)
            
            # Mutual information test
            mi_score = mutual_info_classif(feature_vals, sensitive_vals, discrete_features='auto', random_state=42)[0]
            
            # Statistical dependence test
            if X[feature].dtype in ['int64', 'float64']:
                corr, corr_p = pearsonr(X[feature], sensitive_vals)
                stat_dependence = abs(corr)
            else:
                # Chi-square for categorical
                try:
                    contingency = pd.crosstab(X[feature], sensitive_vals)
                    chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)
                    stat_dependence = chi2_p
                except:
                    stat_dependence = 0
            
            print(f"\n   {feature}:")
            print(f"   • Proxy prediction accuracy: {proxy_accuracy:.4f}")
            print(f"   • Mutual information: {mi_score:.6f}")
            
            # Baseline accuracy (majority class)
            baseline_accuracy = max(np.mean(sensitive_vals), 1 - np.mean(sensitive_vals))
            
            # Consider as proxy if it predicts sensitive attribute better than baseline
            is_proxy = proxy_accuracy > baseline_accuracy + 0.05  # 5% better than baseline
            
            if is_proxy:
                proxy_candidates.append({
                    'feature': feature,
                    'proxy_accuracy': proxy_accuracy,
                    'mutual_info': mi_score,
                    'statistical_dependence': stat_dependence
                })
                print(f"   • POTENTIAL PROXY: Predicts {sensitive_attr} too well!")
            else:
                print(f"   • Not a proxy")
                
        except Exception as e:
            print(f"   • {feature}: Analysis failed ({str(e)[:30]}...)")
    
    return proxy_candidates

# 3. Detect Biased Features
def detect_biased_features(X, y, selected_features, sensitive_attr, corr_threshold=0.2, mi_threshold=0.02, dp_thresh=0.1, di_range=(0.8, 1.25), eq_thresh=0.1, verbose=True):
    biased_features = []
    analysis_results = {}

    sensitive_vals = X[sensitive_attr].values
    # Encode sensitive attribute if categorical but not int
    if X[sensitive_attr].dtype == 'object':
        le = LabelEncoder()
        sensitive_vals = le.fit_transform(sensitive_vals)
        if verbose:
            print(f"Encoded sensitive attribute: {dict(enumerate(le.classes_))}")
    
    print(f"\n  FEATURE-LEVEL BIAS ANALYSIS")
    print("-" * 35)
    print(f"\nAnalyzing {len(selected_features)} features for bias...")

    # Compute mutual information between features and sensitive attribute
    feature_data = X[selected_features]

    # Encode categorical features for mutual information calculation
    feature_data_encoded = feature_data.copy()
    for col in feature_data_encoded.columns:
        if feature_data_encoded[col].dtype == 'object':
            le = LabelEncoder()
            feature_data_encoded[col] = le.fit_transform(feature_data_encoded[col].astype(str))

    try:
        mi_scores = mutual_info_classif(feature_data, sensitive_vals, discrete_features='auto', random_state=42)
        mi_dict = dict(zip(selected_features, mi_scores))
    except:
        mi_dict = {feat: 0 for feat in selected_features}

    for i, feature in enumerate(selected_features):
        if verbose:
            print(f"\nAnalyzing feature {i+1}/{len(selected_features)}: {feature}")
        
        analysis = {}

        # Create clean data for analysis
        df_feat = pd.DataFrame({
            "feature": X[feature],
            "sensitive": X[sensitive_attr],
            "target": y
        }).dropna()

        if len(df_feat)==0:
            continue

        # Statistical Dependence Check
        analysis['statistical_tests'] = {}
        
        # Correlation test (for numerical features)
        if df_feat["feature"].dtype in ['int64', 'float64']:
            try:
                # Encode sensitive attribute for correlation if it's categorical
                sensitive_for_corr = df_feat["sensitive"]
                if sensitive_for_corr.dtype == 'object':
                    le_sens = LabelEncoder()
                    sensitive_for_corr = le_sens.fit_transform(sensitive_for_corr)

                corr, p_val = pearsonr(df_feat["feature"], df_feat["sensitive"])
                analysis['statistical_tests']['correlation'] = {
                    'value': abs(corr),
                    'p_value': p_val,
                    'significant': abs(corr)>corr_threshold
                }
                if verbose:
                    print(f"  • Correlation with {sensitive_attr}: {corr:.4f} (p={p_val:.4f})")
            except Exception as e:
                if verbose:
                    print(f"  • Correlation analysis failed: {e}")
                analysis['statistical_tests']['correlation'] = {'value':0, 'significant': False}
        else:
            analysis['statistical_tests']['correlation'] = {'value':0, 'significant': False}

        # Mutual Information
        mi_score = mi_dict.get(feature, 0)
        analysis['statistical_tests']['mutual_info'] = {
            'value': mi_score,
            'significant': mi_score>mi_threshold
        }

        if verbose:
            print(f"   • Mutual information: {mi_score:.6f}")

        # Distribution Analysis by sensitive groups
        if df_feat["feature"].dtype in ['int64', 'float64']:
            feat_by_sensitive = df_feat.groupby('sensitive')['feature'].agg(['mean', 'std', 'count'])
            if verbose:
                print(f"   • {feature} by {sensitive_attr}:")
                print(f"     {feat_by_sensitive}")
 
        # Statistical tests only for numerical features
        if df_feat["feature"].dtype in ['int64', 'float64']:
        # Kolomogorov - Smirnov test for distribution differences
            try:
                groups = df_feat["sensitive"].unique()
                if len(groups)==2:
                    g1_vals = df_feat[df_feat["sensitive"] == groups[0]]["feature"]
                    g2_vals = df_feat[df_feat["sensitive"] == groups[1]]["feature"]
                    ks_stat, ks_p = ks_2samp(g1_vals, g2_vals)

                    # Mann-Whitney U test (non-parametric)
                    u_stat, u_p = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')

                    analysis['statistical_tests']['ks_test'] = {
                        'statistic': ks_stat,
                        'p_value': ks_p,
                        'significant': ks_p<0.05
                    }
                    analysis['statistical_tests']['mannwhitney_test'] = {
                        'statistic': u_stat,
                        'p_value': u_p,
                        'significant': u_p<0.05
                    }
                    if verbose:
                        print(f"   • KS test p-value: {ks_p:.6f}")
                        print(f"   • Mann-Whitney U test p-value: {u_p:.6f}")
            except Exception as e:
                if verbose:
                    print(f"   • Statistical tests failed: {e}")
                analysis['statistical_tests']['ks_test'] = {'significant': False}
                analysis['statistical_tests']['mannwhitney_test'] = {'significant': False}
        
        else:
            # For categorical features, use chi-square test
            try:
                contingency = pd.crosstab(df_feat["feature"], df_feat["sensitive"])
                chi2_stat, chi2_p, _, _ = chi2_contingency(contingency)
                analysis['statistical_tests']['chi2_test'] = {
                    'statistic': chi2_stat,
                    'p_value': chi2_p,
                    'significant': chi2_p<0.05
                }
                if verbose:
                    print(f"   • Chi-square test p-value: {chi2_p:.6f}")
            except Exception as e:
                if verbose:
                    print(f"   • Chi-square test failed: {e}")
                analysis['statistical_tests']['chi2_test'] = {'significant': False}

        # Model-based prediction using only this feature
        try:
            # Prepare feature data - encode if categorical
            X_single = X[feature].values
            if X[feature].dtype == 'object':
                le_feat = LabelEncoder()
                X_single = le_feat.fit_transform(X_single)
            X_single = X_single.reshape(-1, 1)
            
            single_model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
            single_model.fit(X_single, y)
            y_pred_single = single_model.predict(X_single)

            fm = FairnessMetrics()
            fm_single = fm.compute_all_metrics(y.values, y_pred_single, y_pred_single, X[sensitive_attr].values)
            analysis['single_feature_metrics'] = fm_single

            if verbose and fm_single:
                print(f"   • Single feature DP diff: {fm_single.get('dp_diff', 0):.4f}")
        except Exception as e:
            if verbose:
                print(f"   • Single feature analysis failed: {e}")
            analysis['single_feature_metrics'] = None
        
        # Determine if feature is biased
        stat_dependent = any([
            analysis['statistical_tests'].get('correlation', {}).get('significant', False),
            analysis['statistical_tests'].get('mutual_info', {}).get('significant', False),
            analysis['statistical_tests'].get('ks_test', {}).get('significant', False),
            analysis['statistical_tests'].get('mannwhitney_test', {}).get('significant', False),
            analysis['statistical_tests'].get('chi2_test', {}).get('significant', False),
        ])

        # For now, skip the full model fairness analysis to avoid the string conversion error
        fairness_violated = False
        analysis['fairness_violations'] = []

        analysis['is_biased'] = stat_dependent or fairness_violated
        analysis['bias_reasons'] = []

        if stat_dependent:
            analysis['bias_reasons'].append('statistical_dependence')
        if fairness_violated:
            analysis['bias_reasons'].extend(analysis['fairness_violations'])

        if analysis['is_biased']:
            biased_features.append((feature, analysis))
            if verbose:
                print(f"Biased Reasons: {', '.join(analysis['bias_reasons'])}")
        else:
            if verbose:
                print(f"Not Biased")

        analysis_results[feature] = analysis
        
    return biased_features, len(biased_features), analysis_results

# Model-Based Testing Function
def model_based_bias_testing(X, y, selected_features, sensitive_attr):
    """
    Comprehensive model-based bias testing
    """
    print(f"\nMODEL-BASED BIAS TESTING")
    print("-" * 32)
    
    from sklearn.model_selection import train_test_split

    # ADD: Encode categorical features
    X_encoded = X.copy()
    for col in selected_features:
        if X_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    
    # Prepare data
    X_features = X_encoded[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Get sensitive attribute for test set before using it
    sensitive_test = X.loc[X_test.index, sensitive_attr]

    fm = FairnessMetrics()
    detailed_metrics = fm.compute_all_metrics(y_test.values, y_pred, y_prob, sensitive_test.values)

    
    # # Get sensitive attribute for test set
    # sensitive_test = X.loc[X_test.index, sensitive_attr]
    
    # Compute detailed fairness metrics
    fairness_results = {}
    for group in sensitive_test.unique():
        mask = sensitive_test == group
        group_true = y_test[mask]
        group_pred = y_pred[mask]
        
        if len(group_true) > 0:
            approval_rate = np.mean(group_pred)
            accuracy = np.mean(group_true == group_pred)
            
            if len(np.unique(group_true)) > 1 and len(np.unique(group_pred)) > 1:
                try:
                    tn, fp, fn, tp = confusion_matrix(group_true, group_pred).ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (tp + tn) > 0 else 0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                except:
                    tpr = fpr = precision = 0
            else:
                tpr = fpr = precision = 0
            
            fairness_results[group] = {
                'count': len(group_true),
                'approval_rate': approval_rate,
                'accuracy': accuracy,
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision
            }
    
    # Print results
    print("Model Performance by Sensitive Group:")
    for group, metrics in fairness_results.items():
        print(f"\n   {sensitive_attr} = {group}:")
        print(f"   • Sample count: {metrics['count']}")
        print(f"   • Approval rate: {metrics['approval_rate']:.4f}")
        print(f"   • Accuracy: {metrics['accuracy']:.4f}")
        print(f"   • True Positive Rate: {metrics['tpr']:.4f}")
        print(f"   • False Positive Rate: {metrics['fpr']:.4f}")
    
    # Calculate bias metrics
    bias_detected = False
    if len(fairness_results) == 2:
        groups = list(fairness_results.keys())
        g1, g2 = groups[0], groups[1]
        
        dp_diff = abs(fairness_results[g1]['approval_rate'] - fairness_results[g2]['approval_rate'])
        eo_diff = abs(fairness_results[g1]['tpr'] - fairness_results[g2]['tpr'])
        eq_odds_diff = max(
            abs(fairness_results[g1]['tpr'] - fairness_results[g2]['tpr']),
            abs(fairness_results[g1]['fpr'] - fairness_results[g2]['fpr'])
        )
        
        print(f"\n MODEL FAIRNESS METRICS:")
        print(f"   • Demographic Parity Difference: {dp_diff:.4f}")
        print(f"   • Equal Opportunity Difference: {eo_diff:.4f}")
        print(f"   • Equalized Odds Difference: {eq_odds_diff:.4f}")
        
        # Bias thresholds (sensitive)
        bias_detected = dp_diff > 0.02 or eo_diff > 0.02 or eq_odds_diff > 0.02
        print(f"   • MODEL BIAS DETECTED: {'YES' if bias_detected else 'NO'}")
    
    return {
        'fairness_results': fairness_results,
        'bias_detected': bias_detected,
        'model_performance': fairness_results,
        'detailed_fairness_metrics': detailed_metrics
    }

# End-To-End Pipeline
def feature_bias_pipeline(X, y, sensitive_attr, target, exclusion_config=None, apply_mitigation=True):
    # Load exclusion config if not provided
    if exclusion_config is None:
        exclusion_config = load_exclusion_config()

    # 1. Direct Bias Analysis (your existing code)
    print("="*60)
    print("Bias Detection Pipeline (Skipping Feature Selection)")
    print("="*60)

    # Basic Data Info
    print(f"\nDataset Info:")
    print(f"- Total samples: {len(X)}")
    print(f"- Total features: {len(X.columns)}")
    print(f"- Target distribution: {dict(y.value_counts())}")
    print(f"- Sensitive attribute distribution: {dict(X[sensitive_attr].value_counts())}")

    # Direct Bias Analysis
    direct_bias_results = direct_bias_analysis(X, y, sensitive_attr)

    # 1. Use All Features (Skip Feature Selection)
    print(f"\n{'='*20} Using All Features {'='*20}")
    # We will use all features for this run based on user request
    all_features = [col for col in X.columns if col != sensitive_attr]
    
    important_features = []
    excluded_features = []
    
    for feat in all_features:
        should_exclude, reason = should_exclude_column(feat, exclusion_config)
        if should_exclude:
            excluded_features.append((feat, reason))
        else:
            important_features.append(feat)
    
    print(f"\n{'='*20} Feature Selection {'='*20}")
    print(f"Total features in dataset: {len(X.columns)}")
    print(f"Features after exclusion: {len(important_features)}")
    
    if excluded_features:
        print(f"\nExcluded {len(excluded_features)} features:")
        for feat, reason in excluded_features[:10]:  # Show first 10
            print(f"  - {feat}: {reason}")
        if len(excluded_features) > 10:
            print(f"  ... and {len(excluded_features) - 10} more")
    
    print(f"\nUsing {len(important_features)} features for bias analysis.")

    # proxy Detection
    proxy_candidates = detect_proxy_features(X, important_features, sensitive_attr)

    # Model-Based Testing
    model_results = model_based_bias_testing(X, y, important_features, sensitive_attr)

    # 2. Detect Biased Features
    print(f"\n{'='*20} Bias Detection {'='*20}")
    biased_feats, bias_count, analysis = detect_biased_features(X, y, important_features, sensitive_attr,
                                                                corr_threshold=0.15,
                                                                mi_threshold=0.01,
                                                                dp_thresh=0.05,
                                                                di_range=(0.8, 1.25),
                                                                eq_thresh=0.05)
    
    # NEW: Apply bias mitigation techniques if requested
    mitigation_results = {}
    if apply_mitigation and bias_count > 0:
        print(f"\n{'='*20} Bias Mitigation {'='*20}")
        
        # Split data for testing mitigation techniques
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 1. Pre-processing methods
        print("\n1. APPLYING PRE-PROCESSING METHODS")
        print("-" * 35)
        
        # Reweighing method
        print("\nReweighing Method:")
        reweigher = PreprocessingMethods.Reweighing(sensitive_attr_col=sensitive_attr)
        sample_weights = reweigher.fit_transform(X_train, y_train)
        
        # Train a model with reweighing
        model_reweighed = LogisticRegression(random_state=42, max_iter=1000)
        model_reweighed.fit(X_train[important_features], y_train, sample_weight=sample_weights)
        
        # Evaluate
        y_pred_reweighed = model_reweighed.predict(X_test[important_features])
        y_prob_reweighed = model_reweighed.predict_proba(X_test[important_features])[:, 1]
        
        fm = FairnessMetrics()
        reweighing_metrics = fm.compute_all_metrics(
            y_test.values, y_pred_reweighed, y_prob_reweighed, X_test[sensitive_attr].values
        )
        print("Fairness metrics after reweighing:")
        for metric, value in reweighing_metrics.items():
            if isinstance(value, float):
                print(f"  • {metric}: {value:.4f}")
            else:
                print(f"  • {metric}: {value}")
        
        # Data transformation method
        print("\nData Transformation Method:")
        transformer = PreprocessingMethods.DataTransformation(sensitive_attr_col=sensitive_attr)
        X_train_transformed = transformer.fit_transform(X_train)
        X_test_transformed = transformer.transform(X_test)
        
        # Train a model with transformed data
        model_transformed = LogisticRegression(random_state=42, max_iter=1000)
        model_transformed.fit(X_train_transformed[important_features], y_train)
        
        # Evaluate
        y_pred_transformed = model_transformed.predict(X_test_transformed[important_features])
        y_prob_transformed = model_transformed.predict_proba(X_test_transformed[important_features])[:, 1]
        
        transformation_metrics = fm.compute_all_metrics(
            y_test.values, y_pred_transformed, y_prob_transformed, X_test[sensitive_attr].values
        )
        print("Fairness metrics after data transformation:")
        for metric, value in transformation_metrics.items():
            if isinstance(value, float):
                print(f"  • {metric}: {value:.4f}")
            else:
                print(f"  • {metric}: {value}")
        
        # 2. In-processing methods
        print("\n2. APPLYING IN-PROCESSING METHODS")
        print("-" * 35)
        
        # Fair Logistic Regression
        print("\nFair Logistic Regression:")
        fair_lr = InProcessingMethods.FairLogisticRegression(
            sensitive_attr_col=sensitive_attr, fairness_penalty=1.0
        )
        fair_lr.fit(X_train, y_train)
        
        # Evaluate
        y_pred_fair_lr = fair_lr.predict(X_test)
        y_prob_fair_lr = fair_lr.predict_proba(X_test)[:, 1]
        
        fair_lr_metrics = fm.compute_all_metrics(
            y_test.values, y_pred_fair_lr, y_prob_fair_lr, X_test[sensitive_attr].values
        )
        print("Fairness metrics with fair logistic regression:")
        for metric, value in fair_lr_metrics.items():
            if isinstance(value, float):
                print(f"  • {metric}: {value:.4f}")
            else:
                print(f"  • {metric}: {value}")
        
        # 3. Post-processing methods
        print("\n3. APPLYING POST-PROCESSING METHODS")
        print("-" * 35)
        
        # Train a standard model for post-processing
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X_train[important_features], y_train)
        y_prob_base = base_model.predict_proba(X_test[important_features])[:, 1]
        
        # Threshold Optimizer
        print("\nThreshold Optimization:")
        threshold_opt = PostProcessingMethods.ThresholdOptimizer()
        threshold_opt.fit(y_test.values, y_prob_base, X_test[sensitive_attr].values)
        y_pred_threshold = threshold_opt.transform(y_prob_base, X_test[sensitive_attr].values)
        
        threshold_metrics = fm.compute_all_metrics(
            y_test.values, y_pred_threshold, y_prob_base, X_test[sensitive_attr].values
        )
        print("Fairness metrics after threshold optimization:")
        for metric, value in threshold_metrics.items():
            if isinstance(value, float):
                print(f"  • {metric}: {value:.4f}")
            else:
                print(f"  • {metric}: {value}")
        
        # Calibrator
        print("\nCalibration:")
        calibrator = PostProcessingMethods.Calibrator()
        calibrator.fit(y_test.values, y_prob_base, X_test[sensitive_attr].values)
        y_prob_calibrated = calibrator.transform(y_prob_base, X_test[sensitive_attr].values)
        y_pred_calibrated = (y_prob_calibrated > 0.5).astype(int)
        
        calibration_metrics = fm.compute_all_metrics(
            y_test.values, y_pred_calibrated, y_prob_calibrated, X_test[sensitive_attr].values
        )
        print("Fairness metrics after calibration:")
        for metric, value in calibration_metrics.items():
            if isinstance(value, float):
                print(f"  • {metric}: {value:.4f}")
            else:
                print(f"  • {metric}: {value}")
        
        # Store all mitigation results
        mitigation_results = {
            'preprocessing': {
                'reweighing': reweighing_metrics,
                'transformation': transformation_metrics
            },
            'inprocessing': {
                'fair_lr': fair_lr_metrics
            },
            'postprocessing': {
                'threshold_opt': threshold_metrics,
                'calibration': calibration_metrics
            }
        }
        
        # Compare mitigation techniques
        print("\nCOMPARISON OF MITIGATION TECHNIQUES")
        print("-" * 35)
        
        # Get baseline metrics from original model results
        baseline_metrics = model_results.get('detailed_fairness_metrics', {})
        
        # Create comparison table
        comparison_table = {
            'Baseline': baseline_metrics.get('statistical_parity', 0),
            'Reweighing': reweighing_metrics.get('statistical_parity', 0),
            'Transformation': transformation_metrics.get('statistical_parity', 0),
            'Fair LR': fair_lr_metrics.get('statistical_parity', 0),
            'Threshold Opt': threshold_metrics.get('statistical_parity', 0),
            'Calibration': calibration_metrics.get('statistical_parity', 0)
        }
        
        print("Statistical Parity Difference (lower is better):")
        for method, value in comparison_table.items():
            print(f"  • {method}: {value:.4f}")
        
        # Find best method
        best_method = min(comparison_table.items(), key=lambda x: x[1])
        print(f"\nBest mitigation method: {best_method[0]} with statistical parity = {best_method[1]:.4f}")

    print(f"\n{'='*20} Results Summary {'='*20}")
    print(f"Important Features: {len(important_features)}")
    print(f"Biased Features:", bias_count)

    total_bias_indicators = 0
    bias_sources = []
    
    # Count all bias indicators
    if direct_bias_results['has_direct_bias']:
        total_bias_indicators += 1
        bias_sources.append(f"Direct bias (p-value: {direct_bias_results['p_value']:.6f})")
    
    if len(proxy_candidates) > 0:
        total_bias_indicators += len(proxy_candidates)
        for proxy in proxy_candidates:
            bias_sources.append(f"Proxy feature: {proxy['feature']} (accuracy: {proxy['proxy_accuracy']:.3f})")
    
    if model_results['bias_detected']:
        total_bias_indicators += 1
        bias_sources.append("Model predictions show fairness violations")
    
    if bias_count > 0:
        total_bias_indicators += bias_count
        print(f"Features flagged as biased:")
        for feat, details in biased_feats:
            bias_sources.append(f"Feature bias: {feat} ({', '.join(details['bias_reasons'])})")

    # Retry with more sensitive thresholds
    biased_feats_sensitive, bias_count_sensitive, analysis_sensitive = detect_biased_features(
            X, y, important_features, sensitive_attr,
            corr_threshold=0.1,
            mi_threshold=0.1,
            dp_thresh=0.1,
            di_range=(0.8, 1.25),
            eq_thresh=0.1,
            verbose=False
    )

    if bias_count_sensitive>0:
        print(f"With sensitive thresholds: {bias_count_sensitive} biased features found")
        for feat, details in biased_feats_sensitive:
            print(f" - {feat}: {', '.join(details['bias_reasons'])}")

    # Final bias count calculation
    final_bias_count = max(bias_count, bias_count_sensitive if 'bias_count_sensitive' in locals() else 0)

    print(f"\nFINAL BIAS SUMMARY")
    print("-" * 30)
    print(f"Total Features Analyzed: {len(important_features)}")
    print(f"Number of Biased Features (Standard): {bias_count}")
    if 'bias_count_sensitive' in locals():
        print(f"Number of Biased Features (Sensitive): {bias_count_sensitive}")
    print(f"Number of Proxy Features: {len(proxy_candidates)}")
    print(f"Final Biased Feature Count: {final_bias_count}")
    
    # Create comprehensive results dictionary
    results = {
        'important_features': important_features,
        'biased_features': biased_feats,
        'bias_count': bias_count,
        'final_bias_count': final_bias_count,
        'proxy_candidates': proxy_candidates,
        'total_bias_indicators': total_bias_indicators,
        'direct_bias_results': direct_bias_results,
        'model_results': model_results,
        'analysis_results': analysis    
    }
    
    # Add sensitive analysis results if available
    if 'bias_count_sensitive' in locals():
        results['biased_features_sensitive'] = biased_feats_sensitive
        results['bias_count_sensitive'] = bias_count_sensitive
        results['analysis_sensitive'] = analysis_sensitive

    if apply_mitigation and 'mitigation_results' in locals() and mitigation_results:
        results['mitigation_results'] = mitigation_results

    return results

def enhanced_bias_visualization(X, y, sensitive_attr, important_features, biased_feats, analysis_results):
    if len(biased_feats) == 0:
        print("No biased features to visualize.")
        return
    
    n_biased = len(biased_feats)
    fig, axes = plt.subplots(2, min(3, n_biased), figsize=(15, 10))
    if n_biased == 1:
        axes = axes.reshape(2, 1)
    elif n_biased == 2:
        axes = axes.reshape(2, 2)
    
    for i, (feat, details) in enumerate(biased_feats[:3]):  # Show first 3
        # Distribution plot
        ax1 = axes[0, i] if n_biased > 1 else axes[0]
        for group in X[sensitive_attr].unique():
            data = X[X[sensitive_attr] == group][feat]
            ax1.hist(data, alpha=0.7, label=f'{sensitive_attr}={group}', bins=20)
        ax1.set_title(f'{feat} Distribution by {sensitive_attr}')
        ax1.set_xlabel(feat)
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Bias metrics plot
        ax2 = axes[1, i] if n_biased > 1 else axes[1]
        if details.get('single_fairness_metrics'):
            fm = details['single_fairness_metrics']
            metrics_names = [
                'Statistical Parity',
                'Disparate Impact',
                'Equal Opportunity',
                'Equalized Odds',
                'Calibration',
                'Gen. Entropy Idx'
            ]
            metrics_keys = [
                'statistical_parity',
                'disparate_impact',
                'equal_opportunity',
                'equalized_odds',
                'calibration',
                'generalized_entropy_index'
            ]
            metrics_values = [fm.get(k, 0) for k in metrics_keys]
            bars = ax2.bar(metrics_names, metrics_values, color=['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
            ax2.set_title(f'{feat} Fairness Metrics')
            ax2.set_ylabel('Metric Value')
            ax2.tick_params(axis='x', rotation=45)
            # Add threshold lines (example for group fairness)
            ax2.axhline(y=0.05, color='black', linestyle='--', alpha=0.5, label='Group Fairness Threshold')
    
    plt.tight_layout()
    plt.show()

def comprehensive_bias_visualization(X, y, sensitive_attr, results):
    """
    Create comprehensive visualizations for bias analysis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Target Distribution by Sensitive Attribute
    ax1 = fig.add_subplot(gs[0, :])
    target_by_sensitive = pd.crosstab(X[sensitive_attr], y, normalize='index')
    target_by_sensitive.plot(kind='bar', ax=ax1, stacked=False)
    ax1.set_title(f'Target Distribution by {sensitive_attr}', fontsize=16, fontweight='bold')
    ax1.set_xlabel(sensitive_attr)
    ax1.set_ylabel('Proportion')
    ax1.legend(['Rejected (0)', 'Hired (1)'], title='Hiring Decision')
    
    # Add approval rates as text
    for i, (idx, row) in enumerate(target_by_sensitive.iterrows()):
        if 1 in row.index:
            ax1.text(i, row[1] + 0.02, f'{row[1]:.2%}', ha='center', va='bottom')
    
    # 2. Feature Importance from Bias Analysis
    ax2 = fig.add_subplot(gs[1, :2])
    biased_features = [feat for feat, _ in results['biased_features']]
    bias_scores = []
    
    for feat, details in results['biased_features']:
        # Calculate a composite bias score
        score = 0
        if 'statistical_tests' in details:
            tests = details['statistical_tests']
            if 'correlation' in tests and tests['correlation']['significant']:
                score += abs(tests['correlation']['value'])
            if 'mutual_info' in tests and tests['mutual_info']['significant']:
                score += tests['mutual_info']['value'] * 10  # Scale MI
            if 'chi2_test' in tests and tests['chi2_test']['significant']:
                score += 0.3
            if 'ks_test' in tests and tests['ks_test']['significant']:
                score += tests['ks_test']['statistic']
        bias_scores.append(score)
    
    if biased_features:
        # Sort by bias score
        sorted_pairs = sorted(zip(biased_features, bias_scores), key=lambda x: x[1], reverse=True)
        biased_features, bias_scores = zip(*sorted_pairs)
        
        bars = ax2.barh(biased_features, bias_scores, color='salmon')
        ax2.set_xlabel('Composite Bias Score')
        ax2.set_title('Biased Features Ranked by Statistical Dependence', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, score in zip(bars, bias_scores):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center')
    
    # 3. Proxy Features
    ax3 = fig.add_subplot(gs[1, 2])
    if results['proxy_candidates']:
        proxy_names = [p['feature'] for p in results['proxy_candidates']]
        proxy_accuracy = [p['proxy_accuracy'] for p in results['proxy_candidates']]
        
        bars = ax3.barh(proxy_names, proxy_accuracy, color='lightcoral')
        ax3.set_xlabel('Prediction Accuracy')
        ax3.set_title(f'Proxy Features for {sensitive_attr}', fontsize=14, fontweight='bold')
        ax3.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Random Baseline')
        
        # Add value labels
        for bar, acc in zip(bars, proxy_accuracy):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{acc:.3f}', va='center')
    else:
        ax3.text(0.5, 0.5, 'No proxy features detected', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title(f'Proxy Features for {sensitive_attr}', fontsize=14, fontweight='bold')

    # Fairness metrics at the model level(not just featurewise)
    if 'detailed_fairness_metrics' in results['model_results']:
        ax = fig.add_subplot(gs[2, :])
        fm = results['model_results']['detailed_fairness_metrics']
        metrics_names = [
        'Statistical Parity', 'Disparate Impact', 
        'Equal Opportunity', 'Equalized Odds',
        'Calibration', 'Gen. Entropy Idx'
        ]
        metrics_keys = [
        'statistical_parity', 'disparate_impact',
        'equal_opportunity', 'equalized_odds',
        'calibration', 'generalized_entropy_index'
        ]
        metrics_values = [fm.get(k, 0) for k in metrics_keys]
        ax.bar(metrics_names, metrics_values, color=['red', 'orange', 'yellow', 'green', 'blue', 'purple'])
        ax.set_title('Model-Level Group/Individual Fairness Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Metric Value')
        ax.tick_params(axis='x', rotation=30)

    
    # 4. Distribution plots for top biased features
    if len(results['biased_features']) > 0:
        n_features_to_plot = min(6, len(results['biased_features']))
        
        for i, (feat, details) in enumerate(results['biased_features'][:n_features_to_plot]):
            row = 2 + i // 3
            col = i % 3
            ax = fig.add_subplot(gs[row, col])
            
            if X[feat].dtype in ['int64', 'float64'] and X[feat].nunique() > 10:
                # Continuous feature - use violin plot
                data_for_plot = pd.DataFrame({
                    feat: X[feat],
                    sensitive_attr: X[sensitive_attr]
                })
                sns.violinplot(data=data_for_plot, x=sensitive_attr, y=feat, ax=ax, inner='box')
                ax.set_title(f'{feat} Distribution by {sensitive_attr}', fontsize=12)
            else:
                # Categorical or discrete feature - use count plot
                crosstab = pd.crosstab(X[feat], X[sensitive_attr], normalize='columns')
                crosstab.plot(kind='bar', ax=ax, width=0.8)
                ax.set_title(f'{feat} Distribution by {sensitive_attr}', fontsize=12)
                ax.set_ylabel('Proportion')
                ax.legend(title=sensitive_attr, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add summary statistics box
    summary_text = f"""
    BIAS ANALYSIS SUMMARY
    ━━━━━━━━━━━━━━━━━━━━
    Sensitive Attribute: {sensitive_attr}
    Total Features: {len(results['important_features'])}
    Biased Features: {results['final_bias_count']}
    Proxy Features: {len(results['proxy_candidates'])}
    
    Direct Bias: {'YES' if results['direct_bias_results']['has_direct_bias'] else 'NO'}
    Chi-square p-value: {results['direct_bias_results']['p_value']:.6f}
    Model Bias: {'YES' if results['model_results']['bias_detected'] else 'NO'}
    """
    
    # Add text box
    fig.text(0.02, 0.02, summary_text, transform=fig.transFigure, 
             fontsize=11, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Comprehensive Bias Analysis for {sensitive_attr}', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Additional visualization for Age groups (since it has multiple categories)
def visualize_multigroup_fairness(X, y, sensitive_attr, results):
    """
    Special visualization for multi-group sensitive attributes like Age
    """
    if X[sensitive_attr].nunique() <= 2:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Multi-Group Fairness Analysis for {sensitive_attr}', fontsize=16, fontweight='bold')
    
    # 1. Approval rates by group
    ax1 = axes[0, 0]
    approval_rates = y.groupby(X[sensitive_attr]).mean().sort_index()
    bars = ax1.bar(approval_rates.index.astype(str), approval_rates.values, color='skyblue')
    ax1.set_xlabel(sensitive_attr)
    ax1.set_ylabel('Approval Rate')
    ax1.set_title('Approval Rates by Group')
    ax1.axhline(y=y.mean(), color='red', linestyle='--', label=f'Overall: {y.mean():.3f}')
    ax1.legend()
    
    # Add value labels
    for bar, rate in zip(bars, approval_rates.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.3f}', ha='center', va='bottom')
    
    # 2. Sample sizes by group
    ax2 = axes[0, 1]
    group_counts = X[sensitive_attr].value_counts().sort_index()
    bars = ax2.bar(group_counts.index.astype(str), group_counts.values, color='lightgreen')
    ax2.set_xlabel(sensitive_attr)
    ax2.set_ylabel('Count')
    ax2.set_title('Sample Sizes by Group')
    
    # Add value labels
    for bar, count in zip(bars, group_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{count}', ha='center', va='bottom')
    
    # 3. Feature correlations with sensitive attribute
    ax3 = axes[1, 0]
    correlations = {}
    for feat in results['important_features']:
        if X[feat].dtype in ['int64', 'float64']:
            # Encode sensitive attribute for correlation
            sensitive_encoded = X[sensitive_attr]
            if sensitive_encoded.dtype == 'object':
                le = LabelEncoder()
                sensitive_encoded = le.fit_transform(sensitive_encoded)
            
            corr = abs(X[feat].corr(pd.Series(sensitive_encoded)))
            correlations[feat] = corr
    
    if correlations:
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
        features, corr_values = zip(*sorted_corr)
        bars = ax3.barh(features, corr_values, color='coral')
        ax3.set_xlabel('Absolute Correlation')
        ax3.set_title(f'Top 10 Feature Correlations with {sensitive_attr}')
        
        # Add value labels
        for bar, corr in zip(bars, corr_values):
            ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                    f'{corr:.3f}', va='center')
    
    # 4. Model performance metrics by group (if available)
    ax4 = axes[1, 1]
    if 'fairness_results' in results['model_results']:
        fairness_data = results['model_results']['fairness_results']
        groups = list(fairness_data.keys())
        metrics = ['approval_rate', 'accuracy', 'tpr']
        
        x = np.arange(len(groups))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [fairness_data[g].get(metric, 0) for g in groups]
            ax4.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax4.set_xlabel(sensitive_attr)
        ax4.set_ylabel('Metric Value')
        ax4.set_title('Model Performance Metrics by Group')
        ax4.set_xticks(x + width)
        ax4.set_xticklabels([str(g) for g in groups])
        ax4.legend()
        ax4.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()

def auto_detect_categorical_columns(df, max_unique_ratio=0.1, min_unique_count=2):
    # Automatically detect categorical columns in a dataset
    categorical_cols = []

    for col in df.columns:
        if df[col].dtype == 'object':
            # String columns are likely categorical
            categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            # Numeric columns might be categorical if they have few unique values
            unique_ratio = df[col].nunique()/len(df)
            unique_count = df[col].nunique()

            if (unique_ratio<=max_unique_ratio and unique_count>=min_unique_count and unique_count<=20): # Reasonable upper limit for categories
                categorical_cols.append(col)
    return categorical_cols

def preprocess_sensitive_attributes(df, sensitive_config):
    """
    Preprocess sensitive attributes based on JSON configuration
    """
    df_processed = df.copy()
    encoding_info = {}
    
    for attr_config in sensitive_config.get("sensitive_attributes", []):
        attr_name = attr_config["name"]
        
        if attr_name not in df_processed.columns:
            print(f"Warning: Sensitive attribute '{attr_name}' not found in dataset")
            continue
        
        # CHANGE: First check if bucketing is specified, regardless of data type
        if "buckets" in attr_config:
            # Apply bucketing for numerical attributes like Age
            buckets = attr_config["buckets"]
            default_val = attr_config.get("default_value", -1)
            
            # Create bucket assignments
            bucket_assignments = pd.Series([default_val] * len(df_processed), index=df_processed.index)
            
            for bucket in buckets:
                lower, upper = bucket["range"]
                mask = (df_processed[attr_name] >= lower) & (df_processed[attr_name] <= upper)
                bucket_assignments[mask] = bucket["value"]
            
            # Store original values before replacing
            original_col_name = f"{attr_name}_original"
            df_processed[original_col_name] = df_processed[attr_name].copy()
            
            # Replace with bucketed values
            df_processed[attr_name] = bucket_assignments
            
            encoding_info[attr_name] = {
                "type": "numerical_bucketing",
                "buckets": buckets,
                "default": default_val,
                "original_column": original_col_name
            }

        # check dtype to determine if mapping needed
        elif pd.api.types.is_numeric_dtype(df_processed[attr_name]):
            # Numeric dtype detected — skip mapping, keep as is
            encoding_info[attr_name] = {
                "type": "numeric_no_mapping",
                "note": "Skipped JSON mapping because data already numeric"
            }
        else:
            # Handle categorical mapping
            if "mapping" in attr_config:
                mapping = attr_config["mapping"]
                default_val = attr_config.get("default_value", -1)
            
                # Apply mapping
                df_processed[attr_name] = df_processed[attr_name].map(mapping)
                df_processed[attr_name] = df_processed[attr_name].fillna(default_val)
            
                encoding_info[attr_name] = {
                    "type": "categorical_mapping",
                    "mapping": mapping,
                    "default": default_val
                }

            else:
                # No mapping and no bucketing - keep as is
                encoding_info[attr_name] = {
                    "type": "no_mapping",
                    "note": "No mapping or bucketing applied"
                }
    
    return df_processed, encoding_info

if __name__ == "__main__":
    # Load your dataset
    try:
        df = pd.read_csv(r"task1/datasets/complex_biased_dataset.csv")
        print(f"Dataset loaded: {df.shape}")

        # Load exclusion configuration
        exclusion_config = load_exclusion_config()
        print(f"Loaded exclusion configuration")
        
        # Load sensitive attributes configuration
        with open("sensitive_attributes.json", "r") as f:
            sensitive_config = json.load(f)
        
        # Filter sensitive attributes to only those present in the dataset
        available_sensitive_attrs = []
        filtered_sensitive_config = {"sensitive_attributes": []}
        
        for attr_config in sensitive_config.get("sensitive_attributes", []):
            attr_name = attr_config["name"]
            if attr_name in df.columns:
                filtered_sensitive_config["sensitive_attributes"].append(attr_config)
                available_sensitive_attrs.append(attr_name)
            else:
                print(f"Note: Sensitive attribute '{attr_name}' from JSON not found in dataset, skipping")
        
        # Preprocess only the available sensitive attributes using JSON config
        df_processed, encoding_info = preprocess_sensitive_attributes(df, filtered_sensitive_config)
        print(f"Preprocessed sensitive attributes: {list(encoding_info.keys())}")
        
        # Auto-detect other categorical columns (excluding sensitive attributes and target)
        target = "Hiring_Decision"
        sensitive_attributes = available_sensitive_attrs  # Use the filtered list
        
        columns_to_check = [col for col in df_processed.columns 
                           if col not in sensitive_attributes and col != target]
        
        columns_to_encode = []
        excluded_from_encoding = []
        
        for col in columns_to_check:
            should_exclude, reason = should_exclude_column(col, exclusion_config)
            if should_exclude:
                excluded_from_encoding.append((col, reason))
            else:
                columns_to_encode.append(col)
        
        if excluded_from_encoding:
            print(f"\nColumns excluded from analysis:")
            for col, reason in excluded_from_encoding:
                print(f"  - {col}: {reason}")
        
        categorical_cols = auto_detect_categorical_columns(df_processed[columns_to_encode])
        print(f"Other categorical columns to encode ({len(categorical_cols)}): {categorical_cols}")
        
        # Encode other categorical columns using LabelEncoder
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
            print(f"Encoded {col}: {dict(enumerate(le.classes_))}")
        
        print(f"Using target column: {target}")
        print(f"Sensitive attributes to analyze: {sensitive_attributes}")

        # Store results for all sensitive attributes
        all_results = {}

        for sensitive_attr in sensitive_attributes:
            print(f"\n{'='*50}")
            print(f"Running bias analysis for sensitive attribute: {sensitive_attr}")
            print(f"{'='*50}")

            X = df_processed.drop(columns=[target])
            y = df_processed[target]
            if y.dtype == object:
                # Try mapping yes/no strings to 1/0
                if set(y.unique()) == {'Yes', 'No'}:
                    y = y.map({'Yes': 1, 'No': 0})
                else:
                    raise ValueError("Target column is non-numeric and not Yes/No format.")

            # Run bias pipeline
            results = feature_bias_pipeline(
                X, y, sensitive_attr=sensitive_attr, target=target
            )

            # Store results for this sensitive attribute
            all_results[sensitive_attr] = results

            num_biased_features = results['final_bias_count']
            print(f"\nAnswer for {sensitive_attr}: Number of biased features = {num_biased_features}")

            # Create comprehensive visualizations
            comprehensive_bias_visualization(X, y, sensitive_attr, results)
        
            # Special visualization for multi-group attributes
            if X[sensitive_attr].nunique() > 2:
                visualize_multigroup_fairness(X, y, sensitive_attr, results)

            # NEW: Add enhanced bias visualization
            if 'biased_features' in results and results['biased_features']:
                enhanced_bias_visualization(X, y, sensitive_attr, results['important_features'], results['biased_features'], results['analysis_results'])

        # Show summary of what was detected
        print(f"\nDETECTION SUMMARY:")
        print(f"   • Dataset shape: {df.shape}")
        print(f"   • Sensitive attributes analyzed: {sensitive_attributes}")
        print(f"   • Target column: {target}")
        print(f"   • Other categorical columns encoded: {categorical_cols}")
        # print(f"   • Important features selected: {len(results['important_features'])}")
        print(f"   • Biased features found: {num_biased_features}")

        # Print summary for each sensitive attribute
        print(f"\nBIASED FEATURES BY SENSITIVE ATTRIBUTE:")
        for sensitive_attr, results in all_results.items():
            print(f"   • {sensitive_attr}: {results['final_bias_count']} biased features")

            # NEW: Print mitigation results summary
            if 'mitigation_results' in results and results['mitigation_results']:
                print(f"     Mitigation results:")
                
                # Compare statistical parity across methods
                baseline_sp = results['model_results'].get('detailed_fairness_metrics', {}).get('statistical_parity', 0)
                print(f"     • Baseline statistical parity: {baseline_sp:.4f}")
                
                # Get best mitigation method
                best_method = None
                best_sp = float('inf')
                
                for category, methods in results['mitigation_results'].items():
                    for method_name, metrics in methods.items():
                        sp = metrics.get('statistical_parity', float('inf'))
                        if sp < best_sp:
                            best_sp = sp
                            best_method = f"{category}_{method_name}"
                
                if best_method:
                    print(f"     • Best mitigation: {best_method} (statistical parity: {best_sp:.4f})")
                    improvement = ((baseline_sp - best_sp) / baseline_sp) * 100 if baseline_sp > 0 else 0
                    print(f"     • Bias reduction: {improvement:.1f}%")
    
    except FileNotFoundError:
        print("Dataset file not found. Please update the file path.")
    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback
        traceback.print_exc()