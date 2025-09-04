import pandas as pd
import numpy as np
import traceback
from typing import Dict, List, Tuple, Any
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, pearsonr, ks_2samp, mannwhitneyu
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import uuid
import asyncio
from fastapi import HTTPException

from app.models.schemas import (
    AnalysisJob, AnalysisResults, ModelInfo, SensitiveFeature, 
    BiasMetric, FairnessScore, JobStatus, TaskType
)
from app.services.upload_service import upload_service
from app.utils.file_validators import detect_model_info, detect_sensitive_columns, preprocess_dataset


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


class AdvancedBiasDetection:
    """
    Advanced bias detection methods including direct bias analysis, proxy detection,
    comprehensive statistical testing, and multi-sensitive attribute analysis.
    """
    
    def __init__(self):
        self.fairness_metrics = FairnessMetrics()
    
    def direct_bias_analysis(self, X: pd.DataFrame, y: pd.Series, sensitive_attr: str) -> Dict[str, Any]:
        """
        Analyze direct statistical relationship between sensitive attribute and target
        """
        print(f"\nDIRECT BIAS ANALYSIS")
        print("-" * 30)
        
        # Check direct correlation between sensitive attribute and target
        sensitive_encoded = X[sensitive_attr]
        if X[sensitive_attr].dtype == 'object':
            le = LabelEncoder()
            sensitive_encoded = le.fit_transform(X[sensitive_attr])
        
        target_by_group = pd.DataFrame({
            'target': y, 
            'sensitive': sensitive_encoded
        }).groupby('sensitive')['target'].agg(['mean', 'count'])
        
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
    
    def detect_proxy_features(self, X: pd.DataFrame, selected_features: List[str], 
                            sensitive_attr: str) -> List[Dict[str, Any]]:
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
                mi_score = mutual_info_classif(feature_vals, sensitive_vals, 
                                             discrete_features='auto', random_state=42)[0]
                
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
    
    def comprehensive_statistical_testing(self, X: pd.DataFrame, y: pd.Series, 
                                        selected_features: List[str], sensitive_attr: str,
                                        corr_threshold: float = 0.2, mi_threshold: float = 0.02) -> Dict[str, Any]:
        """
        Comprehensive statistical testing for bias detection including KS test, Mann-Whitney U, etc.
        """
        print(f"\nCOMPREHENSIVE STATISTICAL TESTING")
        print("-" * 40)
        
        analysis_results = {}
        biased_features = []
        
        sensitive_vals = X[sensitive_attr].values
        if X[sensitive_attr].dtype == 'object':
            le = LabelEncoder()
            sensitive_vals = le.fit_transform(sensitive_vals)
        
        # Compute mutual information between features and sensitive attribute
        feature_data = X[selected_features]
        feature_data_encoded = feature_data.copy()
        for col in feature_data_encoded.columns:
            if feature_data_encoded[col].dtype == 'object':
                le = LabelEncoder()
                feature_data_encoded[col] = le.fit_transform(feature_data_encoded[col].astype(str))
        
        try:
            mi_scores = mutual_info_classif(feature_data_encoded, sensitive_vals, 
                                          discrete_features='auto', random_state=42)
            mi_dict = dict(zip(selected_features, mi_scores))
        except:
            mi_dict = {feat: 0 for feat in selected_features}
        
        for i, feature in enumerate(selected_features):
            print(f"\nAnalyzing feature {i+1}/{len(selected_features)}: {feature}")
            
            analysis = {}
            
            # Create clean data for analysis
            df_feat = pd.DataFrame({
                "feature": X[feature],
                "sensitive": X[sensitive_attr],
                "target": y
            }).dropna()
            
            if len(df_feat) == 0:
                continue
            
            # Statistical Dependence Check
            analysis['statistical_tests'] = {}
            
            # Correlation test (for numerical features)
            if df_feat["feature"].dtype in ['int64', 'float64']:
                try:
                    # Encode sensitive attribute for correlation if it's categorical
                    sensitive_encoded = df_feat["sensitive"]
                    if df_feat["sensitive"].dtype == 'object':
                        le = LabelEncoder()
                        sensitive_encoded = le.fit_transform(df_feat["sensitive"])
                    
                    corr, p_val = pearsonr(df_feat["feature"], sensitive_encoded)
                    analysis['statistical_tests']['correlation'] = {
                        'value': corr,
                        'p_value': p_val,
                        'significant': abs(corr) > corr_threshold and p_val < 0.05
                    }
                    print(f"   • Correlation with {sensitive_attr}: {corr:.4f} (p={p_val:.4f})")
                except Exception as e:
                    print(f"   • Correlation analysis failed: {e}")
                    analysis['statistical_tests']['correlation'] = {'value': 0, 'significant': False}
            else:
                analysis['statistical_tests']['correlation'] = {'value': 0, 'significant': False}
            
            # Mutual Information
            mi_score = mi_dict.get(feature, 0)
            analysis['statistical_tests']['mutual_info'] = {
                'value': mi_score,
                'significant': mi_score > mi_threshold
            }
            print(f"   • Mutual information: {mi_score:.6f}")
            
            # Distribution Analysis by sensitive groups
            if df_feat["feature"].dtype in ['int64', 'float64']:
                feat_by_sensitive = df_feat.groupby('sensitive')['feature'].agg(['mean', 'std', 'count'])
                print(f"   • {feature} by {sensitive_attr}:")
                print(f"     {feat_by_sensitive}")
            
            # Statistical tests for numerical features
            if df_feat["feature"].dtype in ['int64', 'float64']:
                # Kolmogorov-Smirnov test for distribution differences
                try:
                    groups = df_feat["sensitive"].unique()
                    if len(groups) == 2:
                        g1_vals = df_feat[df_feat["sensitive"] == groups[0]]["feature"]
                        g2_vals = df_feat[df_feat["sensitive"] == groups[1]]["feature"]
                        ks_stat, ks_p = ks_2samp(g1_vals, g2_vals)
                        
                        # Mann-Whitney U test (non-parametric)
                        u_stat, u_p = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
                        
                        analysis['statistical_tests']['ks_test'] = {
                            'statistic': ks_stat,
                            'p_value': ks_p,
                            'significant': ks_p < 0.05
                        }
                        analysis['statistical_tests']['mannwhitney_test'] = {
                            'statistic': u_stat,
                            'p_value': u_p,
                            'significant': u_p < 0.05
                        }
                        print(f"   • KS test p-value: {ks_p:.6f}")
                        print(f"   • Mann-Whitney U test p-value: {u_p:.6f}")
                except Exception as e:
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
                        'significant': chi2_p < 0.05
                    }
                    print(f"   • Chi-square test p-value: {chi2_p:.6f}")
                except Exception as e:
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
                
                fm_single = self.fairness_metrics.compute_all_metrics(
                    y.values, y_pred_single, y_pred_single, X[sensitive_attr].values
                )
                analysis['single_feature_metrics'] = fm_single
                
                if fm_single:
                    print(f"   • Single feature statistical parity: {fm_single.get('statistical_parity', 0):.4f}")
            except Exception as e:
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
            
            analysis['is_biased'] = stat_dependent
            analysis['bias_reasons'] = []
            
            if stat_dependent:
                analysis['bias_reasons'].append('statistical_dependence')
                biased_features.append((feature, analysis))
                print(f"   • BIASED: {', '.join(analysis['bias_reasons'])}")
            else:
                print(f"   • Not Biased")
            
            analysis_results[feature] = analysis
        
        return {
            'biased_features': biased_features,
            'analysis_results': analysis_results,
            'total_biased': len(biased_features)
        }
    
    def model_based_bias_testing(self, X: pd.DataFrame, y: pd.Series, 
                               selected_features: List[str], sensitive_attr: str) -> Dict[str, Any]:
        """
        Comprehensive model-based bias testing
        """
        print(f"\nMODEL-BASED BIAS TESTING")
        print("-" * 32)
        
        # Encode categorical features
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
        
        # Get sensitive attribute for test set
        sensitive_test = X.loc[X_test.index, sensitive_attr]
        
        # Compute comprehensive fairness metrics
        fairness_metrics = self.fairness_metrics.compute_all_metrics(
            y_test.values, y_pred, y_prob, sensitive_test.values
        )
        
        print("Model-based fairness metrics:")
        for metric, value in fairness_metrics.items():
            if isinstance(value, float):
                print(f"  • {metric}: {value:.4f}")
            else:
                print(f"  • {metric}: {value}")
        
        return {
            'model_accuracy': accuracy_score(y_test, y_pred),
            'fairness_metrics': fairness_metrics,
            'feature_importance': dict(zip(selected_features, model.coef_[0])) if hasattr(model, 'coef_') else {}
        }
    
    def multi_sensitive_attribute_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                         sensitive_attrs: List[str], 
                                         selected_features: List[str]) -> Dict[str, Any]:
        """
        Analyze bias across multiple sensitive attributes simultaneously
        """
        print(f"\nMULTI-SENSITIVE ATTRIBUTE ANALYSIS")
        print("-" * 40)
        print(f"Analyzing {len(sensitive_attrs)} sensitive attributes: {sensitive_attrs}")
        
        multi_analysis = {}
        
        # Individual analysis for each sensitive attribute
        for attr in sensitive_attrs:
            print(f"\n--- Analysis for {attr} ---")
            
            # Direct bias analysis
            direct_bias = self.direct_bias_analysis(X, y, attr)
            
            # Proxy detection
            proxy_features = self.detect_proxy_features(X, selected_features, attr)
            
            # Statistical testing
            statistical_results = self.comprehensive_statistical_testing(
                X, y, selected_features, attr
            )
            
            # Model-based testing
            model_results = self.model_based_bias_testing(X, y, selected_features, attr)
            
            multi_analysis[attr] = {
                'direct_bias': direct_bias,
                'proxy_features': proxy_features,
                'statistical_results': statistical_results,
                'model_results': model_results
            }
        
        # Cross-sensitive attribute analysis
        print(f"\n--- Cross-Attribute Analysis ---")
        cross_analysis = {}
        
        for i, attr1 in enumerate(sensitive_attrs):
            for j, attr2 in enumerate(sensitive_attrs[i+1:], i+1):
                print(f"\nAnalyzing interaction between {attr1} and {attr2}")
                
                # Create combined sensitive attribute
                combined_attr = f"{attr1}_{attr2}"
                X[combined_attr] = X[attr1].astype(str) + "_" + X[attr2].astype(str)
                
                # Analyze combined attribute
                try:
                    combined_direct_bias = self.direct_bias_analysis(X, y, combined_attr)
                    combined_model_results = self.model_based_bias_testing(
                        X, y, selected_features, combined_attr
                    )
                    
                    cross_analysis[f"{attr1}_x_{attr2}"] = {
                        'direct_bias': combined_direct_bias,
                        'model_results': combined_model_results
                    }
                except Exception as e:
                    print(f"Cross-analysis failed for {attr1} x {attr2}: {e}")
                    cross_analysis[f"{attr1}_x_{attr2}"] = {'error': str(e)}
        
        # Summary statistics
        summary = {
            'total_sensitive_attrs': len(sensitive_attrs),
            'attrs_with_direct_bias': len([attr for attr in sensitive_attrs 
                                         if multi_analysis[attr]['direct_bias']['has_direct_bias']]),
            'total_proxy_features': sum(len(multi_analysis[attr]['proxy_features']) 
                                      for attr in sensitive_attrs),
            'total_biased_features': sum(multi_analysis[attr]['statistical_results']['total_biased'] 
                                       for attr in sensitive_attrs)
        }
        
        return {
            'individual_analysis': multi_analysis,
            'cross_analysis': cross_analysis,
            'summary': summary
        }


class AnalysisService:
    def __init__(self):
        # In-memory storage for analysis jobs (use database in production)
        self.analysis_jobs: Dict[str, AnalysisJob] = {}
        self.analysis_results: Dict[str, AnalysisResults] = {}
        # Initialize advanced bias detection
        self.advanced_bias_detector = AdvancedBiasDetection()
    
    async def start_analysis(self, model_upload_id: str, train_dataset_upload_id: str, 
                           test_dataset_upload_id: str = None) -> AnalysisJob:
        """Start bias analysis job"""
        job = AnalysisJob(
            model_upload_id=model_upload_id,
            train_dataset_upload_id=train_dataset_upload_id,
            test_dataset_upload_id=test_dataset_upload_id
        )
        
        self.analysis_jobs[job.job_id] = job
        
        # Start analysis in background
        asyncio.create_task(self._run_analysis(job.job_id))
        
        return job
    
    async def _run_analysis(self, job_id: str):
        """Run the complete bias analysis workflow"""
        try:
            job = self.analysis_jobs[job_id]
            job.status = JobStatus.RUNNING
            job.progress = 10
            
            print(f"Starting analysis for job {job_id}")
            
            # Step 1: Load model and datasets
            print("Loading model and dataset...")
            model = upload_service.load_model(job.model_upload_id)
            train_df = upload_service.load_dataset(job.train_dataset_upload_id)
            print(f"Loaded dataset with shape: {train_df.shape}")
            
            # Preprocess dataset
            print("Preprocessing dataset...")
            train_df, target_column = preprocess_dataset(train_df)
            print(f"Target column identified: {target_column}")
            job.progress = 20
            
            # Step 2: Model analysis and info extraction
            print("Analyzing model...")
            model_info = self._analyze_model(model, train_df, target_column)
            print(f"Model type: {model_info.model_type}, Task: {model_info.task_type}")
            job.progress = 30
            
            # Step 3: Sensitive feature detection
            print("Detecting sensitive features...")
            sensitive_features = self._detect_sensitive_features(train_df, target_column, model_info.task_type)
            print(f"Found {len(sensitive_features)} sensitive features")
            job.progress = 50
            
            # Step 4: Bias detection (three types)
            print("Detecting bias...")
            bias_metrics = self._detect_bias(model, train_df, target_column, sensitive_features)
            print(f"Calculated {len(bias_metrics)} bias metrics")
            job.progress = 70
            
            # Step 5: Calculate fairness scores
            print("Calculating fairness scores...")
            fairness_score = self._calculate_fairness_score(bias_metrics)
            job.progress = 80
            
            # Step 6: Generate visualizations
            print("Generating visualizations...")
            visualizations = self._generate_visualizations(train_df, target_column, sensitive_features, bias_metrics)
            job.progress = 90
            
            # Step 7: Create analysis summary
            print("Creating analysis summary...")
            analysis_summary = self._create_analysis_summary(model_info, sensitive_features, bias_metrics, fairness_score)
            
            # Store results
            results = AnalysisResults(
                job_id=job_id,
                model_info=model_info,
                sensitive_features=sensitive_features,
                bias_metrics=bias_metrics,
                fairness_score=fairness_score,
                visualizations=visualizations,
                analysis_summary=analysis_summary
            )
            
            self.analysis_results[job_id] = results
            
            # Update job status
            job.status = JobStatus.COMPLETED
            job.progress = 100
            print(f"Analysis completed successfully for job {job_id}")
            
        except Exception as e:
            print(f"Analysis failed for job {job_id}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
    
    def _analyze_model(self, model, df: pd.DataFrame, target_column: str) -> ModelInfo:
        """Analyze model and extract information"""
        # Detect basic model info
        model_info_dict = detect_model_info(model)
        
        # Extract additional info from dataset
        feature_columns = [col for col in df.columns if col != target_column]
        
        # Determine task type if not detected
        task_type = model_info_dict["task_type"]
        if task_type is None:
            # Fallback: detect from target column
            if df[target_column].dtype in ['object', 'category', 'bool']:
                task_type = TaskType.CLASSIFICATION
            elif df[target_column].nunique() <= 10:  # Likely classification if few unique values
                task_type = TaskType.CLASSIFICATION
            else:
                task_type = TaskType.REGRESSION
            
            print(f"Task type not detected from model, inferred from data: {task_type}")
        
        # Get number of classes for classification
        n_classes = model_info_dict.get("n_classes")
        if task_type == TaskType.CLASSIFICATION and n_classes is None:
            n_classes = len(df[target_column].unique())
        
        return ModelInfo(
            model_type=model_info_dict["model_type"],
            task_type=task_type,
            framework=model_info_dict["framework"],
            n_features=len(feature_columns),
            n_classes=n_classes,
            feature_names=feature_columns,
            target_column=target_column,
            model_params={}
        )
    
    def _detect_sensitive_features(self, df: pd.DataFrame, target_column: str, 
                                 task_type: TaskType) -> List[SensitiveFeature]:
        """Detect sensitive features using statistical tests"""
        sensitive_features = []
        feature_columns = [col for col in df.columns if col != target_column]
        
        print(f"Analyzing {len(feature_columns)} features for sensitivity...")
        
        for i, feature in enumerate(feature_columns):
            if i % 10 == 0:  # Progress logging
                print(f"Processing feature {i+1}/{len(feature_columns)}: {feature}")
            
            # Skip if feature has too many unique values (likely not categorical sensitive attribute)
            if df[feature].dtype == 'object' and df[feature].nunique() > 20:
                print(f"Skipping {feature}: too many unique values ({df[feature].nunique()})")
                continue
            
            # Skip if feature has too many missing values
            if df[feature].isnull().sum() / len(df) > 0.5:
                print(f"Skipping {feature}: too many missing values")
                continue
            
            try:
                # Perform statistical tests
                correlation_score, p_value, test_type = self._perform_statistical_test(
                    df[feature], df[target_column], task_type
                )
                
                # Determine significance level
                if p_value < 0.001:
                    significance = "highly_significant"
                elif p_value < 0.01:
                    significance = "very_significant"
                elif p_value < 0.05:
                    significance = "significant"
                else:
                    significance = "not_significant"
                
                # Check if feature matches sensitive attribute patterns
                is_potentially_sensitive = any(pattern in feature.lower() 
                                             for pattern in ['gender', 'race', 'age', 'ethnicity', 'sex', 'religion', 'marital'])
                
                # Include if statistically significant or potentially sensitive
                if p_value < 0.05 or is_potentially_sensitive:
                    sensitive_features.append(SensitiveFeature(
                        feature_name=feature,
                        correlation_score=correlation_score,
                        p_value=p_value,
                        test_type=test_type,
                        significance_level=significance,
                        description=f"Statistical correlation with target variable detected using {test_type}"
                    ))
                    print(f"Found sensitive feature: {feature} (p={p_value:.4f})")
                    
            except Exception as e:
                print(f"Error testing feature {feature}: {str(e)}")
                continue
        
        print(f"Completed analysis of sensitive features. Found {len(sensitive_features)} candidates.")
        return sensitive_features
    
    def _perform_statistical_test(self, feature_series: pd.Series, target_series: pd.Series, 
                                task_type: TaskType) -> Tuple[float, float, str]:
        """Perform appropriate statistical test based on data types"""
        
        # Encode categorical variables
        if feature_series.dtype == 'object':
            le_feature = LabelEncoder()
            feature_encoded = le_feature.fit_transform(feature_series.astype(str))
        else:
            feature_encoded = feature_series.values
        
        if target_series.dtype == 'object' or task_type == TaskType.CLASSIFICATION:
            le_target = LabelEncoder()
            target_encoded = le_target.fit_transform(target_series.astype(str))
        else:
            target_encoded = target_series.values
        
        # Choose appropriate test
        if feature_series.dtype == 'object' and (target_series.dtype == 'object' or task_type == TaskType.CLASSIFICATION):
            # Chi-square test for categorical vs categorical
            contingency_table = pd.crosstab(feature_encoded, target_encoded)
            chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
            correlation_score = np.sqrt(chi2_stat / (chi2_stat + len(feature_series)))
            test_type = "chi_square"
            
        elif feature_series.dtype != 'object' and (target_series.dtype == 'object' or task_type == TaskType.CLASSIFICATION):
            # ANOVA F-test for numerical vs categorical
            groups = [target_encoded[feature_encoded == val] for val in np.unique(feature_encoded)]
            f_stat, p_value = stats.f_oneway(*groups)
            correlation_score = np.sqrt(f_stat / (f_stat + len(feature_series) - len(groups)))
            test_type = "anova_f_test"
            
        else:
            # Pearson correlation for numerical vs numerical
            correlation_score, p_value = stats.pearsonr(feature_encoded, target_encoded)
            correlation_score = abs(correlation_score)
            test_type = "pearson_correlation"
        
        return float(correlation_score), float(p_value), test_type
    
    def _detect_bias(self, model, df: pd.DataFrame, target_column: str, 
                    sensitive_features: List[SensitiveFeature]) -> List[BiasMetric]:
        """Detect comprehensive bias using advanced fairness metrics and detection methods"""
        bias_metrics = []
        fairness_calculator = FairnessMetrics()
        
        # Prepare data for model predictions if possible
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        print(f"Starting advanced bias detection for {len(sensitive_features)} sensitive features")
        
        # Get all feature columns (excluding target)
        feature_columns = [col for col in df.columns if col != target_column]
        
        try:
            # Phase 1: Advanced bias detection for each sensitive feature
            for sensitive_feature in sensitive_features:
                feature_name = sensitive_feature.feature_name
                print(f"\n=== Advanced Analysis for {feature_name} ===")
                
                # 1. Direct Bias Analysis
                direct_bias_results = self.advanced_bias_detector.direct_bias_analysis(
                    df, y, feature_name
                )
                
                # Convert direct bias results to BiasMetric
                if direct_bias_results['has_direct_bias']:
                    bias_metrics.append(BiasMetric(
                        metric_name=f"Direct Bias (Chi-square) - {feature_name}",
                        value=direct_bias_results['p_value'],
                        threshold=0.05,
                        is_biased=True,
                        severity="high" if direct_bias_results['p_value'] < 0.01 else "medium",
                        description=f"Direct statistical relationship between {feature_name} and target (p-value: {direct_bias_results['p_value']:.6f})"
                    ))
                
                # 2. Proxy Feature Detection
                proxy_features = self.advanced_bias_detector.detect_proxy_features(
                    df, feature_columns, feature_name
                )
                
                # Convert proxy detection results to BiasMetrics
                for proxy in proxy_features:
                    bias_metrics.append(BiasMetric(
                        metric_name=f"Proxy Feature Detection - {proxy['feature']} for {feature_name}",
                        value=proxy['proxy_accuracy'],
                        threshold=0.6,  # Baseline + 5% improvement threshold
                        is_biased=True,
                        severity="high" if proxy['proxy_accuracy'] > 0.8 else "medium",
                        description=f"Feature {proxy['feature']} can predict {feature_name} with {proxy['proxy_accuracy']:.3f} accuracy"
                    ))
                
                # 3. Comprehensive Statistical Testing
                statistical_results = self.advanced_bias_detector.comprehensive_statistical_testing(
                    df, y, feature_columns, feature_name
                )
                
                # Convert statistical testing results to BiasMetrics
                for feature, analysis in statistical_results['analysis_results'].items():
                    if analysis['is_biased']:
                        # Create metrics for significant statistical tests
                        for test_name, test_result in analysis['statistical_tests'].items():
                            if test_result.get('significant', False):
                                test_value = test_result.get('value', test_result.get('p_value', 0))
                                bias_metrics.append(BiasMetric(
                                    metric_name=f"Statistical Test ({test_name}) - {feature} vs {feature_name}",
                                    value=float(test_value),
                                    threshold=0.05 if 'p_value' in test_result else 0.2,
                                    is_biased=True,
                                    severity="high" if test_value < 0.01 or abs(test_value) > 0.5 else "medium",
                                    description=f"{test_name.replace('_', ' ').title()} shows significant relationship between {feature} and {feature_name}"
                                ))
                
                # 4. Model-Based Testing (if model is available)
                if hasattr(model, 'predict'):
                    try:
                        model_results = self.advanced_bias_detector.model_based_bias_testing(
                            df, y, feature_columns, feature_name
                        )
                        
                        # Convert model-based fairness metrics to BiasMetrics
                        for metric_name, metric_value in model_results['fairness_metrics'].items():
                            if metric_name in ['num_sensitive_groups', 'is_multiclass_sensitive', 'group_prediction_rates']:
                                continue  # Skip metadata
                            
                            if isinstance(metric_value, (int, float)):
                                threshold, severity_high, severity_medium = self._get_metric_thresholds(metric_name)
                                is_biased = self._is_metric_biased(metric_name, metric_value, threshold)
                                severity = self._get_metric_severity(metric_name, metric_value, severity_high, severity_medium)
                                
                                bias_metrics.append(BiasMetric(
                                    metric_name=f"Model-Based {metric_name.replace('_', ' ').title()} - {feature_name}",
                                    value=float(metric_value),
                                    threshold=threshold,
                                    is_biased=is_biased,
                                    severity=severity,
                                    description=self._get_metric_description(metric_name, feature_name)
                                ))
                    except Exception as e:
                        print(f"Model-based testing failed for {feature_name}: {e}")
            
            # Phase 2: Multi-Sensitive Attribute Analysis (if multiple sensitive features)
            if len(sensitive_features) > 1:
                print(f"\n=== Multi-Sensitive Attribute Analysis ===")
                sensitive_attr_names = [sf.feature_name for sf in sensitive_features]
                
                try:
                    multi_results = self.advanced_bias_detector.multi_sensitive_attribute_analysis(
                        df, y, sensitive_attr_names, feature_columns
                    )
                    
                    # Add summary metrics for multi-attribute analysis
                    summary = multi_results['summary']
                    
                    bias_metrics.append(BiasMetric(
                        metric_name="Multi-Attribute Direct Bias Count",
                        value=float(summary['attrs_with_direct_bias']),
                        threshold=1.0,
                        is_biased=summary['attrs_with_direct_bias'] > 0,
                        severity="high" if summary['attrs_with_direct_bias'] > 1 else "medium",
                        description=f"{summary['attrs_with_direct_bias']} out of {summary['total_sensitive_attrs']} sensitive attributes show direct bias"
                    ))
                    
                    bias_metrics.append(BiasMetric(
                        metric_name="Multi-Attribute Proxy Features Count",
                        value=float(summary['total_proxy_features']),
                        threshold=1.0,
                        is_biased=summary['total_proxy_features'] > 0,
                        severity="high" if summary['total_proxy_features'] > 3 else "medium" if summary['total_proxy_features'] > 0 else "low",
                        description=f"{summary['total_proxy_features']} potential proxy features detected across all sensitive attributes"
                    ))
                    
                    # Add cross-attribute interaction metrics
                    for interaction_name, interaction_results in multi_results['cross_analysis'].items():
                        if 'error' not in interaction_results and interaction_results.get('direct_bias', {}).get('has_direct_bias', False):
                            bias_metrics.append(BiasMetric(
                                metric_name=f"Cross-Attribute Bias - {interaction_name}",
                                value=interaction_results['direct_bias']['p_value'],
                                threshold=0.05,
                                is_biased=True,
                                severity="high" if interaction_results['direct_bias']['p_value'] < 0.01 else "medium",
                                description=f"Significant bias detected in interaction between {interaction_name.replace('_x_', ' and ')}"
                            ))
                
                except Exception as e:
                    print(f"Multi-sensitive attribute analysis failed: {e}")
            
            # Phase 3: Legacy comprehensive fairness metrics (for compatibility)
            if hasattr(model, 'predict'):
                try:
                    X_values = X.values
                    y_pred = model.predict(X_values)
                    
                    # Get prediction probabilities if available
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_values)
                        if y_prob.ndim > 1:
                            y_prob = y_prob[:, 1]  # Use positive class probabilities
                    else:
                        y_prob = y_pred.astype(float)
                    
                    # Convert to binary if needed
                    if len(np.unique(y.values)) > 2:
                        y_true_binary = (y.values > np.median(y.values)).astype(int)
                        y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
                    else:
                        y_true_binary = y.values.astype(int)
                        y_pred_binary = y_pred.astype(int)
                    
                    # Calculate comprehensive fairness metrics for each sensitive feature
                    for sensitive_feature in sensitive_features:
                        feature_name = sensitive_feature.feature_name
                        sensitive_attr = df[feature_name].values
                        
                        # Encode sensitive attribute if categorical
                        if df[feature_name].dtype == 'object':
                            le = LabelEncoder()
                            sensitive_attr = le.fit_transform(sensitive_attr.astype(str))
                        
                        # Compute all fairness metrics
                        all_metrics = fairness_calculator.compute_all_metrics(
                            y_true_binary, y_pred_binary, y_prob, sensitive_attr
                        )
                        
                        # Add standard fairness metrics for comprehensive reporting
                        for metric_name, metric_value in all_metrics.items():
                            if metric_name in ['num_sensitive_groups', 'is_multiclass_sensitive', 'group_prediction_rates']:
                                continue  # Skip metadata
                            
                            if isinstance(metric_value, (int, float)):
                                threshold, severity_high, severity_medium = self._get_metric_thresholds(metric_name)
                                is_biased = self._is_metric_biased(metric_name, metric_value, threshold)
                                severity = self._get_metric_severity(metric_name, metric_value, severity_high, severity_medium)
                                
                                bias_metrics.append(BiasMetric(
                                    metric_name=f"Standard {metric_name.replace('_', ' ').title()} - {feature_name}",
                                    value=float(metric_value),
                                    threshold=threshold,
                                    is_biased=is_biased,
                                    severity=severity,
                                    description=self._get_metric_description(metric_name, feature_name)
                                ))
                
                except Exception as e:
                    print(f"Error in standard fairness metrics calculation: {e}")
            
            else:
                print("Model does not support predictions. Using dataset-based metrics only.")
                
                # Fallback to dataset-based metrics when model predictions are not available
                for sensitive_feature in sensitive_features:
                    feature_name = sensitive_feature.feature_name
                    
                    # Calculate demographic parity difference
                    dp_diff = self._calculate_demographic_parity_difference(df, feature_name, target_column)
                    
                    bias_metrics.append(BiasMetric(
                        metric_name=f"Demographic Parity Difference - {feature_name}",
                        value=dp_diff,
                        threshold=0.1,
                        is_biased=abs(dp_diff) > 0.1,
                        severity="high" if abs(dp_diff) > 0.2 else "medium" if abs(dp_diff) > 0.1 else "low",
                        description=f"Difference in positive outcome rates between groups in {feature_name}"
                    ))
                    
                    # Calculate disparate impact ratio
                    di_ratio = self._calculate_disparate_impact_ratio(df, feature_name, target_column)
                    
                    bias_metrics.append(BiasMetric(
                        metric_name=f"Disparate Impact Ratio - {feature_name}",
                        value=di_ratio,
                        threshold=0.8,
                        is_biased=di_ratio < 0.8 or di_ratio > 1.25,
                        severity="high" if di_ratio < 0.6 or di_ratio > 1.67 else "medium" if di_ratio < 0.8 or di_ratio > 1.25 else "low",
                        description=f"Ratio of positive outcome rates between groups in {feature_name}"
                    ))
                    
        except Exception as e:
            print(f"Error in advanced bias detection: {str(e)}")
            traceback.print_exc()
        
        print(f"Completed advanced bias analysis. Generated {len(bias_metrics)} bias metrics.")
        return bias_metrics
    
    def _get_metric_thresholds(self, metric_name: str) -> tuple:
        """Get appropriate thresholds for different fairness metrics"""
        thresholds = {
            'statistical_parity': (0.1, 0.2, 0.1),
            'disparate_impact': (0.8, 0.6, 0.8),
            'equal_opportunity': (0.1, 0.2, 0.1),
            'equalized_odds': (0.1, 0.2, 0.1),
            'calibration': (0.1, 0.2, 0.1),
            'generalized_entropy_index': (0.5, 1.0, 0.5)
        }
        return thresholds.get(metric_name, (0.1, 0.2, 0.1))
    
    def _is_metric_biased(self, metric_name: str, value: float, threshold: float) -> bool:
        """Determine if a metric indicates bias"""
        if metric_name == 'disparate_impact':
            return value < 0.8 or value > 1.25
        else:
            return abs(value) > threshold
    
    def _get_metric_severity(self, metric_name: str, value: float, high_threshold: float, medium_threshold: float) -> str:
        """Determine severity level of bias"""
        if metric_name == 'disparate_impact':
            if value < 0.6 or value > 1.67:
                return "high"
            elif value < 0.8 or value > 1.25:
                return "medium"
            else:
                return "low"
        else:
            abs_value = abs(value)
            if abs_value > high_threshold:
                return "high"
            elif abs_value > medium_threshold:
                return "medium"
            else:
                return "low"
    
    def _get_metric_description(self, metric_name: str, feature_name: str) -> str:
        """Get description for each fairness metric"""
        descriptions = {
            'statistical_parity': f"Difference in positive prediction rates between groups in {feature_name}",
            'disparate_impact': f"Ratio of positive prediction rates between groups in {feature_name}",
            'equal_opportunity': f"Difference in true positive rates between groups in {feature_name}",
            'equalized_odds': f"Maximum difference in TPR/FPR between groups in {feature_name}",
            'calibration': f"Difference in calibration (reliability) between groups in {feature_name}",
            'generalized_entropy_index': f"Individual fairness measure for predictions in {feature_name}"
        }
        return descriptions.get(metric_name, f"Fairness metric for {feature_name}")
    
    def _calculate_demographic_parity_difference(self, df: pd.DataFrame, 
                                               feature_column: str, target_column: str) -> float:
        """Calculate demographic parity difference"""
        try:
            groups = df.groupby(feature_column)[target_column].mean()
            max_rate = groups.max()
            min_rate = groups.min()
            return float(max_rate - min_rate)
        except:
            return 0.0
    
    def _calculate_disparate_impact_ratio(self, df: pd.DataFrame, 
                                        feature_column: str, target_column: str) -> float:
        """Calculate disparate impact ratio"""
        try:
            groups = df.groupby(feature_column)[target_column].mean()
            max_rate = groups.max()
            min_rate = groups.min()
            if max_rate == 0:
                return 1.0
            return float(min_rate / max_rate)
        except:
            return 1.0
    
    def _calculate_equalized_odds_difference(self, y_true, y_pred, sensitive_feature) -> float:
        """Calculate equalized odds difference"""
        try:
            # Convert to binary if needed
            if len(np.unique(y_true)) > 2:
                y_true_binary = (y_true > np.median(y_true)).astype(int)
                y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
            else:
                y_true_binary = y_true
                y_pred_binary = y_pred
            
            groups = np.unique(sensitive_feature)
            if len(groups) < 2:
                return 0.0
            
            tpr_diff = 0.0
            fpr_diff = 0.0
            
            for i in range(len(groups) - 1):
                for j in range(i + 1, len(groups)):
                    mask_i = sensitive_feature == groups[i]
                    mask_j = sensitive_feature == groups[j]
                    
                    # Calculate TPR for each group
                    tpr_i = np.sum((y_true_binary[mask_i] == 1) & (y_pred_binary[mask_i] == 1)) / max(1, np.sum(y_true_binary[mask_i] == 1))
                    tpr_j = np.sum((y_true_binary[mask_j] == 1) & (y_pred_binary[mask_j] == 1)) / max(1, np.sum(y_true_binary[mask_j] == 1))
                    
                    # Calculate FPR for each group
                    fpr_i = np.sum((y_true_binary[mask_i] == 0) & (y_pred_binary[mask_i] == 1)) / max(1, np.sum(y_true_binary[mask_i] == 0))
                    fpr_j = np.sum((y_true_binary[mask_j] == 0) & (y_pred_binary[mask_j] == 1)) / max(1, np.sum(y_true_binary[mask_j] == 0))
                    
                    tpr_diff = max(tpr_diff, abs(tpr_i - tpr_j))
                    fpr_diff = max(fpr_diff, abs(fpr_i - fpr_j))
            
            return float(max(tpr_diff, fpr_diff))
            
        except:
            return 0.0
    
    def _calculate_fairness_score(self, bias_metrics: List[BiasMetric]) -> FairnessScore:
        """Calculate overall fairness score with improved handling of comprehensive metrics"""
        if not bias_metrics:
            return FairnessScore(
                overall_score=100.0,
                bias_score=0.0,
                fairness_level="excellent",
                metrics_breakdown={},
                recommendations=["No bias detected"]
            )
        
        # Calculate weighted bias score with metric-specific normalization
        bias_scores = []
        metrics_breakdown = {}
        
        for metric in bias_metrics:
            # Improved normalization based on metric type
            normalized_score = self._normalize_metric_score(metric.metric_name, metric.value)
            
            bias_scores.append(normalized_score)
            metrics_breakdown[metric.metric_name] = normalized_score
        
        bias_score = np.mean(bias_scores)
        fairness_score = 100 - bias_score
        
        # Determine fairness level with more nuanced thresholds
        if fairness_score >= 90:
            fairness_level = "excellent"
        elif fairness_score >= 80:
            fairness_level = "very_good"
        elif fairness_score >= 70:
            fairness_level = "good"
        elif fairness_score >= 60:
            fairness_level = "fair"
        elif fairness_score >= 40:
            fairness_level = "poor"
        else:
            fairness_level = "very_poor"
        
        # Generate comprehensive recommendations
        recommendations = self._generate_comprehensive_recommendations(bias_metrics, fairness_score)
        
        return FairnessScore(
            overall_score=float(fairness_score),
            bias_score=float(bias_score),
            fairness_level=fairness_level,
            metrics_breakdown=metrics_breakdown,
            recommendations=recommendations
        )
    
    def _normalize_metric_score(self, metric_name: str, value: float) -> float:
        """Normalize metric scores based on their specific characteristics"""
        try:
            # Handle different metric types with appropriate scaling
            if "Statistical Parity" in metric_name:
                return min(100, abs(value) * 1000)  # Scale 0-0.1 to 0-100
            elif "Disparate Impact" in metric_name:
                deviation = abs(1 - value)
                return min(100, deviation * 500)  # Scale deviation from 1.0
            elif "Equal Opportunity" in metric_name:
                return min(100, abs(value) * 1000)  # Scale 0-0.1 to 0-100
            elif "Equalized Odds" in metric_name:
                return min(100, abs(value) * 1000)  # Scale 0-0.1 to 0-100
            elif "Calibration" in metric_name:
                return min(100, abs(value) * 1000)  # Scale 0-0.1 to 0-100
            elif "Generalized Entropy" in metric_name:
                return min(100, abs(value) * 100)   # Scale 0-1 to 0-100
            else:
                # Default scaling for other metrics
                return min(100, abs(value) * 500)
        except:
            return 0.0
    
    def _generate_comprehensive_recommendations(self, bias_metrics: List[BiasMetric], fairness_score: float) -> List[str]:
        """Generate comprehensive recommendations based on specific bias patterns"""
        recommendations = []
        
        # Categorize metrics by type
        parity_metrics = [m for m in bias_metrics if "Statistical Parity" in m.metric_name or "Demographic Parity" in m.metric_name]
        equality_metrics = [m for m in bias_metrics if "Equal" in m.metric_name and "Odds" in m.metric_name]
        impact_metrics = [m for m in bias_metrics if "Disparate Impact" in m.metric_name]
        calibration_metrics = [m for m in bias_metrics if "Calibration" in m.metric_name]
        
        high_bias_metrics = [m for m in bias_metrics if m.severity == "high"]
        medium_bias_metrics = [m for m in bias_metrics if m.severity == "medium"]
        
        # Overall fairness assessment
        if fairness_score < 40:
            recommendations.append("🚨 CRITICAL: Severe bias detected across multiple metrics")
            recommendations.append("⛔ Model deployment strongly discouraged without bias mitigation")
        elif fairness_score < 60:
            recommendations.append("⚠️ Significant bias detected. Immediate attention required")
            recommendations.append("🔄 Consider comprehensive bias mitigation strategies")
        elif fairness_score < 80:
            recommendations.append("🔍 Moderate bias levels detected. Improvement recommended")
        else:
            recommendations.append("✅ Good overall fairness level achieved")
        
        # Specific metric-based recommendations
        if any(m.is_biased for m in parity_metrics):
            recommendations.append("� Statistical Parity violated: Different approval rates across groups")
            recommendations.append("💡 Consider: Data rebalancing, fairness constraints, or post-processing")
        
        if any(m.is_biased for m in equality_metrics):
            recommendations.append("⚖️ Equal Opportunity/Equalized Odds violated: Unequal error rates")
            recommendations.append("💡 Consider: Threshold optimization or in-processing fairness methods")
        
        if any(m.is_biased for m in impact_metrics):
            recommendations.append("� Disparate Impact detected: Legal compliance concerns")
            recommendations.append("💡 Consider: Feature selection review and bias-aware preprocessing")
        
        if any(m.is_biased for m in calibration_metrics):
            recommendations.append("🎯 Calibration issues: Model reliability varies across groups")
            recommendations.append("💡 Consider: Group-specific calibration or probability calibration methods")
        
        # Severity-based recommendations
        if high_bias_metrics:
            affected_features = list(set([m.metric_name.split(" - ")[1] for m in high_bias_metrics if " - " in m.metric_name]))
            recommendations.append(f"🚨 High-severity bias in: {', '.join(affected_features)}")
            recommendations.append("� Priority: Review data collection and feature engineering for these attributes")
        
        if medium_bias_metrics:
            recommendations.append("⚡ Medium-severity bias detected. Monitor and consider mitigation")
        
        # General improvement suggestions
        if fairness_score < 80:
            recommendations.append("🛠️ Suggested approaches:")
            recommendations.append("   • Pre-processing: Reweighting, synthetic data generation")
            recommendations.append("   • In-processing: Fairness-constrained optimization")
            recommendations.append("   • Post-processing: Threshold optimization, outcome redistribution")
        
        if fairness_score >= 75:
            recommendations.append("👀 Continue monitoring for bias drift over time")
            recommendations.append("📈 Consider A/B testing with fairness metrics")
        
        return recommendations
    
    def _generate_visualizations(self, df: pd.DataFrame, target_column: str, 
                               sensitive_features: List[SensitiveFeature], 
                               bias_metrics: List[BiasMetric]) -> Dict[str, any]:
        """Generate visualization data for frontend charts"""
        visualizations = {}
        
        try:
            # 1. Target distribution by sensitive groups - return data for bar charts
            if sensitive_features:
                target_distribution_data = []
                
                for sensitive_feature in sensitive_features[:4]:  # Limit to 4 features
                    feature_name = sensitive_feature.feature_name
                    
                    # Create grouped data
                    grouped_data = df.groupby([feature_name, target_column]).size().unstack(fill_value=0)
                    
                    # Convert to format suitable for Recharts
                    chart_data = []
                    for index_val in grouped_data.index:
                        data_point = {"group": str(index_val)}
                        for col in grouped_data.columns:
                            data_point[str(col)] = int(grouped_data.loc[index_val, col])
                        chart_data.append(data_point)
                    
                    target_distribution_data.append({
                        "feature_name": feature_name,
                        "data": chart_data,
                        "keys": [str(col) for col in grouped_data.columns]
                    })
                
                visualizations["target_distribution"] = target_distribution_data
            
            # 2. Bias metrics data for radar/bar chart
            if bias_metrics:
                bias_chart_data = []
                
                for metric in bias_metrics[:12]:  # All metrics
                    bias_chart_data.append({
                        "metric": metric.metric_name.replace("_", " ").title(),
                        "value": float(metric.value),
                        "threshold": float(metric.threshold),
                        "is_biased": metric.is_biased,
                        "severity": metric.severity
                    })
                
                visualizations["bias_metrics"] = bias_chart_data
                
                # Radar chart data (top 6 metrics)
                radar_data = []
                for metric in bias_metrics[:6]:
                    radar_data.append({
                        "metric": metric.metric_name.replace("_", " ").title()[:15],  # Truncate long names
                        "value": min(1.0, abs(float(metric.value))),
                        "fullName": metric.metric_name.replace("_", " ").title()
                    })
                
                visualizations["bias_radar"] = radar_data
            
            # 3. Fairness score breakdown - create meaningful fairness categories
            fairness_breakdown = []
            
            if bias_metrics:
                # Group metrics by category
                demographic_metrics = [m for m in bias_metrics if "demographic" in m.metric_name.lower() or "parity" in m.metric_name.lower()]
                equality_metrics = [m for m in bias_metrics if "equality" in m.metric_name.lower() or "odds" in m.metric_name.lower()]
                predictive_metrics = [m for m in bias_metrics if "predictive" in m.metric_name.lower() or "value" in m.metric_name.lower()]
                calibration_metrics = [m for m in bias_metrics if "calibration" in m.metric_name.lower()]
                
                # Calculate category scores
                if demographic_metrics:
                    avg_score = np.mean([1 - abs(m.value) for m in demographic_metrics])
                    fairness_breakdown.append({
                        "category": "Demographic Parity",
                        "score": float(max(0, min(1, avg_score))),
                        "description": "Equal selection rates across groups"
                    })
                
                if equality_metrics:
                    avg_score = np.mean([1 - abs(m.value) for m in equality_metrics])
                    fairness_breakdown.append({
                        "category": "Equality of Odds",
                        "score": float(max(0, min(1, avg_score))),
                        "description": "Equal true/false positive rates"
                    })
                
                if predictive_metrics:
                    avg_score = np.mean([1 - abs(m.value) for m in predictive_metrics])
                    fairness_breakdown.append({
                        "category": "Predictive Parity",
                        "score": float(max(0, min(1, avg_score))),
                        "description": "Equal predictive values across groups"
                    })
                
                if calibration_metrics:
                    avg_score = np.mean([1 - abs(m.value) for m in calibration_metrics])
                    fairness_breakdown.append({
                        "category": "Calibration",
                        "score": float(max(0, min(1, avg_score))),
                        "description": "Consistent probability calibration"
                    })
                
                # If no specific categories, create general breakdown
                if not fairness_breakdown:
                    fairness_breakdown = [
                        {
                            "category": "Overall Fairness",
                            "score": float(max(0, min(1, 1 - np.mean([abs(m.value) for m in bias_metrics])))),
                            "description": "Combined fairness assessment"
                        },
                        {
                            "category": "Bias Detection",
                            "score": float(max(0, min(1, len([m for m in bias_metrics if not m.is_biased]) / len(bias_metrics)))),
                            "description": "Percentage of unbiased metrics"
                        }
                    ]
            
            if fairness_breakdown:
                visualizations["fairness_breakdown"] = fairness_breakdown
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print(f"Generated visualizations keys: {list(visualizations.keys())}")
        for key, value in visualizations.items():
            if isinstance(value, list):
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {type(value)}")
        
        return visualizations
    
    def _create_analysis_summary(self, model_info: ModelInfo, sensitive_features: List[SensitiveFeature], 
                                bias_metrics: List[BiasMetric], fairness_score: FairnessScore) -> str:
        """Create a comprehensive analysis summary with advanced fairness insights"""
        
        # Categorize metrics by type for better organization
        parity_metrics = [m for m in bias_metrics if "statistical parity" in m.metric_name.lower() or "demographic parity" in m.metric_name.lower()]
        equality_metrics = [m for m in bias_metrics if ("equal opportunity" in m.metric_name.lower() or "equalized odds" in m.metric_name.lower())]
        impact_metrics = [m for m in bias_metrics if "disparate impact" in m.metric_name.lower()]
        calibration_metrics = [m for m in bias_metrics if "calibration" in m.metric_name.lower()]
        entropy_metrics = [m for m in bias_metrics if "entropy" in m.metric_name.lower()]
        
        # Count biased metrics by severity
        high_bias_count = len([m for m in bias_metrics if m.severity == "high"])
        medium_bias_count = len([m for m in bias_metrics if m.severity == "medium"])
        total_biased = len([m for m in bias_metrics if m.is_biased])
        
        summary = f"""
# Comprehensive Bias Analysis Report

## Model Information
- **Model Type**: {model_info.model_type.value}
- **Framework**: {model_info.framework}
- **Task Type**: {model_info.task_type.value if model_info.task_type else 'Unknown'}
- **Features Analyzed**: {model_info.n_features}
- **Target Variable**: {model_info.target_column}

## Sensitive Features Analysis
**{len(sensitive_features)} sensitive features detected** through statistical testing:
{chr(10).join([f"- **{sf.feature_name}**: {sf.test_type.replace('_', ' ').title()} (p-value: {sf.p_value:.4f}, significance: {sf.significance_level})" for sf in sensitive_features[:7]])}
{f"{chr(10)}... and {len(sensitive_features) - 7} more" if len(sensitive_features) > 7 else ""}

## Fairness Assessment
- **Overall Fairness Score**: {fairness_score.overall_score:.1f}/100
- **Bias Severity Score**: {fairness_score.bias_score:.1f}/100
- **Fairness Level**: {fairness_score.fairness_level.replace('_', ' ').title()}
- **Metrics Analyzed**: {len(bias_metrics)} comprehensive fairness metrics
- **Biased Metrics**: {total_biased}/{len(bias_metrics)} ({(total_biased/len(bias_metrics)*100):.1f}% if bias_metrics else 0)

## Detailed Fairness Metrics

### Statistical Parity & Demographic Fairness
{chr(10).join([f"- **{m.metric_name}**: {m.value:.3f} ({'❌ BIASED' if m.is_biased else '✅ FAIR'}) [{m.severity.upper()}]" for m in parity_metrics[:3]]) if parity_metrics else "- No demographic parity metrics available"}

### Equal Opportunity & Equalized Odds
{chr(10).join([f"- **{m.metric_name}**: {m.value:.3f} ({'❌ BIASED' if m.is_biased else '✅ FAIR'}) [{m.severity.upper()}]" for m in equality_metrics[:3]]) if equality_metrics else "- No equality metrics available"}

### Disparate Impact Analysis
{chr(10).join([f"- **{m.metric_name}**: {m.value:.3f} ({'❌ BIASED' if m.is_biased else '✅ FAIR'}) [{m.severity.upper()}]" for m in impact_metrics[:3]]) if impact_metrics else "- No disparate impact metrics available"}

### Calibration & Individual Fairness
{chr(10).join([f"- **{m.metric_name}**: {m.value:.3f} ({'❌ BIASED' if m.is_biased else '✅ FAIR'}) [{m.severity.upper()}]" for m in (calibration_metrics + entropy_metrics)[:3]]) if (calibration_metrics or entropy_metrics) else "- No calibration/individual fairness metrics available"}

## Bias Severity Analysis
- **High Severity Issues**: {high_bias_count} metrics
- **Medium Severity Issues**: {medium_bias_count} metrics
- **Low/No Issues**: {len(bias_metrics) - high_bias_count - medium_bias_count} metrics

## Key Findings & Insights
{chr(10).join([f"- {metric.metric_name}: {metric.value:.3f} - {metric.description}" for metric in sorted(bias_metrics, key=lambda x: (x.severity == 'low', x.severity == 'medium', x.severity == 'high'), reverse=True)[:5]])}

## Actionable Recommendations
{chr(10).join([f"- {rec}" for rec in fairness_score.recommendations])}

## Legal & Ethical Considerations
{self._generate_legal_considerations(bias_metrics, fairness_score)}

---
*This analysis was conducted using {len(bias_metrics)} state-of-the-art fairness metrics including Statistical Parity, Disparate Impact, Equal Opportunity, Equalized Odds, Calibration, and Individual Fairness measures.*
        """
        
        return summary.strip()
    
    def _generate_legal_considerations(self, bias_metrics: List[BiasMetric], fairness_score: FairnessScore) -> str:
        """Generate legal and ethical considerations based on bias analysis"""
        considerations = []
        
        disparate_impact_issues = [m for m in bias_metrics if "disparate impact" in m.metric_name.lower() and m.is_biased]
        parity_issues = [m for m in bias_metrics if "parity" in m.metric_name.lower() and m.is_biased]
        
        if fairness_score.overall_score < 40:
            considerations.append("🚨 **CRITICAL**: Model may violate anti-discrimination laws")
            considerations.append("⚖️ Legal review strongly recommended before deployment")
        elif fairness_score.overall_score < 60:
            considerations.append("⚠️ **WARNING**: Potential compliance risks identified")
            considerations.append("📋 Document bias analysis for regulatory compliance")
        
        if disparate_impact_issues:
            considerations.append("📊 Disparate impact detected - may violate Equal Employment Opportunity laws")
        
        if parity_issues:
            considerations.append("⚖️ Statistical parity violations may indicate systemic bias")
        
        if fairness_score.overall_score >= 80:
            considerations.append("✅ Model demonstrates good fairness practices")
            considerations.append("👥 Continue monitoring for protected class impacts")
        
        if not considerations:
            considerations.append("📋 Consider consulting legal experts on AI fairness requirements")
            considerations.append("📖 Stay updated on evolving AI governance regulations")
        
        return chr(10).join(considerations)
    
    def get_analysis_job(self, job_id: str) -> AnalysisJob:
        """Get analysis job by ID"""
        if job_id not in self.analysis_jobs:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        return self.analysis_jobs[job_id]
    
    def get_analysis_results(self, job_id: str) -> AnalysisResults:
        """Get analysis results by job ID"""
        if job_id not in self.analysis_results:
            raise HTTPException(status_code=404, detail="Analysis results not found")
        return self.analysis_results[job_id]


# Global analysis service instance
analysis_service = AnalysisService()
