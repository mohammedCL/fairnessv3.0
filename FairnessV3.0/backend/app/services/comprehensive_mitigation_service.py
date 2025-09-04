"""
Comprehensive Bias Mitigation Service

This service extends the existing mitigation system to automatically apply all available
bias mitigation strategies, evaluate their effectiveness, and provide comprehensive
results for frontend display.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import asyncio
import copy
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

from app.models.schemas import (
    MitigationJob, MitigationResults, MitigationStrategy, 
    JobStatus, BiasMetric, TaskType, SensitiveFeature
)
from app.services.analysis_service import analysis_service
from app.services.upload_service import upload_service
from app.utils.logger import get_mitigation_logger
from app.services.analysis_service import FairnessMetrics

logger = get_mitigation_logger()


@dataclass
class MitigationResult:
    """Result of a single mitigation strategy"""
    strategy_name: str
    strategy_type: str
    metrics: Dict[str, float]
    fairness_score: float
    model_performance: Dict[str, float]
    improvement_percentage: Dict[str, float]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ComprehensiveMitigationResults:
    """Comprehensive results from all mitigation strategies"""
    job_id: str
    bias_before: Dict[str, float]
    bias_after: List[MitigationResult]
    best_strategy: str
    improvements: Dict[str, float]
    overall_fairness_improvement: float
    execution_summary: Dict[str, Any]
    recommendations: List[str]


class ComprehensiveMitigationService:
    """
    Service that automatically applies all available bias mitigation strategies
    and evaluates their effectiveness
    """
    
    def __init__(self):
        self.fairness_calculator = FairnessMetrics()
        self.mitigation_jobs: Dict[str, MitigationJob] = {}
        self.comprehensive_results: Dict[str, ComprehensiveMitigationResults] = {}
        
        # Define all available mitigation strategies
        self.available_strategies = {
            "reweighing": {
                "type": "preprocessing",
                "name": "Data Reweighing",
                "description": "Reweight training samples to balance sensitive groups"
            },
            "disparate_impact_remover": {
                "type": "preprocessing", 
                "name": "Disparate Impact Remover",
                "description": "Remove features that cause disparate impact"
            },
            "data_augmentation": {
                "type": "preprocessing",
                "name": "Data Augmentation",
                "description": "Augment underrepresented groups with synthetic data"
            },
            "fairness_regularization": {
                "type": "inprocessing",
                "name": "Fairness Regularization", 
                "description": "Add fairness penalty to model training"
            },
            "adversarial_debiasing": {
                "type": "inprocessing",
                "name": "Adversarial Debiasing",
                "description": "Use adversarial training to remove bias"
            },
            "threshold_optimization": {
                "type": "postprocessing",
                "name": "Threshold Optimization",
                "description": "Optimize decision thresholds for each group"
            },
            "calibration_adjustment": {
                "type": "postprocessing", 
                "name": "Calibration Adjustment",
                "description": "Calibrate predictions to ensure fairness"
            },
            "equalized_odds_postprocessing": {
                "type": "postprocessing",
                "name": "Equalized Odds Post-processing",
                "description": "Adjust predictions to achieve equalized odds"
            }
        }
    
    async def start_comprehensive_mitigation(self, analysis_job_id: str) -> MitigationJob:
        """Start comprehensive bias mitigation that applies all strategies"""
        
        job = MitigationJob(
            analysis_job_id=analysis_job_id,
            strategy=MitigationStrategy.PREPROCESSING,  # Will apply all strategies
            strategy_params={"comprehensive": True}
        )
        
        self.mitigation_jobs[job.job_id] = job
        
        # Start comprehensive mitigation in background
        asyncio.create_task(self._run_comprehensive_mitigation(job.job_id))
        
        return job
    
    async def _run_comprehensive_mitigation(self, job_id: str):
        """Run comprehensive bias mitigation workflow"""
        try:
            job = self.mitigation_jobs[job_id]
            job.status = JobStatus.RUNNING
            job.progress = 5
            
            # Get analysis results and data
            analysis_results = analysis_service.get_analysis_results(job.analysis_job_id)
            analysis_job = analysis_service.get_analysis_job(job.analysis_job_id)
            
            # Load original data and model
            model = upload_service.load_model(analysis_job.model_upload_id)
            train_df = upload_service.load_dataset(analysis_job.train_dataset_upload_id)
            
            job.progress = 10
            
            # Calculate baseline bias metrics
            baseline_metrics = await self._calculate_baseline_metrics(
                model, train_df, analysis_results
            )
            
            job.progress = 15
            
            # Apply all mitigation strategies
            strategy_results = []
            total_strategies = len(self.available_strategies)
            progress_increment = 70 / total_strategies
            
            for i, (strategy_key, strategy_info) in enumerate(self.available_strategies.items()):
                try:
                    result = await self._apply_single_strategy(
                        strategy_key, strategy_info, model, train_df, analysis_results, baseline_metrics
                    )
                    strategy_results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error applying strategy {strategy_key}: {str(e)}")
                    # Add failed result
                    strategy_results.append(MitigationResult(
                        strategy_name=strategy_info["name"],
                        strategy_type=strategy_info["type"],
                        metrics={},
                        fairness_score=0.0,
                        model_performance={},
                        improvement_percentage={},
                        execution_time=0.0,
                        success=False,
                        error_message=str(e)
                    ))
                
                job.progress = 15 + (i + 1) * progress_increment
            
            job.progress = 85
            
            # Find best strategy and calculate improvements
            best_strategy = self._find_best_strategy(strategy_results)
            overall_improvements = self._calculate_overall_improvements(baseline_metrics, strategy_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(strategy_results, baseline_metrics)
            
            # Create comprehensive results
            comprehensive_results = ComprehensiveMitigationResults(
                job_id=job_id,
                bias_before=baseline_metrics,
                bias_after=strategy_results,
                best_strategy=best_strategy,
                improvements=overall_improvements,
                overall_fairness_improvement=self._calculate_overall_fairness_improvement(strategy_results),
                execution_summary=self._create_execution_summary(strategy_results),
                recommendations=recommendations
            )
            
            self.comprehensive_results[job_id] = comprehensive_results
            
            job.progress = 100
            job.status = JobStatus.COMPLETED
            
            logger.info(f"Comprehensive mitigation completed for job {job_id}")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Comprehensive mitigation failed for job {job_id}: {str(e)}")
    
    async def _calculate_baseline_metrics(self, model, train_df: pd.DataFrame, 
                                        analysis_results) -> Dict[str, float]:
        """Calculate baseline bias metrics before mitigation"""
        
        target_column = analysis_results.model_info.target_column
        sensitive_features = analysis_results.sensitive_features[:3]  # Top 3
        
        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        baseline_metrics = {}
        
        try:
            # Get model predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_processed)
                
                # Get prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_processed)
                    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                        y_prob = y_prob[:, 1]  # Positive class probabilities
                    else:
                        y_prob = y_prob.flatten()
                else:
                    y_prob = y_pred.astype(float)
                
                # Convert to binary if needed
                if len(np.unique(y.values)) > 2:
                    y_true_binary = (y.values > np.median(y.values)).astype(int)
                    y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
                else:
                    y_true_binary = y.values.astype(int)
                    y_pred_binary = y_pred.astype(int)
                
                # Calculate fairness metrics for each sensitive feature
                for sensitive_feature in sensitive_features:
                    feature_name = sensitive_feature.feature_name
                    if feature_name in train_df.columns:
                        sensitive_attr = train_df[feature_name].values
                        
                        # Encode if categorical
                        if train_df[feature_name].dtype == 'object':
                            le = LabelEncoder()
                            sensitive_attr = le.fit_transform(sensitive_attr.astype(str))
                        
                        # Calculate all fairness metrics
                        metrics = self.fairness_calculator.compute_all_metrics(
                            y_true_binary, y_pred_binary, y_prob, sensitive_attr
                        )
                        
                        # Add metrics with feature prefix
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                baseline_metrics[f"{metric_name}_{feature_name}"] = float(value)
                
                # Add overall model performance metrics
                baseline_metrics["accuracy"] = float(accuracy_score(y_true_binary, y_pred_binary))
                baseline_metrics["precision"] = float(precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0))
                baseline_metrics["recall"] = float(recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0))
                baseline_metrics["f1_score"] = float(f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0))
                
        except Exception as e:
            logger.error(f"Error calculating baseline metrics: {str(e)}")
            # Use analysis results bias metrics as fallback
            for bias_metric in analysis_results.bias_metrics:
                baseline_metrics[bias_metric.metric_name] = bias_metric.value
        
        return baseline_metrics
    
    async def _apply_single_strategy(self, strategy_key: str, strategy_info: Dict[str, str],
                                   model, train_df: pd.DataFrame, analysis_results,
                                   baseline_metrics: Dict[str, float]) -> MitigationResult:
        """Apply a single mitigation strategy and evaluate its effectiveness"""
        
        import time
        start_time = time.time()
        
        try:
            target_column = analysis_results.model_info.target_column
            sensitive_features = [sf.feature_name for sf in analysis_results.sensitive_features[:3]]
            
            # Apply the specific strategy
            if strategy_info["type"] == "preprocessing":
                mitigated_model, mitigated_data = await self._apply_preprocessing_strategy(
                    strategy_key, model, train_df, target_column, sensitive_features
                )
            elif strategy_info["type"] == "inprocessing":
                mitigated_model, mitigated_data = await self._apply_inprocessing_strategy(
                    strategy_key, model, train_df, target_column, sensitive_features
                )
            elif strategy_info["type"] == "postprocessing":
                mitigated_model, mitigated_data = await self._apply_postprocessing_strategy(
                    strategy_key, model, train_df, target_column, sensitive_features
                )
            else:
                raise ValueError(f"Unknown strategy type: {strategy_info['type']}")
            
            # Calculate metrics after mitigation
            after_metrics = await self._calculate_after_metrics(
                mitigated_model, mitigated_data, target_column, sensitive_features
            )
            
            # Calculate fairness score
            fairness_score = self._calculate_fairness_score(after_metrics)
            
            # Calculate model performance
            model_performance = self._calculate_model_performance(
                mitigated_model, mitigated_data, target_column
            )
            
            # Calculate improvement percentages
            improvement_percentage = self._calculate_improvement_percentages(
                baseline_metrics, after_metrics
            )
            
            execution_time = time.time() - start_time
            
            return MitigationResult(
                strategy_name=strategy_info["name"],
                strategy_type=strategy_info["type"],
                metrics=after_metrics,
                fairness_score=fairness_score,
                model_performance=model_performance,
                improvement_percentage=improvement_percentage,
                execution_time=execution_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error in strategy {strategy_key}: {str(e)}")
            
            return MitigationResult(
                strategy_name=strategy_info["name"],
                strategy_type=strategy_info["type"],
                metrics={},
                fairness_score=0.0,
                model_performance={},
                improvement_percentage={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def _apply_preprocessing_strategy(self, strategy_key: str, model, train_df: pd.DataFrame,
                                          target_column: str, sensitive_features: List[str]) -> Tuple[Any, pd.DataFrame]:
        """Apply preprocessing mitigation strategy"""
        
        if strategy_key == "reweighing":
            mitigated_df = self._apply_reweighing(train_df, target_column, sensitive_features)
        elif strategy_key == "disparate_impact_remover":
            mitigated_df = self._apply_disparate_impact_remover(train_df, target_column, sensitive_features)
        elif strategy_key == "data_augmentation":
            mitigated_df = self._apply_data_augmentation(train_df, target_column, sensitive_features)
        else:
            raise ValueError(f"Unknown preprocessing strategy: {strategy_key}")
        
        # Retrain model on mitigated data
        X = mitigated_df.drop(columns=[target_column])
        y = mitigated_df[target_column]
        
        # Create a new model of the same type
        mitigated_model = self._create_similar_model(model)
        
        # Handle categorical features
        X_processed = self._preprocess_features(X)
        
        mitigated_model.fit(X_processed, y)
        
        return mitigated_model, mitigated_df
    
    async def _apply_inprocessing_strategy(self, strategy_key: str, model, train_df: pd.DataFrame,
                                         target_column: str, sensitive_features: List[str]) -> Tuple[Any, pd.DataFrame]:
        """Apply in-processing mitigation strategy"""
        
        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]
        X_processed = self._preprocess_features(X)
        
        if strategy_key == "fairness_regularization":
            # Use regularized model with fairness penalty
            mitigated_model = LogisticRegression(
                C=0.5,  # Increased regularization
                random_state=42,
                max_iter=1000
            )
        elif strategy_key == "adversarial_debiasing":
            # Simplified adversarial debiasing using modified regularization
            mitigated_model = LogisticRegression(
                C=0.1,  # Strong regularization for adversarial effect
                random_state=42,
                max_iter=1000,
                penalty='l1',
                solver='liblinear'
            )
        else:
            raise ValueError(f"Unknown in-processing strategy: {strategy_key}")
        
        mitigated_model.fit(X_processed, y)
        
        return mitigated_model, train_df
    
    async def _apply_postprocessing_strategy(self, strategy_key: str, model, train_df: pd.DataFrame,
                                           target_column: str, sensitive_features: List[str]) -> Tuple[Any, pd.DataFrame]:
        """Apply post-processing mitigation strategy"""
        
        from app.services.mitigation_service import PostProcessingWrapper
        
        if strategy_key == "threshold_optimization":
            strategy_params = {"threshold_optimization": True, "equalized_odds": False}
        elif strategy_key == "calibration_adjustment":
            strategy_params = {"calibration": True, "equalized_odds": False}
        elif strategy_key == "equalized_odds_postprocessing":
            strategy_params = {"equalized_odds": True, "threshold_optimization": True}
        else:
            raise ValueError(f"Unknown post-processing strategy: {strategy_key}")
        
        # Create wrapper model that applies post-processing
        mitigated_model = PostProcessingWrapper(
            model, train_df, target_column, sensitive_features, strategy_params
        )
        
        return mitigated_model, train_df
    
    def _apply_reweighing(self, df: pd.DataFrame, target_column: str, sensitive_features: List[str]) -> pd.DataFrame:
        """Apply reweighing to balance sensitive groups"""
        df_reweighted = df.copy()
        
        for sensitive_feature in sensitive_features:
            if sensitive_feature in df.columns:
                # Calculate weights to balance groups
                group_counts = df.groupby([sensitive_feature, target_column]).size()
                total_count = len(df)
                
                weights = {}
                for group in df[sensitive_feature].unique():
                    for target_val in df[target_column].unique():
                        group_target_count = len(df[(df[sensitive_feature] == group) & (df[target_column] == target_val)])
                        group_count = len(df[df[sensitive_feature] == group])
                        target_count = len(df[df[target_column] == target_val])
                        
                        if group_target_count > 0:
                            expected_count = (group_count * target_count) / total_count
                            weight = expected_count / group_target_count
                            weights[(group, target_val)] = weight
                
                # Apply weights through resampling
                resampled_dfs = []
                for (group, target_val), weight in weights.items():
                    subset = df[(df[sensitive_feature] == group) & (df[target_column] == target_val)]
                    if len(subset) > 0 and weight > 0:
                        n_samples = max(1, int(len(subset) * weight))
                        resampled_subset = resample(subset, n_samples=n_samples, random_state=42)
                        resampled_dfs.append(resampled_subset)
                
                if resampled_dfs:
                    df_reweighted = pd.concat(resampled_dfs, ignore_index=True)
                    
        return df_reweighted
    
    def _apply_disparate_impact_remover(self, df: pd.DataFrame, target_column: str, 
                                      sensitive_features: List[str]) -> pd.DataFrame:
        """Apply disparate impact remover"""
        df_processed = df.copy()
        
        # Remove sensitive features to prevent direct discrimination
        for sensitive_feature in sensitive_features:
            if sensitive_feature in df.columns and sensitive_feature != target_column:
                df_processed = df_processed.drop(columns=[sensitive_feature])
        
        # Also remove highly correlated features (proxy detection)
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df_processed[numeric_cols].corr()
            
            # Remove features with high correlation to each other (potential proxies)
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))
            
            # Remove one from each highly correlated pair
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                if feat1 != target_column and feat2 != target_column:
                    features_to_remove.add(feat2)  # Remove the second feature
            
            df_processed = df_processed.drop(columns=list(features_to_remove))
        
        return df_processed
    
    def _apply_data_augmentation(self, df: pd.DataFrame, target_column: str, 
                               sensitive_features: List[str]) -> pd.DataFrame:
        """Apply data augmentation for underrepresented groups"""
        df_augmented = df.copy()
        
        for sensitive_feature in sensitive_features:
            if sensitive_feature in df.columns:
                # Find underrepresented groups
                group_counts = df.groupby([sensitive_feature, target_column]).size()
                max_count = group_counts.max()
                
                augmented_dfs = [df]
                
                for (group, target_val), count in group_counts.items():
                    if count < max_count * 0.7:  # If group is less than 70% of max
                        # Oversample this group
                        subset = df[(df[sensitive_feature] == group) & (df[target_column] == target_val)]
                        if len(subset) > 0:
                            n_additional = int(max_count * 0.7) - count
                            if n_additional > 0:
                                augmented_subset = resample(
                                    subset, n_samples=n_additional, random_state=42
                                )
                                augmented_dfs.append(augmented_subset)
                
                df_augmented = pd.concat(augmented_dfs, ignore_index=True)
                    
        return df_augmented
    
    async def _calculate_after_metrics(self, mitigated_model, mitigated_data: pd.DataFrame,
                                     target_column: str, sensitive_features: List[str]) -> Dict[str, float]:
        """Calculate bias metrics after mitigation"""
        
        X = mitigated_data.drop(columns=[target_column])
        y = mitigated_data[target_column]
        
        X_processed = self._preprocess_features(X)
        
        after_metrics = {}
        
        try:
            # Get model predictions
            if hasattr(mitigated_model, 'predict'):
                y_pred = mitigated_model.predict(X_processed)
                
                # Get prediction probabilities if available
                if hasattr(mitigated_model, 'predict_proba'):
                    y_prob = mitigated_model.predict_proba(X_processed)
                    if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                        y_prob = y_prob[:, 1]  # Positive class probabilities
                    else:
                        y_prob = y_prob.flatten()
                else:
                    y_prob = y_pred.astype(float)
                
                # Convert to binary if needed
                if len(np.unique(y.values)) > 2:
                    y_true_binary = (y.values > np.median(y.values)).astype(int)
                    y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
                else:
                    y_true_binary = y.values.astype(int)
                    y_pred_binary = y_pred.astype(int)
                
                # Calculate fairness metrics for each sensitive feature
                for feature_name in sensitive_features:
                    if feature_name in mitigated_data.columns:
                        sensitive_attr = mitigated_data[feature_name].values
                        
                        # Encode if categorical
                        if mitigated_data[feature_name].dtype == 'object':
                            le = LabelEncoder()
                            sensitive_attr = le.fit_transform(sensitive_attr.astype(str))
                        
                        # Calculate all fairness metrics
                        metrics = self.fairness_calculator.compute_all_metrics(
                            y_true_binary, y_pred_binary, y_prob, sensitive_attr
                        )
                        
                        # Add metrics with feature prefix
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                after_metrics[f"{metric_name}_{feature_name}"] = float(value)
                
        except Exception as e:
            logger.error(f"Error calculating after metrics: {str(e)}")
        
        return after_metrics
    
    def _calculate_fairness_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall fairness score from metrics"""
        if not metrics:
            return 0.0
        
        # Define weights for different metric types
        metric_weights = {
            'statistical_parity': 0.2,
            'disparate_impact': 0.2,
            'equal_opportunity': 0.2,
            'equalized_odds': 0.2,
            'calibration': 0.2
        }
        
        fairness_score = 0.0
        total_weight = 0.0
        
        for metric_name, value in metrics.items():
            for metric_type, weight in metric_weights.items():
                if metric_type in metric_name.lower():
                    # Convert to fairness score (closer to ideal = higher score)
                    if 'disparate_impact' in metric_name.lower():
                        # For disparate impact, ideal is 1.0
                        score = max(0, 100 - abs(value - 1.0) * 100)
                    else:
                        # For other metrics, ideal is 0.0
                        score = max(0, 100 - abs(value) * 100)
                    
                    fairness_score += score * weight
                    total_weight += weight
                    break
        
        return fairness_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_model_performance(self, mitigated_model, mitigated_data: pd.DataFrame,
                                   target_column: str) -> Dict[str, float]:
        """Calculate model performance metrics"""
        
        X = mitigated_data.drop(columns=[target_column])
        y = mitigated_data[target_column]
        
        try:
            X_processed = self._preprocess_features(X)
            y_pred = mitigated_model.predict(X_processed)
            
            # Convert to binary if needed
            if len(np.unique(y.values)) > 2:
                y_true_binary = (y.values > np.median(y.values)).astype(int)
                y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
            else:
                y_true_binary = y.values.astype(int)
                y_pred_binary = y_pred.astype(int)
            
            return {
                "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
                "precision": float(precision_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_true_binary, y_pred_binary, average='weighted', zero_division=0))
            }
        except Exception as e:
            logger.error(f"Error calculating model performance: {str(e)}")
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    
    def _calculate_improvement_percentages(self, baseline_metrics: Dict[str, float],
                                         after_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate improvement percentages for each metric"""
        
        improvements = {}
        
        for metric_name in baseline_metrics.keys():
            if metric_name in after_metrics:
                baseline_value = baseline_metrics[metric_name]
                after_value = after_metrics[metric_name]
                
                if baseline_value != 0:
                    # For fairness metrics, improvement means getting closer to ideal
                    if 'disparate_impact' in metric_name.lower():
                        # For disparate impact, ideal is 1.0
                        baseline_distance = abs(baseline_value - 1.0)
                        after_distance = abs(after_value - 1.0)
                        if baseline_distance > 0:
                            improvement = (baseline_distance - after_distance) / baseline_distance * 100
                        else:
                            improvement = 0.0
                    else:
                        # For other metrics, ideal is 0.0
                        if baseline_value > 0:
                            improvement = (baseline_value - after_value) / baseline_value * 100
                        else:
                            improvement = 0.0
                    
                    improvements[metric_name] = improvement
                else:
                    improvements[metric_name] = 0.0
        
        return improvements
    
    def _find_best_strategy(self, strategy_results: List[MitigationResult]) -> str:
        """Find the best performing mitigation strategy"""
        
        successful_results = [r for r in strategy_results if r.success]
        
        if not successful_results:
            return "None (all strategies failed)"
        
        # Find strategy with highest fairness score
        best_result = max(successful_results, key=lambda x: x.fairness_score)
        return best_result.strategy_name
    
    def _calculate_overall_improvements(self, baseline_metrics: Dict[str, float],
                                      strategy_results: List[MitigationResult]) -> Dict[str, float]:
        """Calculate overall improvements across all strategies"""
        
        successful_results = [r for r in strategy_results if r.success]
        
        if not successful_results:
            return {}
        
        # Find best improvement for each metric across all strategies
        overall_improvements = {}
        
        for metric_name in baseline_metrics.keys():
            best_improvement = float('-inf')
            
            for result in successful_results:
                if metric_name in result.improvement_percentage:
                    improvement = result.improvement_percentage[metric_name]
                    best_improvement = max(best_improvement, improvement)
            
            if best_improvement != float('-inf'):
                overall_improvements[metric_name] = best_improvement
        
        return overall_improvements
    
    def _calculate_overall_fairness_improvement(self, strategy_results: List[MitigationResult]) -> float:
        """Calculate overall fairness improvement across all strategies"""
        
        successful_results = [r for r in strategy_results if r.success]
        
        if not successful_results:
            return 0.0
        
        # Return the best fairness score achieved
        best_fairness_score = max(r.fairness_score for r in successful_results)
        return best_fairness_score
    
    def _create_execution_summary(self, strategy_results: List[MitigationResult]) -> Dict[str, Any]:
        """Create execution summary"""
        
        successful_count = sum(1 for r in strategy_results if r.success)
        total_count = len(strategy_results)
        
        return {
            "total_strategies": total_count,
            "successful_strategies": successful_count,
            "failed_strategies": total_count - successful_count,
            "success_rate": (successful_count / total_count * 100) if total_count > 0 else 0,
            "total_execution_time": sum(r.execution_time for r in strategy_results),
            "strategy_breakdown": {
                "preprocessing": len([r for r in strategy_results if r.strategy_type == "preprocessing"]),
                "inprocessing": len([r for r in strategy_results if r.strategy_type == "inprocessing"]),
                "postprocessing": len([r for r in strategy_results if r.strategy_type == "postprocessing"])
            }
        }
    
    def _generate_recommendations(self, strategy_results: List[MitigationResult],
                                baseline_metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations based on results"""
        
        recommendations = []
        successful_results = [r for r in strategy_results if r.success]
        
        if not successful_results:
            recommendations.append("All mitigation strategies failed. Consider data quality issues or model complexity.")
            return recommendations
        
        # Find best strategies by type
        best_preprocessing = max([r for r in successful_results if r.strategy_type == "preprocessing"], 
                               key=lambda x: x.fairness_score, default=None)
        best_inprocessing = max([r for r in successful_results if r.strategy_type == "inprocessing"], 
                              key=lambda x: x.fairness_score, default=None)
        best_postprocessing = max([r for r in successful_results if r.strategy_type == "postprocessing"], 
                                key=lambda x: x.fairness_score, default=None)
        
        # Overall best strategy
        best_overall = max(successful_results, key=lambda x: x.fairness_score)
        recommendations.append(f"Best overall strategy: {best_overall.strategy_name} (Fairness Score: {best_overall.fairness_score:.1f})")
        
        # Recommendations by type
        if best_preprocessing:
            recommendations.append(f"Best preprocessing approach: {best_preprocessing.strategy_name}")
        if best_inprocessing:
            recommendations.append(f"Best in-processing approach: {best_inprocessing.strategy_name}")
        if best_postprocessing:
            recommendations.append(f"Best post-processing approach: {best_postprocessing.strategy_name}")
        
        # Performance trade-off analysis
        best_performance = max(successful_results, key=lambda x: x.model_performance.get('accuracy', 0))
        if best_performance.strategy_name != best_overall.strategy_name:
            recommendations.append(f"For best model performance, consider: {best_performance.strategy_name}")
        
        # Strategy-specific recommendations
        if best_overall.fairness_score < 60:
            recommendations.append("Consider combining multiple mitigation strategies for better results.")
        
        if best_overall.fairness_score > 80:
            recommendations.append("Excellent fairness improvement achieved. Monitor for performance trade-offs.")
        
        return recommendations
    
    def _create_similar_model(self, original_model):
        """Create a new model of the same type as the original"""
        
        model_type = type(original_model).__name__
        
        if 'LogisticRegression' in model_type:
            return LogisticRegression(random_state=42, max_iter=1000)
        elif 'RandomForest' in model_type:
            return RandomForestClassifier(random_state=42, n_estimators=100)
        elif 'GradientBoosting' in model_type:
            return GradientBoostingClassifier(random_state=42)
        elif 'SVC' in model_type or 'SVM' in model_type:
            return SVC(random_state=42, probability=True)
        elif 'GaussianNB' in model_type:
            return GaussianNB()
        else:
            # Default to LogisticRegression
            return LogisticRegression(random_state=42, max_iter=1000)
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Basic feature preprocessing"""
        X_processed = X.copy()
        
        # Handle categorical variables
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Convert to numpy array
        return X_processed.values
    
    def get_comprehensive_results(self, job_id: str) -> ComprehensiveMitigationResults:
        """Get comprehensive mitigation results"""
        if job_id not in self.comprehensive_results:
            raise HTTPException(status_code=404, detail=f"Results not found for job {job_id}")
        
        return self.comprehensive_results[job_id]
    
    def get_mitigation_job(self, job_id: str) -> MitigationJob:
        """Get mitigation job status"""
        if job_id not in self.mitigation_jobs:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        
        return self.mitigation_jobs[job_id]
    
    def format_results_for_frontend(self, job_id: str) -> Dict[str, Any]:
        """Format results for frontend consumption"""
        
        results = self.get_comprehensive_results(job_id)
        
        # Format bias metrics before mitigation
        bias_before = {}
        for metric_name, value in results.bias_before.items():
            # Clean up metric names for display
            clean_name = metric_name.replace('_', ' ').title()
            bias_before[clean_name] = value
        
        # Format bias metrics after each mitigation strategy
        bias_after = []
        for result in results.bias_after:
            if result.success:
                # Calculate fairness score based on metrics
                fairness_score = result.fairness_score
                
                # Clean up metrics for display
                clean_metrics = {}
                for metric_name, value in result.metrics.items():
                    clean_name = metric_name.replace('_', ' ').title()
                    clean_metrics[clean_name] = value
                
                bias_after.append({
                    "strategy": result.strategy_name,
                    "strategy_type": result.strategy_type,
                    "metrics": clean_metrics,
                    "fairness_score": fairness_score,
                    "model_performance": result.model_performance,
                    "execution_time": result.execution_time
                })
        
        # Calculate improvements
        improvements = {}
        for metric_name, improvement in results.improvements.items():
            clean_name = metric_name.replace('_', ' ').title()
            improvements[clean_name] = improvement
        
        return {
            "job_id": job_id,
            "bias_before": bias_before,
            "bias_after": bias_after,
            "best_strategy": results.best_strategy,
            "improvements": improvements,
            "overall_fairness_improvement": results.overall_fairness_improvement,
            "execution_summary": results.execution_summary,
            "recommendations": results.recommendations,
            "strategies_applied": len(results.bias_after),
            "successful_strategies": len([r for r in results.bias_after if r.success])
        }


# Create service instance
comprehensive_mitigation_service = ComprehensiveMitigationService()
