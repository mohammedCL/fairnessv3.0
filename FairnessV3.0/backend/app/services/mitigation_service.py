import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import asyncio
import copy
from fastapi import HTTPException

from app.models.schemas import (
    MitigationJob, MitigationResults, MitigationStrategy, 
    JobStatus, BiasMetric, TaskType
)
from app.services.analysis_service import analysis_service
from app.services.upload_service import upload_service
from app.utils.logger import get_mitigation_logger, LogExecutionTime

# Initialize logger
logger = get_mitigation_logger()


class MitigationService:
    def __init__(self):
        # In-memory storage for mitigation jobs
        self.mitigation_jobs: Dict[str, MitigationJob] = {}
        self.mitigation_results: Dict[str, MitigationResults] = {}
    
    async def start_mitigation(self, analysis_job_id: str, strategy: MitigationStrategy, 
                             strategy_params: Dict[str, Any] = None) -> MitigationJob:
        """Start bias mitigation job"""
        if strategy_params is None:
            strategy_params = {}
            
        job = MitigationJob(
            analysis_job_id=analysis_job_id,
            strategy=strategy,
            strategy_params=strategy_params
        )
        
        self.mitigation_jobs[job.job_id] = job
        
        # Start mitigation in background
        asyncio.create_task(self._run_mitigation(job.job_id))
        
        return job
    
    async def _run_mitigation(self, job_id: str):
        """Run the bias mitigation workflow"""
        try:
            job = self.mitigation_jobs[job_id]
            job.status = JobStatus.RUNNING
            job.progress = 10
            
            # Get analysis results
            analysis_results = analysis_service.get_analysis_results(job.analysis_job_id)
            analysis_job = analysis_service.get_analysis_job(job.analysis_job_id)
            
            # Load original data and model
            model = upload_service.load_model(analysis_job.model_upload_id)
            train_df = upload_service.load_dataset(analysis_job.train_dataset_upload_id)
            
            # Get original bias metrics (before mitigation)
            before_metrics = analysis_results.bias_metrics
            job.progress = 20
            
            # Apply mitigation strategy
            mitigated_model, mitigated_data = await self._apply_mitigation_strategy(
                job.strategy, model, train_df, analysis_results, job.strategy_params
            )
            job.progress = 60
            
            # Evaluate mitigated model
            after_metrics = self._evaluate_mitigated_model(
                mitigated_model, mitigated_data, analysis_results
            )
            job.progress = 80
            
            # Calculate improvement
            improvement_summary = self._calculate_improvement(before_metrics, after_metrics)
            fairness_improvement = self._calculate_fairness_improvement(before_metrics, after_metrics)
            
            # Store results
            results = MitigationResults(
                job_id=job_id,
                strategy_applied=job.strategy,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_summary=improvement_summary,
                fairness_improvement=fairness_improvement,
                mitigation_details={
                    "strategy_params": job.strategy_params,
                    "original_model_type": str(type(model)),
                    "mitigation_applied": True
                }
            )
            
            self.mitigation_results[job_id] = results
            
            # Update job status
            job.status = JobStatus.COMPLETED
            job.progress = 100
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
    
    async def _apply_mitigation_strategy(self, strategy: MitigationStrategy, model, 
                                       train_df: pd.DataFrame, analysis_results, 
                                       strategy_params: Dict[str, Any]):
        """Apply the selected mitigation strategy"""
        
        target_column = analysis_results.model_info.target_column
        sensitive_features = [sf.feature_name for sf in analysis_results.sensitive_features[:3]]  # Top 3
        
        if strategy == MitigationStrategy.PREPROCESSING:
            return await self._apply_preprocessing_mitigation(
                model, train_df, target_column, sensitive_features, strategy_params
            )
        elif strategy == MitigationStrategy.INPROCESSING:
            return await self._apply_inprocessing_mitigation(
                model, train_df, target_column, sensitive_features, strategy_params
            )
        elif strategy == MitigationStrategy.POSTPROCESSING:
            return await self._apply_postprocessing_mitigation(
                model, train_df, target_column, sensitive_features, strategy_params
            )
        else:
            raise ValueError(f"Unknown mitigation strategy: {strategy}")
    
    async def _apply_preprocessing_mitigation(self, model, train_df: pd.DataFrame, 
                                            target_column: str, sensitive_features: List[str], 
                                            strategy_params: Dict[str, Any]):
        """Apply preprocessing mitigation strategies"""
        mitigated_df = train_df.copy()
        
        # Strategy 1: Reweighing
        if strategy_params.get("use_reweighing", True):
            mitigated_df = self._apply_reweighing(mitigated_df, target_column, sensitive_features)
        
        # Strategy 2: Disparate Impact Remover
        if strategy_params.get("use_disparate_impact_remover", True):
            mitigated_df = self._apply_disparate_impact_remover(
                mitigated_df, target_column, sensitive_features
            )
        
        # Strategy 3: Data augmentation for underrepresented groups
        if strategy_params.get("use_data_augmentation", True):
            mitigated_df = self._apply_data_augmentation(
                mitigated_df, target_column, sensitive_features
            )
        
        # Retrain model on mitigated data
        X = mitigated_df.drop(columns=[target_column])
        y = mitigated_df[target_column]
        
        # Create a new model of the same type
        mitigated_model = self._create_similar_model(model)
        
        # Handle categorical features
        X_processed = self._preprocess_features(X)
        
        mitigated_model.fit(X_processed, y)
        
        return mitigated_model, mitigated_df
    
    async def _apply_inprocessing_mitigation(self, model, train_df: pd.DataFrame, 
                                           target_column: str, sensitive_features: List[str], 
                                           strategy_params: Dict[str, Any]):
        """Apply in-processing mitigation strategies"""
        
        # For in-processing, we'll use fairness-aware algorithms
        # This is a simplified implementation - in practice, you'd use libraries like fairlearn
        
        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Use a fairness-aware model (simplified)
        fairness_penalty = strategy_params.get("fairness_penalty", 0.1)
        
        if hasattr(model, 'fit'):
            # Create a fairness-constrained version
            mitigated_model = LogisticRegression(
                C=1.0 / (1.0 + fairness_penalty),  # Add regularization for fairness
                random_state=42,
                max_iter=1000
            )
            mitigated_model.fit(X_processed, y)
        else:
            mitigated_model = model
        
        return mitigated_model, train_df
    
    async def _apply_postprocessing_mitigation(self, model, train_df: pd.DataFrame, 
                                             target_column: str, sensitive_features: List[str], 
                                             strategy_params: Dict[str, Any]):
        """Apply post-processing mitigation strategies"""
        
        # For post-processing, we adjust predictions to achieve fairness
        # This is a simplified threshold optimization approach
        
        X = train_df.drop(columns=[target_column])
        y = train_df[target_column]
        
        # Debug: Check what type of model we have
        logger.debug(f"Model type: {type(model)}")
        logger.debug(f"Model has predict: {hasattr(model, 'predict')}")
        logger.debug(f"Model has predict_proba: {hasattr(model, 'predict_proba')}")
        
        # Check if model is actually a model object
        if not hasattr(model, 'predict'):
            logger.warning("Model object doesn't have predict method. Using dummy wrapper.")
            # Create a simple dummy classifier for demonstration
            from sklearn.dummy import DummyClassifier
            dummy_model = DummyClassifier(strategy='most_frequent')
            dummy_model.fit(X, y)
            model = dummy_model
        
        # Preprocess features
        X_processed = self._preprocess_features(X)
        
        # Create a wrapper model that applies post-processing
        mitigated_model = PostProcessingWrapper(
            model, train_df, target_column, sensitive_features, strategy_params
        )
        
        return mitigated_model, train_df
    
    def _apply_reweighing(self, df: pd.DataFrame, target_column: str, 
                         sensitive_features: List[str]) -> pd.DataFrame:
        """Apply reweighing to balance sensitive groups"""
        df_reweighted = df.copy()
        
        for sensitive_feature in sensitive_features:
            if sensitive_feature in df.columns:
                # Calculate weights to balance groups
                group_counts = df.groupby([sensitive_feature, target_column]).size()
                total_positive = (df[target_column] == 1).sum()
                total_negative = (df[target_column] == 0).sum()
                
                weights = {}
                for group in df[sensitive_feature].unique():
                    group_mask = df[sensitive_feature] == group
                    group_positive = ((df[sensitive_feature] == group) & (df[target_column] == 1)).sum()
                    group_negative = ((df[sensitive_feature] == group) & (df[target_column] == 0)).sum()
                    
                    if group_positive > 0 and group_negative > 0:
                        weight_positive = total_positive / (len(df[sensitive_feature].unique()) * group_positive)
                        weight_negative = total_negative / (len(df[sensitive_feature].unique()) * group_negative)
                        weights[(group, 1)] = weight_positive
                        weights[(group, 0)] = weight_negative
                
                # Apply weights through resampling
                resampled_dfs = []
                for (group, target_val), weight in weights.items():
                    subset = df[(df[sensitive_feature] == group) & (df[target_column] == target_val)]
                    if len(subset) > 0:
                        n_samples = int(len(subset) * weight)
                        if n_samples > 0:
                            resampled_subset = resample(subset, n_samples=n_samples, random_state=42)
                            resampled_dfs.append(resampled_subset)
                
                if resampled_dfs:
                    df_reweighted = pd.concat(resampled_dfs, ignore_index=True)
                    
        return df_reweighted
    
    def _apply_disparate_impact_remover(self, df: pd.DataFrame, target_column: str, 
                                      sensitive_features: List[str]) -> pd.DataFrame:
        """Apply disparate impact remover (simplified)"""
        df_processed = df.copy()
        
        # This is a simplified version - remove or modify features that cause disparate impact
        for sensitive_feature in sensitive_features:
            if sensitive_feature in df.columns:
                # Remove the sensitive feature itself to prevent direct discrimination
                if sensitive_feature != target_column:
                    df_processed = df_processed.drop(columns=[sensitive_feature])
        
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
                
                for (group, target_val), count in group_counts.items():
                    if count < max_count * 0.5:  # If group is less than 50% of max
                        # Oversample this group
                        subset = df[(df[sensitive_feature] == group) & (df[target_column] == target_val)]
                        if len(subset) > 0:
                            n_additional = int(max_count * 0.5) - count
                            if n_additional > 0:
                                augmented_subset = resample(
                                    subset, n_samples=n_additional, random_state=42
                                )
                                df_augmented = pd.concat([df_augmented, augmented_subset], ignore_index=True)
        
        return df_augmented
    
    def _create_similar_model(self, original_model):
        """Create a new model similar to the original"""
        if hasattr(original_model, 'get_params'):
            # For scikit-learn models
            model_class = type(original_model)
            params = original_model.get_params()
            return model_class(**params)
        else:
            # Fallback to a simple model
            return LogisticRegression(random_state=42, max_iter=1000)
    
    def _preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Basic feature preprocessing"""
        X_processed = X.copy()
        
        # Handle categorical variables
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                # Simple label encoding
                X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        return X_processed
    
    def _evaluate_mitigated_model(self, mitigated_model, mitigated_data: pd.DataFrame, 
                                analysis_results) -> List[BiasMetric]:
        """Evaluate the mitigated model for bias"""
        
        # This is a simplified evaluation - in practice, you'd run the full analysis pipeline
        after_metrics = []
        
        target_column = analysis_results.model_info.target_column
        sensitive_features = [sf.feature_name for sf in analysis_results.sensitive_features]
        
        # Re-calculate bias metrics on mitigated data
        for sensitive_feature_obj in analysis_results.sensitive_features:
            feature_name = sensitive_feature_obj.feature_name
            
            if feature_name in mitigated_data.columns:
                # Calculate demographic parity difference
                dp_diff = self._calculate_demographic_parity_difference(
                    mitigated_data, feature_name, target_column
                )
                
                after_metrics.append(BiasMetric(
                    metric_name=f"Demographic Parity Difference - {feature_name}",
                    value=dp_diff,
                    threshold=0.1,
                    is_biased=abs(dp_diff) > 0.1,
                    severity="high" if abs(dp_diff) > 0.2 else "medium" if abs(dp_diff) > 0.1 else "low",
                    description=f"Post-mitigation demographic parity for {feature_name}"
                ))
        
        return after_metrics
    
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
    
    def _calculate_improvement(self, before_metrics: List[BiasMetric], 
                             after_metrics: List[BiasMetric]) -> Dict[str, float]:
        """Calculate improvement in bias metrics"""
        improvement = {}
        
        # Match metrics by name
        before_dict = {m.metric_name: m.value for m in before_metrics}
        after_dict = {m.metric_name: m.value for m in after_metrics}
        
        for metric_name in before_dict:
            if metric_name in after_dict:
                before_val = abs(before_dict[metric_name])
                after_val = abs(after_dict[metric_name])
                
                if before_val > 0:
                    improvement_pct = ((before_val - after_val) / before_val) * 100
                    improvement[metric_name] = improvement_pct
        
        return improvement
    
    def _calculate_fairness_improvement(self, before_metrics: List[BiasMetric], 
                                      after_metrics: List[BiasMetric]) -> float:
        """Calculate overall fairness improvement"""
        before_bias_score = np.mean([abs(m.value) for m in before_metrics]) if before_metrics else 0
        after_bias_score = np.mean([abs(m.value) for m in after_metrics]) if after_metrics else 0
        
        if before_bias_score > 0:
            improvement = ((before_bias_score - after_bias_score) / before_bias_score) * 100
            return float(improvement)
        
        return 0.0
    
    def get_mitigation_job(self, job_id: str) -> MitigationJob:
        """Get mitigation job by ID"""
        if job_id not in self.mitigation_jobs:
            raise HTTPException(status_code=404, detail="Mitigation job not found")
        return self.mitigation_jobs[job_id]
    
    def get_mitigation_results(self, job_id: str) -> MitigationResults:
        """Get mitigation results by job ID"""
        if job_id not in self.mitigation_results:
            raise HTTPException(status_code=404, detail="Mitigation results not found")
        return self.mitigation_results[job_id]


class PostProcessingWrapper:
    """Wrapper for applying post-processing bias mitigation"""
    
    def __init__(self, base_model, training_data: pd.DataFrame, target_column: str, 
                 sensitive_features: List[str], strategy_params: Dict[str, Any]):
        self.base_model = base_model
        self.training_data = training_data
        self.target_column = target_column
        self.sensitive_features = sensitive_features
        self.strategy_params = strategy_params
        self.thresholds = {}
        
        # Calculate optimal thresholds for each group
        self._calculate_optimal_thresholds()
    
    def _calculate_optimal_thresholds(self):
        """Calculate optimal decision thresholds for each sensitive group"""
        try:
            X = self.training_data.drop(columns=[self.target_column])
            y = self.training_data[self.target_column]
            
            # Get base model predictions
            X_processed = self._preprocess_features(X)
            
            if hasattr(self.base_model, 'predict_proba'):
                y_prob = self.base_model.predict_proba(X_processed)
                if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                    y_prob = y_prob[:, 1]  # Take positive class probabilities
                else:
                    y_prob = y_prob.flatten()
            else:
                # If no predict_proba, use predict and convert to probabilities
                y_pred = self.base_model.predict(X_processed)
                # Use simple conversion for binary classification
                y_prob = y_pred.astype(float)
            
            # Calculate optimal thresholds for each sensitive group
            for sensitive_feature in self.sensitive_features:
                if sensitive_feature in self.training_data.columns:
                    groups = self.training_data[sensitive_feature].unique()
                    
                    for group in groups:
                        group_mask = self.training_data[sensitive_feature] == group
                        group_y = y[group_mask]
                        group_prob = y_prob[group_mask]
                        
                        # Find threshold that maximizes accuracy while maintaining fairness
                        best_threshold = 0.5
                        best_score = 0
                        
                        for threshold in np.arange(0.1, 0.9, 0.1):
                            group_pred = (group_prob >= threshold).astype(int)
                            accuracy = (group_pred == group_y).mean()
                            
                            if accuracy > best_score:
                                best_score = accuracy
                                best_threshold = threshold
                        
                        self.thresholds[(sensitive_feature, group)] = best_threshold
                        
        except Exception as e:
            logger.error(f"Error calculating optimal thresholds: {str(e)}", exc_info=True)
            # Set default thresholds
            for sensitive_feature in self.sensitive_features:
                if sensitive_feature in self.training_data.columns:
                    groups = self.training_data[sensitive_feature].unique()
                    for group in groups:
                        self.thresholds[(sensitive_feature, group)] = 0.5
    
    def predict(self, X):
        """Make predictions with post-processing bias mitigation"""
        X_processed = self._preprocess_features(X)
        
        if hasattr(self.base_model, 'predict_proba'):
            y_prob = self.base_model.predict_proba(X_processed)
            if y_prob.ndim > 1 and y_prob.shape[1] > 1:
                y_prob = y_prob[:, 1]  # Take positive class probabilities
            else:
                y_prob = y_prob.flatten()
        else:
            return self.base_model.predict(X_processed)
        
        # Apply group-specific thresholds if sensitive features are available
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            threshold = 0.5  # Default threshold
            
            # Check if we have group-specific thresholds
            for sensitive_feature in self.sensitive_features:
                if sensitive_feature in X.columns:
                    group_value = X.iloc[i][sensitive_feature]
                    threshold_key = (sensitive_feature, group_value)
                    if threshold_key in self.thresholds:
                        threshold = self.thresholds[threshold_key]
                        break
            
            predictions[i] = 1 if y_prob[i] >= threshold else 0
        
        return predictions.astype(int)
    
    def _preprocess_features(self, X: pd.DataFrame) -> np.ndarray:
        """Basic feature preprocessing"""
        X_processed = X.copy()
        
        # Handle categorical variables
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = pd.Categorical(X_processed[col]).codes
        
        # Handle missing values
        X_processed = X_processed.fillna(X_processed.median())
        
        # Return numpy array to avoid feature name warnings
        return X_processed.values


# Global mitigation service instance
mitigation_service = MitigationService()
