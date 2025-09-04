import pickle
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.base import BaseEstimator
import os

from app.models.schemas import ModelType, TaskType


def validate_model_file(file_path: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate model file and extract metadata"""
    try:
        # Load the model
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
        elif file_path.endswith('.joblib'):
            model = joblib.load(file_path)
        elif file_path.endswith('.onnx'):
            # For ONNX models, just check if file can be read
            with open(file_path, 'rb') as f:
                f.read()
            return True, {
                "model_type": ModelType.UNKNOWN,
                "framework": "onnx",
                "is_sklearn_compatible": False
            }
        else:
            return False, {"error": "Unsupported file format"}
        
        # Detect model info
        model_info = detect_model_info(model)
        
        return True, model_info
        
    except Exception as e:
        return False, {"error": f"Failed to load model: {str(e)}"}


def validate_dataset_file(file_path: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate dataset file and extract metadata"""
    try:
        # Load the dataset
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            return False, {"error": "Unsupported file format"}
        
        # Basic validation
        if df.empty:
            return False, {"error": "Dataset is empty"}
        
        if df.shape[0] < 10:
            return False, {"error": "Dataset too small (minimum 10 rows required)"}
        
        # Extract metadata
        metadata = {
            "n_rows": df.shape[0],
            "n_columns": df.shape[1],
            "column_names": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2)
        }
        
        return True, metadata
        
    except Exception as e:
        return False, {"error": f"Failed to load dataset: {str(e)}"}


def detect_model_info(model) -> Dict[str, Any]:
    """Detect model type and extract information"""
    model_info = {
        "model_type": ModelType.UNKNOWN,
        "task_type": None,
        "framework": "unknown",
        "is_sklearn_compatible": False,
        "n_features": None,
        "n_classes": None,
        "feature_names": None
    }
    
    # Check if it's a scikit-learn model
    if isinstance(model, BaseEstimator):
        model_info["model_type"] = ModelType.SKLEARN
        model_info["framework"] = "scikit-learn"
        model_info["is_sklearn_compatible"] = True
        
        # Get number of features
        if hasattr(model, 'n_features_in_'):
            model_info["n_features"] = model.n_features_in_
        
        # Get feature names
        if hasattr(model, 'feature_names_in_'):
            model_info["feature_names"] = model.feature_names_in_.tolist()
        
        # Determine task type
        if hasattr(model, 'classes_'):
            model_info["task_type"] = TaskType.CLASSIFICATION
            model_info["n_classes"] = len(model.classes_)
        elif hasattr(model, 'predict') and not hasattr(model, 'predict_proba'):
            model_info["task_type"] = TaskType.REGRESSION
    
    # Check for XGBoost
    elif str(type(model)).find('xgboost') != -1:
        model_info["model_type"] = ModelType.XGBOOST
        model_info["framework"] = "xgboost"
        model_info["is_sklearn_compatible"] = True
        
        # Try to get feature info from XGBoost
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            feature_names = booster.feature_names
            if feature_names:
                model_info["feature_names"] = feature_names
                model_info["n_features"] = len(feature_names)
        
        # Determine objective
        if hasattr(model, 'objective'):
            if 'reg:' in str(model.objective):
                model_info["task_type"] = TaskType.REGRESSION
            elif 'binary:' in str(model.objective) or 'multi:' in str(model.objective):
                model_info["task_type"] = TaskType.CLASSIFICATION
    
    # Check for LightGBM
    elif str(type(model)).find('lightgbm') != -1:
        model_info["model_type"] = ModelType.LIGHTGBM
        model_info["framework"] = "lightgbm"
        model_info["is_sklearn_compatible"] = True
        
        # Get feature info from LightGBM
        if hasattr(model, 'feature_name_'):
            model_info["feature_names"] = model.feature_name_
            model_info["n_features"] = len(model.feature_name_)
        
        # Determine objective
        if hasattr(model, 'params') and 'objective' in model.params:
            objective = model.params['objective']
            if 'regression' in objective:
                model_info["task_type"] = TaskType.REGRESSION
            elif 'binary' in objective or 'multiclass' in objective:
                model_info["task_type"] = TaskType.CLASSIFICATION
    
    return model_info


def detect_sensitive_columns(df: pd.DataFrame) -> list:
    """Detect potentially sensitive columns based on naming patterns"""
    sensitive_patterns = [
        'gender', 'sex', 'race', 'ethnicity', 'age', 'religion', 'nationality',
        'marital', 'disability', 'sexual_orientation', 'political', 'income',
        'education', 'occupation', 'zip', 'postal', 'address', 'location'
    ]
    
    sensitive_columns = []
    
    for col in df.columns:
        col_lower = col.lower()
        for pattern in sensitive_patterns:
            if pattern in col_lower:
                sensitive_columns.append(col)
                break
    
    return sensitive_columns


def preprocess_dataset(df: pd.DataFrame, target_column: str = None) -> Tuple[pd.DataFrame, str]:
    """Basic preprocessing of dataset"""
    # Make a copy
    df_processed = df.copy()
    
    # Auto-detect target column if not provided
    if target_column is None:
        potential_targets = ['target', 'label', 'y', 'outcome', 'class', 'prediction']
        for col in df_processed.columns:
            if col.lower() in potential_targets:
                target_column = col
                break
        
        # If still no target found, use the last column
        if target_column is None:
            target_column = df_processed.columns[-1]
    
    # Handle missing values (simple strategy)
    for col in df_processed.columns:
        if df_processed[col].dtype in ['object', 'string']:
            df_processed[col].fillna('Unknown', inplace=True)
        else:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    return df_processed, target_column
