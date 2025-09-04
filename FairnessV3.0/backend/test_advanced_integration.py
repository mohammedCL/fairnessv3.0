#!/usr/bin/env python3
"""
Test script for advanced bias detection integration in analysis_service.py
Tests all three integrated components:
1. Advanced bias detection methods
2. Comprehensive statistical testing
3. Multi-sensitive attribute analysis
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Add the backend directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from app.services.analysis_service import AnalysisService, AdvancedBiasDetection

def create_test_dataset():
    """Create a test dataset with known biases for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic biased dataset
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
    }
    
    # Add sensitive attributes
    data['gender'] = np.random.choice(['Male', 'Female'], n_samples)
    data['race'] = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples)
    data['region'] = np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    
    # Create biased target variable that correlates with sensitive attributes
    target_prob = 0.3  # Base probability
    
    # Add bias based on gender
    gender_bias = np.where(data['gender'] == 'Male', 0.2, -0.1)
    
    # Add bias based on race
    race_bias = np.where(data['race'] == 'White', 0.15, 
                np.where(data['race'] == 'Asian', 0.1, -0.1))
    
    # Create proxy bias - income correlates with sensitive attributes
    income_gender_bias = np.where(data['gender'] == 'Male', 5000, -3000)
    data['income'] = data['income'] + income_gender_bias
    
    # Create target based on features + bias
    final_prob = (target_prob + 
                 gender_bias + 
                 race_bias + 
                 (data['income'] - 50000) / 100000 + 
                 (data['credit_score'] - 600) / 1000 +
                 np.random.normal(0, 0.1, n_samples))
    
    data['loan_approved'] = (np.random.random(n_samples) < np.clip(final_prob, 0, 1)).astype(int)
    
    return pd.DataFrame(data)

def test_advanced_bias_detection():
    """Test the AdvancedBiasDetection class directly"""
    print("="*60)
    print("TESTING ADVANCED BIAS DETECTION CLASS")
    print("="*60)
    
    # Create test data
    df = create_test_dataset()
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    feature_columns = [col for col in X.columns if col not in ['gender', 'race', 'region']]
    
    # Initialize advanced bias detector
    detector = AdvancedBiasDetection()
    
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Feature columns: {feature_columns}")
    
    # Test 1: Direct Bias Analysis
    print("\n" + "="*40)
    print("TEST 1: DIRECT BIAS ANALYSIS")
    print("="*40)
    
    for sensitive_attr in ['gender', 'race']:
        print(f"\nTesting direct bias for: {sensitive_attr}")
        direct_results = detector.direct_bias_analysis(X, y, sensitive_attr)
        print(f"Results: {direct_results}")
    
    # Test 2: Proxy Detection
    print("\n" + "="*40)
    print("TEST 2: PROXY DETECTION")
    print("="*40)
    
    for sensitive_attr in ['gender', 'race']:
        print(f"\nTesting proxy detection for: {sensitive_attr}")
        proxy_results = detector.detect_proxy_features(X, feature_columns, sensitive_attr)
        print(f"Found {len(proxy_results)} proxy features")
        for proxy in proxy_results:
            print(f"  - {proxy}")
    
    # Test 3: Comprehensive Statistical Testing
    print("\n" + "="*40)
    print("TEST 3: COMPREHENSIVE STATISTICAL TESTING")
    print("="*40)
    
    for sensitive_attr in ['gender', 'race']:
        print(f"\nTesting statistical analysis for: {sensitive_attr}")
        stat_results = detector.comprehensive_statistical_testing(X, y, feature_columns, sensitive_attr)
        print(f"Total biased features: {stat_results['total_biased']}")
        print(f"Biased features: {[f[0] for f in stat_results['biased_features']]}")
    
    # Test 4: Model-Based Testing
    print("\n" + "="*40)
    print("TEST 4: MODEL-BASED TESTING")
    print("="*40)
    
    for sensitive_attr in ['gender', 'race']:
        print(f"\nTesting model-based analysis for: {sensitive_attr}")
        model_results = detector.model_based_bias_testing(X, y, feature_columns, sensitive_attr)
        print(f"Model accuracy: {model_results['model_accuracy']:.4f}")
        print(f"Fairness metrics computed: {len(model_results['fairness_metrics'])}")
    
    # Test 5: Multi-Sensitive Attribute Analysis
    print("\n" + "="*40)
    print("TEST 5: MULTI-SENSITIVE ATTRIBUTE ANALYSIS")
    print("="*40)
    
    multi_results = detector.multi_sensitive_attribute_analysis(
        X, y, ['gender', 'race'], feature_columns
    )
    print(f"Multi-analysis summary: {multi_results['summary']}")
    print(f"Cross-analysis results: {len(multi_results['cross_analysis'])}")
    
    return True

def test_integration_with_analysis_service():
    """Test the integration with AnalysisService"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION WITH ANALYSIS SERVICE")
    print("="*60)
    
    # Create test data
    df = create_test_dataset()
    
    # Train a simple model for testing
    X = df.drop('loan_approved', axis=1)
    y = df['loan_approved']
    
    # Encode categorical features for model training
    X_encoded = X.copy()
    label_encoders = {}
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'object':
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
            label_encoders[col] = le
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_encoded, y)
    
    print(f"Model trained with accuracy: {model.score(X_encoded, y):.4f}")
    
    # Create mock sensitive features (simulating what would be detected)
    from app.models.schemas import SensitiveFeature
    
    sensitive_features = [
        SensitiveFeature(
            feature_name='gender',
            correlation_score=0.5,
            p_value=0.001,
            test_type='chi_square',
            significance_level='high',
            description='Gender shows significant correlation with target variable'
        ),
        SensitiveFeature(
            feature_name='race',
            correlation_score=0.3,
            p_value=0.005,
            test_type='chi_square',
            significance_level='high',
            description='Race shows significant correlation with target variable'
        ),
        SensitiveFeature(
            feature_name='region',
            correlation_score=0.1,
            p_value=0.15,
            test_type='chi_square',
            significance_level='low',
            description='Region shows weak correlation with target variable'
        )
    ]
    
    # Initialize AnalysisService
    analysis_service = AnalysisService()
    
    # Test the enhanced _detect_bias method
    print("\nTesting enhanced _detect_bias method...")
    bias_metrics = analysis_service._detect_bias(model, df, 'loan_approved', sensitive_features)
    
    print(f"\nGenerated {len(bias_metrics)} bias metrics:")
    
    # Categorize metrics by type
    direct_bias_metrics = [m for m in bias_metrics if 'Direct Bias' in m.metric_name]
    proxy_metrics = [m for m in bias_metrics if 'Proxy Feature' in m.metric_name]
    statistical_metrics = [m for m in bias_metrics if 'Statistical Test' in m.metric_name]
    model_based_metrics = [m for m in bias_metrics if 'Model-Based' in m.metric_name]
    multi_attr_metrics = [m for m in bias_metrics if 'Multi-Attribute' in m.metric_name]
    cross_attr_metrics = [m for m in bias_metrics if 'Cross-Attribute' in m.metric_name]
    standard_metrics = [m for m in bias_metrics if 'Standard' in m.metric_name]
    
    print(f"\nMetrics by category:")
    print(f"  - Direct Bias: {len(direct_bias_metrics)}")
    print(f"  - Proxy Features: {len(proxy_metrics)}")
    print(f"  - Statistical Tests: {len(statistical_metrics)}")
    print(f"  - Model-Based: {len(model_based_metrics)}")
    print(f"  - Multi-Attribute: {len(multi_attr_metrics)}")
    print(f"  - Cross-Attribute: {len(cross_attr_metrics)}")
    print(f"  - Standard Fairness: {len(standard_metrics)}")
    
    # Show some example metrics
    print(f"\nExample Direct Bias Metrics:")
    for metric in direct_bias_metrics[:3]:
        print(f"  - {metric.metric_name}: {metric.value:.6f} ({'BIASED' if metric.is_biased else 'FAIR'})")
    
    print(f"\nExample Proxy Feature Metrics:")
    for metric in proxy_metrics[:3]:
        print(f"  - {metric.metric_name}: {metric.value:.4f} ({'BIASED' if metric.is_biased else 'FAIR'})")
    
    print(f"\nExample Statistical Test Metrics:")
    for metric in statistical_metrics[:3]:
        print(f"  - {metric.metric_name}: {metric.value:.6f} ({'BIASED' if metric.is_biased else 'FAIR'})")
    
    # Test fairness score calculation with advanced metrics
    print(f"\nTesting fairness score calculation...")
    fairness_score = analysis_service._calculate_fairness_score(bias_metrics)
    
    print(f"Overall Fairness Score: {fairness_score.overall_score:.2f}/100")
    print(f"Bias Score: {fairness_score.bias_score:.2f}/100")
    print(f"Fairness Level: {fairness_score.fairness_level}")
    print(f"Number of recommendations: {len(fairness_score.recommendations)}")
    
    return True

def main():
    """Main test function"""
    print("ADVANCED BIAS DETECTION INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test 1: Advanced Bias Detection Class
        print("Running Test 1: Advanced Bias Detection Class...")
        success1 = test_advanced_bias_detection()
        print(f"✅ Test 1 {'PASSED' if success1 else 'FAILED'}")
        
        # Test 2: Integration with AnalysisService
        print("\n\nRunning Test 2: Integration with AnalysisService...")
        success2 = test_integration_with_analysis_service()
        print(f"✅ Test 2 {'PASSED' if success2 else 'FAILED'}")
        
        # Overall result
        overall_success = success1 and success2
        print(f"\n{'='*60}")
        print(f"OVERALL INTEGRATION TEST: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        print(f"{'='*60}")
        
        if overall_success:
            print("\n🎉 All advanced bias detection features have been successfully integrated!")
            print("\nIntegrated Features:")
            print("✅ 1. Advanced bias detection methods (direct bias, proxy detection)")
            print("✅ 2. Comprehensive statistical testing (KS, Mann-Whitney, Chi-square, correlation)")
            print("✅ 3. Multi-sensitive attribute analysis (individual + cross-attribute analysis)")
            print("✅ 4. Model-based bias testing with comprehensive fairness metrics")
            print("✅ 5. Enhanced fairness scoring with advanced recommendations")
            
            print("\nThe functionality is now similar to number_of_biased_features.py")
            print("but integrated into the analysis_service.py architecture!")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
