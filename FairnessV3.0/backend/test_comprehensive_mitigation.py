"""
Test script for the comprehensive bias mitigation system

This script demonstrates how to use the new comprehensive mitigation system
that automatically applies all available bias mitigation strategies and
evaluates their effectiveness.
"""

import asyncio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
import pickle
import time

# Add the backend directory to Python path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.comprehensive_mitigation_service import comprehensive_mitigation_service
from app.services.analysis_service import analysis_service
from app.services.upload_service import upload_service
from app.models.schemas import SensitiveFeature, ModelInfo, AnalysisResults, FairnessScore
from app.utils.logger import get_mitigation_logger

logger = get_mitigation_logger()


def create_biased_dataset():
    """Create a synthetic biased dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create sensitive attributes
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    age_group = np.random.choice(['Young', 'Old'], n_samples, p=[0.7, 0.3])
    
    # Create features with bias
    income = np.random.normal(50000, 15000, n_samples)
    # Add gender bias to income
    income = np.where(gender == 'Male', income * 1.2, income * 0.8)
    
    education_score = np.random.normal(3.0, 1.0, n_samples)
    # Add age bias to education
    education_score = np.where(age_group == 'Young', education_score + 0.5, education_score - 0.3)
    
    experience_years = np.random.normal(8, 4, n_samples)
    experience_years = np.clip(experience_years, 0, 40)
    
    # Create biased target variable (loan approval)
    # Base probability
    base_prob = 0.5
    
    # Add feature effects
    prob = base_prob
    prob += (income - 50000) / 100000 * 0.3  # Income effect
    prob += (education_score - 3.0) / 4.0 * 0.2  # Education effect
    prob += experience_years / 40 * 0.1  # Experience effect
    
    # Add bias effects
    prob += np.where(gender == 'Male', 0.15, -0.15)  # Gender bias
    prob += np.where(age_group == 'Young', 0.1, -0.1)  # Age bias
    
    # Convert to binary outcome
    loan_approved = np.random.binomial(1, np.clip(prob, 0.1, 0.9), n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'gender': gender,
        'age_group': age_group,
        'income': income,
        'education_score': education_score,
        'experience_years': experience_years,
        'loan_approved': loan_approved
    })
    
    return df


def create_test_model(df):
    """Create a test model for bias mitigation"""
    # Prepare features and target
    X = df.drop(columns=['loan_approved'])
    y = df['loan_approved']
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_age = LabelEncoder()
    
    X_encoded = X.copy()
    X_encoded['gender'] = le_gender.fit_transform(X['gender'])
    X_encoded['age_group'] = le_age.fit_transform(X['age_group'])
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_encoded, y)
    
    return model, le_gender, le_age


async def test_comprehensive_mitigation():
    """Test the comprehensive mitigation system"""
    print("🚀 Starting Comprehensive Bias Mitigation Test")
    print("=" * 60)
    
    # Step 1: Create test data and model
    print("\n1. Creating biased test dataset...")
    df = create_biased_dataset()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Target distribution: {df['loan_approved'].value_counts().to_dict()}")
    print(f"   Gender distribution: {df['gender'].value_counts().to_dict()}")
    print(f"   Age distribution: {df['age_group'].value_counts().to_dict()}")
    
    # Step 2: Train biased model
    print("\n2. Training biased model...")
    model, le_gender, le_age = create_test_model(df)
    print(f"   Model type: {type(model).__name__}")
    
    # Step 3: Save model and dataset (simulate upload)
    print("\n3. Saving model and dataset...")
    os.makedirs("test_uploads", exist_ok=True)
    
    model_path = "test_uploads/biased_model.pkl"
    dataset_path = "test_uploads/biased_dataset.csv"
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'encoders': {'gender': le_gender, 'age_group': le_age}
        }, f)
    
    df.to_csv(dataset_path, index=False)
    print(f"   Model saved to: {model_path}")
    print(f"   Dataset saved to: {dataset_path}")
    
    # Step 4: Mock analysis results
    print("\n4. Creating mock analysis results...")
    
    # Create sensitive features
    sensitive_features = [
        SensitiveFeature(feature_name="gender", bias_score=0.85, is_protected=True),
        SensitiveFeature(feature_name="age_group", bias_score=0.72, is_protected=True)
    ]
    
    # Create model info
    model_info = ModelInfo(
        model_type="LogisticRegression",
        target_column="loan_approved",
        feature_columns=["gender", "age_group", "income", "education_score", "experience_years"],
        model_size="1.2 KB",
        training_accuracy=0.78
    )
    
    # Create fairness score
    fairness_score = FairnessScore(
        overall_score=45.2,
        fairness_level="Poor",
        bias_detected=True,
        total_metrics=12,
        failing_metrics=8
    )
    
    # Mock analysis results
    analysis_results = AnalysisResults(
        job_id="test_job_123",
        model_info=model_info,
        sensitive_features=sensitive_features,
        bias_metrics=[],  # Will be populated by comprehensive mitigation
        fairness_score=fairness_score,
        visualizations={},
        analysis_summary={}
    )
    
    # Step 5: Register test analysis results
    print("\n5. Registering test analysis...")
    analysis_job_id = "test_analysis_job_123"
    analysis_service.analysis_results[analysis_job_id] = analysis_results
    
    # Mock analysis job
    from app.models.schemas import AnalysisJob, JobStatus
    analysis_job = AnalysisJob(
        model_upload_id="test_model_upload",
        train_dataset_upload_id="test_dataset_upload",
        test_dataset_upload_id=None
    )
    analysis_job.job_id = analysis_job_id
    analysis_job.status = JobStatus.COMPLETED
    analysis_service.analysis_jobs[analysis_job_id] = analysis_job
    
    # Mock upload service
    upload_service.uploaded_models["test_model_upload"] = model_path
    upload_service.uploaded_datasets["test_dataset_upload"] = dataset_path
    
    # Step 6: Start comprehensive mitigation
    print("\n6. Starting comprehensive bias mitigation...")
    mitigation_job = await comprehensive_mitigation_service.start_comprehensive_mitigation(
        analysis_job_id=analysis_job_id
    )
    
    print(f"   Mitigation job ID: {mitigation_job.job_id}")
    print(f"   Initial status: {mitigation_job.status}")
    
    # Step 7: Wait for completion and monitor progress
    print("\n7. Monitoring mitigation progress...")
    max_wait_time = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        job_status = comprehensive_mitigation_service.get_mitigation_job(mitigation_job.job_id)
        print(f"   Progress: {job_status.progress}% - Status: {job_status.status}")
        
        if job_status.status.value in ["completed", "failed"]:
            break
        
        await asyncio.sleep(2)
    
    # Step 8: Get and display results
    print("\n8. Retrieving comprehensive mitigation results...")
    
    try:
        results = comprehensive_mitigation_service.format_results_for_frontend(mitigation_job.job_id)
        
        print("\n" + "=" * 60)
        print("📊 COMPREHENSIVE MITIGATION RESULTS")
        print("=" * 60)
        
        # Display bias before mitigation
        print(f"\n📈 BIAS METRICS BEFORE MITIGATION:")
        for metric_name, value in results['bias_before'].items():
            print(f"   {metric_name}: {value:.4f}")
        
        # Display results for each strategy
        print(f"\n🔧 MITIGATION STRATEGIES APPLIED ({len(results['bias_after'])}):")
        for i, strategy_result in enumerate(results['bias_after'], 1):
            print(f"\n   {i}. {strategy_result['strategy']} ({strategy_result['strategy_type']})")
            print(f"      Fairness Score: {strategy_result['fairness_score']:.2f}")
            print(f"      Execution Time: {strategy_result['execution_time']:.2f}s")
            print(f"      Model Performance:")
            for perf_metric, value in strategy_result['model_performance'].items():
                print(f"        {perf_metric}: {value:.4f}")
        
        # Display best strategy
        print(f"\n🏆 BEST STRATEGY: {results['best_strategy']}")
        print(f"   Overall Fairness Improvement: {results['overall_fairness_improvement']:.2f}")
        
        # Display improvements
        print(f"\n📈 METRIC IMPROVEMENTS:")
        for metric_name, improvement in results['improvements'].items():
            print(f"   {metric_name}: {improvement:.2f}%")
        
        # Display execution summary
        print(f"\n⏱️ EXECUTION SUMMARY:")
        summary = results['execution_summary']
        print(f"   Total Strategies: {summary['total_strategies']}")
        print(f"   Successful: {summary['successful_strategies']}")
        print(f"   Failed: {summary['failed_strategies']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Total Execution Time: {summary['total_execution_time']:.2f}s")
        
        # Display recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        for i, recommendation in enumerate(results['recommendations'], 1):
            print(f"   {i}. {recommendation}")
        
        print("\n" + "=" * 60)
        print("✅ Comprehensive mitigation test completed successfully!")
        print("=" * 60)
        
        # Step 9: Test API response format
        print("\n9. Testing API response formats...")
        
        # Test comparison endpoint
        comparison_data = comprehensive_mitigation_service.get_comprehensive_results(mitigation_job.job_id)
        print(f"   Comparison data available: {len(comparison_data.bias_after)} strategies")
        
        # Test best strategy details
        best_strategy_results = None
        for result in comparison_data.bias_after:
            if result.success and result.strategy_name == comparison_data.best_strategy:
                best_strategy_results = result
                break
        
        if best_strategy_results:
            print(f"   Best strategy details: {best_strategy_results.strategy_name}")
            print(f"   Best strategy fairness score: {best_strategy_results.fairness_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error retrieving results: {str(e)}")
        job_status = comprehensive_mitigation_service.get_mitigation_job(mitigation_job.job_id)
        if job_status.error_message:
            print(f"   Job error: {job_status.error_message}")
        return False


async def test_api_endpoints():
    """Test the API endpoints with sample data"""
    print("\n🔗 Testing API Endpoints")
    print("=" * 40)
    
    # This would typically be done via HTTP requests
    # For now, we'll test the service methods directly
    
    try:
        # Test available strategies endpoint
        from app.api.comprehensive_mitigation import get_comprehensive_strategies
        strategies_info = await get_comprehensive_strategies()
        
        print(f"✅ Available strategies: {len(strategies_info['strategies'])}")
        print(f"✅ Strategy types: {list(strategies_info['strategy_types'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ API endpoint test failed: {str(e)}")
        return False


def main():
    """Main test function"""
    print("🧪 COMPREHENSIVE BIAS MITIGATION SYSTEM TEST")
    print("=" * 60)
    print("This test demonstrates the automatic application of all")
    print("available bias mitigation strategies and their evaluation.")
    print("=" * 60)
    
    async def run_tests():
        # Test comprehensive mitigation
        mitigation_success = await test_comprehensive_mitigation()
        
        # Test API endpoints
        api_success = await test_api_endpoints()
        
        print(f"\n📋 TEST SUMMARY")
        print("=" * 30)
        print(f"Comprehensive Mitigation: {'✅ PASSED' if mitigation_success else '❌ FAILED'}")
        print(f"API Endpoints: {'✅ PASSED' if api_success else '❌ FAILED'}")
        
        if mitigation_success and api_success:
            print("\n🎉 All tests passed! The comprehensive mitigation system is ready.")
        else:
            print("\n⚠️ Some tests failed. Please check the error messages above.")
    
    # Run the async tests
    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
