import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Create directory for outputs
os.makedirs('fairness_test_data', exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

def generate_unbiased_hiring_data(n=1000):
    """Generate unbiased hiring dataset - target depends only on qualifications"""
    
    # Demographics (should not influence decisions)
    age = np.random.randint(22, 65, n)
    gender = np.random.choice([0, 1], size=n, p=[0.6, 0.4])  # 0: Male, 1: Female
    race = np.random.choice([0, 1, 2, 3], size=n, p=[0.6, 0.2, 0.1, 0.1])  # 0: White, 1: Black, 2: Hispanic, 3: Asian
    
    # Qualifications (should influence decisions)
    education_years = np.random.normal(16, 3, n)
    education_years = np.clip(education_years, 12, 22)
    
    work_experience = np.random.normal(8, 5, n)
    work_experience = np.clip(work_experience, 0, 30)
    
    skill_score = np.random.normal(75, 15, n)
    skill_score = np.clip(skill_score, 40, 100)
    
    # Previous salary (might be correlated with demographics due to historical bias)
    base_salary = 30000 + (education_years - 12) * 5000 + work_experience * 2000 + skill_score * 300
    salary_noise = np.random.normal(0, 5000, n)
    previous_salary = base_salary + salary_noise
    
    # Target: Hiring decision based ONLY on qualifications (unbiased)
    hiring_score = (
        0.3 * (education_years - 12) / 10 +
        0.4 * work_experience / 30 +
        0.3 * skill_score / 100
    )
    
    # Add some randomness
    hiring_prob = hiring_score + np.random.normal(0, 0.1, n)
    hired = (hiring_prob > np.median(hiring_prob)).astype(int)
    
    df = pd.DataFrame({
        'age': age.astype(int),
        'gender': gender,
        'race': race,
        'education_years': education_years.round(1),
        'work_experience': work_experience.round(1),
        'skill_score': skill_score.round(1),
        'previous_salary': previous_salary.round(0).astype(int),
        'hired': hired
    })
    
    return df

def generate_biased_hiring_data(n=1000):
    """Generate biased hiring dataset - target unfairly influenced by demographics"""
    
    age = np.random.randint(22, 65, n)
    gender = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    race = np.random.choice([0, 1, 2, 3], size=n, p=[0.6, 0.2, 0.1, 0.1])
    
    education_years = np.random.normal(16, 3, n)
    education_years = np.clip(education_years, 12, 22)
    
    work_experience = np.random.normal(8, 5, n)
    work_experience = np.clip(work_experience, 0, 30)
    
    skill_score = np.random.normal(75, 15, n)
    skill_score = np.clip(skill_score, 40, 100)
    
    base_salary = 30000 + (education_years - 12) * 5000 + work_experience * 2000 + skill_score * 300
    
    # Introduce historical bias in salary based on demographics
    gender_salary_penalty = np.where(gender == 1, -8000, 0)  # Women earn less
    race_salary_penalty = np.where(race == 1, -5000, np.where(race == 2, -3000, 0))  # Racial wage gap
    
    salary_noise = np.random.normal(0, 5000, n)
    previous_salary = base_salary + gender_salary_penalty + race_salary_penalty + salary_noise
    
    # Target: Hiring decision BIASED by demographics
    base_score = (
        0.3 * (education_years - 12) / 10 +
        0.4 * work_experience / 30 +
        0.3 * skill_score / 100
    )
    
    # Add demographic bias
    gender_bias = np.where(gender == 1, -0.15, 0)  # Bias against women
    race_bias = np.where(race == 1, -0.12, np.where(race == 2, -0.08, 0))  # Bias against minorities
    age_bias = np.where(age > 50, -0.1, 0)  # Age discrimination
    
    biased_score = base_score + gender_bias + race_bias + age_bias
    hiring_prob = biased_score + np.random.normal(0, 0.1, n)
    hired = (hiring_prob > np.median(hiring_prob)).astype(int)
    
    df = pd.DataFrame({
        'age': age.astype(int),
        'gender': gender,
        'race': race,
        'education_years': education_years.round(1),
        'work_experience': work_experience.round(1),
        'skill_score': skill_score.round(1),
        'previous_salary': previous_salary.round(0).astype(int),
        'hired': hired
    })
    
    return df

def generate_loan_approval_data(n=1000, biased=False):
    """Generate loan approval dataset"""
    
    age = np.random.randint(18, 80, n)
    gender = np.random.choice([0, 1], size=n, p=[0.52, 0.48])
    race = np.random.choice([0, 1, 2, 3], size=n, p=[0.65, 0.18, 0.12, 0.05])
    
    # Financial features
    annual_income = np.random.lognormal(10.5, 0.8, n)
    annual_income = np.clip(annual_income, 20000, 500000)
    
    credit_score = np.random.normal(700, 100, n)
    credit_score = np.clip(credit_score, 300, 850)
    
    loan_amount = np.random.normal(150000, 75000, n)
    loan_amount = np.clip(loan_amount, 10000, 800000)
    
    debt_to_income = (np.random.beta(2, 5, n) * 0.6)  # 0 to 0.6
    employment_length = np.random.exponential(5, n)
    employment_length = np.clip(employment_length, 0, 40)
    
    # Base approval probability
    approval_score = (
        0.4 * (credit_score - 300) / 550 +
        0.3 * np.log(annual_income / 20000) / np.log(25) +
        0.2 * (1 - debt_to_income) +
        0.1 * np.minimum(employment_length / 10, 1)
    )
    
    if biased:
        # Add demographic bias
        gender_bias = np.where(gender == 1, -0.1, 0)
        race_bias = np.where(race == 1, -0.15, np.where(race == 2, -0.1, 0))
        approval_score += gender_bias + race_bias
    
    approval_prob = 1 / (1 + np.exp(-5 * (approval_score - 0.5)))  # Sigmoid
    approved = (np.random.rand(n) < approval_prob).astype(int)
    
    df = pd.DataFrame({
        'age': age.astype(int),
        'gender': gender,
        'race': race,
        'annual_income': annual_income.round(0).astype(int),
        'credit_score': credit_score.round(0).astype(int),
        'loan_amount': loan_amount.round(0).astype(int),
        'debt_to_income_ratio': debt_to_income.round(3),
        'employment_length': employment_length.round(1),
        'approved': approved
    })
    
    return df

def train_and_save_models(X_train, y_train, X_test, y_test, dataset_name):
    """Train multiple models and save them"""
    
    # Scale features for neural networks and SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'gradient_boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'svm': SVC(random_state=42, probability=True),
        'naive_bayes': GaussianNB(),
        'neural_network': MLPClassifier(random_state=42, hidden_layer_sizes=(100, 50), max_iter=500)
    }
    
    model_scores = {}
    
    for name, model in models.items():
        if name in ['svm', 'neural_network']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            # Save scaler along with scaled models
            joblib.dump(scaler, f'fairness_test_data/{dataset_name}_{name}_scaler.pkl')
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        model_scores[name] = accuracy
        
        # Save model
        joblib.dump(model, f'fairness_test_data/{dataset_name}_{name}_model.pkl')
    
    return model_scores

def main():
    print("Generating datasets and training models...")
    
    # Generate datasets
    datasets = {
        'hiring_unbiased': generate_unbiased_hiring_data(1200),
        'hiring_biased': generate_biased_hiring_data(1200),
        'loan_unbiased': generate_loan_approval_data(1200, biased=False),
        'loan_biased': generate_loan_approval_data(1200, biased=True)
    }
    
    all_scores = {}
    
    for dataset_name, df in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        print(f"Dataset shape: {df.shape}")
        print(f"Target distribution: {df.iloc[:, -1].value_counts().to_dict()}")
        
        # Split into train/test
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df.iloc[:, -1])
        
        # Save datasets
        train_df.to_csv(f'fairness_test_data/{dataset_name}_train.csv', index=False)
        test_df.to_csv(f'fairness_test_data/{dataset_name}_test.csv', index=False)
        
        # Prepare features and target
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        
        # Train and save models
        scores = train_and_save_models(X_train, y_train, X_test, y_test, dataset_name)
        all_scores[dataset_name] = scores
        
        print(f"Model accuracies for {dataset_name}:")
        for model_name, score in scores.items():
            print(f"  {model_name}: {score:.3f}")
    
    # Save summary
    summary_df = pd.DataFrame(all_scores).T
    summary_df.to_csv('fairness_test_data/model_accuracy_summary.csv')
    
    print(f"\n{'='*50}")
    print("Generation complete!")
    print(f"Created {len(os.listdir('fairness_test_data'))} files in 'fairness_test_data/' directory")
    print("\nDatasets created:")
    for dataset_name in datasets.keys():
        print(f"  - {dataset_name}_train.csv, {dataset_name}_test.csv")
    
    print("\nModels trained for each dataset:")
    print("  - logistic_regression, random_forest, gradient_boosting")
    print("  - svm, naive_bayes, neural_network")
    
    return all_scores

if __name__ == "__main__":
    scores = main()
