"""
Simple Model Improvements for Term Deposit Prediction
====================================================
Focus on proven techniques to boost your 89% accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def engineer_features(df):
    """Feature engineering - creates new predictive features"""
    df = df.copy()
    
    print("🔧 Engineering new features...")
    
    # 1. Interaction features (often very powerful)
    df['age_duration'] = df['age'] * df['duration']
    df['campaign_duration'] = df['campaign'] * df['duration']
    
    # 2. Binning continuous variables
    df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                            labels=['young', 'adult', 'middle', 'senior', 'elderly'])
    
    df['duration_category'] = pd.cut(df['duration'], 
                                   bins=[0, 120, 300, 600, float('inf')],
                                   labels=['short', 'medium', 'long', 'very_long'])
    
    # 3. Economic indicators combined
    df['economic_confidence'] = df['cons.conf.idx'] * df['euribor3m']
    
    # 4. Previous campaign success rate by segment
    df['job_success_rate'] = df.groupby('job')['y'].transform(
        lambda x: (x == 'yes').sum() / len(x)
    )
    
    # 5. Contact intensity
    df['contact_intensity'] = df['campaign'] / (df['pdays'].replace(999, 0) + 1)
    
    print(f"✅ Added 6 new engineered features")
    return df

def compare_models():
    """Compare baseline vs improved model"""
    
    # Load data
    print("📊 Loading bank marketing data...")
    df = pd.read_csv('data/bank-additional-full.csv', sep=';')
    print(f"   Dataset: {len(df):,} records, {len(df.columns)} features")
    
    # Feature engineering
    df_enhanced = engineer_features(df)
    
    # Prepare data
    X = df_enhanced.drop('y', axis=1)
    y = df_enhanced['y'].map({'yes': 1, 'no': 0})
    
    # Identify column types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"📊 Features: {len(numerical_cols)} numerical, {len(categorical_cols)} categorical")
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Training set: {len(X_train):,} samples")
    print(f"📊 Test set: {len(X_test):,} samples")
    print(f"📊 Class distribution: {(y_train==1).mean():.1%} positive class")
    
    # Models to compare
    models = {
        'Baseline Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        
        'Improved Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=200,          # More trees
                max_depth=15,              # Control overfitting
                min_samples_split=5,       # Prevent overfitting
                min_samples_leaf=2,        # Prevent overfitting
                max_features='sqrt',       # Feature randomness
                class_weight='balanced',   # Handle imbalance
                random_state=42
            ))
        ]),
        
        'Tuned Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=0.1,                     # Regularization
                class_weight='balanced',   # Handle imbalance
                max_iter=1000,
                random_state=42
            ))
        ])
    }
    
    print("\n" + "="*60)
    print("🚀 TRAINING AND COMPARING MODELS")
    print("="*60)
    
    results = {}
    best_f1 = 0
    best_model = None
    best_name = ""
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name
        
        # Print results
        print(f"   ✅ Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
        print(f"   ✅ Precision: {precision:.4f} ({precision:.1%})")
        print(f"   ✅ Recall:    {recall:.4f} ({recall:.1%})")
        print(f"   ✅ F1 Score:  {f1:.4f} ({f1:.1%})")
        print(f"   ✅ ROC AUC:   {roc_auc:.4f}")
    
    # Results comparison
    print("\n" + "="*60)
    print("📈 RESULTS COMPARISON")
    print("="*60)
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    print(results_df)
    
    # Best model details
    print(f"\n🏆 Best Model: {best_name}")
    print(f"🎯 Best F1 Score: {best_f1:.1%}")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report for {best_name}:")
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best))
    
    # Feature importance
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        print(f"\n🔍 Top 10 Most Important Features ({best_name}):")
        feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
        importances = best_model.named_steps['classifier'].feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        for idx, row in feature_importance_df.iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save best model
    joblib.dump(best_model, '../models/improved_model.pkl')
    print(f"\n💾 Best model saved as 'models/improved_model.pkl'")
    
    return best_model, results

def improvement_recommendations():
    """Show specific recommendations for further improvement"""
    print("\n" + "="*60)
    print("🚀 RECOMMENDATIONS FOR FURTHER IMPROVEMENT")
    print("="*60)
    
    techniques = [
        ("🎯 Feature Engineering", [
            "• Create more interaction features (age × education, job × marital)",
            "• Time-based features (season, month from last contact)",
            "• Customer lifetime value estimates",
            "• Ratio features (duration/previous, campaign/success_rate)"
        ]),
        
        ("⚖️ Handle Class Imbalance", [
            "• SMOTE (Synthetic Minority Oversampling)",
            "• ADASYN (Adaptive Synthetic Sampling)",
            "• Cost-sensitive learning",
            "• Ensemble methods with balanced sampling"
        ]),
        
        ("🧠 Advanced Algorithms", [
            "• XGBoost or LightGBM (often 2-5% better)",
            "• Neural Networks with dropout",
            "• Stacking ensembles",
            "• Bayesian optimization for hyperparameters"
        ]),
        
        ("📊 Data Quality", [
            "• Handle outliers in duration and age",
            "• Feature selection (remove redundant features)",
            "• Cross-validation for robust evaluation",
            "• Collect more data if possible"
        ]),
        
        ("🎛️ Hyperparameter Tuning", [
            "• Grid search or Random search",
            "• Bayesian optimization (optuna, hyperopt)",
            "• Early stopping for tree-based models",
            "• Learning rate scheduling"
        ])
    ]
    
    for category, items in techniques:
        print(f"\n{category}:")
        for item in items:
            print(f"  {item}")

if __name__ == "__main__":
    print("🏦 Simple Model Improvements for Term Deposit Prediction")
    print("=========================================================")
    
    # Compare models
    best_model, results = compare_models()
    
    # Show improvement recommendations
    improvement_recommendations()
    
    # Calculate improvement
    baseline_f1 = results['Baseline Random Forest']['f1']
    best_f1 = max(results[model]['f1'] for model in results)
    improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
    
    print(f"\n🎉 SUMMARY")
    print(f"{'='*40}")
    print(f"📈 F1 Score improvement: {improvement:.1f}%")
    print(f"📊 From {baseline_f1:.1%} to {best_f1:.1%}")
    
    if best_f1 > 0.60:
        print("🎯 Excellent! You've achieved >60% F1 score")
    elif best_f1 > 0.55:
        print("👍 Good improvement! Consider advanced techniques")
    else:
        print("🔄 Try the advanced techniques mentioned above") 