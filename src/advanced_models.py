"""
XGBoost Example for Term Deposit Prediction
==========================================
Demonstration of advanced gradient boosting for maximum performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import joblib

def prepare_data_for_xgboost():
    """Prepare data optimized for XGBoost"""
    
    # Load data
    print("📊 Loading and preparing data for XGBoost...")
    df = pd.read_csv('data/bank-additional-full.csv', sep=';')
    
    # Feature engineering (same as before)
    df['age_duration'] = df['age'] * df['duration']
    df['campaign_duration'] = df['campaign'] * df['duration']
    df['economic_confidence'] = df['cons.conf.idx'] * df['euribor3m']
    df['contact_intensity'] = df['campaign'] / (df['pdays'].replace(999, 0) + 1)
    
    # Age groups as numerical (XGBoost handles this well)
    df['age_group_num'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], labels=[1,2,3,4,5]).astype(int)
    
    # Duration categories as numerical
    df['duration_cat_num'] = pd.cut(df['duration'], bins=[0, 120, 300, 600, float('inf')], labels=[1,2,3,4]).astype(int)
    
    # Job success rate
    df['job_success_rate'] = df.groupby('job')['y'].transform(lambda x: (x == 'yes').sum() / len(x))
    
    # Label encode categorical variables (XGBoost prefers this over one-hot)
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Prepare features and target
    feature_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 
                   'cons.conf.idx', 'euribor3m', 'nr.employed', 'age_duration', 'campaign_duration', 
                   'economic_confidence', 'contact_intensity', 'age_group_num', 'duration_cat_num', 
                   'job_success_rate'] + [col + '_encoded' for col in categorical_cols]
    
    X = df[feature_cols]
    y = (df['y'] == 'yes').astype(int)
    
    return X, y, label_encoders

def train_xgboost_model():
    """Train XGBoost model with optimal parameters"""
    
    try:
        import xgboost as xgb
        print("✅ XGBoost available")
    except ImportError:
        print("❌ XGBoost not installed. Install with: pip install xgboost")
        return None
    
    # Prepare data
    X, y, label_encoders = prepare_data_for_xgboost()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"📊 Training set: {len(X_train):,} samples")
    print(f"📊 Test set: {len(X_test):,} samples")
    print(f"📊 Features: {len(X.columns)} total")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"📊 Scale pos weight: {scale_pos_weight:.2f}")
    
    # XGBoost model with optimized parameters
    print("\n🚀 Training XGBoost model...")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle imbalance
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train model
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 XGBoost Results:")
    print(f"   ✅ Accuracy:  {accuracy:.4f} ({accuracy:.1%})")
    print(f"   ✅ F1 Score:  {f1:.4f} ({f1:.1%})")
    print(f"   ✅ ROC AUC:   {roc_auc:.4f}")
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print(f"\n🔍 Top 10 Most Important Features (XGBoost):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    for idx, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    joblib.dump(xgb_model, '../models/xgboost_model.pkl')
    joblib.dump(label_encoders, '../models/label_encoders.pkl')
    print(f"\n💾 XGBoost model saved as 'models/xgboost_model.pkl'")
    
    return xgb_model, f1

if __name__ == "__main__":
    print("🚀 XGBoost Example for Term Deposit Prediction")
    print("===============================================")
    
    model, f1_score_result = train_xgboost_model()
    
    if model is not None:
        print(f"\n🎯 XGBoost F1 Score: {f1_score_result:.1%}")
        
        if f1_score_result > 0.67:
            print("🏆 Excellent! XGBoost achieved >67% F1 score")
        elif f1_score_result > 0.60:
            print("👍 Good! XGBoost shows solid improvement")
        else:
            print("🔄 Consider hyperparameter tuning for better results")
        
        print("\n💡 Next steps for even better performance:")
        print("   • Hyperparameter tuning with optuna or GridSearch")
        print("   • Feature selection with recursive feature elimination")
        print("   • Ensemble XGBoost with Random Forest")
        print("   • Try LightGBM or CatBoost")
        print("   • Cross-validation for robust evaluation")
    else:
        print("\n💡 To use XGBoost:")
        print("   Run: pip install xgboost")
        print("   Then rerun this script") 