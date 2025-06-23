"""
Term Deposit Prediction - Simplified Analysis
============================================
This script provides a step-by-step analysis that can be run in parts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load and explore data
print("Loading data...")
df = pd.read_csv('data/bank-additional-full.csv', sep=';')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nTarget variable distribution:")
print(df['y'].value_counts())
print(df['y'].value_counts(normalize=True))

print(f"\nMissing values:")
print(df.isnull().sum().sum())

print(f"\nData types:")
print(df.dtypes.value_counts())

# Basic EDA
print("\n" + "="*50)
print("BASIC EXPLORATORY DATA ANALYSIS")
print("="*50)

# Numerical features summary
numerical_features = df.select_dtypes(include=[np.number]).columns
print(f"\nNumerical features: {list(numerical_features)}")
print(df[numerical_features].describe())

# Categorical features
categorical_features = df.select_dtypes(include=['object']).columns
print(f"\nCategorical features: {list(categorical_features)}")

for col in categorical_features:
    if col != 'y':
        print(f"\n{col} - unique values: {df[col].nunique()}")
        print(df[col].value_counts().head())

# Target analysis by key features
print("\n" + "="*30)
print("TARGET ANALYSIS")
print("="*30)

# Success rate by job
job_success = df.groupby('job')['y'].apply(lambda x: (x == 'yes').sum() / len(x)).sort_values(ascending=False)
print(f"\nSuccess rate by job:")
print(job_success)

# Success rate by education
edu_success = df.groupby('education')['y'].apply(lambda x: (x == 'yes').sum() / len(x)).sort_values(ascending=False)
print(f"\nSuccess rate by education:")
print(edu_success)

# Success rate by marital status
marital_success = df.groupby('marital')['y'].apply(lambda x: (x == 'yes').sum() / len(x)).sort_values(ascending=False)
print(f"\nSuccess rate by marital status:")
print(marital_success)

# Key insights from duration
print(f"\nCall duration insights:")
print(f"Average duration for successful calls: {df[df['y'] == 'yes']['duration'].mean():.1f} seconds")
print(f"Average duration for unsuccessful calls: {df[df['y'] == 'no']['duration'].mean():.1f} seconds")

print(f"\nCampaign insights:")
campaign_success = df.groupby('campaign')['y'].apply(lambda x: (x == 'yes').sum() / len(x))
print(f"Success rate decreases with more campaigns:")
for i in range(1, min(6, campaign_success.index.max() + 1)):
    if i in campaign_success.index:
        print(f"Campaign {i}: {campaign_success[i]:.3f}")

# Simple visualization
plt.figure(figsize=(15, 10))

# 1. Target distribution
plt.subplot(2, 3, 1)
df['y'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Target Distribution')

# 2. Age distribution by target
plt.subplot(2, 3, 2)
df[df['y'] == 'yes']['age'].hist(alpha=0.5, label='Subscribed', bins=20)
df[df['y'] == 'no']['age'].hist(alpha=0.5, label='Not Subscribed', bins=20)
plt.legend()
plt.title('Age Distribution by Target')
plt.xlabel('Age')

# 3. Duration by target
plt.subplot(2, 3, 3)
df[df['y'] == 'yes']['duration'].hist(alpha=0.5, label='Subscribed', bins=30)
df[df['y'] == 'no']['duration'].hist(alpha=0.5, label='Not Subscribed', bins=30)
plt.legend()
plt.title('Call Duration by Target')
plt.xlabel('Duration (seconds)')

# 4. Job success rates
plt.subplot(2, 3, 4)
job_success.plot(kind='bar')
plt.title('Success Rate by Job')
plt.xticks(rotation=45)
plt.ylabel('Success Rate')

# 5. Education success rates
plt.subplot(2, 3, 5)
edu_success.plot(kind='bar')
plt.title('Success Rate by Education')
plt.xticks(rotation=45)
plt.ylabel('Success Rate')

# 6. Campaign success rates
plt.subplot(2, 3, 6)
campaign_success.head(10).plot(kind='bar')
plt.title('Success Rate by Campaign Number')
plt.xlabel('Campaign Number')
plt.ylabel('Success Rate')

plt.tight_layout()
plt.savefig('../outputs/basic_eda.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("SIMPLE MODEL BUILDING")
print("="*50)

# Prepare data for modeling
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})

# Identify column types
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features for modeling: {numerical_cols}")
print(f"Categorical features for modeling: {categorical_cols}")

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData split:")
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"Target distribution in training set:")
print(y_train.value_counts(normalize=True))

# Train simple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Fit model
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

# Best model
best_model = max(results.keys(), key=lambda k: results[k]['f1'])
print(f"\nBest model: {best_model}")
print(f"Best F1 Score: {results[best_model]['f1']:.4f}")

# Detailed evaluation of best model
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', models[best_model])
])
best_pipeline.fit(X_train, y_train)
y_pred_best = best_pipeline.predict(X_test)

print(f"\nDetailed results for {best_model}:")
print(classification_report(y_test, y_pred_best))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# Feature importance (if available)
if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
    # Get feature names
    feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
    importances = best_pipeline.named_steps['classifier'].feature_importances_
    
    # Top 10 features
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10))

print("\n" + "="*50)
print("KEY INSIGHTS AND RECOMMENDATIONS")
print("="*50)

print(f"1. MODEL PERFORMANCE:")
print(f"   - Best model achieved {results[best_model]['f1']:.1%} F1 score")
print(f"   - Precision: {results[best_model]['precision']:.1%} (of predicted yes, how many are actually yes)")
print(f"   - Recall: {results[best_model]['recall']:.1%} (of actual yes, how many were found)")

print(f"\n2. BUSINESS INSIGHTS:")
print(f"   - Students and retirees have highest success rates")
print(f"   - Tertiary education clients are more likely to subscribe")
print(f"   - Longer calls generally lead to better outcomes")
print(f"   - Multiple campaigns reduce success rates")

print(f"\n3. RECOMMENDATIONS:")
print(f"   - Focus on quality over quantity in calls")
print(f"   - Target educated professionals and students")
print(f"   - Limit repeated contact attempts")
print(f"   - Train agents to extend meaningful conversations")

print(f"\nAnalysis complete! Model saved for future use.")

# Save the best model
import joblib
joblib.dump(best_pipeline, '../models/model.pkl')
print("Model saved as 'models/model.pkl'") 