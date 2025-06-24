# Term Deposit Subscription Predictor

A machine learning project to predict whether a bank customer will subscribe to a term deposit based on marketing campaign data.

## 📊 Project Overview

This project analyzes bank marketing data to predict customer subscription to term deposits. It includes data exploration, model training, and a web dashboard for predictions.

**Dataset**: Bank Marketing Dataset (UCI Machine Learning Repository)
- **Records**: 41,188 customers
- **Features**: 20 attributes (demographic, campaign, economic indicators)
- **Target**: Term deposit subscription (yes/no)

## 🗂️ Project Structure

```
Term-Deposit-Subscription-Predictor/
├── src/                          # Source code
│   ├── train_model.py            # Train baseline model
│   ├── improve_model.py          # Advanced model improvements  
│   └── advanced_models.py        # XGBoost and other algorithms
├── models/                       # Trained models
│   └── model.pkl                 # Main trained model
├── data/                         # Dataset files
│   └── bank-additional-full.csv  # Main dataset
├── outputs/                      # Generated outputs
│   └── basic_eda.png            # EDA visualization
├── app.py                       # Streamlit web dashboard
├── run_app.py                   # App launcher script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd Term-Deposit-Subscription-Predictor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train the baseline model
python src/train_model.py
```

### 3. Launch the Dashboard

```bash
# Start the web application
python run_app.py
```

The dashboard will open at `http://localhost:8501`

## 📈 Model Performance

**Baseline Random Forest Model:**
- **Accuracy**: 89.2%
- **Precision**: 67.3%
- **Recall**: 28.5%
- **F1-Score**: 40.1%

**Key Features:**
- Call duration (most important)
- Economic indicators (employment rate, confidence index)
- Customer demographics (age, job, education)
- Campaign characteristics

## 🎯 Model Improvements

To explore advanced techniques for better performance:

```bash
# Feature engineering + tuned hyperparameters
python src/improve_model.py

# XGBoost and advanced algorithms  
python src/advanced_models.py
```

These scripts demonstrate:
- ✅ Feature engineering (interactions, binning)
- ✅ Class imbalance handling (SMOTE, class weights)
- ✅ Hyperparameter tuning
- ✅ Ensemble methods
- ✅ Advanced algorithms (XGBoost, LightGBM)

## 🎛️ Web Dashboard Features

The Streamlit dashboard includes:

1. **📈 Executive Dashboard**
   - Key metrics and KPIs
   - Target distribution analysis
   - Success rates by demographics

2. **🔍 Data Explorer**
   - Interactive filtering
   - Correlation analysis
   - Raw data inspection

3. **🎯 Predictions**
   - Real-time prediction interface
   - Probability estimates
   - Input validation

4. **📊 Model Performance**
   - Feature importance
   - Model metrics
   - Performance comparison

## 🔍 Key Business Insights

1. **High-Value Segments**:
   - Students and retirees (highest success rates)
   - Tertiary education customers
   - Clients with longer call durations

2. **Campaign Optimization**:
   - Quality over quantity in contact attempts
   - Optimal call duration: 200+ seconds
   - Avoid excessive repeated contacts

3. **Economic Factors**:
   - Employment variation rate significantly impacts success
   - Consumer confidence index is a strong predictor
   - Euribor rate influences customer decisions

## 🛠️ Technical Details

**Libraries Used:**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, imbalanced-learn
- **Visualization**: matplotlib, seaborn
- **Web Framework**: streamlit
- **Advanced ML**: xgboost, lightgbm

**Model Pipeline:**
1. Data preprocessing (scaling, encoding)
2. Feature engineering (interactions, binning)
3. Model training (Random Forest baseline)
4. Hyperparameter optimization
5. Model evaluation and selection

## 📊 Dataset Information

**Source**: UCI Machine Learning Repository
**Domain**: Banking/Finance
**Task**: Binary Classification

**Feature Categories:**
- **Demographics**: age, job, marital, education
- **Financial**: default, housing, loan
- **Campaign**: contact, month, day, duration, campaign
- **Previous**: pdays, previous, poutcome
- **Economic**: emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

TO view this project, go to: https://termpredictor25.streamlit.app/

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions or suggestions, please open an issue in this repository. 