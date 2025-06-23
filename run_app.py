#!/usr/bin/env python3
"""
Simple script to run the Term Deposit Predictor Streamlit app
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'pandas', 'numpy', 'sklearn', 'matplotlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package if package != 'sklearn' else 'scikit-learn')
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'data/bank-additional-full.csv',
        'models/model.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        
        if 'models/model.pkl' in missing_files:
            print("\n💡 To create the model file, run:")
            print("   python src/train_model.py")
        
        return False
    
    return True

def main():
    """Main function to run the Streamlit app"""
    print("🏦 Term Deposit Predictor - Starting App...")
    print("=" * 50)
    
    # Check requirements
    print("🔍 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    
    # Check data files
    print("📂 Checking data files...")
    if not check_data_files():
        print("\n⚠️  Some files are missing, but the app will still run with limited functionality.")
    
    # Run Streamlit app
    print("🚀 Starting Streamlit app...")
    print("\n📱 The app will open in your default web browser at:")
    print("   http://localhost:8501")
    print("\n🛑 To stop the app, press Ctrl+C in this terminal")
    print("=" * 50)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")

if __name__ == "__main__":
    main() 