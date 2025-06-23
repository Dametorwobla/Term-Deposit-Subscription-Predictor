#!/usr/bin/env python3
"""
Setup script for Term Deposit Subscription Predictor
Verifies installation and runs basic tests
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'data/bank-additional-full.csv',
        'src/train_model.py',
        'app.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ Found {file_path}")
    
    if missing_files:
        print("❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    return True

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed to install dependencies:")
        print(e.stderr.decode())
        return False

def test_imports():
    """Test if key packages can be imported"""
    packages = ['pandas', 'numpy', 'sklearn', 'streamlit', 'matplotlib']
    
    print("\n🔍 Testing package imports...")
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    
    return True

def run_model_training():
    """Run a quick model training test"""
    print("\n🤖 Testing model training...")
    try:
        # Change to src directory and run training
        original_dir = os.getcwd()
        os.chdir('src')
        
        result = subprocess.run([sys.executable, "train_model.py"], 
                               capture_output=True, text=True, timeout=120)
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully")
            return True
        else:
            print("❌ Model training failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Model training timed out")
        return False
    except Exception as e:
        print(f"❌ Model training error: {e}")
        return False

def main():
    """Main setup function"""
    print("🏦 Term Deposit Predictor - Setup & Verification")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check required files
    print("\n📁 Checking project files...")
    if not check_files():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test model training
    if not run_model_training():
        print("\n⚠️  Model training failed, but you can still run the app manually")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Train the model: python src/train_model.py")
    print("2. Launch the app:  python run_app.py")
    print("3. Open browser:    http://localhost:8501")
    print("\n💡 For advanced models:")
    print("   python src/improve_model.py")
    print("   python src/advanced_models.py")

if __name__ == "__main__":
    main() 