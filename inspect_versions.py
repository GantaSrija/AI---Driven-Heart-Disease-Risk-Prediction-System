#!/usr/bin/env python3
"""
Script to inspect the versions of scikit-learn and numpy used to create the model files.
This helps ensure compatibility when deploying across different systems.
"""

import joblib
import sys

def inspect_model_versions():
    """Inspect the versions used in model.pkl and scaler.pkl"""

    print("🔍 Inspecting model file versions...")
    print("=" * 50)

    try:
        # Load model and scaler
        print("Loading model.pkl...")
        model = joblib.load('model.pkl')

        print("Loading scaler.pkl...")
        scaler = joblib.load('scaler.pkl')

        print("\n✅ Files loaded successfully!")
        print("-" * 30)

        # Check model type and attributes
        print(f"Model type: {type(model).__name__}")
        if hasattr(model, 'n_features_in_'):
            print(f"Number of features: {model.n_features_in_}")

        # Try to get version information from model
        if hasattr(model, '_sklearn_version'):
            print(f"Scikit-learn version used for model: {model._sklearn_version}")
        else:
            print("⚠️  Could not find _sklearn_version in model")

        # Try to get version information from scaler
        if hasattr(scaler, '_sklearn_version'):
            print(f"Scikit-learn version used for scaler: {scaler._sklearn_version}")
        else:
            print("⚠️  Could not find _sklearn_version in scaler")

        # Get current versions
        import sklearn
        import numpy as np
        import pandas as pd

        print("\n📊 Current environment versions:")
        print(f"  Python: {sys.version.split()[0]}")
        print(f"  Scikit-learn: {sklearn.__version__}")
        print(f"  NumPy: {np.__version__}")
        print(f"  Pandas: {pd.__version__}")

        # Check compatibility
        print("\n🔄 Compatibility Check:")
        if hasattr(model, '_sklearn_version'):
            model_sklearn_ver = model._sklearn_version
            current_sklearn_ver = sklearn.__version__

            if model_sklearn_ver == current_sklearn_ver:
                print("✅ Scikit-learn versions match!")
            else:
                print(f"⚠️  Version mismatch - Model created with {model_sklearn_ver}, current is {current_sklearn_ver}")
                print("   Consider installing the matching version: pip install scikit-learn==" + model_sklearn_ver)

    except Exception as e:
        print(f"❌ Error loading files: {e}")
        print("\n💡 Suggestions:")
        print("   1. Ensure model.pkl and scaler.pkl exist in the current directory")
        print("   2. Try retraining the model: python train_model.py")
        print("   3. Check file permissions")
        return False

    return True

def inspect_numpy_version():
    """Try to detect numpy version used for arrays"""

    print("\n🔍 Inspecting NumPy array versions...")
    print("-" * 40)

    try:
        import numpy as np

        # Load scaler to check its arrays
        scaler = joblib.load('scaler.pkl')

        # Check scaler attributes
        if hasattr(scaler, 'mean_'):
            print(f"Scaler mean array dtype: {scaler.mean_.dtype}")
            print(f"Scaler mean array shape: {scaler.mean_.shape}")

        if hasattr(scaler, 'scale_'):
            print(f"Scaler scale array dtype: {scaler.scale_.dtype}")
            print(f"Scaler scale array shape: {scaler.scale_.shape}")

        # Check model arrays if it's a tree-based model
        model = joblib.load('model.pkl')

        if hasattr(model, 'estimators_'):
            print(f"Random Forest with {len(model.estimators_)} estimators")
            # Check first tree's structure
            first_tree = model.estimators_[0]
            if hasattr(first_tree, 'tree_'):
                tree = first_tree.tree_
                print(f"Tree node array dtype: {tree.value.dtype}")
                print(f"Tree node array shape: {tree.value.shape}")

    except Exception as e:
        print(f"❌ Error inspecting arrays: {e}")

if __name__ == "__main__":
    print("🔧 Heart Failure Model Version Inspector")
    print("This tool helps identify the versions used to create your model files.\n")

    # Change to the script's directory
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    success = inspect_model_versions()
    if success:
        inspect_numpy_version()

    print("\n📝 Next Steps:")
    print("1. Note the versions shown above")
    print("2. On your target system, install matching versions:")
    print("   pip install scikit-learn==[version] numpy==[version]")
    print("3. If versions don't match, consider retraining the model")
    print("4. For production, consider using Docker for environment consistency")
