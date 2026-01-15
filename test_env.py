import sys
print("Python version:", sys.version)
print("Testing imports...")

try:
    import numpy as np
    print("✅ numpy imported")
except Exception as e:
    print("❌ numpy failed:", e)

try:
    import pandas as pd
    print("✅ pandas imported")
except Exception as e:
    print("❌ pandas failed:", e)

try:
    import sklearn
    print("✅ sklearn imported")
except Exception as e:
    print("❌ sklearn failed:", e)

try:
    import statsmodels
    print("✅ statsmodels imported")
except Exception as e:
    print("❌ statsmodels failed:", e)

print("Testing basic operations...")
try:
    df = pd.read_csv('data1.csv')
    print(f"✅ Data loaded: shape={df.shape}")
    print(f"Columns: {list(df.columns)[:5]}...")
except Exception as e:
    print(f"❌ Data loading failed: {e}")

print("Test completed.")
