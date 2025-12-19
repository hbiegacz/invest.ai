from random_forest_train import load_dataset, FEATURE_COLUMNS, TARGET_COLUMN, SINGLE_RF_PARAMS, base_rf_params, build_model, train_model, print_metrics
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import shap
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Setup output directory
OUTPUT_DIR = Path("shap/plots")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. LOAD DATA 
# ============================================================================
print("1. Loading dataset...")
try:
    df = load_dataset()
    print(f"   Dataset shape: {df.shape}")
except Exception as e:
    print(f"   Error loading dataset: {e}")
    exit(1)

X = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

print(f"   Features selected ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

# 2. TRAIN RANDOM FOREST MODEL 
# ============================================================================
print("2. Training Random Forest Model...")
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

rf_params = {**base_rf_params(), **SINGLE_RF_PARAMS}
model, metrics = train_model(df, test_size=0.2, rf_params=rf_params)

print("\n   Model Metrics:")
print_metrics(metrics)

# 3. SHAP ANALYSIS SETUP
# ============================================================================
print("3. Setting up SHAP analysis...")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test, check_additivity=False)

# SCALE THE VALUES
# ============================================================================
# Since the values the model predicts are rather small, 
# we scale them by 100, so that when we look at the plots we dont just see +0 and -0
# it makes the plots more redable

shap_values.values = shap_values.values * 100
shap_values.base_values = shap_values.base_values * 100

print(f"   SHAP values shape: {shap_values.shape}")
print(f"   (Values scaled by 100 for percentage display)")

# 4. SHAP SUMMARY PLOT
# ============================================================================
print("4. Generating Summary Plot...")
try:
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary_plot.png")
    plt.close()
    print("   Saved shap_summary_plot.png")
except Exception as e:
    print(f"   Error generating summary plot: {e}")

# 5. DECISION PLOT
# ============================================================================
print("5. Generating Decision Plot (Stacked Waterfall alternative)...")
try:
    plt.figure(figsize=(10, 10))
    shap.decision_plot(
        shap_values.base_values[0],  
        shap_values.values[:20],     
        X_test.iloc[:20],            
        show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_decision_plot.png")
    plt.close()
    print("   Saved shap_decision_plot.png (Stacked view of first 20 samples)")
except Exception as e:
    print(f"   Error generating decision plot: {e}")

# 6. FORCE PLOT (single prediction)
# ============================================================================
print("6. Generating Force Plot (single)...")
try:
    plt.figure(figsize=(20, 3))
    shap.force_plot(
        explainer.expected_value * 100, # scaling again
        shap_values.values[0], 
        X_test.iloc[0], 
        matplotlib=True,
        show=False
    )
    plt.savefig(OUTPUT_DIR / "shap_force_plot_0.png", bbox_inches='tight', dpi=150)
    plt.close()
    print("   Saved shap_force_plot_0.png")
except Exception as e:
    print(f"   Error generating force plot: {e}")

# 7. BAR PLOT 
# ============================================================================
print("7. Generating Bar Plot...")
try:
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_bar_plot.png", bbox_inches='tight')
    plt.close()
    print("   Saved shap_bar_plot.png")
except Exception as e:
    print(f"   Error generating bar plot: {e}")

# 8. BEESWARM PLOT
# ============================================================================
print("8. Generating Beeswarm Plot...")
try:
    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_beeswarm_plot.png", bbox_inches='tight')
    plt.close()
    print("   Saved shap_beeswarm_plot.png")
except Exception as e:
    print(f"   Error generating beeswarm plot: {e}")

# SUMMARY
# ============================================================================
print("\nAnalysis Complete. Plots saved to 'shap_plots' directory.")