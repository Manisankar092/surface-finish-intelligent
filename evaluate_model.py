import matplotlib.pyplot as plt
import numpy as np
import shap
import os

def evaluate_model(model, X_test, y_test, history, feature_names):
    # Predictions
    y_pred = model.predict(X_test).flatten()

    # 1. Actual vs Predicted Surface Finish
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel("Actual Surface Finish")
    plt.ylabel("Predicted Surface Finish")
    plt.title("Actual vs Predicted Surface Finish")
    plt.tight_layout()
    plt.savefig("static/plots/actual_vs_pred.png")
    plt.close()

    # 2. Training vs Validation Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/plots/loss_curve.png")
    plt.close()

    # 3. MAE Curve
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.title("Training vs Validation MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig("static/plots/mae_curve.png")
    plt.close()

    # 4. Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Surface Finish")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig("static/plots/residuals.png")
    plt.close()
