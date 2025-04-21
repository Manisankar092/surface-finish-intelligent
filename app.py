import os
import traceback
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib

# Force CPU usage and suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Log where TensorFlow is running
print("TensorFlow running on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

# Flask app setup
app = Flask(__name__)
app.secret_key = "my_secret_key"

# Model and scaler paths
model_path = os.path.join(app.root_path, "model_v2")  # Update to your SavedModel directory
scaler_path = os.path.join(app.root_path, "scaler.pkl")

# Load model and scaler (Only load once to save memory)
def load_model_and_scaler():
    try:
        # Load model in SavedModel format
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        print("Error loading model/scaler:", traceback.format_exc())
        return None, None

expected_columns = ["Speed", "Feed", "DOC"]
users = {}

@app.route('/about')
def about():
    return render_template('about.html')



@app.route("/predict", methods=["GET", "POST"])
def predict():
    if 'user' not in session:
        flash("You need to login first.", "warning")
        return redirect(url_for('login'))

    predictions = None
    if request.method == "POST":
        file = request.files.get("excel_file")
        if file:
            try:
                df = pd.read_excel(file)
                df.columns = df.columns.str.strip()

                if not all(col in df.columns for col in expected_columns):
                    flash("The uploaded file is missing required columns!", "danger")
                    return redirect(url_for('predict'))

                input_data = df[expected_columns]
                X_scaled = scaler.transform(input_data)
                y_pred = model.predict(X_scaled).flatten()
                df["Predicted Surface Finish"] = np.round(y_pred, 3)
                predictions = df.to_dict(orient="records")

                generate_graphs(df)

            except Exception as e:
                print("Error during prediction:", traceback.format_exc())
                flash("Something went wrong while processing the file.", "danger")
                return redirect(url_for('predict'))

    return render_template("predict.html", predictions=predictions)

def generate_graphs(df):
    static_path = os.path.join(app.root_path, 'static')

    # Example Loss Curve
    plt.figure()
    plt.plot(np.arange(100), np.random.random(100))
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(static_path, 'loss_curve.png'))

    # MAE Curve
    plt.figure()
    plt.plot(np.arange(100), np.random.random(100))
    plt.title("MAE Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.savefig(os.path.join(static_path, 'mae_curve.png'))

    # Actual vs Predicted
    plt.figure()
    plt.plot(df["Predicted Surface Finish"], label='Predicted', color='blue')
    plt.title("Actual vs Predicted Surface Finish")
    plt.xlabel("Sample")
    plt.ylabel("Surface Finish")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(static_path, 'actual_vs_predicted.png'))

    # Residual Plot
    plt.figure()
    residuals = np.random.randn(len(df))  # Replace with actual residuals if available
    plt.scatter(np.arange(len(residuals)), residuals, alpha=0.6)
    plt.title("Residual Plot")
    plt.xlabel("Sample")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.savefig(os.path.join(static_path, 'residual_plot.png'))


@app.route("/health")
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
