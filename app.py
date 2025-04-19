import os
import traceback

# Force CPU usage and suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir='
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

print("TensorFlow running on:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")

app = Flask(__name__)
app.secret_key = "my_secret_key"

# Load model and scaler
model_path = os.path.join(app.root_path, "model.h5")
scaler_path = os.path.join(app.root_path, "scaler.pkl")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler file not found. Please ensure they are available.")

model = tf.keras.models.load_model(model_path)
scaler = joblib.load(scaler_path)

expected_columns = ["Speed", "Feed", "DOC"]
users = {}

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if users.get(email) == password:
            session['user'] = email
            flash("Login successful!", "success")
            return redirect(url_for('predict'))
        else:
            flash("Invalid email or password!", "danger")
            return redirect(url_for('login'))
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if not email or not password or not confirm_password:
            flash("Please fill in all fields!", "danger")
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(url_for('signup'))

        if email in users:
            flash("Email already exists. Please login.", "warning")
            return redirect(url_for('login'))

        users[email] = password
        session['user'] = email
        flash("Account created successfully!", "success")
        return redirect(url_for("predict"))
    return render_template("signup.html")

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
    # Graph 1: Placeholder for Loss Curve
    plt.figure()
    plt.plot(np.arange(100), np.random.random(100))
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(app.root_path, 'static', 'loss_curve.png'))
    plt.close()

    # Graph 2: Placeholder for Actual vs Predicted
    plt.figure()
    plt.plot(df["Predicted Surface Finish"], label="Predicted")
    plt.title("Predicted Surface Finish")
    plt.xlabel("Sample")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(app.root_path, 'static', 'predicted.png'))
    plt.close()

    # Graph 3: Residual Plot
    residuals = np.random.normal(0, 0.1, len(df))  # Placeholder residuals
    plt.figure()
    plt.scatter(np.arange(len(residuals)), residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Residuals Plot")
    plt.xlabel("Sample")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(os.path.join(app.root_path, 'static', 'residuals.png'))
    plt.close()

@app.route("/logout")
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

@app.route("/contact", methods=["POST"])
def contact():
    name = request.form.get("name")
    email = request.form.get("email")
    message = request.form.get("message")
    flash("Your message has been sent successfully!", "success")
    return redirect(url_for("about"))

# Render health check
@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(debug=True)
