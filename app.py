from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "my_secret_key"  # Needed for flash and session handling

# Load model and scaler (Ensure the correct path to the model and scaler)
model_path = os.path.join(app.root_path, "model.h5")
scaler_path = os.path.join(app.root_path, "scaler.pkl")

# Check if model and scaler exist
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
   
   raise FileNotFoundError("Model or scaler file not found. Please ensure they are available.")

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Expected columns in Excel
expected_columns = ["Speed", "Feed", "DOC"]

# Simulated in-memory user storage (for demonstration)
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
        
        # Check if the user exists and the password is correct
        if users.get(email) == password:
            session['user'] = email  # Store email in session after successful login
            flash("Login successful!", "success")
            return redirect(url_for('predict'))  # Redirect to prediction page
        else:
            flash("Invalid email or password!", "danger")
            return redirect(url_for('login'))  # Redirect back to login page

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
        session['user'] = email  # Store user email in session upon signup
        flash("Account created successfully!", "success")
        return redirect(url_for("predict"))

    return render_template("signup.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if 'user' not in session:
        flash("You need to login first.", "warning")
        return redirect(url_for('login'))  # Redirect to login if no session exists
    
    predictions = None
    if request.method == "POST":
        file = request.files.get("excel_file")
        if file:
            try:
                df = pd.read_excel(file)
                df.columns = df.columns.str.strip()  # Ensure no leading/trailing spaces in column names

                # Ensure the file has the required columns
                if not all(col in df.columns for col in expected_columns):
                    flash("The uploaded file is missing required columns!", "danger")
                    return redirect(url_for('predict'))

                input_data = df[expected_columns]
                X_scaled = scaler.transform(input_data)
                y_pred = model.predict(X_scaled).flatten()
                df["Predicted Surface Finish"] = np.round(y_pred, 3)
                predictions = df.to_dict(orient="records")

                # Generate and save graphs
                generate_graphs(df)

            except Exception as e:
                flash(f"Error processing file: {e}", "danger")
                return redirect(url_for('predict'))

    return render_template("predict.html", predictions=predictions)

def generate_graphs(df):
    # Example graph for loss curve (you should replace with actual data logic)
    plt.figure()
    plt.plot(np.arange(100), np.random.random(100))  # Placeholder for actual loss curve data
    plt.title("Loss Curve")
    loss_curve_path = os.path.join(app.root_path, 'static', 'loss_curve.png')
    plt.savefig(loss_curve_path)

    # Example graph for residual plot (you should replace with actual logic)
    plt.figure()
    plt.plot(np.arange(100), np.random.random(100))  # Placeholder for residual data
    plt.title("MAE Curve")
    residual_plot_path = os.path.join(app.root_path, 'static', 'residuals.png')
    plt.savefig(residual_plot_path)

    plt.figure()
    plt.plot(np.arange(100), np.random.random(100))  # Placeholder for actual loss curve data
    plt.title("Actual vs Predicted")
    loss_curve_path = os.path.join(app.root_path, 'static', 'loss_curve.png')
    plt.savefig(loss_curve_path)

    # Example graph for residual plot (you should replace with actual logic)
    plt.figure()
    plt.plot(np.arange(100), np.random.random(100))  # Placeholder for residual data
    plt.title("Residuals Plot")
    residual_plot_path = os.path.join(app.root_path, 'static', 'residuals.png')
    plt.savefig(residual_plot_path)


@app.route("/logout")
def logout():
    session.pop('user', None)  # Clear the session
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))  # Redirect to login page after logout

@app.route("/contact", methods=["POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")
        
        # Process the message, e.g., save to a database or send an email
        
        flash("Your message has been sent successfully!", "success")
        return redirect(url_for("about"))

if __name__ == "__main__":
    app.run(debug=True)
