from flask import Flask, render_template, request, redirect, session
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- LOAD MODELS ----------------
rf_model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
bigru_model = load_model("feature_extractor.h5")

# ---------------- LOGIN ----------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    if request.form["username"] == "admin" and request.form["password"] == "admin":
        session["user"] = "admin"
        return redirect("/dashboard")
    return "Invalid Login"

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    return render_template("dashboard.html")

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect("/")

    # -------- INPUT --------
    screen = float(request.form['screen'])
    session_time = float(request.form['session'])
    night = float(request.form['night'])
    brightness = float(request.form['brightness'])
    dark = int(request.form['dark'])
    gender = int(request.form['gender'])
    ambient = float(request.form['ambient'])

    # -------- STRICT VALIDATION --------
    if not (0 <= screen <= 24):
        return "Screen time must be between 0–24 hours"
    if not (0 <= night <= 11):
        return "Night usage must be between 0–11 hours"
    if not (1 <= brightness <= 10):
        return "Brightness must be between 1–10"
    if not (1 <= ambient <= 10):
        return "Ambient light must be between 1–10"

    # -------- FEATURE ENGINEERING --------
    session_hour = session_time / 60
    interaction_feature = screen * night
    noise = np.random.normal(0, 0.1)

    input_dict = {
        "daily_screen_time": screen,
        "avg_session_duration": session_time,
        "night_usage": night,
        "brightness_level": brightness,
        "dark_mode_usage": dark,
        "Gender": gender,
        "ambient_light": ambient,
        "session_hour": session_hour,
        "interaction_feature": interaction_feature,
        "noise": noise
    }

    input_df = pd.DataFrame([input_dict])

    # -------- FIX FEATURE ORDER --------
    expected_features = scaler.feature_names_in_

    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_features]

    # -------- SCALE --------
    input_scaled = scaler.transform(input_df)

    # -------- BiGRU FEATURE EXTRACTION --------
    input_reshaped = input_scaled.reshape(1, input_scaled.shape[1], 1)
    deep_features = bigru_model.predict(input_reshaped)

    # -------- RANDOM FOREST --------
    pred = rf_model.predict(deep_features)[0]
    prob = rf_model.predict_proba(deep_features)[0][1]

    # -------- 🔥 FINAL DYNAMIC SCORE (FIXED) --------
    ml_score = prob * 100

    rule_score = (
        screen * 2.5 +
        night * 4 +
        brightness * 2 +
        (1 - dark) * 5 +
        ambient * 2 +
        (session_time / 60) * 3
    )

    score = int(np.clip((ml_score * 0.5 + rule_score * 0.5), 0, 100))

    # -------- RESULT --------
    result = "High Risk" if score > 60 else "Low Risk"

    tips = [
    "Follow 20-20-20 rule",
    "Reduce night usage",
    "Lower brightness",
    "Take regular breaks"
]

    return render_template("result.html",
                           result=result,
                           score=score,
                           tips=tips)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)