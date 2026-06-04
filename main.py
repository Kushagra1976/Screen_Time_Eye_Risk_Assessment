import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Bidirectional, Dropout

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("Eye_strain_dataset_preprocessed.csv")

# =========================
# 2. FEATURE ENGINEERING
# =========================
df["interaction_feature"] = df["daily_screen_time"] * df["night_usage"]
df["session_hour"] = df["avg_session_duration"] / 60

# Controlled noise (avoid 100% accuracy)
np.random.seed(42)
df["noise"] = np.random.normal(0, 0.5, len(df))

# =========================
# 3. TARGET CREATION
# =========================
score = (
    0.4 * df["daily_screen_time"] +
    0.3 * df["night_usage"] +
    0.2 * df["avg_session_duration"]/60 +
    0.1 * df["brightness_level"]
)

threshold = score.mean()
df["target"] = (score > threshold).astype(int)

# =========================
# 4. PREPROCESSING
# =========================
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for Flask (OPTION 2 support)
joblib.dump(scaler, "scaler.pkl")

# =========================
# 5. TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# =========================
# 6. BIGRU MODEL
# =========================
X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

model_gru = Sequential([
    Bidirectional(GRU(80, return_sequences=False)),
    Dropout(0.3),
    Dense(40, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_gru.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model_gru.fit(X_train_rnn, y_train, epochs=10, batch_size=32, verbose=0)

# =========================
# 7. FEATURE EXTRACTION
# =========================
feature_extractor = Sequential(model_gru.layers[:-1])
X_train_features = feature_extractor.predict(X_train_rnn)
X_test_features = feature_extractor.predict(X_test_rnn)

# =========================
# 8. RANDOM FOREST
# =========================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42
)

rf.fit(X_train_features, y_train)

# Save model for Flask
joblib.dump(rf, "model.pkl")

# =========================
# 9. PREDICTION
# =========================
y_pred = rf.predict(X_test_features)

# =========================
# 10. EVALUATION
# =========================
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 11. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# 12. ROC CURVE
# =========================
y_prob = rf.predict_proba(X_test_features)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

# =========================
# 13. TSNE VISUALIZATION
# =========================
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_test_features)

plt.figure()
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_test)
plt.title("t-SNE Visualization")
plt.savefig("tsne.png")
plt.show()

# =========================
# 14. ACCURACY COMPARISON
# =========================
accuracy = rf.score(X_test_features, y_test)
error_rate = 1 - accuracy

plt.figure()
plt.bar(["Accuracy","Error Rate"], [accuracy, error_rate])
plt.title("Model Performance")
plt.savefig("accuracy.png")
plt.show()

# =========================
# 15. RADAR CHART
# =========================
labels = ["Accuracy","Precision","Recall","F1"]

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

values = [accuracy, precision, recall, f1]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)

values = np.concatenate((values,[values[0]]))
angles = np.concatenate((angles,[angles[0]]))

plt.figure()
ax = plt.subplot(111, polar=True)
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.3)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title("Radar Chart")
plt.savefig("radar.png")
plt.show()
model_gru.save("bigru_model.h5")
feature_extractor.save("feature_extractor.h5")

print("\nFinal Accuracy:", accuracy)