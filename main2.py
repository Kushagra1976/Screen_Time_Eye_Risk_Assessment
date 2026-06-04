import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import seaborn as sns

print("KAGGLE HYBRID MODEL STARTED")

# ---------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------
df = pd.read_csv("Kaggle_dataset.csv")

# Drop ID column
df = df.drop("User_ID", axis=1)

# ---------------------------------------------
# 2. ENCODE CATEGORICAL FEATURES
# ---------------------------------------------
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["Location"] = le.fit_transform(df["Location"])

# ---------------------------------------------
# 3. CREATE STRONG STRUCTURED TARGET
# ---------------------------------------------
np.random.seed(42)

score = (
    0.6 * df["Daily_Screen_Time_Hours"] +
    0.4 * df["Social_Media_Usage_Hours"] +
    0.3 * df["Gaming_App_Usage_Hours"] +
    0.2 * df["Total_App_Usage_Hours"] +
    0.15 * df["Number_of_Apps_Used"] +
    0.1 * (df["Daily_Screen_Time_Hours"] * df["Gaming_App_Usage_Hours"])
)

# Add small controlled noise
score += np.random.normal(0, 0.05 * score.std(), len(score))

# Normalize safely
score_min = score.min()
score_max = score.max()
score_norm = (score - score_min) / (score_max - score_min + 1e-9)

# Manual classification
eye_strain = np.zeros(len(score_norm))
eye_strain[score_norm > 0.30] = 1
eye_strain[score_norm > 0.65] = 2

df["Eye_Strain_Level"] = eye_strain

# ---------------------------------------------
# 4. SPLIT FEATURES & TARGET
# ---------------------------------------------
X = df.drop("Eye_Strain_Level", axis=1)
y = df["Eye_Strain_Level"]

# ---------------------------------------------
# 5. SCALE FEATURES
# ---------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------
# 6. TRAIN TEST SPLIT (70-30)
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# ---------------------------------------------
# 7. BiGRU MODEL
# ---------------------------------------------
X_train_r = np.expand_dims(X_train, axis=2)
X_test_r = np.expand_dims(X_test, axis=2)

input_layer = Input(shape=(X_train_r.shape[1], 1))

x = Bidirectional(GRU(80))(input_layer)
x = Dropout(0.3)(x)
feature_layer = Dense(40, activation='relu')(x)
output_layer = Dense(3, activation='softmax')(feature_layer)

bigru_model = Model(inputs=input_layer, outputs=output_layer)

bigru_model.compile(
    optimizer=Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

bigru_model.fit(
    X_train_r,
    y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# ---------------------------------------------
# 8. FEATURE EXTRACTION
# ---------------------------------------------
feature_extractor = Model(
    inputs=bigru_model.input,
    outputs=feature_layer
)

train_features = feature_extractor.predict(X_train_r)
test_features = feature_extractor.predict(X_test_r)

# ---------------------------------------------
# 9. RANDOM FOREST CLASSIFIER
# ---------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    random_state=42
)

rf_model.fit(train_features, y_train)

y_pred = rf_model.predict(test_features)

accuracy = accuracy_score(y_test, y_pred)

print("\nKaggle Dataset Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save accuracy for comparison
with open("kaggle_results.txt", "w") as f:
    f.write(str(accuracy))

# ---------------------------------------------
# 10. CONFUSION MATRIX
# ---------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Kaggle Hybrid")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 11. ROC CURVE
# ---------------------------------------------
y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_score = rf_model.predict_proba(test_features)

plt.figure(figsize=(7,6))
for i in range(3):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f}')

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Kaggle Hybrid")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 12. t-SNE VISUALIZATION (SAFE)
# ---------------------------------------------
tsne_input = test_features + 1e-6

tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(tsne_input)

plt.figure(figsize=(7,6))
sns.scatterplot(
    x=tsne_features[:,0],
    y=tsne_features[:,1],
    hue=y_test,
    palette="deep"
)
plt.title("t-SNE Visualization - Kaggle Hybrid")
plt.tight_layout()
plt.show()

print("\nKAGGLE MODEL COMPLETED SUCCESSFULLY ✅")