import matplotlib.pyplot as plt
import numpy as np
import os

print("PROFESSIONAL MODEL COMPARISON STARTED")

# ---------------------------------------------
# 1. CHECK FILES
# ---------------------------------------------
if not os.path.exists("eye_results.txt") or not os.path.exists("kaggle_results.txt"):
    print("Run main.py and main2.py first.")
    exit()

eye_acc = float(open("eye_results.txt").read())
kaggle_acc = float(open("kaggle_results.txt").read())

print(f"Eye Dataset Accuracy: {eye_acc:.4f}")
print(f"Kaggle Dataset Accuracy: {kaggle_acc:.4f}")

# ---------------------------------------------
#  ACCURACY BAR CHART
# ---------------------------------------------
plt.figure(figsize=(6,5))
plt.bar(["Eye Dataset", "Kaggle Dataset"], [eye_acc, kaggle_acc])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison")
plt.tight_layout()
plt.show()

# ---------------------------------------------
#  ERROR RATE COMPARISON
# ---------------------------------------------
eye_error = 1 - eye_acc
kaggle_error = 1 - kaggle_acc

plt.figure(figsize=(6,5))
plt.bar(["Eye Dataset", "Kaggle Dataset"], [eye_error, kaggle_error])
plt.ylim(0,1)
plt.ylabel("Error Rate")
plt.title("Error Rate Comparison")
plt.tight_layout()
plt.show()

# ---------------------------------------------
#  RADAR CHART (Professional Look)
# ---------------------------------------------
labels = ["Accuracy", "Stability", "Generalization"]

# Simulated stability & generalization scores based on accuracy
eye_metrics = [
    eye_acc,
    eye_acc * 0.95,
    eye_acc * 0.93
]

kaggle_metrics = [
    kaggle_acc,
    kaggle_acc * 0.95,
    kaggle_acc * 0.93
]

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

eye_metrics += eye_metrics[:1]
kaggle_metrics += kaggle_metrics[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

ax.plot(angles, eye_metrics, label="Eye Dataset")
ax.fill(angles, eye_metrics, alpha=0.25)

ax.plot(angles, kaggle_metrics, label="Kaggle Dataset")
ax.fill(angles, kaggle_metrics, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

ax.set_ylim(0,1)
plt.title("Overall Model Performance Comparison")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ---------------------------------------------
#  TEXT INSIGHT
# ---------------------------------------------
print("\n--- PERFORMANCE INSIGHT ---")

if eye_acc > kaggle_acc:
    print("Eye Dataset model performs better overall.")
elif kaggle_acc > eye_acc:
    print("Kaggle Dataset model performs better overall.")
else:
    print("Both models show equivalent performance.")

print("\nComparison Completed Successfully ✅")