import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from prepare_data import load_rfm_splits

X_train, X_test, y_train, y_test = load_rfm_splits()

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()

print("Classification Report")
print(classification_report(y_test, y_pred,
                            target_names=[f"Cluster {i}" for i in range(4)]))

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels([f"C{i}" for i in range(4)])
ax.set_yticklabels([f"C{i}" for i in range(4)])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Logistic Regression Confusion Matrix (acc={accuracy:.4f})")
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig("src/figures/logistic_regression.png",
            dpi=150, bbox_inches="tight")
plt.show()

print(f"\nTest accuracy: {accuracy:.4f}")

joblib.dump(model, "models/logistic_regression.pkl")
print("Model saved to models/logistic_regression.pkl")
