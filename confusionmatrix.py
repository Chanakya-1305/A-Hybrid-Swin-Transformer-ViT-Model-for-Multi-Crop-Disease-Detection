import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14,12))
sns.heatmap(cm,
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes,
            cmap="Blues",
            fmt="d")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Crop Disease Model")
plt.show()