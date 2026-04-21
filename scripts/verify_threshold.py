import yaml
import numpy as np
from sklearn.metrics import confusion_matrix

y_true = np.load("results/oof_y_true_thresh2.npy")
y_prob = np.load("results/oof_y_prob_thresh2.npy")

with open("configs/pipeline_config.yaml") as f:
    config = yaml.safe_load(f)

CHOSEN_THRESHOLD = config['bcr']['decision_threshold']
print(f"Loaded threshold from config: {CHOSEN_THRESHOLD}")

y_pred = (y_prob >= CHOSEN_THRESHOLD).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
print(f"Threshold: {CHOSEN_THRESHOLD}")
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"Recall (sensitivity): {tp/(tp+fn):.4f}")
print(f"Precision:            {tp/(tp+fp):.4f}")
print(f"FN rate:              {fn/(fn+tp):.4f}  ← fraction of bad channels missed")
print()
print(f"Day 11 reference: TN=17786  FP=382  FN=355  TP=377  (threshold=0.604)")

