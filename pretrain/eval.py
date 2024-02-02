import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, f1_score

def classify(model, device, dataset, batch_size=128):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    y_trues = np.empty((0, len(dataset.CLASSES)))
    y_preds = np.empty((0, len(dataset.CLASSES)))

    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)

            y_pred = torch.sigmoid(y_hat).cpu().numpy().round()
            y = y.cpu()

            y_preds = np.concatenate((y_preds, y_pred), axis=0)
            y_trues = np.concatenate((y_trues, y), axis=0)

    return y_trues, y_preds

def get_f1(y_trues, y_preds):
    f1 = []
    for j in range(y_trues.shape[1]):
        f1.append(f1_score(y_trues[:, j], y_preds[:, j]))
    return np.array(f1)

def get_auprc(y_trues, y_scores):
    auprc = []
    for j in range(y_trues.shape[1]):
        p, r, thresholds = precision_recall_curve(y_trues[:, j], y_scores[:, j])
        auprc.append(auc(r, p))

    return np.array(auprc)

