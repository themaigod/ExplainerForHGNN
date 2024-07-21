import torch
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score as roc_auc_sklearn
from sklearn.metrics import confusion_matrix as confusion_matrix_sklearn


def accuracy(logits: torch.Tensor, labels: torch.Tensor):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    return (prediction == labels).sum() / len(prediction)


def MacroF1(logits: torch.Tensor, labels: torch.Tensor):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    return f1_score(labels, prediction, average='macro')


def MicroF1(logits: torch.Tensor, labels: torch.Tensor):
    return accuracy(logits, labels)


def roc_auc_score(logits: torch.Tensor, labels: torch.Tensor):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    return roc_auc_sklearn(labels, prediction)


def confusion_matrix(logits: torch.Tensor, labels: torch.Tensor):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    return confusion_matrix_sklearn(labels, prediction)


node_classification_support_metrics = {
    'accuracy': accuracy,
    'Macro-F1': MacroF1,
    'Micro-F1': MicroF1,
    'roc_auc_score': roc_auc_score,
    'confusion_matrix': confusion_matrix
}
