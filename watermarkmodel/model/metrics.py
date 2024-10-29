import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(x: np.ndarray, y: np.ndarray):
    cm = confusion_matrix(x, y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['non_watermark', 'watermark'])

    #compute tp, tp_and_fn and tp_and_fp w.r.t all classes
    tp_and_fn = cm.sum(1)
    tp_and_fp = cm.sum(0)
    tp = cm.diagonal()

    precision = tp / tp_and_fp
    recall = tp / tp_and_fn
    return disp, precision, recall