import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(x: np.ndarray, y: np.ndarray):
    cm = confusion_matrix(x, y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['non_watermark', 'watermark'])
    disp.savefig('confmatrix.png')
    return disp.plot()