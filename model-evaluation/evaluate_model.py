from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import model_testing

def main():
    ## Wrangle prediction outputs
    y_true, y_pred = model_testing.evaluate()
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ## Score your model.
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    print("Precision:", precision)
    recall = recall_score(y_true, y_pred)
    print("Recall:", recall)

    ## Make plots
    heatmap = sns.heatmap(cm, annot=True, cmap="YlGnBu") # Plot confusion matrix
    fig = heatmap.get_figure()
    fig.savefig("confusion_matrix.png")
    p, r, thresholds = precision_recall_curve(y_true, y_pred)
    fig, ax = plt.subplots() # Plot precision-recall curve
    ax.plot(p, r)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    fig.savefig("precision-recall.png")
    accuracy = np.mean(y_true == y_pred)
    print("Accuracy:", accuracy)
    precision = np.mean(y_true == y_pred) / np.mean(y_pred == 1)
    print("Precision:", precision)

if __name__ == '__main__':
    main()