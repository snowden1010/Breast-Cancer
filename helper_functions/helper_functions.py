from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model,model_label, x_testing, y_true):

    """
    
    """
    y_pred = model.predict(x_testing)
    profile = {
    "model" : model_label, 
    "accuracy" : f"{accuracy_score(y_true, y_pred) * 100:.3f} %",
    "recall" : f"{recall_score(y_true, y_pred) * 100 :.3f} %",
    "f1-score" : f"{f1_score(y_true, y_pred) * 100 :.3f} %",
    "precision-score" : f"{precision_score(y_true, y_pred) * 100 :.3f} %",
    }
    conf_mat = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(data=conf_mat, annot=True, cmap="viridis")
    plt.title(f"Confusion Matrix {model_label}", fontsize=20)
    plt.xlabel("Predicted Label", fontsize=16, c="c")
    plt.ylabel("Actual Label", fontsize=16, c="g")
    plt.show()
    return profile

