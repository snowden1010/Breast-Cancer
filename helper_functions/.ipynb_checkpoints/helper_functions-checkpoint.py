from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score



def evaluate_model(model,model_label, x_testing, y_true):

    """
    
    """
    y_pred = model.predict(x_testing)
    return {
    "model" : model_label, 
    "accuracy" : f"{accuracy_score(y_true, y_pred) * 100:.3f} %",
    "recall" : f"{recall_score(y_true, y_pred) * 100 :.3f} %",
    "f1-score" : f"{f1_score(y_true, y_pred) * 100 :.3f} %",
    "precision-score" : f"{precision_score(y_true, y_pred) * 100 :.3f} %",
    }

