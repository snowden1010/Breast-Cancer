def evaluate_model(model,model_label, x_testing, y_true):

    """
    
    """
    preds = model.predict(x_testing)
    model_accuracy_score = accuracy_score(y_testing, y_true)
    model_precision_score = precision_score(y_testing, y_true)
    model_recall_score = recall_score(y_testing, y_true)
    model_f1_score = f1_score(y_testing, y_true)
    return {
    "model" : model_label, 
    "accuracy" : f"{model_accuracy_score * 100:.3f} %",
    "recall" : model_recall_score,
    "f1-score" : model_f1_score,
    "precision-score" : model_precision_score
    }

