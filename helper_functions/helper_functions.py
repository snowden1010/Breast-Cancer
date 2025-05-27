from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, model_label, x_testing, y_true):
    """Evaluates a given model's performance and plots its confusion matrix.

    This function calculates and displays key classification metrics: accuracy,
    recall, f1-score, and precision. It also generates and shows a heatmap
    of the confusion matrix for visual inspection of model predictions.

    Args:
        model: The trained machine learning model object. This object should
            have a `predict()` method that takes `x_testing` as input.
            Examples include scikit-learn classifiers (e.g., LogisticRegression,
            RandomForestClassifier).
        model_label (str): A descriptive label for the model being evaluated
            (e.g., "Logistic Regression Model", "SVM Classifier"). This label
            will be used in the output profile and confusion matrix title.
        x_testing (pd.DataFrame or np.ndarray): The feature dataset used for
            testing the model. This should be the independent variables on which
            the model will make predictions.
        y_true (pd.Series or np.ndarray): The true target labels corresponding
            to `x_testing`. This represents the actual outcomes for comparison
            with the model's predictions.

    Returns:
        dict: A dictionary containing the model's performance profile with the
        following keys:
            - "model" (str): The `model_label` provided.
            - "accuracy" (str): The accuracy score formatted as a percentage.
            - "recall" (str): The recall score formatted as a percentage.
            - "f1-score" (str): The f1-score formatted as a percentage.
            - "precision-score" (str): The precision score formatted as a percentage.
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

