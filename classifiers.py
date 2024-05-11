from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the classifiers
nb_classifier = MultinomialNB()
logistic_classifier = LogisticRegression(max_iter=1000)
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

def train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test, feature_type):
    """
    Train and evaluate the classifier.
    :param classifier: The classifier to be trained and evaluated.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Testing features.
    :param y_test: Testing labels.
    :param feature_type: Type of feature (TF or TF-IDF).
    :return: y_pred.
    """
    print(f"Training {classifier.__class__.__name__} with {feature_type} features...")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{classifier.__class__.__name__} {feature_type} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    return y_pred

def calculate_performance_metrics(y_true, y_pred):
    """
    Calculate performance metrics for the classifier.
    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Tuple of accuracy, true positive rate (TPR), and false positive rate (FPR).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn)  # True Positive Rate (Sensitivity)
    fpr = fp / (fp + tn)  # False Positive Rate
    return accuracy, tpr, fpr
