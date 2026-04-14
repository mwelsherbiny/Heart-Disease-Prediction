import matplotlib.pyplot as plt
from pprint import pprint

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_validate

def evaluate_model(model, x, y):
    y_pred = model.predict(x)

    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"Accuracy        : {acc:.4f}")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Greens)
    plt.title("Confusion Matrix")
    plt.show()


def cross_validate_model(model, x, y):
    scores = cross_validate(
        model,
        x,
        y,
        cv=10,
        scoring={
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
    )

    print(f"CV Mean Accuracy: {scores['test_accuracy'].mean():.4f}")
    print(f"CV Mean Precision: {scores['test_precision'].mean():.4f}")
    print(f"CV Mean Recall: {scores['test_recall'].mean():.4f}")
    print(f"CV Mean F1 Score: {scores['test_f1'].mean():.4f}")


def print_search_cv_metrics(model_search):
    best_index = model_search.best_index_
    
    print(f"CV Mean Accuracy: {model_search.cv_results_['mean_test_accuracy'][best_index]:.4f}")
    print(f"CV Mean Precision: {model_search.cv_results_['mean_test_precision'][best_index]:.4f}")
    print(f"CV Mean Recall: {model_search.cv_results_['mean_test_recall'][best_index]:.4f}")
    print(f"CV Mean F1 Score: {model_search.cv_results_['mean_test_f1'][best_index]:.4f}")

    print("\nBest Parameters:")
    pprint(model_search.best_params_)