import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
import numpy as np
import joblib

MODEL_PATH = "decision_tree_model.joblib"


def load_data(csv_path="creditcard.csv"):
    """Load dataset from CSV path."""
    data = pd.read_csv(csv_path)
    return data


def resample_data(X, y, random_state=42):
    """Apply SMOTE to balance the dataset and return resampled X and y."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def create_model():
    """Create a DecisionTreeClassifier with the chosen hyperparameters."""
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        criterion='entropy',
        class_weight='balanced',
        random_state=42
    )
    return model


def train_and_save_model(model, X_train, y_train, model_path=MODEL_PATH):
    """Train the model and save it to disk."""
    print("Training Decision Tree model...")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    return model


def load_model(model_path=MODEL_PATH):
    """Load a saved model from disk, or return None if not found."""
    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path} ...")
        return joblib.load(model_path)
    return None


def get_model(model, X_train, y_train, force_retrain=False, model_path=MODEL_PATH):
    """Return a trained model. Load from disk if available unless force_retrain is True."""
    if not force_retrain:
        loaded = load_model(model_path)
        if loaded is not None:
            return loaded
    return train_and_save_model(model, X_train, y_train, model_path)


def evaluate_and_report(model, X_test, y_test, y_probs=None):
    """Evaluate model and print reports and plots (as in the original script)."""
    y_pred = model.predict(X_test)
    if y_probs is None:
        if hasattr(model, "predict_proba"):
            y_probs = model.predict_proba(X_test)[:, 1]
        else:
            # If model has no predict_proba, use predictions as probabilities (not ideal)
            y_probs = y_pred

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Model Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Detailed classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Decision Tree')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc_score = auc(fpr, tpr)

    print(f"\nAUC Score: {auc_score:.4f}")

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Decision Tree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== Top 10 Most Important Features ===")
    print(feature_importance.head(10))

    # Plot Feature Importance
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances - Decision Tree')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Plot partial decision tree (first few levels only)
    plt.figure(figsize=(20, 10))
    plot_tree(model,
              feature_names=X.columns,
              class_names=['Normal', 'Fraud'],
              filled=True,
              max_depth=3,  # Show only first 3 levels for readability
              fontsize=8)
    plt.title('Decision Tree Visualization (First 3 Levels)')
    plt.tight_layout()
    plt.show()

    # Tree depth and complexity analysis
    print("\n=== Decision Tree Structure ===")
    print(f"Tree depth: {model.get_depth()}")
    print(f"Number of leaves: {model.get_n_leaves()}")
    print(f"Number of nodes: {model.tree_.node_count}")

    # Model parameters summary
    print("\n=== Decision Tree Model Parameters ===")
    print(f"Max depth: {model.max_depth}")
    print(f"Min samples split: {model.min_samples_split}")
    print(f"Min samples leaf: {model.min_samples_leaf}")
    print(f"Max features: {model.max_features}")
    print(f"Criterion: {model.criterion}")
    print(f"Class weight: {model.class_weight}")

    # Decision rules for top features (first few rules)
    print("\n=== Sample Decision Rules ===")
    tree_rules = export_text(model, feature_names=list(X.columns), max_depth=3)
    print("First few decision rules (depth 3):")
    print(tree_rules[:1000] + "...")

    # Feature splits analysis
    print("\n=== Tree Split Information ===")
    print(f"Total number of features used in tree: {np.sum(model.feature_importances_ > 0)}")
    print(f"Features with importance > 0.01: {np.sum(model.feature_importances_ > 0.01)}")

    # Prediction distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_probs[y_test == 0], bins=30, alpha=0.7, label='Normal Transactions', density=True)
    plt.hist(y_probs[y_test == 1], bins=30, alpha=0.7, label='Fraud Transactions', density=True)
    plt.xlabel('Prediction Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Probabilities - Decision Tree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Leaf node distribution
    leaves_samples = model.tree_.n_node_samples[model.tree_.children_left == -1]
    plt.figure(figsize=(10, 6))
    plt.hist(leaves_samples, bins=30, edgecolor='black')
    plt.xlabel('Number of Samples in Leaf Nodes')
    plt.ylabel('Frequency')
    plt.title('Distribution of Samples in Leaf Nodes')
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n=== Model Training/Evaluation Complete ===")
    print(f"Decision Tree has {model.get_n_leaves()} leaf nodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree for Fraud Detection (with model persistence)")
    parser.add_argument("--retrain", action="store_true", help="Force retraining even if a saved model exists")
    parser.add_argument("--csv", default="creditcard.csv", help="Path to the creditcard.csv file")
    args = parser.parse_args()

    # Load the dataset
    data = load_data(args.csv)
    print(data.head())
    print('\n\nRows and columns: ', data.shape)
    print('\n\nColumn names: ', data.columns)
    print('\n\n', data['Class'].value_counts())
    print('Statistical summary: \n\n', data.describe())

    # Data Visualization (amount and class plots)
    plt.figure(figsize=(10, 6))
    data['Amount'].hist(bins=50)
    plt.title("Distribution of Transaction Amounts")
    plt.xlabel("Amount")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Class', y='Amount', data=data)
    plt.title("Transaction Amounts by Class")
    plt.show()

    plt.figure(figsize=(8, 6))
    data['Class'].value_counts().plot(kind='bar')
    plt.title("Class Distribution (0: Normal, 1: Fraud)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()

    # Prepare features and labels
    X = data.drop('Class', axis=1)  # Features
    y = data['Class']  # Labels

    # Apply SMOTE for handling imbalanced dataset
    X_res, y_res = resample_data(X, y, random_state=42)
    print("Original Shape:", X.shape)
    print("Balanced Shape:", X_res.shape)
    print("Original Class Distribution:\n", pd.Series(y).value_counts())
    print("Balanced Class Distribution:\n", pd.Series(y_res).value_counts())

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Create model object
    model = create_model()

    # Get model: load if present unless --retrain is provided
    model = get_model(model, X_train, y_train, force_retrain=args.retrain, model_path=MODEL_PATH)

    # Evaluate and report (will run regardless of whether model was loaded or just trained)
    evaluate_and_report(model, X_test, y_test)
