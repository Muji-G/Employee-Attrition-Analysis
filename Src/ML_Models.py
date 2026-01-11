"""
This file contains all machine learning–related logic:
- train/test split
- model training
- evaluation
- feature importance
- ROC and Precision–Recall curves
"""

import pandas as pd
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

# Preparing train / test data
def prepare_data(df, target_col="Attrition", test_size=0.3, stratify=True, scale=True):
    """Splits data into train and test sets.Optionally applies feature scaling."""

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    stratify_col = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_col
    )

    # Scale features (important for Logistic Regression)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train = X_train.values
        X_test = X_test.values

    return X_train, X_test, y_train, y_test, X.columns


# Model training functions
def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(
        solver="liblinear",
        penalty="l1",
        max_iter=1000
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    """Train Decision Tree model."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None):
    """Train Random Forest model."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# Model evaluation
def evaluate(model, X_train, X_test, y_train, y_test):
    """ Prints confusion matrix, accuracy, and classification report for train and test sets."""

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"\n{model.__class__.__name__} (TRAIN)")
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_train_pred))
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print(
        pd.DataFrame(
            classification_report(y_train, y_train_pred, output_dict=True)
        )
    )

    print(f"\n{model.__class__.__name__} (TEST)")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print(
        pd.DataFrame(
            classification_report(y_test, y_test_pred, output_dict=True)
        )
    )


# Feature importance / coefficients
def show_feature_importance(model, feature_names):
    """Displays feature importance for tree models or coefficients for linear models."""

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        print(f"\n=== {model.__class__.__name__} Feature Importance ===")
        print(importance)

    elif hasattr(model, "coef_"):
        coef = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": model.coef_[0]
        }).sort_values(by="Coefficient", key=abs, ascending=False)

        print(f"\n=== {model.__class__.__name__} Coefficients ===")
        print(coef)


# Precision–Recall Curve (Plotly)
def plot_pr_curve_plotly(model, X_test, y_test, model_name="Model"):
    """Interactive Precision–Recall curve using probabilities."""

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=precisions[:-1],
        mode="lines",
        name="Precision"
    ))
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=recalls[:-1],
        mode="lines",
        name="Recall"
    ))

    fig.update_layout(
        title=f"Precision–Recall vs Threshold ({model_name})",
        xaxis_title="Threshold",
        yaxis_title="Score",
        template="plotly_white"
    )
    fig.show()


# ROC Curve (Plotly)
def plot_roc_curve_plotly(model, X_test, y_test, model_name="Model"):
    """Interactive ROC curve using probabilities."""

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_scores)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name="ROC Curve"
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line=dict(dash="dash"),
        name="Random Guess"
    ))

    fig.update_layout(
        title=f"ROC Curve ({model_name})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="plotly_white"
    )
    fig.show()
