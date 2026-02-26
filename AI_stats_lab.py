"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Split into train and test (80-20).
            Use random_state=42.
    STEP 3: Standardize features using StandardScaler.
            IMPORTANT:
            - Fit scaler only on X_train
            - Transform both X_train and X_test
    STEP 4: Train LinearRegression model.
    STEP 5: Compute:
            - train_mse
            - test_mse
            - train_r2
            - test_r2
    STEP 6: Identify indices of top 3 features
            with largest absolute coefficients.

    RETURN:
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices (list length 3)
    """
    
    # STEP 1: Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # STEP 2: Split into train and test (80-20) with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # STEP 3: Standardize features using StandardScaler
    # IMPORTANT: Fit scaler only on X_train, then transform both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # STEP 4: Train LinearRegression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # STEP 5: Compute metrics
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # STEP 6: Identify top 3 features with largest absolute coefficients
    # Get absolute coefficients and find indices of top 3
    top_3_feature_indices = np.argsort(np.abs(model.coef_))[-3:][::-1].tolist()
    
    # COMMENTS:
    # 1. Does the model overfit?
    #    Compare train vs test metrics:
    #    - If test_mse >> train_mse or test_r2 << train_r2, model is overfitting
    #    - If they're similar, the model generalizes well
    #    - Typically train metrics are slightly better than test metrics
    #
    # 2. Why is feature scaling important for linear regression?
    #    - Linear Regression does NOT strictly require scaling, but scaling helps:
    #    - Prevents features with larger scales from dominating
    #    - Improves numerical stability
    #    - Makes gradient descent converge faster (useful for other algorithms)
    #    - Makes coefficient interpretation more uniform
    #    - Particularly important when using regularization (Ridge, Lasso)
    
    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Standardize entire dataset (after splitting is NOT needed for CV,
            but use pipeline logic manually).
    STEP 3: Perform 5-fold cross-validation
            using LinearRegression.
            Use scoring='r2'.

    STEP 4: Compute:
            - mean_r2
            - std_r2

    RETURN:
        mean_r2,
        std_r2
    """
    
    # STEP 1: Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # STEP 2: Standardize entire dataset
    # For cross-validation, we standardize the full dataset
    # (CV will handle the internal train-test splits)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # STEP 3: Perform 5-fold cross-validation with LinearRegression
    model = LinearRegression()
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    # STEP 4: Compute mean and standard deviation of R² scores
    mean_r2 = cv_scores.mean()
    std_r2 = cv_scores.std()
    
    # COMMENTS:
    # 1. What does the standard deviation of CV scores represent?
    #    - The standard deviation measures the variability of model performance
    #    - across different folds.
    #    - Low std: Model performs consistently across all folds (good generalization)
    #    - High std: Model performance varies significantly (potential overfitting
    #      on some folds or high variance in predictions)
    #
    # 2. How does cross-validation help reduce variance risk?
    #    - CV uses multiple train-test splits, reducing luck/randomness
    #    - Provides multiple performance estimates instead of just one
    #    - More reliable estimate of true model performance
    #    - Reduces bias in model evaluation
    #    - Better utilizes limited data for both training and evaluation
    #
    # 3. Compare CV mean R² with test R² from Q1 - are they similar? Why?
    #    - They should be reasonably similar (typically within 0.05-0.1)
    #    - If CV mean R² > test R² from Q1: The specific test split might be harder
    #    - If CV mean R² < test R² from Q1: The specific test split might be easier
    #    - CV mean is more robust because it averages over multiple splits
    #    - CV provides better estimate of true generalization performance
    
    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
            Use random_state=42.
    STEP 3: Standardize features.
    STEP 4: Train LogisticRegression(max_iter=5000).
    STEP 5: Compute:
            - train_accuracy
            - test_accuracy
            - precision
            - recall
            - f1
            - confusion matrix (optional to compute but not return)

    In comments:
        Explain what a False Negative represents medically.

    RETURN:
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    """
    
    # STEP 1: Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # STEP 2: Split into train-test (80-20) with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # STEP 3: Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # STEP 4: Train LogisticRegression with max_iter=5000
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    
    # STEP 5: Compute predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Compute metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    # Compute confusion matrix (not returned but informative for analysis)
    cm = confusion_matrix(y_test, y_test_pred)
    # cm structure: [[TN, FP],
    #                [FN, TP]]
    
    # COMMENTS:
    # Medical Context - What does a False Negative represent?
    # - False Negative (FN): Patient HAS cancer, but model predicts NO cancer
    # - Why is it dangerous?
    #   1. Patient is not treated and cancer goes undetected
    #   2. Cancer can progress to advanced stages without intervention
    #   3. Reduces patient survival chances
    #   4. Medical cost vastly increases if caught later
    #   5. In medical diagnosis, FN is often more critical than FP
    #      (False Positive causes unnecessary but manageable anxiety/testing)
    # - This is why Recall (sensitivity) is crucial for medical diagnosis:
    #   Recall = TP / (TP + FN) measures ability to catch actual cases
    
    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
    STEP 3: Standardize features.
    STEP 4: For C in [0.01, 0.1, 1, 10, 100]:
            - Train LogisticRegression(max_iter=5000, C=value)
            - Compute train accuracy
            - Compute test accuracy

    STEP 5: Store results in dictionary:
            {
                C_value: (train_accuracy, test_accuracy)
            }

    In comments:
        - What happens when C is very small?
        - What happens when C is very large?
        - Which case causes overfitting?

    RETURN:
        results_dictionary
    """
    
    # STEP 1: Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # STEP 2: Split into train-test (80-20) with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # STEP 3: Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # STEP 4 & 5: Test different C values and store results
    C_values = [0.01, 0.1, 1, 10, 100]
    results = {}
    
    for C in C_values:
        # Train LogisticRegression with current C value
        # C is the inverse of regularization strength (C = 1/λ)
        model = LogisticRegression(max_iter=5000, C=C, random_state=42)
        model.fit(X_train, y_train)
        
        # Compute accuracies
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        # Store in dictionary
        results[C] = (train_acc, test_acc)
    
    # COMMENTS:
    # Understanding C parameter (C = 1/λ, where λ is regularization strength):
    #
    # 1. What happens when C is very small? (e.g., C = 0.01)
    #    - High regularization strength (strong penalty on coefficients)
    #    - Large λ means model coefficients are heavily constrained
    #    - Model becomes simpler (lower complexity)
    #    - Underfitting risk: Model may be too simple to learn patterns
    #    - Both train and test accuracy may be lower
    #    - But more stable and generalizable
    #
    # 2. What happens when C is very large? (e.g., C = 100)
    #    - Low regularization strength (weak penalty on coefficients)
    #    - Small λ means model coefficients can grow large
    #    - Model becomes more complex (fits training data more tightly)
    #    - Overfitting risk: Model memorizes noise in training data
    #    - Train accuracy >> test accuracy (large gap)
    #    - Test accuracy may not improve significantly
    #
    # 3. Which C value(s) might lead to overfitting? Underfitting?
    #    - Overfitting: Large C (e.g., 100) - if train >> test accuracy
    #    - Underfitting: Small C (e.g., 0.01) - if both accuracies are low
    #    - Optimal: Usually C = 1 or where gap (train - test) is smallest
    #
    # 4. Based on results, optimal C:
    #    - Look for C value with highest test accuracy (generalization)
    #    - But also consider stability (train and test close together)
    #    - Typically find sweet spot where test accuracy peaks
    
    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Standardize entire dataset.
    STEP 3: Perform 5-fold cross-validation
            using LogisticRegression(C=1, max_iter=5000).
            Use scoring='accuracy'.

    STEP 4: Compute:
            - mean_accuracy
            - std_accuracy

    In comments:
        Explain why cross-validation is especially
        important in medical diagnosis problems.

    RETURN:
        mean_accuracy,
        std_accuracy
    """
    
    # STEP 1: Load breast cancer dataset
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # STEP 2: Standardize entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # STEP 3: Perform 5-fold cross-validation
    model = LogisticRegression(C=1, max_iter=5000, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    # STEP 4: Compute mean and standard deviation of accuracy scores
    mean_accuracy = cv_scores.mean()
    std_accuracy = cv_scores.std()
    
    # COMMENTS:
    # 1. Compare CV mean accuracy with test accuracy from Q3:
    #    - They should be reasonably similar (typically within 0.02-0.05)
    #    - If different:
    #      - CV mean might be higher: lucky test split in Q3
    #      - CV mean might be lower: unlucky test split in Q3
    #    - Consistency is good: suggests model generalizes well
    #
    # 2. Why is cross-validation especially important in medical diagnosis?
    #    - Limited data: Medical datasets are often small (expensive to collect)
    #    - High stakes: Errors have serious consequences for patients
    #    - Reliability: Need robust estimate of true performance (not lucky split)
    #    - Balanced folds: CV ensures each patient data is used for validation
    #    - Overfitting detection: Can catch if model overfits particular subsets
    #    - Risk assessment: std_accuracy shows if model is consistently reliable
    #    - Regulatory/ethical: Provide high confidence in model performance
    #      for clinical deployment
    #
    # 3. What does the standard deviation tell us about model stability?
    #    - Low std: Model performs consistently across all patient subsets
    #      (Reliable for clinical use)
    #    - High std: Model performance varies significantly
    #      (May fail for certain patient types - concerning for medical use)
    #    - Medical context: High std is riskier because we need consistent
    #      performance across all patients
    
    return mean_accuracy, std_accuracy
