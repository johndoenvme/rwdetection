from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np

#  The O(1) features for the decision tree
o1_features = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,  # IO Statistics
               26, 27,  # Var, CoV
               34, 35, 36, 37, 38, 39, 40, 41]  # Access on LBA Head Region

with_xgboost = True
dt_max_depth = 6


class DeftPunkModel:

    def __init__(self):
        self.with_xgboost = with_xgboost
        self.decision_tree_max_depth = dt_max_depth

    def get_pipeline(self):
        decision_tree = SubsetDecisionTreeClassifier(max_depth=self.decision_tree_max_depth, random_state=0)
        if self.with_xgboost:
            xgb = XGBClassifier(random_state=0)
            steps = [("DT+XGBoost", DeftPunkClassifier(decision_tree, xgb))]
        else:
            steps = [("DT", decision_tree)]
        return Pipeline(steps=steps)


class SubsetDecisionTreeClassifier(DecisionTreeClassifier):
    """ Works on a subset of features"""

    def fit(self, X, y, sample_weight=None, check_input=True):
        # Only the O(1) features
        X = X[:, o1_features]

        # Call the parent's fit method
        super().fit(X, y, sample_weight=sample_weight, check_input=check_input)
        return self

    def predict(self, X, check_input=True):
        # Only the O(1) features
        X = X[:, o1_features]

        # Call the parent's predict method
        predictions = super().predict(X, check_input=check_input)

        # Add custom post-processing logic if needed
        return predictions

    def predict_proba(self, X, check_input=True):
        # Only the O(1) features
        X = X[:, o1_features]

        # Call the parent's predict_proba method
        probabilities = super().predict_proba(X, check_input=check_input)

        # Add custom post-processing logic if needed
        return probabilities[:, 1]


class DeftPunkClassifier(BaseEstimator, TransformerMixin):
    """ This class fits on an outer classifier and then passes to the inner classifier positively predicted classes """

    def __init__(self, outer_clf, inner_clf):
        self.outer_clf = outer_clf
        self.inner_clf = inner_clf

    def fit(self, X, y):
        # Fit Outer Classifier (A)
        self.outer_clf.fit(X, y)
        # Use Outer Classifier (A) to predict and identify positive samples
        y_pred_outer = self.outer_clf.predict(X)

        # Fit Inner Classifier (B) only on the positively classified samples from Outer Classifier (A)
        X_positive = X[y_pred_outer == 1]
        y_positive = y[y_pred_outer == 1]
        self.inner_clf.fit(X_positive, y_positive)

        return self

    def predict(self, X):
        # Predict using Outer Classifier (A)
        y_pred_outer = self.outer_clf.predict(X)

        # Predict using Inner Classifier (B), but only on the positive samples from Outer Classifier (A)
        X_positive = X[y_pred_outer == 1]
        y_pred_inner = np.zeros_like(y_pred_outer)  # Default to 0 for negative predictions
        if X_positive.shape[0] > 0:  # Ensure there are positive samples for Inner Classifier (B) to predict on
            y_pred_inner_pos = self.inner_clf.predict(X_positive)
            y_pred_inner[y_pred_outer == 1] = y_pred_inner_pos

        # Final predictions:
        # Negative if Outer Classifier (A) or Inner Classifier (B) predicts negative (0)
        # Positive only if both Outer Classifier (A) and Inner Classifier (B) predict positive (1)
        y_final = np.where((y_pred_outer == 0) | (y_pred_inner == 0), 0, 1)

        return y_final

    def predict_proba(self, X):
        # Predict using Outer Classifier (A)
        pred_outer = self.outer_clf.predict(X)

        # Predict probabilities using Inner Classifier (B), but only on the positive samples from Outer Classifier (A)
        X_positive = X[pred_outer == 1]
        prob_inner = np.zeros((pred_outer.shape[0], 2))
        prob_inner[:, 0] = 1  # Default to 0 for negative predictions

        if X_positive.shape[0] > 0:
            prob_inner_pos = self.inner_clf.predict_proba(X_positive)
            prob_inner[pred_outer == 1] = prob_inner_pos  # Get the probability for positive class (1)

        # Final probabilities: combine Outer Classifier (A) and Inner Classifier (B) probabilities
        final_prob = np.zeros((pred_outer.shape[0], 2))
        final_prob[:, 0] = 1  # Default to 0 for negative predictions

        final_prob[pred_outer == 1] = prob_inner[pred_outer == 1]  # Inner Classifier (B) overrides

        return final_prob  # Return as a 2D array for consistency with predict_proba format
