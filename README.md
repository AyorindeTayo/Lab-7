# Ensemble Learning Lab: Combining Models for Improved Performance

## Overview
This lab is based on Chapter 7 of *Python Machine Learning (Second Edition)* by Sebastian Raschka and Vahid Mirjalili. The chapter covers ensemble learning methods that combine multiple models to achieve better predictive performance than individual models.

In this hands-on lab, you will:

- Implement majority voting classifiers from scratch
- Use bagging with decision trees
- Apply AdaBoost for boosting weak learners
- Compare ensemble methods with individual classifiers
- Evaluate and tune ensemble classifiers

The lab uses the **Wine dataset** and **Iris dataset** for classification tasks.

---

## Objectives
- Understand the concept of ensemble learning and why it often outperforms individual models
- Implement and evaluate majority voting classifiers
- Apply bagging with bootstrap sampling
- Understand and implement AdaBoost for sequential learning
- Compare the performance of different ensemble methods
- Learn hyperparameter tuning for ensemble classifiers

---

## Prerequisites
- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`
---

## Part 1: Implementing a Majority Vote Classifier
### Step 1.1: Understanding Ensemble Concepts

Ensemble methods combine multiple classifiers to make more accurate predictions than individual classifiers.

### Step 1.2: Implement Majority Voting Classifier
```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
from sklearn.externals import six

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """A majority vote ensemble classifier"""

    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """Fit classifiers"""
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """Predict class labels"""
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1, arr=predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """Predict class probabilities"""
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba

    def get_params(self, deep=True):
        """Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
