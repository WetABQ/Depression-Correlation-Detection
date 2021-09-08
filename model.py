from typing import Sequence
from IPython.core.display import HTML
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Index
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import density

from sklearn import metrics

import eli5
from eli5.sklearn import PermutationImportance
from utils import printTime

class Model:

  def __init__(self, df: DataFrame) -> None:
    self.df = df

  @printTime
  def init_data_set(self, enable_pca=True) -> None:
    # split the data into features and target
    self.X = self.df.iloc[:, 1:]
    self.y = self.df.iloc[:, 0]

    if enable_pca:
      pca = PCA(0.95)
      pca.fit(self.X)
      self.X = pca.transform(self.X)
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.7)
    else:
    # split the data into test and train
      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.7)

  @printTime
  def train_logistic(self, train=False) -> LogisticRegression:
    logreg = LogisticRegression()
    if train: self.train_model(logreg)
    return logreg

  @printTime
  def train_svm(self, train=False) -> SVC:
    svm = SVC()
    if train: self.train_model(svm)
    return svm

  @printTime
  def train_mlp(self, train=False) -> MLPClassifier:
    mlp = MLPClassifier(verbose=1, max_iter=100000, early_stopping=True)
    if train: 
      self.train_model(mlp)
      plt.plot(mlp.loss_curve_)
      plt.plot(mlp.validation_scores_)
      plt.show()
    return mlp
  
  @printTime
  def train_decision_tree(self, train=False) -> DecisionTreeClassifier:
    dtc = DecisionTreeClassifier()
    if train: self.train_model(dtc)
    return dtc

  @printTime
  def train_random_forest(self, train=False) -> RandomForestClassifier:
    rfc = RandomForestClassifier()
    if train: self.train_model(rfc)
    return rfc

  # KNN and Naive Bayes

  def evaluation_score(self, model) -> None: print(f"{model.__class__.__name__}: Score - {model.score(self.X_test, self.y_test)}")
    
  @printTime
  def evaluation_cv(self, model) -> None: print(f"{model.__class__.__name__}: Cross Validation - {cross_val_score(model, self.X, self.y, cv=3)}")

  def train_model(self, model):
    model.fit(self.X_train, self.y_train)
    self.evaluation_score(model)
    return model

  @printTime
  def evaluation_model(self, model):
    from sklearn.model_selection import learning_curve

    train_size, train_score, test_score = learning_curve(model, self.X, self.y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5))

    train_scores_mean = np.mean(train_score, axis=1)
    train_scores_std = np.std(train_score, axis=1)
    test_scores_mean = np.mean(test_score, axis=1)
    test_scores_std = np.std(test_score, axis=1)

    plt.fill_between(train_size, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
    plt.fill_between(train_size, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_size, train_scores_mean, 'o--', color="r",
                label="Training score")
    plt.plot(train_size, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")

    plt.grid()
    plt.title(f'Learn Curve for {model.__class__.__name__}')
    plt.legend(loc="best")
    plt.show()

  @printTime
  def feature_importance(self, model: DecisionTreeClassifier, showGraph=False):
    feature_importances = model.feature_importances_
    feature_names = self.X.columns
    indices = np.argsort(feature_importances)[::-1][:20]
    rankTop = []

    for index in indices:
      rankTop.append(feature_names[index])
      print("feature %s (%f)" %(feature_names[index], feature_importances[index]))
    
    if showGraph: 
      plt.figure(figsize=(16,8))
      plt.title("Feature Importance")
      plt.bar(range(len(indices)), feature_importances[indices], color='b')
      plt.xticks(range(len(indices)), np.array(feature_names) [indices], color='b', rotation=90)
      plt.show()
    
  @printTime
  def permutation_importance_graph(self, model) -> HTML:
    perm = PermutationImportance(model, random_state=1).fit(self.X_test, self.y_test) 
    return eli5.show_weights(perm, feature_names=self.X.columns.tolist(),top=5000)