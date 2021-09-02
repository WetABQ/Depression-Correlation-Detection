from typing import Sequence
from IPython.core.display import HTML
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import Index
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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
  def init_data_set(self, test_size = 0.1) -> None:
    # split the data into features and target
    self.X = self.df.iloc[:, 1:]
    self.y = self.df.iloc[:, 0]

    # split the data into test and train
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, shuffle=True)

  @printTime
  def train_logistic(self) -> LogisticRegression:
    logreg = LogisticRegression().fit(self.X_train, self.y_train)
    print(f"LogReg: {logreg.score(self.X_test, self.y_test)}")
    return logreg

  @printTime
  def train_svm(self) -> SVC:
    svm = SVC().fit(self.X_train, self.y_train)
    print(f"SVM: {svm.score(self.X_test, self.y_test)}")
    return svm

  @printTime
  def train_mlp(self) -> MLPClassifier:
    dnn = MLPClassifier().fit(self.X_train, self.y_train)
    print(f"MLP: {dnn.score(self.X_test, self.y_test)}")
    return dnn
  
  @printTime
  def train_decision_tree(self) -> DecisionTreeClassifier:
    dtc = DecisionTreeClassifier().fit(self.X_train, self.y_train)
    print(f"DTC: {dtc.score(self.X_test, self.y_test)}")
    return dtc

  @printTime
  def train_random_forest(self) -> RandomForestClassifier:
    rfc = RandomForestClassifier().fit(self.X_train, self.y_train)
    print(f"RFC: {rfc.score(self.X_test, self.y_test)}")
    return rfc

  @printTime
  def feature_importance(self, model: DecisionTreeClassifier, showGraph=False):
    feature_importances = model.feature_importances_
    feature_names = self.X.columns
    indices = np.argsort(feature_importances)[::-1][:500]
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