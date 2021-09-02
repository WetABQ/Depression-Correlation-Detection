import numpy as np
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce

from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neural_network import MLPClassifier as DNN
from sklearn import svm

from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import density

from sklearn import metrics

def read_chns_dataframe() -> dict[str, DataFrame]:
  return {path.stem:pd.read_sas(path) for path in Path("./chns/").glob("Master_*/*.sas7bdat") if not "Agriculture" in str(path) } # TODO: Exclude more pattern

# print(f"Total file: {len(chns_datas)}")
# idind_count = 0
# hhid_count = 0

# for (name, chns_data) in chns_datas.items():
#   columns_l = [column.lower() for column in chns_data.columns]
#   #print(f"---{name}---")
#   if "idind" in columns_l:
#     idind_count += 1
#   else:
#     pass
#    #print(f"{name} don't contain idind")
#   if "hhid" in columns_l:
#     hhid_count += 1
#   else:
#     print(f"{name} don't contain hhid")

# print(f"idint: {idind_count}, hhid: {hhid_count}")