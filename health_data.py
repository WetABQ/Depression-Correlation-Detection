import numpy as np
import pandas as pd
from pandas import DataFrame
from pathlib import Path

from pandas.core.series import Series

from utils import printTime
from model import Model

class DepressionHealthData:

  def __init__(self) -> None:
      self.raw_health_data = self.__read_health_dataframe()

  def __read_health_dataframe(self) -> dict[str, DataFrame]:
    return {path.stem[0:2]:pd.read_csv(path, encoding="latin-1") for path in Path("./healthdata/").glob("*.csv")}

  @printTime
  def __play_healthdata(self):
    print("Read Health Data")
    #qu = reduce(lambda left, right: pd.merge(left, right, how='inner', on=['SEQN']), health_datas.values()) # Merge All
    self.df = self.raw_health_data["qu"]
    #self.df = self.raw_health_data["qu"].merge(self.raw_health_data["de"], how='inner', on=['SEQN']).merge(self.raw_health_data["me"], how='inner', on=['SEQN']) # Only Merge Questionnaire and Demographics
    print(self.df.head())
    print("Merged Data")

  @printTime
  def __evaluate_depression_status(self):
    print("Evaluating Depression Status")
    sum = self.get_depression_status()
    self.df.insert(1, 'SUM', sum)
    self.df = self.df[self.df['SUM'] >= 0]
    print(self.df.head())
    print("Evaluated Depression Status, inserted SUM column")

  @printTime
  def get_depression_status(self, classify = False, depression_level = 8) -> Series:
    dpq = self.df.loc[:,['SEQN','DPQ010',	'DPQ020',	'DPQ030',	'DPQ040',	'DPQ050',	'DPQ060',	'DPQ070',	'DPQ080',	'DPQ090']].copy()
    dpq = dpq.dropna(thresh = 9) # drop when > 9 NaN
    a = dpq.loc[:, ['DPQ010', 'DPQ020',	'DPQ030',	'DPQ040',	'DPQ050',	'DPQ060',	'DPQ070',	'DPQ080',	'DPQ090']]
    a[a > 3] = 0 # 将大于 3 的DPQ值转为 0
    sum = a.sum(axis = 1)
    if classify:
      sum[sum <= depression_level] = 0
      sum[sum > depression_level] = 1
    return sum


  @printTime
  def prepare_data(self):
    self.__play_healthdata()
    self.__evaluate_depression_status()
    #self.clean_data()
    
  
  @printTime
  def clean_data(self):
    self.df = self.df.drop(['SEQN','DPQ010', 'DPQ020',	'DPQ030',	'DPQ040',	'DPQ050',	'DPQ060',	'DPQ070',	'DPQ080',	'DPQ090', "DPQ100"], axis=1)
    self.df = self.df.select_dtypes(exclude=['object'])
    self.df = self.df.fillna(0)
  
  def get_model(self) -> Model:
    m = Model(self.df)
    m.init_data_set()
    return m
