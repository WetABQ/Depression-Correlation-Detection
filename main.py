import pandas as pd
from health_data import DepressionHealthData 
from utils import printTime
from model import Model

def init():
  #显示所有列
  #pd.set_option('display.max_columns', None)
  #显示所有行
  #pd.set_option('display.max_rows', None)
  pass

@printTime
def main():
  init()
  dhd = DepressionHealthData()
  dhd.prepare_data()
  m = dhd.get_model()
  m.train_random_forest()

if __name__ == "__main__":
  main()