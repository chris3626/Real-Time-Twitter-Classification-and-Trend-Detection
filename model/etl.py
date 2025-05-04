 ## Process to clean and trasform the data##
#SaulM#
import pandas as pd
import numpy as np
import re

class ETL:
  def __init__(self, data):
    self.data = data

  def clean_data(self):
    self.data = self.data.dropna()
    self.data['text'] = self.data['text'].replace(r'@\w+', '', regex =True)
    self.data['text'] = self.data['text'].replace(r'http?:\S+|www.\S+', '', regex = True)
    self.data['text'] = self.data['text'].replace(r'#\w+', '', regex = True)
    self.data['text'] = self.data['text'].replace(r'\s+', ' ', regex=True).str.strip()

    return self

  def transform_data(self):
    # if to int
    if self.data['target'].dtype != 'int':
      self.data['target'] = self.data['target'].astype(int)

  # check the date
    if self.data['date'].dtype != 'datetime64[ns]':
      self.data['date'] = self.data['date'].str.replace(r'\s[A-Z]{3}\s', ' ', regex=True)

        # Parse cleaned date strings
      self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
      # add MM-DD-YYYY to the data set also time for analysis
      self.data['year'] = self.data['date'].dt.year
      self.data['month'] = self.data['date'].dt.month
      self.data['day'] = self.data['date'].dt.day
      self.data['time'] = pd.to_datetime(self.data['date'], format='%H:%M:%S', errors = 'coerce').dt.time
      # remove the date col since the we already extracted the info
      self.data.drop('date', axis=1, inplace=True)

    if self.data['text'].dtype != 'object':
      self.data['text'] = self.data['text'].astype(str)

    if self.data['user'].dtype != 'object':
      self.data['user'] = self.data['user'].astype(str)

    if np.array_equal(np.sort(self.data['target'].unique()), np.array([0,4])):
      self.data['target'] = self.data['target'].replace({4:1})

    return self
  
  def get_data(self):
    return self.data
