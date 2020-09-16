from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import csv
import matplotlib.pyplot as plt
import numpy as np
import os


def build_dataset(csv_file_path, val=0.17):
  dataset = {'x': [], 'y':[]}

  with open(csv_file_path, 'r') as f:
    csv_reader = csv.reader(f, delimiter=',')
    len_row = 180

    for row in list(csv_reader)[1:]:
      if len(row) > 0:
        x = np.array(row[1:len_row-1]).astype(np.int16)
        y = int(row[-1])
        dataset['x'].append(x)
        dataset['y'].append(y)
  return dataset


def normalize_vector(vector):
  normalized_vector = (vector - vector.min()) / (vector.max() - vector.min())
  return normalized_vector

def preprocess_dataset(dataset, val=0.17):
  x = np.array([normalize_vector(i) for i in dataset['x']], dtype=np.float)
  y = np.array(dataset['y'], dtype=np.uint8)

  ohe = OneHotEncoder()
  y = ohe.fit_transform(y.reshape((-1,1))).toarray()
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=val, random_state=42)
  return x_train, x_test, y_train, y_test

# if __name__ == "__main__":
#   dataset = build_dataset('../datasets/data.csv')
#   x_train, x_test, y_train, y_test = preprocess_dataset(dataset)