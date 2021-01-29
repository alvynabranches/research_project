import os
import numpy as np
import pandas as pd
import pickle as pkl
from time import perf_counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

start = perf_counter()

mkdir = lambda dir_: os.mkdir(dir_) if not os.path.isdir(dir_) else 0
df = pd.read_csv('Master-Data_2019.csv', header=0)
base_dir = os.path.dirname(os.path.abspath(__file__))

model_dir = f'{base_dir}/website/models'
encoder_dir = f'{base_dir}/website/encoders'
scaler_dir = f'{base_dir}/website/scalers'
unique_dir = f'{base_dir}/website/unique'
data_dir = f'{base_dir}/website/data'

mkdir(model_dir)
mkdir(encoder_dir)
mkdir(scaler_dir)
mkdir(unique_dir)
mkdir(data_dir)

model_dir = f'{model_dir}/placement_prediction'
encoder_dir = f'{encoder_dir}/placement_prediction'
scaler_dir = f'{scaler_dir}/placement_prediction'
unique_dir = f'{unique_dir}/placement_prediction'
data_dir = f'{data_dir}/placement_prediciton'

mkdir(model_dir)
mkdir(encoder_dir)
mkdir(scaler_dir)
mkdir(unique_dir)
mkdir(data_dir)

