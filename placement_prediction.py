import os
import numpy as np
import pandas as pd
import pickle as pkl
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils import preprocess_data_placement

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

df.drop(['No.', 'Degree', 'Roll No','First Name', 'Middle Name', 'Last Name','Date of Birth', 'Back Papers','Pending Back Papers','Eligible But Not Registered Count', 'Registered But Not Offer Count','Semester 8 Aggregate Marks','Back Papers.8', 'Pending Back Papers.8','year_down'], axis=1, inplace=True)
df = preprocess_data_placement(df)

m = df['Diploma_Marks'].notnull()
df.loc[m, 'HSC_Marks'] = df.loc[m, 'HSC_Marks'].fillna(df['HSC_Marks'])
[
    'BRANCH', 'Campus', 'Gender', 'BE_Aggregate_Marks', 'Semester1_Marks', 'BackPapers1', 'P_BackPapers1', 'Semester2_Marks', 
    'BackPapers2', 'P_BackPapers2', 'Semester3_Marks', 'BackPapers3', 'P_BackPapers3', 'Semester4_Marks', 'BackPapers4', 
    'P_BackPapers4', 'Semester5_Marks', 'BackPapers5', 'P_BackPapers5', 'Semester6_Marks', 'BackPapers6', 'P_BackPapers6', 
    'Semester7_Marks', 'BackPapers7', 'P_BackPapers7', 'HSC_Marks', 'SSC_Marks', 'Diploma_Marks', 'dead_back_log', 'live_atkt', 
    'Job Offer Count'
]

branch_encoder, campus_encoder, gender_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()
pkl.dump(df['Campus'], open(f'{unique_dir}/Campus.sav','wb'))
df['Campus'] = campus_encoder.fit_transform(df['BRANCH'])
pkl.dump(campus_encoder, open(f'{encoder_dir}/Campus.sav','wb'))
df['BRANCH'] = branch_encoder.fit_transform(df['BRANCH'])
pkl.dump(df['Campus'], open(f'{unique_dir}/Gender.sav','wb'))
df['Gender'] = gender_encoder.fit_transform(df['Gender'])
pkl.dump(campus_encoder, open(f'{encoder_dir}/Gender.sav','wb'))

X = df.iloc[:,df.columns!='Job Offer Count']
y = df['Job Offer Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=2019)

scaler = StandardScaler()
