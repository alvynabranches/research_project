import os, sys
import numpy as np
import pandas as pd
import pickle as pkl
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

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

branch_encoder, campus_encoder, gender_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()

pkl.dump(df['Campus'].unique().tolist(), open(f'{unique_dir}/Campus.sav','wb'))
df['Campus'] = campus_encoder.fit_transform(df['Campus'])
pkl.dump(campus_encoder, open(f'{encoder_dir}/Campus.sav','wb'))
pkl.dump(df['BRANCH'].unique().tolist(), open(f'{unique_dir}/Branch.sav','wb'))
df['BRANCH'] = branch_encoder.fit_transform(df['BRANCH'])
pkl.dump(branch_encoder, open(f'{encoder_dir}/Branch.sav','wb'))
pkl.dump(df['Gender'].unique().tolist(), open(f'{unique_dir}/Gender.sav','wb'))
df['Gender'] = gender_encoder.fit_transform(df['Gender'])
pkl.dump(gender_encoder, open(f'{encoder_dir}/Gender.sav','wb'))

X = df.iloc[:,df.columns!='Job Offer Count']
y = df['Job Offer Count']
pkl.dump(df['Job Offer Count'].unique().tolist(), open(f'{unique_dir}/JobOfferCount.sav','wb'))
# print(X.columns.values.tolist())
# print(df['Job Offer Count'].unique().tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=2019)

pkl.dump(X_train, open(f'{data_dir}/X_train.sav', 'wb'))
pkl.dump(X_test, open(f'{data_dir}/X_test.sav', 'wb'))
pkl.dump(y_train, open(f'{data_dir}/y_train.sav', 'wb'))
pkl.dump(y_test, open(f'{data_dir}/y_test.sav', 'wb'))

# print(X_test.reset_index().drop(['index'], axis=1).head(2))
for col in X_test.reset_index().drop(['index'], axis=1).columns.values.tolist():
    print(f'{col}: {X_test[col][0]}')

scaler = StandardScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pkl.dump(scaler, open(f'{scaler_dir}/scaler.sav', 'wb'))
pkl.dump(X_train, open(f'{data_dir}/X_train_scaled.sav', 'wb'))
pkl.dump(X_test, open(f'{data_dir}/X_test_scaled.sav', 'wb'))

# print(df.iloc[:,df.columns!='Job Offer Count'].columns.values.tolist())
# [
#     'BRANCH', 'Campus', 'Gender', 'BE_Aggregate_Marks', 'Semester1_Marks', 'BackPapers1', 'P_BackPapers1', 'Semester2_Marks', 
#     'BackPapers2', 'P_BackPapers2', 'Semester3_Marks', 'BackPapers3', 'P_BackPapers3', 'Semester4_Marks', 'BackPapers4', 
#     'P_BackPapers4', 'Semester5_Marks', 'BackPapers5', 'P_BackPapers5', 'Semester6_Marks', 'BackPapers6', 'P_BackPapers6', 
#     'Semester7_Marks', 'BackPapers7', 'P_BackPapers7', 'HSC_Marks', 'SSC_Marks', 'Diploma_Marks', 'dead_back_log', 'live_atkt'
# ]

models_list = []
# models_list.append(dict(model_name='LogisticRegression', model=LogisticRegression(max_iter=1000))) # max_iter=1000
# models_list.append(dict(model_name='SupportVectorClassifier', model=SVC()))
# models_list.append(dict(model_name='DecisionTreeClassifier', model=DecisionTreeClassifier()))
models_list.append(dict(model_name='RandomForestClassifier', model=RandomForestClassifier(n_estimators=300, max_features=3)))
# models_list.append(dict(model_name='GaussianNB', model=GaussianNB()))
# models_list.append(dict(model_name='KNeighborsClassifier', model=KNeighborsClassifier()))
# models_list.append(dict(model_name='XGBoostClassifier', model=XGBClassifier(n_estimators=175,seed=41)))

results = dict(model_name=[], accuracy=[])
info = ''
s = perf_counter()
for model_dict in models_list:
    kfold = KFold(n_splits=10, random_state=2019, shuffle=True)
    cross_val_results = cross_val_score(model_dict['model'], X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=4)
    results['model_name'].append(model_dict['model_name'])
    results['accuracy'].append(cross_val_results)
    info += f"{model_dict['model_name']}: {cross_val_results.mean()} ({cross_val_results.std()})\n"
    model_dict['model'].fit(X_train, y_train)
    model_dict['model'].fit(X_test, y_test)
    pkl.dump(model_dict['model'], open(f'{model_dir}/{model_dict["model_name"]}.sav', 'wb'))
e = perf_counter()
# print(info)
print(f'Time Taken: {e-s:.2f} seconds')

print(models_list[0]['model'].predict([X_test[0]])[0])
print(scaler.inverse_transform([X_test[0]]))
print(X_test[0].shape)

models_list = []
# models_list.append(dict(model_name='LogisticRegression', model=pkl.load(open(f'{model_dir}/LogisticRegression.sav','rb'))))
# models_list.append(dict(model_name='SupportVectorClassifier', model=pkl.load(open(f'{model_dir}/SupportVectorClassifier.sav','rb'))))
# models_list.append(dict(model_name='DecisionTreeClassifier', model=pkl.load(open(f'{model_dir}/DecisionTreeClassifier.sav','rb'))))
models_list.append(dict(model_name='RandomForestClassifier', model=pkl.load(open(f'{model_dir}/RandomForestClassifier.sav','rb'))))
# models_list.append(dict(model_name='GaussianNB', model=pkl.load(open(f'{model_dir}/GaussianNB.sav','rb'))))
# models_list.append(dict(model_name='KNeighborsClassifier', model=pkl.load(open(f'{model_dir}/KNeighborsClassifier.sav','rb'))))
# models_list.append(dict(model_name='XGBoostClassifier', model=pkl.load(open(f'{model_dir}/XGBoostClassifier.sav','rb'))))

# fig = plt.figure()
# fig.suptitle('Machine Learning Algorithms performance comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results['accuracy'])
# ax.set_xticklabels(results['model_name'])
# plt.show()

end = perf_counter()
print(f'Total Time Taken: {end-start:.2f} seconds')