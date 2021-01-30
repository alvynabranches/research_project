import os, sys
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.utils import to_categorical

from utils import preprocess_data, get_HSC_binned, get_Merit_Marks_binned, EDA_After_PP

start = perf_counter()

mkdir = lambda dir_: os.mkdir(dir_) if not os.path.isdir(dir_) else 0
df = pd.read_csv('STUDENT_DATA13.csv', sep=',', encoding='gbk')
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

model_dir = f'{model_dir}/college_prediction'
encoder_dir = f'{encoder_dir}/college_prediction'
scaler_dir = f'{scaler_dir}/college_prediction'
unique_dir = f'{unique_dir}/college_prediction'
data_dir = f'{data_dir}/college_prediciton'

mkdir(model_dir)
mkdir(encoder_dir)
mkdir(scaler_dir)
mkdir(unique_dir)
mkdir(data_dir)

df.drop(['Main Serial No.','Sr. No.','Candidate Name', 'Seat Type', 'Fees Paid', 'Admitted/Uploaded Late'], axis=1, inplace=True)

df['HSC Eligibility'] = df['HSC Eligibility'].astype(float)
df['Merit Marks'] = df['Merit Marks'].astype(float)

df = preprocess_data(df)
    
df['HSC Eligibility'] = df['HSC Eligibility'].astype(float)
get_HSC_binned(df)
get_Merit_Marks_binned(df)

college_name_encoder = LabelEncoder()
df['College Name'] = np.where(pd.isnull(df['College Name']), 'SINHGAD SKN, VADGAON', df['College Name'])
pkl.dump(df['College Name'].unique().tolist(), open(f'{unique_dir}/college_names.sav','wb'))
df['College Name'] = college_name_encoder.fit_transform(df['College Name'])
pkl.dump(college_name_encoder, open(f'{encoder_dir}/CollegeName.sav','wb'))

college_names = college_name_encoder.classes_

df = EDA_After_PP(df)

df['Category_PH Type'] = df['Category'].str.cat(df['PH Type'], sep =" ") 
df['Category_PH_Defence Type'] = df['Category_PH Type'].str.cat(df['Defence Type'], sep =" ") 
df.drop(['PH Type','Defence Type','Category_PH Type','Category'], axis=1, inplace=True)

df['NATIONALITY'] = np.where(pd.isnull(df['NATIONALITY']), 'INDIAN', df['NATIONALITY'])
df['BRANCH'] = np.where(pd.isnull(df['BRANCH']), 'COMPUTER', df['BRANCH'])
df['Gender'] = np.where(pd.isnull(df['Gender']), 'M', df['Gender'])
df['Candidate Type'] = np.where(pd.isnull(df['Candidate Type']), 'TYPE A', df['Candidate Type'])
df['College Code'] = np.where(pd.isnull(df['College Code']), 6178, df['College Code'])
df['Merit No'] = np.where(pd.isnull(df['Merit No']), 6000, df['Merit No'])
df['Merit Marks'] = np.where(pd.isnull(df['Merit Marks']), 50, df['Merit Marks'])
df['HSC Eligibility'] = np.where(pd.isnull(df['HSC Eligibility']), 50, df['HSC Eligibility'])

home_university_encoder, gender_encoder, category_ph_defence_type_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()
candidate_type_encoder, branch_encoder, nationality_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()
cap_round_encoder = LabelEncoder()

pkl.dump(df['Home University'].unique().tolist(), open(f'{unique_dir}/home_universities.sav','wb'))
df['Home University'] = home_university_encoder.fit_transform(df['Home University'])
pkl.dump(home_university_encoder, open(f'{encoder_dir}/HomeUniversity.sav','wb'))
pkl.dump(df['Gender'].unique().tolist(), open(f'{unique_dir}/genders.sav','wb'))
df['Gender'] = gender_encoder.fit_transform(df['Gender'])
pkl.dump(gender_encoder, open(f'{encoder_dir}/Gender.sav','wb'))
pkl.dump(df['Category_PH_Defence Type'].unique().tolist(), open(f'{unique_dir}/category_ph_defence_types.sav','wb'))
df['Category_PH_Defence Type'] = category_ph_defence_type_encoder.fit_transform(df['Category_PH_Defence Type'])
pkl.dump(category_ph_defence_type_encoder, open(f'{encoder_dir}/Category_PH_DefenceType.sav','wb'))
pkl.dump(df['Candidate Type'].unique().tolist(), open(f'{unique_dir}/candidate_types.sav','wb'))
df['Candidate Type'] = candidate_type_encoder.fit_transform(df['Candidate Type'])
pkl.dump(candidate_type_encoder, open(f'{encoder_dir}/CandidateType.sav','wb'))
pkl.dump(df['BRANCH'].unique().tolist(), open(f'{unique_dir}/branches.sav','wb'))
df['BRANCH'] = branch_encoder.fit_transform(df['BRANCH'])
pkl.dump(branch_encoder, open(f'{encoder_dir}/BRANCH.sav','wb'))
pkl.dump(df['NATIONALITY'].unique().tolist(), open(f'{unique_dir}/nationalities.sav','wb'))
df['NATIONALITY'] = nationality_encoder.fit_transform(df['NATIONALITY'])
pkl.dump(nationality_encoder, open(f'{encoder_dir}/NATIONALITY.sav','wb'))
pkl.dump(df['CAP Round'].unique().tolist(), open(f'{unique_dir}/cap_rounds.sav','wb'))
df['CAP Round'] = cap_round_encoder.fit_transform(df['CAP Round'])
pkl.dump(cap_round_encoder, open(f'{encoder_dir}/CAPRound.sav','wb'))

df.drop(['College Code'], axis=1, inplace=True)
df.drop(['HSC_binned'], axis=1, inplace=True)
df.drop(['Merit_Marks_binned'], axis=1, inplace=True)
df.dropna(inplace=True)

con = ['Merit No', 'Merit Marks', 'HSC Eligibility']
cat = ['Gender', 'Candidate Type', 'Home University', 'CAP Round', 'BRANCH', 'NATIONALITY', 'Category_PH_Defence Type']
all_fields = con + cat
# X = np.append(df[cat].values, df[con].values, axis=1)

X = df.iloc[:,df.columns!='College Name']
y = df['College Name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=2019)

pkl.dump(X_train, open(f'{data_dir}/X_train.sav', 'wb'))
pkl.dump(X_test, open(f'{data_dir}/X_test.sav', 'wb'))
pkl.dump(y_train, open(f'{data_dir}/y_train.sav', 'wb'))
pkl.dump(y_test, open(f'{data_dir}/y_test.sav', 'wb'))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pkl.dump(scaler, open(f'{scaler_dir}/scaler.sav', 'wb'))
pkl.dump(X_train, open(f'{data_dir}/X_train_scaled.sav', 'wb'))
pkl.dump(X_test, open(f'{data_dir}/X_test_scaled.sav', 'wb'))

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# model = Sequential()
# model.add(Dense(600, activation='tanh', input_shape=(13,)))
# model.add(Dense(200, activation='tanh'))
# model.add(Dense(60, activation='tanh'))
# model.add(Dense(25, activation='softmax'))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# s = perf_counter()
# model.fit(X_train, y_train, epochs=500)
# model.save(f'{model_dir}/NeuralNetwork.h5')
# e = perf_counter()
# print(f'Time Taken: {e-s:.2f} seconds')

models_list = []
# models_list.append(dict(model_name='LogisticRegression', model=LogisticRegression(max_iter=1000))) # max_iter=1000
# models_list.append(dict(model_name='SupportVectorClassifier', model=SVC()))
# models_list.append(dict(model_name='DecisionTreeClassifier', model=DecisionTreeClassifier()))
# models_list.append(dict(model_name='RandomForestClassifier', model=RandomForestClassifier(n_estimators=300, max_features=3)))
# models_list.append(dict(model_name='GaussianNB', model=GaussianNB()))
# models_list.append(dict(model_name='KNeighborsClassifier', model=KNeighborsClassifier()))
models_list.append(dict(model_name='XGBoostClassifier', model=XGBClassifier(n_estimators=175,seed=41)))

results = dict(model_name=[], accuracy=[])
info = ''
s = perf_counter()
for model_dict in models_list:
    kfold = KFold(n_splits=10, random_state=2019)
    cross_val_results = cross_val_score(model_dict['model'], X_train, y_train, cv=kfold, scoring='accuracy', n_jobs=4)
    results['model_name'].append(model_dict['model_name'])
    results['accuracy'].append(cross_val_results)
    info += f"{model_dict['model_name']}: {cross_val_results.mean()} ({cross_val_results.std()})\n"
    model_dict['model'].fit(X_train, y_train)
    model_dict['model'].fit(X_test, y_test)
    pkl.dump(model_dict['model'], open(f'{model_dir}/{model_dict["model_name"]}.sav', 'wb'))
e = perf_counter()
print(info)
print(f'Time Taken: {e-s:.2f} seconds')

models_list = []
# models_list.append(dict(model_name='LogisticRegression', model=pkl.load(open(f'{model_dir}/LogisticRegression.sav','rb'))))
# models_list.append(dict(model_name='SupportVectorClassifier', model=pkl.load(open(f'{model_dir}/SupportVectorClassifier.sav','rb'))))
# models_list.append(dict(model_name='DecisionTreeClassifier', model=pkl.load(open(f'{model_dir}/DecisionTreeClassifier.sav','rb'))))
# models_list.append(dict(model_name='RandomForestClassifier', model=pkl.load(open(f'{model_dir}/RandomForestClassifier.sav','rb'))))
# models_list.append(dict(model_name='GaussianNB', model=pkl.load(open(f'{model_dir}/GaussianNB.sav','rb'))))
# models_list.append(dict(model_name='KNeighborsClassifier', model=pkl.load(open(f'{model_dir}/KNeighborsClassifier.sav','rb'))))
models_list.append(dict(model_name='XGBoostClassifier', model=pkl.load(open(f'{model_dir}/XGBoostClassifier.sav','rb'))))

# fig = plt.figure()
# fig.suptitle('Machine Learning Algorithms performance comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results['accuracy'])
# ax.set_xticklabels(results['model_name'])
# plt.show()

end = perf_counter()
print(f'Total Time Taken: {end-start:.2f} seconds')