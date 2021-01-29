import os, sys
import numpy as np
import pandas as pd
import pickle as pkl
from time import perf_counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

start = perf_counter()
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = f'{base_dir}/website/models'
encoder_dir = f'{base_dir}/website/encoders'
scaler_dir = f'{base_dir}/website/scalers'
unique_dir = f'{base_dir}/website/unique'
data_dir = f'{base_dir}/website/data'
df = pd.read_csv('STUDENT_DATA13.csv', sep=',', encoding='gbk')

mkdir = lambda dir_: os.mkdir(dir_) if not os.path.isdir(dir_) else 0

mkdir(model_dir)
mkdir(encoder_dir)
mkdir(scaler_dir)
mkdir(unique_dir)
mkdir(data_dir)

model_dir = f'{model_dir}/branch_prediction'
encoder_dir = f'{encoder_dir}/branch_prediction'
scaler_dir = f'{scaler_dir}/branch_prediction'
unique_dir = f'{unique_dir}/branch_prediction'
data_dir = f'{data_dir}/branch_prediction'
mkdir(model_dir)
mkdir(encoder_dir)
mkdir(scaler_dir)
mkdir(unique_dir)
mkdir(data_dir)

df['HSC Eligibility'] = df['HSC Eligibility'].astype(float)
df['Merit Marks'] = df['Merit Marks'].astype(float)
df.drop(['Main Serial No.','Sr. No.','Candidate Name', 'Fees Paid','Seat Type', 'Admitted/Uploaded Late'], axis=1, inplace=True)

df["PH Type"] = np.where(pd.isnull(df['PH Type']), 'NA', df['PH Type'])
df['Category'] = np.where(pd.isnull(df['Category']), 'OPEN', df['Category'])
df['Home University'] = np.where(pd.isnull(df['Home University']), 'PUNE', df['Home University'])
df['College Name'] = np.where(pd.isnull(df['College Name']), 'SINHGAD SKN, VADGAON', df['College Name'])
df['NATIONALITY'] = np.where(pd.isnull(df['NATIONALITY']), 'INDIAN', df['NATIONALITY'])
df['BRANCH'] = np.where(pd.isnull(df['BRANCH']), 'COMPUTER', df['BRANCH'])
df['Defence Type'] = np.where(pd.isnull(df['Defence Type']), 'NA', df['Defence Type'])
df['Gender'] = np.where(pd.isnull(df['Gender']), 'M', df['Gender'])
df['HSC Eligibility'] = np.where(pd.isnull(df['HSC Eligibility']), 50, df['HSC Eligibility'])
df['CAP Round'] = np.where(pd.isnull(df['CAP Round']), 'ROUND-1', df['CAP Round'])
df['Candidate Type'] = np.where(pd.isnull(df['Candidate Type']), 'TYPE A', df['Candidate Type'])
df['College Code'] = np.where(pd.isnull(df['College Code']), 6178, df['College Code'])
df['Merit No'] = np.where(pd.isnull(df['Merit No']), 67390, df['Merit No'])
df['Merit Marks'] = np.where(pd.isnull(df['Merit Marks']), 50, df['Merit Marks'])

gender_encoder, category_encoder, candidate_type_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()
college_name_encoder, nationality_encoder, defence_type_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()
cap_round_encoder, ph_type_encoder, branch_encoder = LabelEncoder(), LabelEncoder(), LabelEncoder()
home_university_encoder, college_code_encoder = LabelEncoder(), LabelEncoder()

pkl.dump(df['Gender'].unique().tolist(), open(f'{unique_dir}/genders.sav','wb'))
df['Gender'] = gender_encoder.fit_transform(df['Gender'])
pkl.dump(gender_encoder, open(f'{encoder_dir}/Gender.sav','wb'))
pkl.dump(df['Category'].unique().tolist(), open(f'{unique_dir}/categories.sav','wb'))
df['Category'] = category_encoder.fit_transform(df['Category'])
pkl.dump(category_encoder, open(f'{encoder_dir}/Category.sav','wb'))
pkl.dump(df['Candidate Type'].unique().tolist(), open(f'{unique_dir}/candidate_types.sav','wb'))
df['Candidate Type'] = candidate_type_encoder.fit_transform(df['Candidate Type'])
pkl.dump(candidate_type_encoder, open(f'{encoder_dir}/CandidateType.sav','wb'))
pkl.dump(df['College Name'].unique().tolist(), open(f'{unique_dir}/college_names.sav','wb'))
df['College Name'] = college_name_encoder.fit_transform(df['College Name'])
pkl.dump(college_name_encoder, open(f'{encoder_dir}/CollegeName.sav','wb'))
pkl.dump(df['NATIONALITY'].unique().tolist(), open(f'{unique_dir}/nationalities.sav','wb'))
df['NATIONALITY'] = nationality_encoder.fit_transform(df['NATIONALITY'])
pkl.dump(nationality_encoder, open(f'{encoder_dir}/NATIONALITY.sav','wb'))
pkl.dump(df['Defence Type'].unique().tolist(), open(f'{unique_dir}/defence_types.sav','wb'))
df['Defence Type'] = defence_type_encoder.fit_transform(df['Defence Type'])
pkl.dump(defence_type_encoder, open(f'{encoder_dir}/DefenceType.sav','wb'))
pkl.dump(df['CAP Round'].unique().tolist(), open(f'{unique_dir}/cap_rounds.sav','wb'))
df['CAP Round'] = cap_round_encoder.fit_transform(df['CAP Round'])
pkl.dump(cap_round_encoder, open(f'{encoder_dir}/CAPRound.sav','wb'))
pkl.dump(df['PH Type'].unique().tolist(), open(f'{unique_dir}/ph_types.sav','wb'))
df['PH Type'] = ph_type_encoder.fit_transform(df['PH Type'])
pkl.dump(ph_type_encoder, open(f'{encoder_dir}/PHType.sav','wb'))
pkl.dump(df['BRANCH'].unique().tolist(), open(f'{unique_dir}/branches.sav','wb'))
df['BRANCH'] = branch_encoder.fit_transform(df['BRANCH'])
pkl.dump(branch_encoder, open(f'{encoder_dir}/BRANCH.sav','wb'))
pkl.dump(df['Home University'].unique().tolist(), open(f'{unique_dir}/home_universities.sav','wb'))
df['Home University'] = home_university_encoder.fit_transform(df['Home University'])
pkl.dump(home_university_encoder, open(f'{encoder_dir}/HomeUniversity.sav','wb'))
pkl.dump(df['College Code'].unique().tolist(), open(f'{unique_dir}/college_codes.sav','wb'))
df['College Code'] = college_code_encoder.fit_transform(df['College Code'])
pkl.dump(college_code_encoder, open(f'{encoder_dir}/CollegeCode.sav','wb'))

con = ['Merit No', 'Merit Marks', 'HSC Eligibility']
cat = ['College Name', 'College Code', 'Gender', 'Candidate Type', 'Category', 'Home University', 'PH Type', 'Defence Type', 'NATIONALITY', 'CAP Round']
all_fields = con + cat
# X = np.append(df[cat].values, df[con].values, axis=1)
print(df.columns.values.tolist())
X = df.iloc[:,df.columns != 'BRANCH']
y = df['BRANCH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=2019)

pkl.dump(X_train, open(f'{data_dir}/X_train.sav', 'wb'))
pkl.dump(X_test, open(f'{data_dir}/X_test.sav', 'wb'))
pkl.dump(y_train, open(f'{data_dir}/y_train.sav', 'wb'))
pkl.dump(y_test, open(f'{data_dir}/y_test.sav', 'wb'))

models_list = []
models_list.append(dict(model_name='LogisticRegression', model=LogisticRegression(max_iter=1000))) # max_iter=1000
models_list.append(dict(model_name='SupportVectorClassifier', model=SVC()))
models_list.append(dict(model_name='DecisionTreeClassifier', model=DecisionTreeClassifier()))
models_list.append(dict(model_name='RandomForestClassifier', model=RandomForestClassifier(n_estimators=300, max_features=3)))
models_list.append(dict(model_name='GaussianNB', model=GaussianNB()))
models_list.append(dict(model_name='KNeighborsClassifier', model=KNeighborsClassifier()))

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
    pkl.dump(model_dict['model'], open(f'{model_dir}/{model_dict["model_name"]}.sav', 'wb'))
e = perf_counter()
print(info)
print(f'Time Taken: {e-s:.2f} seconds')

models_list = []
models_list.append(dict(model_name='LogisticRegression', model=pkl.load(open(f'{model_dir}/LogisticRegression.sav','rb'))))
models_list.append(dict(model_name='SupportVectorClassifier', model=pkl.load(open(f'{model_dir}/SupportVectorClassifier.sav','rb'))))
models_list.append(dict(model_name='DecisionTreeClassifier', model=pkl.load(open(f'{model_dir}/DecisionTreeClassifier.sav','rb'))))
models_list.append(dict(model_name='RandomForestClassifier', model=pkl.load(open(f'{model_dir}/RandomForestClassifier.sav','rb'))))
models_list.append(dict(model_name='GaussianNB', model=pkl.load(open(f'{model_dir}/GaussianNB.sav','rb'))))
models_list.append(dict(model_name='KNeighborsClassifier', model=pkl.load(open(f'{model_dir}/KNeighborsClassifier.sav','rb'))))

end = perf_counter()
print(f'Total Time Taken: {end-start:.2f} seconds')