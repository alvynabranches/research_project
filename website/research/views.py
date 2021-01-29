import os
from socket import gethostbyname, gethostname
import numpy as np
import pandas as pd
import pickle as pkl
from django.shortcuts import render, HttpResponse

# Create your views here.
def index(request):
    show_variables = dict(
        online=gethostbyname(gethostname())!='127.0.0.1',
        offline=gethostbyname(gethostname())=='127.0.0.1'
    )
    return render(request, 'research/index.html', context=show_variables)

def branch_prediction(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '..', 'models', 'branch_prediction')
    encoder_dir = os.path.join(base_dir, '..', 'encoders', 'branch_prediction')
    scaler_dir = os.path.join(base_dir, '..', 'scalers', 'branch_prediction')
    unique_dir = os.path.join(base_dir, '..', 'unique', 'branch_prediction')
    data_dir = os.path.join(base_dir, '..', 'data', 'branch_prediction')
    
    # df = pd.read_csv(os.path.join(data_dir, 'STUDENT_DATA13.csv'))
    
    unique_college_names = pkl.load(open(f'{unique_dir}/college_names.sav','rb'))
    unique_college_codes = pkl.load(open(f'{unique_dir}/college_codes.sav','rb'))
    unique_genders = pkl.load(open(f'{unique_dir}/genders.sav','rb'))
    unique_candidate_types = pkl.load(open(f'{unique_dir}/candidate_types.sav','rb'))
    unique_categories = pkl.load(open(f'{unique_dir}/categories.sav','rb'))
    unique_home_universities = pkl.load(open(f'{unique_dir}/home_universities.sav','rb'))
    unique_ph_types = pkl.load(open(f'{unique_dir}/ph_types.sav','rb'))
    unique_defence_types = pkl.load(open(f'{unique_dir}/defence_types.sav','rb'))
    unique_nationalities = pkl.load(open(f'{unique_dir}/nationalities.sav','rb'))
    unique_cap_rounds = pkl.load(open(f'{unique_dir}/cap_rounds.sav','rb'))
    unique_branches = pkl.load(open(f'{unique_dir}/branches.sav','rb'))
    
    models = []
    models.append(dict(model_name='LogisticRegression', model=pkl.load(open(f'{model_dir}/LogisticRegression.sav','rb'))))
    models.append(dict(model_name='SupportVectorClassifier', model=pkl.load(open(f'{model_dir}/SupportVectorClassifier.sav','rb'))))
    models.append(dict(model_name='DecisionTreeClassifier', model=pkl.load(open(f'{model_dir}/DecisionTreeClassifier.sav','rb'))))
    models.append(dict(model_name='RandomForestClassifier', model=pkl.load(open(f'{model_dir}/RandomForestClassifier.sav','rb'))))
    models.append(dict(model_name='GaussianNB', model=pkl.load(open(f'{model_dir}/GaussianNB.sav','rb'))))
    models.append(dict(model_name='KNeighborsClassifier', model=pkl.load(open(f'{model_dir}/KNeighborsClassifier.sav','rb'))))
        
    if request.method == 'POST':
        merit_no = request.POST['merit_no']
        merit_marks = request.POST['merit_marks']
        hsc_eligibility = request.POST['hsc_eligibility']
        college_name = request.POST['college_name']
        college_code = request.POST['college_code']
        gender = request.POST['gender']
        candidate_type = request.POST['candidate_type']
        category = request.POST['category']
        home_university = request.POST['home_university']
        ph_type = request.POST['ph_type']
        defence_type = request.POST['defence_type']
        nationality = request.POST['nationality']
        cap_round = request.POST['cap_round']
        # branch = request.POST['branch']
        
        gender_encoder = pkl.load(open(f'{encoder_dir}/Gender.sav','rb'))
        category_encoder = pkl.load(open(f'{encoder_dir}/Category.sav','rb'))
        candidate_type_encoder = pkl.load(open(f'{encoder_dir}/CandidateType.sav','rb'))
        college_name_encoder = pkl.load(open(f'{encoder_dir}/CollegeName.sav','rb'))
        nationality_encoder = pkl.load(open(f'{encoder_dir}/NATIONALITY.sav','rb'))
        defence_type_encoder = pkl.load(open(f'{encoder_dir}/DefenceType.sav','rb'))
        cap_round_encoder = pkl.load(open(f'{encoder_dir}/CAPRound.sav','rb'))
        ph_type_encoder = pkl.load(open(f'{encoder_dir}/PHType.sav','rb'))
        branch_encoder = pkl.load(open(f'{encoder_dir}/BRANCH.sav','rb'))
        home_university_encoder = pkl.load(open(f'{encoder_dir}/HomeUniversity.sav','rb'))
        college_code_encoder = pkl.load(open(f'{encoder_dir}/CollegeCode.sav','rb'))
        
        data = np.array([[
            college_name_encoder.transform([college_name]), college_code_encoder.transform([float(college_code)]), 
            merit_no, merit_marks, gender_encoder.transform([gender]), candidate_type_encoder.transform([candidate_type]), 
            category_encoder.transform([category]), home_university_encoder.transform([home_university]), 
            ph_type_encoder.transform([ph_type]), defence_type_encoder.transform([defence_type]), hsc_eligibility, 
            cap_round_encoder.transform([cap_round]), nationality_encoder.transform([nationality])
        ]])
        results = []
        for model in models:
            results.append([model['model_name'], branch_encoder.inverse_transform(model['model'].predict(data).tolist())[0]])
        show_variables = dict(
            online=gethostbyname(gethostname())!='127.0.0.1',
            offline=gethostbyname(gethostname())=='127.0.0.1',
            results=results,
        )
        return render(request, 'research/branch_prediction_result.html', context=show_variables)
    show_variables = dict(
        online=gethostbyname(gethostname())!='127.0.0.1',offline=gethostbyname(gethostname())=='127.0.0.1',
        ucn=unique_college_names,
        ucc=unique_college_codes,
        ug=unique_genders,
        uct=unique_candidate_types,
        uc=unique_categories,
        uhu=unique_home_universities,
        upt=unique_ph_types,
        udt=unique_defence_types,
        un=unique_nationalities,
        ucr=unique_cap_rounds,
        ub=unique_branches,
    )
    return render(request, 'research/branch_prediction.html', context=show_variables)

def college_prediction(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '..', 'models', 'college_prediction')
    encoder_dir = os.path.join(base_dir, '..', 'encoders', 'college_prediction')
    scaler_dir = os.path.join(base_dir, '..', 'scalers', 'college_prediction')
    unique_dir = os.path.join(base_dir, '..', 'unique', 'college_prediction')
    data_dir = os.path.join(base_dir, '..', 'data', 'college_prediction')
    
    unique_genders = pkl.load(open(f'{unique_dir}/genders.sav','rb'))
    unique_candidate_types = pkl.load(open(f'{unique_dir}/candidate_types.sav','rb'))
    unique_home_universities = pkl.load(open(f'{unique_dir}/home_universities.sav','rb'))
    unique_cap_rounds = pkl.load(open(f'{unique_dir}/cap_rounds.sav','rb'))
    unique_branches = pkl.load(open(f'{unique_dir}/branches.sav','rb'))
    unique_nationalities = pkl.load(open(f'{unique_dir}/nationalities.sav','rb'))
    unique_categories_ph_defence_types = pkl.load(open(f'{unique_dir}/category_ph_defence_types.sav','rb'))
    unique_college_names = pkl.load(open(f'{unique_dir}/college_names.sav','rb'))
    
    models = []
    models.append(dict(model_name='LogisticRegression', model=pkl.load(open(f'{model_dir}/LogisticRegression.sav','rb'))))
    models.append(dict(model_name='SupportVectorClassifier', model=pkl.load(open(f'{model_dir}/SupportVectorClassifier.sav','rb'))))
    models.append(dict(model_name='DecisionTreeClassifier', model=pkl.load(open(f'{model_dir}/DecisionTreeClassifier.sav','rb'))))
    models.append(dict(model_name='RandomForestClassifier', model=pkl.load(open(f'{model_dir}/RandomForestClassifier.sav','rb'))))
    models.append(dict(model_name='GaussianNB', model=pkl.load(open(f'{model_dir}/GaussianNB.sav','rb'))))
    models.append(dict(model_name='KNeighborsClassifier', model=pkl.load(open(f'{model_dir}/KNeighborsClassifier.sav','rb'))))
    
    if request.method == 'POST':
        merit_no = request.POST['merit_no']
        merit_marks = request.POST['merit_marks']
        hsc_eligibility = request.POST['hsc_eligibility']
        gender = request.POST['gender']
        candidate_type = request.POST['candidate_type']
        home_university = request.POST['home_university']
        cap_round = request.POST['cap_round']
        branch = request.POST['branch']
        nationality = request.POST['nationality']
        category_ph_defence_type = request.POST['category_ph_defence_type']
        # college_name = request.POST['college_name']
        
        gender_encoder = pkl.load(open(f'{encoder_dir}/Gender.sav','rb'))
        candidate_type_encoder = pkl.load(open(f'{encoder_dir}/CandidateType.sav','rb'))
        home_university_encoder = pkl.load(open(f'{encoder_dir}/HomeUniversity.sav','rb'))
        cap_round_encoder = pkl.load(open(f'{encoder_dir}/CAPRound.sav','rb'))
        branch_encoder = pkl.load(open(f'{encoder_dir}/BRANCH.sav','rb'))
        nationality_encoder = pkl.load(open(f'{encoder_dir}/NATIONALITY.sav','rb'))
        category_ph_defence_encoder = pkl.load(open(f'{encoder_dir}/Category_PH_DefenceType.sav','rb'))
        college_name_encoder = pkl.load(open(f'{encoder_dir}/CollegeName.sav','rb'))
        
        data = np.array([[
            merit_no, merit_marks, gender_encoder.transform([gender]), candidate_type_encoder.transform([candidate_type]), 
            home_university_encoder.transform([home_university]), hsc_eligibility, cap_round_encoder.transform([cap_round]), 
            branch_encoder.transform([branch]), nationality_encoder.transform([nationality]), 
            category_ph_defence_encoder.transform([category_ph_defence_type])
        ]])
        scaler = pkl.load(open(f'{scaler_dir}/scaler.sav', 'rb'))
        data = scaler.transform(data)
        results = []
        for model in models:
            results.append([model['model_name'], college_name_encoder.inverse_transform(model['model'].predict(data).tolist())[0]])
        show_variables = dict(
            online=gethostbyname(gethostname())!='127.0.0.1',
            offline=gethostbyname(gethostname())=='127.0.0.1',
            results=results,
        )
        return render(request, 'research/college_prediction_result.html', context=show_variables)

    show_variables = dict(
        online=gethostbyname(gethostname())!='127.0.0.1',
        offline=gethostbyname(gethostname())=='127.0.0.1',
        ug=unique_genders,
        uct=unique_candidate_types,
        uhu=unique_home_universities,
        ucr=unique_cap_rounds,
        ub=unique_branches,
        un=unique_nationalities,
        ucpdt=unique_categories_ph_defence_types,
        ucn=unique_college_names,
    )
    return render(request, 'research/college_prediction.html', context=show_variables)

def placement_prediction(request):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '..', 'models', 'college_prediction')
    encoder_dir = os.path.join(base_dir, '..', 'encoders', 'college_prediction')
    scaler_dir = os.path.join(base_dir, '..', 'scalers', 'college_prediction')
    unique_dir = os.path.join(base_dir, '..', 'unique', 'college_prediction')
    data_dir = os.path.join(base_dir, '..', 'data', 'college_prediction')
    
    
    if request.method == 'POST':
        
        results = []
        # for model in models:
        #     results.append([model['model_name'], college_name_encoder.inverse_transform(model['model'].predict(data).tolist())[0]])
        show_variables = dict(
            online=gethostbyname(gethostname())!='127.0.0.1',
            offline=gethostbyname(gethostname())=='127.0.0.1',
            results=results
        )
        return render(request, 'research/placement_prediction_result.html')
    show_variables = dict(
        online=gethostbyname(gethostname())!='127.0.0.1',
        offline=gethostbyname(gethostname())=='127.0.0.1'
    )
    return render(request, 'research/placement_prediction.html', context=show_variables)