import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(df):
    df2=pd.read_csv('STUDENT_DATA13.csv',usecols=['Merit Marks'])
    for Marks in range(len(df2)):
        if(df2.iloc[Marks,0])>1000000:
            (df2.iloc[Marks,0])=(df2.iloc[Marks,0])/100000
        if(df2.iloc[Marks,0])>100000:
            (df2.iloc[Marks,0])=(df2.iloc[Marks,0])/10000
        elif(df2.iloc[Marks,0])>10000:
            (df2.iloc[Marks,0])=(df2.iloc[Marks,0])/1000
        elif(df2.iloc[Marks,0])>1000:
            (df2.iloc[Marks,0])=(df2.iloc[Marks,0])/100
        elif(df2.iloc[Marks,0])>100:
            (df2.iloc[Marks,0])=(df2.iloc[Marks,0])/10
    df["Merit Marks"]=df2["Merit Marks"]
    print('\nMax Merit Marks: ',df['Merit Marks'].max())
    df2=pd.read_csv(r"STUDENT_DATA13.csv",usecols=["HSC Eligibility"])
    for Marks in range(len(df2)):
        if(df2.iloc[Marks,0])>100000:
            (df2.iloc[Marks,0])=(df2.iloc[Marks,0])/10000
        elif(df2.iloc[Marks,0])>10000:
            (df2.iloc[Marks,0])=(df2.iloc[Marks,0])/1000
        elif(df2.iloc[Marks,0])>1000:
            (df2.iloc[Marks,0])=(df2.iloc[Marks,0])/100
        elif(df2.iloc[Marks,0])>100:
            (df2.iloc[Marks,0])=(df2.iloc[Marks,0])/10
    df["HSC Eligibility"]=df2["HSC Eligibility"]
    print('\nMax HSC Eligibility: ',df["HSC Eligibility"].max())
    df['College Name'].replace('PVG CET, PUNE', 'PVG CET', inplace=True)
    df['College Name'].replace('DY PATIL AKURDI', 'DY PATIL AK', inplace=True)
    df['College Name'].replace('PICT, PUNE', 'PICT', inplace=True)
    df['College Name'].replace('MMMIT, LOHGAON, PUNE', 'MMMIT, LOH', inplace=True)
    df['College Name'].replace('KJCOEMR, PISOLI', 'KJCOEMR', inplace=True)
    df['College Name'].replace('KJ TRINITY COER', 'KJ TRINITY', inplace=True)
    df['College Name'].replace('SINHGAD KONDHWA (BK)', 'SINHGAD KONDHWA', inplace=True)
    df['College Name'].replace('INDIRA COEM, PUNE', 'INDIRA COEM', inplace=True)
    df['College Name'].replace('SINHGAD SKN, VADGAON', 'SINHGAD SKN', inplace=True)
    df['College Name'].replace('PES MODERN COE, PUNE', 'PES MODERN', inplace=True)
    df['College Name'].replace('TSSM PVPIT, BAVDHAN', 'PVPIT, BAVDHAN', inplace=True)
    df['College Name'].replace('SINHGAD ACADEMY, KONDHWA', 'SINHGAD ACAD, KONDHWA', inplace=True)
    df['College Name'].replace('PIMPRI CHINCHWAD COLLEGE OF ENGG AND RESEARCH', 'PCCOE, Ravet', inplace=True)
    df['College Name'].replace('PIMPRI\xa0CHINCHWAD\xa0COLLEGE OF\xa0ENGG\xa0AND\xa0RESEARCH', 'PCCOE, Ravet', inplace=True)
    df['College Name'].replace('PIMPRI燙HINCHWAD燙OLLEGE OF燛NGG燗ND燫ESEARCH', 'PCCOE, Ravet', inplace=True)
    df['College Name'].replace('SC0E, VADGAON', 'SCOE, VADGAON', inplace=True)
    df['College Name'].replace('PIMPRI CHINCHWAD COLLEGE OF ENGINEERING', 'PCCOE', inplace=True)
    df['College Name'].replace('MAEERS M.I.T COLLEGE OF ENGINEERING', 'M.I.T COE', inplace=True)
    df['College Name'].replace('DY PATIL IET,PIMPRI', 'DY PATIL,PIMPRI', inplace=True)
    df['College Name'].replace('MMCOE, KARVENAGAR, PUNE', 'MMCOE,PUNE', inplace=True)
    
    df['Defence Type'].replace('DEF-3', 'DEF3', inplace=True)
    df['Defence Type'].replace('DEF 2', 'DEF2', inplace=True)
    df['Defence Type'].replace('DEF 3', 'DEF3', inplace=True)
    df['Defence Type'].replace('DEF 1', 'DEF1', inplace=True)
    df['Defence Type'].replace('DEF-1', 'DEF1', inplace=True)
    df['Defence Type'].replace('DEF-3', 'DEF3', inplace=True)
    
    df['Category'].replace('OPEN ', 'OPEN', inplace=True)
    
    df['Home University'].replace('S.R.T.M.U', 'S.R.T.M.U.', inplace=True)
    df['Home University'].replace('SRTMU', 'S.R.T.M.U.', inplace=True)
    df['Home University'].replace('B.A.M.U', 'B.A.M.U.', inplace=True)
    df['Home University'].replace('BAMU', 'B.A.M.U.', inplace=True)
    df['Home University'].replace('GONDWANE', 'GONDWANA', inplace=True)
    df['Home University'].replace('PUME', 'PUNE', inplace=True)
    df['Home University'].replace('M', 'MUMBAI', inplace=True) 
    df['Home University'].replace('OHU ', 'OHU', inplace=True)  
    df['Home University'].replace('0', 'OHU', inplace=True)
    df['Home University'].replace(' SHIVAJI ', 'SHIVAJI', inplace=True)
    df['Home University'].replace('SH', 'SHIVAJI', inplace=True)
    df['Home University'].replace('AMRAVATI ', 'AMRAVATI', inplace=True)
    df['Home University'].replace('SO', 'SOLAPUR', inplace=True)
    df['Home University'].replace('A', 'SOLAPUR', inplace=True)
    df['Home University'].replace('NAGAPUR', 'NAGPUR', inplace=True)
    df['Home University'].replace('NORTH MAHATRASHTRA', 'NORTH MAHARASHTRA', inplace=True)
    
    df['Category'].replace('O', 'OPEN', inplace=True)
    df['Category'].replace('PO', 'OPEN', inplace=True)
    df['Category'].replace('OBC ', 'OBC', inplace=True)
    df['Category'].replace('SCS', 'SC', inplace=True)
    df['Category'].replace('ST ', 'ST', inplace=True)
    df['Category'].replace('O', 'OPEN', inplace=True)
    df['Category'].replace('NY-C', 'NT-C', inplace=True)
    df['Category'].replace('NY-D', 'NT-D', inplace=True)
    df['Category'].replace('NT', 'NT-D', inplace=True)
    df['Category'].replace('NT--B', 'NT-B', inplace=True)
    df['Category'].replace('NA ', 'NT-D', inplace=True)
    df['Category'].replace('NT-D ', 'NT-D', inplace=True)
    df['Category'].replace('OMS', 'OBC', inplace=True)
    
    df['BRANCH'].replace('ELECTRONIC AND TELECOMMUNICATION ENG.', 'ENTC', inplace=True)
    df['BRANCH'].replace('INFORMATION TECHNOLOGY', 'IT', inplace=True)
    df['BRANCH'].replace('ELECTRONIC AND TELECOMMUNICATION ENGG', 'ENTC', inplace=True)
    df['BRANCH'].replace('ELECTRONICS AND TELECOMMUNICATION ENG.', 'ENTC', inplace=True)
    df['BRANCH'].replace('ELECTRONICS AND TELECOMUNICATION ENGG', 'ENTC', inplace=True)
    df['BRANCH'].replace('ELECTRONICS AND TELCOMMUNICATION ENGG', 'ENTC', inplace=True)
    df['BRANCH'].replace('PRODUCTION ENGINEERING', 'PRODUCTION', inplace=True)
    df['BRANCH'].replace('COMPUTER ENGINEERING', 'COMPUTER', inplace=True)
    df['BRANCH'].replace('COMPUTER  ENGINEERING', 'COMPUTER', inplace=True)    
    df['BRANCH'].replace('CIVIL ENGINEERING', 'CIVIL', inplace=True)
    df['BRANCH'].replace('MECHANICAL ENGINEERING', 'MECHANICAL', inplace=True)
    df['BRANCH'].replace('MECHANICAL  ENGINEERING', 'MECHANICAL', inplace=True)
    df['BRANCH'].replace('MACHANICAL  ENGINEERING', 'MECHANICAL', inplace=True)
    df['BRANCH'].replace('ELECTRIC ENGINEERING', 'ELECTRICAL', inplace=True)
    df['BRANCH'].replace('ELECTRICAL ENGINEERING', 'ELECTRICAL', inplace=True)
    df['BRANCH'].replace('ELECTRONIC', 'ELECTRONICS', inplace=True)
    df['BRANCH'].replace('AUTOMOBILE ENG.', 'AUTOMOBILE', inplace=True)
    df['BRANCH'].replace('POLYMER ENGINEERING', 'POLYMER', inplace=True)
    df['BRANCH'].replace('PETRO CHEMICAL ENGINNEERING', 'PETRO CHEMICAL', inplace=True)
    df['BRANCH'].replace('MECHANICAL ENGINEERING', 'MECHANICAL', inplace=True)
    df['BRANCH'].replace('PETROLEUM ENGINEERING', 'PETROLEUM', inplace=True)
    df['BRANCH'].replace('AUTOMOBILE ENG.', 'AUTOMOBILE', inplace=True)
    df['BRANCH'].replace('POLYMER ENGINEERING', 'POLYMER', inplace=True)
    df['BRANCH'].replace('ELECTRONICS AND TELECOMMUNICATION ENGG', 'ENTC', inplace=True)
    df['BRANCH'].replace('MECHANICAL ENGINEERING [SANDWICH]', 'MECHANICAL[SANDWICH]', inplace=True)
    df['BRANCH'].replace('ELECTRONICS AND TELTCOMMNICATION ENGG', 'ENTC', inplace=True)
    df['BRANCH'].replace('INSTRUMENTATION  ENGINEERING', 'INSTRUMENTATION', inplace=True)
    df['BRANCH'].replace('INSTRUMENTAION ENGINEERING', 'INSTRUMENTATION', inplace=True)
                         
    df['Candidate Type'].replace('TYPE  A', 'TYPE A', inplace=True)
    df['Candidate Type'].replace('TYPE A ',  'TYPE A', inplace=True)
    df['Candidate Type'].replace('TYPE  B', 'TYPE B', inplace=True)
    df['Candidate Type'].replace('TYPE B ', 'TYPE B', inplace=True)
    df['Candidate Type'].replace('TYPE  C', 'TYPE C', inplace=True)
    df['Candidate Type'].replace('MAHARASHTRA TYPE E', 'TYPE E', inplace=True)
    df['Candidate Type'].replace('TYPEC', 'TYPE C', inplace=True)
    df['Candidate Type'].replace('J & K', 'J&K', inplace=True)
    df['Candidate Type'].replace('TYPE ', 'TYPE E', inplace=True)
    df['Candidate Type'].replace('J&K SPECIAL SCHOLARSHIP SCHEME CANDIDATE', 'J&K SSSC', inplace=True)
    df['Candidate Type'].replace('J&K SPECIAL SCHOLARSHIP SCHEME', 'J&K SSSC', inplace=True)
    df['Candidate Type'].replace('JKSSS', 'J&K SSSC', inplace=True)
    df['Candidate Type'].replace('JKSSS', 'J&K SSSC', inplace=True)
    df['Candidate Type'].replace('J & K SPECIAL SCHOLARSHIP SCHEME CANDIDATE', 'J&K SSSC', inplace=True)
    df['Candidate Type'].replace('J & K SPECIAL SCHOLORSHIP SCHEME CANDIDATE', 'J&K SSSC', inplace=True)
    df['Candidate Type'].replace('J&K SPECIAL SCHOLARSHIP CANDIDATE', 'J&K SSSC', inplace=True)
    df['Candidate Type'].replace('TYEP A', 'TYPE A', inplace=True)
    df['Candidate Type'].replace('TYPEA', 'TYPE A', inplace=True)
    df['Candidate Type'].replace('GoI', 'PIO', inplace=True)
    df['Candidate Type'].replace('OMS ', 'OMS', inplace=True)

    df['CAP Round'].replace('ROUND 1', 'ROUND-1', inplace=True)
    df['CAP Round'].replace('ROUND 2',  'ROUND-2', inplace=True)
    df['CAP Round'].replace('ROUND 3', 'ROUND-3', inplace=True)
    df['CAP Round'].replace('ROUND 4', 'ROUND-4', inplace=True)
    df['CAP Round'].replace('ROUND 5', 'ROUND-5', inplace=True)
    df['CAP Round'].replace('ROIND 1', 'ROUND-1', inplace=True)
    df['CAP Round'].replace('ROUND 6', 'ROUND-6', inplace=True)
    df['CAP Round'].replace('ROUND 7',  'ROUND-7', inplace=True)
    df['CAP Round'].replace('ROUND 8', 'ROUND-8', inplace=True)
    df['CAP Round'].replace('ROUND 9', 'ROUND-9', inplace=True)
    df['CAP Round'].replace('ROUND 10', 'ROUND-10', inplace=True)
    df['CAP Round'].replace('ROIND 11', 'ROUND-11', inplace=True)
    df['CAP Round'].replace('ROUND 11', 'ROUND-11', inplace=True)
    df['CAP Round'].replace('ROUND 12', 'ROUND-12', inplace=True)
    df['CAP Round'].replace('ROUND 13',  'ROUND-13', inplace=True)
    df['CAP Round'].replace('ROUND 14', 'ROUND-14', inplace=True)
    df['CAP Round'].replace('  INSTITUTE LEVEL', 'IL', inplace=True)
    df['CAP Round'].replace('INSITITUT LEVEL', 'IL', inplace=True)
    df['CAP Round'].replace('INSTITUTE LEVEL', 'IL', inplace=True)
    df['CAP Round'].replace('COUNSELING ROUND', 'CR', inplace=True)
    df['CAP Round'].replace('COUNSELING ROUND ',  'CR', inplace=True)
    df['CAP Round'].replace('COUNSELLING ROUND', 'CR', inplace=True)
    df['CAP Round'].replace('COUNSLING ROUND', 'CR', inplace=True)
    df['CAP Round'].replace('COUNSELING  ROUND', 'CR', inplace=True)
    df['CAP Round'].replace('J & K COUNSELING', 'J&K COUNSELING', inplace=True)
    df['CAP Round'].replace('J & K COUNSELING ', 'J&K COUNSELING', inplace=True)
    df['CAP Round'].replace('J &K COUNSELLING', 'J&K COUNSELING', inplace=True)
    df['CAP Round'].replace('J & K COUNSINGH', 'J&K COUNSELING', inplace=True)
    df['CAP Round'].replace('J&K COUNCILING', 'J&K COUNSELING', inplace=True)
    df['CAP Round'].replace('J & K COUNSELLING', 'J&K COUNSELING', inplace=True)
    df['CAP Round'].replace('RANUD 2',  'ROUND-2', inplace=True)
    df['CAP Round'].replace('RANUD2',  'ROUND-2', inplace=True)
    df['CAP Round'].replace('ROUND 2 ',  'ROUND-2', inplace=True)
    df['CAP Round'].replace('RUNND-2',  'ROUND-2', inplace=True)
    df['CAP Round'].replace('ROUND2',  'ROUND-2', inplace=True)
    df['CAP Round'].replace('ROUN  1',  'ROUND-1', inplace=True)
    df['CAP Round'].replace('ROUN  1',  'ROUND-1', inplace=True)
    df['CAP Round'].replace('ROUND -1',  'ROUND-1', inplace=True)
    df['CAP Round'].replace(' ROUND-1',  'ROUND-1', inplace=True)
    df['CAP Round'].replace('0',  'ROUND-0', inplace=True)
    df['CAP Round'].replace('`',  'ROUND-1', inplace=True)

                                     
    df['Gender'].replace('m', 'M', inplace=True)
    df['Gender'].replace('MM', 'M', inplace=True)
    df['Gender'].replace('MF', 'M', inplace=True)
                         
    df['NATIONALITY'].replace('J & K', 'J&K', inplace=True)
    df["PH Type"]=np.where(pd.isnull(df['PH Type']), 'NA', df['PH Type'])
    df["Defence Type"]=np.where(pd.isnull(df['Defence Type']), 'NA', df['Defence Type'])
    df['Category'] = np.where(pd.isnull(df['Category']), 'OPEN', df['Category'])
    df['Home University'] = np.where(pd.isnull(df['Home University']), 'NA', df['Home University'])
    df['CAP Round'] = np.where(pd.isnull(df['CAP Round']), 'NA', df['CAP Round'])
       
    return df

def get_HSC_binned(df):
    bins = [0, 35, 50, 60, 75, 100]
    group_names = ['Fail', 'Pass Class', 'Second Class', 'First Class', 'Distinction']
    get_min_max_stats = lambda group: {'min': group.min(), 'max': group.max(),  'mean': group.mean()}
    df['HSC_binned'] = pd.cut(df['HSC Eligibility'], bins, labels=group_names)
    print('\n',df['HSC Eligibility'].groupby(df['HSC_binned']).apply(get_min_max_stats).unstack())
    df['HSC Eligibility'].groupby(df['HSC_binned']).apply(get_min_max_stats).unstack().plot(kind='bar')
    pd_ct = pd.crosstab([df['College Name']],df['HSC_binned'],margins=True)
    print('\n',pd_ct)
    pd_ct.plot(kind='bar',title='College Name-HSC Marks wise group')

def get_Merit_Marks_binned(df):
    bins = [0, 35, 50, 60, 75, 100]
    group_names = ['Fail', 'Pass Class', 'Second Class', 'First Class', 'Distinction']
    get_min_max_stats = lambda group: {'min': group.min(), 'max': group.max(),  'mean': group.mean()}
    df['Merit_Marks_binned'] = pd.cut(df['Merit Marks'], bins, labels=group_names)
    print('\n',df['Merit Marks'].groupby(df['Merit_Marks_binned']).apply(get_min_max_stats).unstack())
    df['Merit Marks'].groupby(df['Merit_Marks_binned']).apply(get_min_max_stats).unstack().plot(kind='bar')
    pd_ct = pd.crosstab([df['College Name']],df['Merit_Marks_binned'],margins=True)
    print('\n',pd_ct)
    pd_ct.plot(kind='bar',title='College Name-Merit_Marks wise group')
    
def EDA_After_PP(df):
    df_college=df[['College Code', 'Category', 'HSC Eligibility', 'Merit Marks','BRANCH']]
    title="Heatmap for 2D representation of Features"
    plt.figure(figsize=(9,5))
    sns.heatmap(df_college.corr(),annot=True,linewidth = 0.9, cmap='coolwarm')
    fig, ax = plt.subplots(figsize=(16,10))
    sns.countplot(x = "College Name", hue="Gender" , data = df, ax=ax)
    
    fig, ax = plt.subplots(figsize=(20,7))
    sns.countplot(x = "College Name", hue="Category" , data = df, ax=ax)
    
    fig, ax = plt.subplots(figsize=(16,10))
    sns.countplot(x = "College Name", hue="Home University" , data = df, ax=ax)
    
    fig, ax = plt.subplots(figsize=(16,10))
    sns.countplot(x = "College Name", hue="Candidate Type" , data = df, ax=ax)
    
    # Boxplot
    plt.figure(figsize = (16,10))
    sns.boxplot( x = 'College Name', y = 'Merit Marks', data = df)
    
    plt.figure(figsize = (16,10))
    sns.boxplot( x = 'College Name', y = "HSC Eligibility", data = df)
    
    #Management Quota Admission 
    df.boxplot(column=['Merit Marks'],by=["College Code","BRANCH"],rot=45,figsize = (20,14))
    
     
    plt.figure(figsize = (16,10))
    sns.boxplot( x = 'BRANCH', y = 'Merit Marks', data = df)
    
    fig, ax = plt.subplots(figsize=(16,14))
    sns.countplot(x = "College Name", hue="BRANCH" , data = df, ax=ax)

    return df

