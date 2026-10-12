"""FINAL health-claims scenario features & reason-code groups.
7 reason codes, each an FWA fraud scenario; provider archetype decomposed into
Setting (where) x Specialty (what) so the archetype emerges as their interaction.
Raw Claim_Amount excluded (predictive but illogical: mid-range=fraud, synthetic artifact)."""
import pandas as pd, numpy as np

def _z(df,col,by,ref): return ((df[col]-df[by].map(ref[(col,by,'mean')]))/df[by].map(ref[(col,by,'std')]).replace(0,np.nan)).fillna(0)

def fit_ref(df):
    d=df.copy(); d['amt_per_proc']=d['Claim_Amount']/d['Number_of_Procedures'].clip(lower=1); ref={}
    for col,by in [('Number_of_Procedures','Diagnosis_Code'),('amt_per_proc','Procedure_Code')]:
        g=d.groupby(by)[col]; ref[(col,by,'mean')]=g.mean(); ref[(col,by,'std')]=g.std()
    ref['app_q85']=d['amt_per_proc'].quantile(.85); return ref

def engineer(df,ref):
    d=df.copy(); d['amt_per_proc']=d['Claim_Amount']/d['Number_of_Procedures'].clip(lower=1); o=pd.DataFrame(index=d.index)
    o['Provider_Type_indexed']=pd.factorize(d['Provider_Type'],sort=True)[0]
    o['specialist_office_flag']=(d['Provider_Type']=='Specialist Office').astype(int)
    o['Provider_Specialty_indexed']=pd.factorize(d['Provider_Specialty'],sort=True)[0]
    o['high_risk_specialty_flag']=d['Provider_Specialty'].isin(['Cardiology','Orthopedics']).astype(int)
    o['Number_of_Procedures']=d['Number_of_Procedures']
    o['high_procedure_flag']=(d['Number_of_Procedures']>=8).astype(int)
    o['procs_vs_dx_z']=_z(d,'Number_of_Procedures','Diagnosis_Code',ref)
    o['unbundling_nonInpatient']=((d['Number_of_Procedures']>=8)&(d['Service_Type']!='Inpatient')).astype(int)
    o['amt_per_proc']=d['amt_per_proc']
    o['app_vs_proc_z']=_z(d,'amt_per_proc','Procedure_Code',ref)
    o['upcoding_flag']=((d['amt_per_proc']>ref['app_q85'])&(d['Number_of_Procedures']<=2)).astype(int)
    o['Provider_Patient_Distance_Miles']=d['Provider_Patient_Distance_Miles']
    o['far_encounter_flag']=(d['Provider_Patient_Distance_Miles']>600).astype(int)
    o['far_nonInpatient']=((d['Provider_Patient_Distance_Miles']>600)&(d['Service_Type']!='Inpatient')).astype(int)
    o['Admission_Type_indexed']=pd.factorize(d['Admission_Type'],sort=True)[0]
    o['Service_Type_indexed']=pd.factorize(d['Service_Type'],sort=True)[0]
    o['elective_high_intensity']=((d['Admission_Type']=='Elective')&(d['Number_of_Procedures']>=6)).astype(int)
    o['newborn_adult_specialty']=((d['Admission_Type']=='Newborn')&d['Provider_Specialty'].isin(['Cardiology','Orthopedics','Oncology'])).astype(int)
    o['Diagnosis_Code_indexed']=pd.factorize(d['Diagnosis_Code'],sort=True)[0]
    o['Procedure_Code_indexed']=pd.factorize(d['Procedure_Code'],sort=True)[0]
    o['high_risk_dx_flag']=d['Diagnosis_Code'].isin(['R05','J02.9']).astype(int)
    return o

GROUPS={
 'Provider Setting Risk':['Provider_Type_indexed','specialist_office_flag'],
 'Provider Specialty Risk':['Provider_Specialty_indexed','high_risk_specialty_flag'],
 'Unbundling / Overutilization':['Number_of_Procedures','high_procedure_flag','procs_vs_dx_z','unbundling_nonInpatient'],
 'Upcoding':['amt_per_proc','app_vs_proc_z','upcoding_flag'],
 'Phantom / Implausible Encounter':['Provider_Patient_Distance_Miles','far_encounter_flag','far_nonInpatient'],
 'Medical Necessity / Setting':['Admission_Type_indexed','Service_Type_indexed','elective_high_intensity','newborn_adult_specialty'],
 'Clinical Coding':['Diagnosis_Code_indexed','Procedure_Code_indexed','high_risk_dx_flag'],
}
SCENARIO_MEANING={
 'Provider Setting Risk':'Where care is delivered (specialist-office setting)',
 'Provider Specialty Risk':'What specialty (cardiology/orthopedics elevated)',
 'Unbundling / Overutilization':'Excess procedures for the case',
 'Upcoding':'High amount for few procedures',
 'Phantom / Implausible Encounter':'Encounter unlikely to have occurred (distance)',
 'Medical Necessity / Setting':'Care setting/admission does not fit',
 'Clinical Coding':'Diagnosis/procedure codes carry risk',
}
MONO={'specialist_office_flag':1,'high_risk_specialty_flag':1,'high_procedure_flag':1,'procs_vs_dx_z':1,'unbundling_nonInpatient':1,
      'amt_per_proc':1,'app_vs_proc_z':1,'upcoding_flag':1,'Provider_Patient_Distance_Miles':1,'far_encounter_flag':1,
      'far_nonInpatient':1,'elective_high_intensity':1,'newborn_adult_specialty':1,'high_risk_dx_flag':1}
