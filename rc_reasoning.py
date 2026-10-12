"""Reason-code-level SHAP attribution plots + plain-language reasoning per code,
grounded in the claim's actual feature values."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from features_health_final import GROUPS, SCENARIO_MEANING

# ---------- reason-code-level attribution plots ----------
def rc_global_plot(G, y, path=None):
    """Left: distribution of each code's SHAP contribution, fraud vs legit. Right: mean |contribution|."""
    names=list(GROUPS); fig,(a1,a2)=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[2.2,1]})
    rng=np.random.RandomState(0)
    for k,g in enumerate(names):
        for lab,col,off in [(0,'#1f77b4',-0.18),(1,'#d62728',+0.18)]:
            v=G.loc[y.values==lab,g].values; jit=rng.uniform(-0.12,0.12,len(v))
            a1.scatter(v,np.full(len(v),k)+off+jit,s=4,alpha=0.35,c=col,edgecolors='none')
    a1.set_yticks(range(len(names))); a1.set_yticklabels(names); a1.invert_yaxis(); a1.axvline(0,c='k',lw=.8)
    a1.set_xlabel('Reason-code SHAP contribution (log-odds)'); a1.set_title('Per-claim contribution by reason code  (blue=legit, red=fraud)')
    imp=G.abs().mean()[names]; a2.barh(range(len(names)),imp.values,color='#555'); a2.set_yticks(range(len(names))); a2.set_yticklabels([]); a2.invert_yaxis()
    a2.set_xlabel('mean |contribution|'); a2.set_title('Global reason-code importance')
    plt.tight_layout()
    if path: plt.savefig(path,dpi=150,bbox_inches='tight')
    plt.show()

def rc_waterfall(G, base_value, i, title=None, path=None):
    """Waterfall from base log-odds to f(x) using the 7 reason-code contributions."""
    names=list(GROUPS); row=G.loc[i,names].sort_values(key=lambda s: -s.abs())
    fig,ax=plt.subplots(figsize=(8.5,4.8)); cum=base_value; ys=[]
    for k,(g,v) in enumerate(row.items()):
        ax.barh(k,v,left=cum,color='#d62728' if v>0 else '#1f77b4'); ax.text(cum+v+(0.05 if v>=0 else -0.05),k,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
        cum+=v; ys.append(g)
    ax.axvline(base_value,c='gray',ls='--',lw=.8); ax.axvline(cum,c='k',lw=1)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys); ax.invert_yaxis()
    ax.set_xlabel(f'log-odds   (base E[f]={base_value:.2f}  →  f(x)={cum:.2f},  P={1/(1+np.exp(-cum)):.3f})')
    ax.set_title(title or f'Reason-code waterfall — claim {i}'); plt.tight_layout()
    if path: plt.savefig(path,dpi=150,bbox_inches='tight')
    plt.show()

# ---------- plain-language reasoning ----------
def _phrase(f, raw, X, z_ref=None):
    v=X[f]
    if f=='specialist_office_flag': return 'care delivered at a Specialist Office (elevated-risk setting)' if v==1 else 'non-specialist setting'
    if f=='Provider_Type_indexed': return f"provider type is {raw['Provider_Type']}"
    if f=='high_risk_specialty_flag': return f"specialty is {raw['Provider_Specialty']} (elevated-risk specialty)" if v==1 else f"specialty is {raw['Provider_Specialty']}"
    if f=='Provider_Specialty_indexed': return f"specialty is {raw['Provider_Specialty']}"
    if f=='unbundling_nonInpatient': return f"{raw['Number_of_Procedures']} procedures billed on a non-inpatient ({raw['Service_Type']}) claim" if v==1 else None
    if f=='high_procedure_flag': return f"{raw['Number_of_Procedures']} procedures on a single claim" if v==1 else None
    if f=='Number_of_Procedures': return f"{int(v)} procedures billed"
    if f=='procs_vs_dx_z': return f"procedure count is {v:+.1f} sd vs the norm for diagnosis {raw['Diagnosis_Code']}"
    if f=='amt_per_proc': return f"${v:,.0f} billed per procedure"
    if f=='app_vs_proc_z': return f"amount per procedure is {v:+.1f} sd vs the norm for procedure {raw['Procedure_Code']}"
    if f=='upcoding_flag': return f"high amount (${raw['Claim_Amount']:,.0f}) spread over only {raw['Number_of_Procedures']} procedure(s)" if v==1 else None
    if f=='Provider_Patient_Distance_Miles': return f"patient is {v:,.0f} miles from the provider"
    if f=='far_encounter_flag': return 'encounter >600 miles from patient' if v==1 else None
    if f=='far_nonInpatient': return f">600 miles travelled for a non-inpatient ({raw['Service_Type']}) service" if v==1 else None
    if f=='Admission_Type_indexed': return f"admission type is {raw['Admission_Type']}"
    if f=='Service_Type_indexed': return f"service type is {raw['Service_Type']}"
    if f=='elective_high_intensity': return f"elective admission with {raw['Number_of_Procedures']} procedures" if v==1 else None
    if f=='newborn_adult_specialty': return f"newborn admission billed by {raw['Provider_Specialty']} (adult specialty)" if v==1 else None
    if f=='Diagnosis_Code_indexed': return f"diagnosis {raw['Diagnosis_Code']}"
    if f=='Procedure_Code_indexed': return f"procedure {raw['Procedure_Code']}"
    if f=='high_risk_dx_flag': return f"diagnosis {raw['Diagnosis_Code']} is an elevated-risk code" if v==1 else None
    return None

TOPIC={'specialist_office_flag':'setting','Provider_Type_indexed':'setting','high_risk_specialty_flag':'specialty',
       'Provider_Specialty_indexed':'specialty','unbundling_nonInpatient':'procs','high_procedure_flag':'procs',
       'Number_of_Procedures':'procs','procs_vs_dx_z':'procs_norm','amt_per_proc':'amount','app_vs_proc_z':'amount_norm',
       'upcoding_flag':'amount','Provider_Patient_Distance_Miles':'distance','far_encounter_flag':'distance',
       'far_nonInpatient':'distance','Admission_Type_indexed':'admission','Service_Type_indexed':'service',
       'elective_high_intensity':'admission','newborn_adult_specialty':'admission','Diagnosis_Code_indexed':'dx',
       'Procedure_Code_indexed':'proc','high_risk_dx_flag':'dx'}

def reasoning(i, SV, G, RC, Xte, df_raw, k=3, n_feat=2):
    """For claim i: top-k codes, each with a one-line reason built from the top contributing features and their actual values.
    Deduplicates by underlying attribute so a flag and its base categorical do not both appear."""
    raw=df_raw.loc[i]; X=Xte.loc[i]; out=[]
    top=G.loc[i].clip(lower=0).sort_values(ascending=False).head(k)
    for g in top.index:
        fs=GROUPS[g]; contrib=SV.loc[i,fs].sort_values(ascending=False)
        phrases=[]; used=set()
        for f in contrib.index:
            if contrib[f]<=0 or len(phrases)>=n_feat: break
            t=TOPIC.get(f,f)
            if t in used: continue
            ph=_phrase(f,raw,X)
            if ph: phrases.append(ph); used.add(t)
        because='; '.join(phrases) if phrases else SCENARIO_MEANING[g].lower()
        out.append(f"{g} — score {RC.loc[i,g]:.2f} ({G.loc[i,g]:+.2f} log-odds)\n      because: {because}")
    return out
