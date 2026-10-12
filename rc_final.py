"""Reason-code scoring (standard Shapley grouping), top-k output, diversity metrics,
and the interaction-value compound diagnostic."""
import numpy as np, pandas as pd, itertools
from features_health_final import GROUPS

def shifted_sigmoid(x,target_at_zero=0.3,k=1.0):
    b=np.log(target_at_zero/(1-target_at_zero)); return 1/(1+np.exp(-(k*x+b)))

def group_shap(shap_df):   # standard Shapley grouping: each interaction split equally (axiomatic)
    return pd.DataFrame({g:shap_df[fs].sum(axis=1) for g,fs in GROUPS.items()},index=shap_df.index)

def reason_scores(G,target_at_zero=0.3): return G.apply(lambda c: shifted_sigmoid(c,target_at_zero))

def top_k(G,scores,k=3):
    """Per-prediction: top-k reason codes by positive contribution, with log-odds contribution and 0-1 score."""
    rows=[]
    for i in G.index:
        pos=G.loc[i].clip(lower=0).sort_values(ascending=False).head(k)
        rows.append({f'rc{j+1}':f'{g} ({G.loc[i,g]:+.2f} | score {scores.loc[i,g]:.2f})' for j,g in enumerate(pos.index)})
    return pd.DataFrame(rows,index=G.index)

def diversity(G,y):
    """Own-scale diversity: each code ranked against its own distribution; top-driver share among fraud cases."""
    names=list(GROUPS); P=G.rank(pct=True)[y==1]
    top=P.idxmax(axis=1).value_counts(normalize=True).reindex(names).fillna(0); t=top[top>0]
    ent=-(t*np.log(t)).sum()/np.log(len(names)); high=(P>=0.8).mean()
    fr=G[y==1]; corr=np.abs(fr.corr().values[np.triu_indices(len(names),1)]).mean()
    return pd.DataFrame({'top_driver_share':top,'fires_high_on_fraud':high}),ent,corr

def compound_diagnostic(INT,cols,y):
    """Interaction-value decomposition: how much attribution is cross-scenario, and which named pairs dominate."""
    names=list(GROUPS); gidx={g:[cols.index(f) for f in fs] for g,fs in GROUPS.items()}
    own=np.column_stack([INT[:,gidx[g]][:,:,gidx[g]].sum(axis=(1,2)) for g in names])
    cross={f'{a} x {b}':2*INT[:,gidx[a]][:,:,gidx[b]].sum(axis=(1,2)) for a,b in itertools.combinations(names,2)}
    C=pd.DataFrame(cross); fr=(y==1).values
    share_all=np.abs(C.values).sum()/np.abs(INT).sum()
    pos_fr=np.clip(C.values[fr],0,None).sum(); tot_fr=np.clip(INT[fr],0,None).sum()
    dom=C[fr].idxmax(axis=1).value_counts(normalize=True)
    return share_all,pos_fr/tot_fr,dom
