import pandas as pd, numpy as np, glob, os
cols=pd.read_csv('/home/claude/data/unsw/NUSW-NB15_features.csv',encoding='latin1')['Name'].str.strip().tolist()
dfs=[]
for i in range(1,5):
    d=pd.read_csv(f'/home/claude/data/unsw/UNSW-NB15_{i}.csv',header=None,names=cols,low_memory=False)
    d['file']=i; dfs.append(d)
df=pd.concat(dfs,ignore_index=True)
df['attack_cat']=df['attack_cat'].fillna('Normal').astype(str).str.strip().replace({'Backdoors':'Backdoor','':'Normal'})
for c in ['sport','dsport']:
    df[c]=pd.to_numeric(df[c],errors='coerce')
df=df.sort_values(['Stime','Ltime'],kind='stable').reset_index(drop=True)
df['day']=pd.to_datetime(df['Stime'],unit='s').dt.strftime('%Y-%m-%d')
print(df.shape); print(df.groupby(['day','attack_cat']).size().unstack(0).fillna(0).astype(int))
print(df.groupby('file')['day'].agg(lambda s:s.value_counts().to_dict()))
print(pd.to_datetime(df.groupby('day')['Stime'].agg(['min','max']).stack(),unit='s'))
df.to_pickle('/home/claude/work/unsw_all.pkl')
