import pandas as pd
import scipy as sp 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import random
import statsmodels.api as sm
import copy
from sklearn.model_selection import ShuffleSplit

sns.set(style="white",font_scale=1.5) #whitegrid

df=pd.read_csv('../ON_freqs.csv')

df=df[df['time point']>=5]

# coefficient_of_dermination = r2_score(y, p(x))

# dimension, lag, maximum time
dimslags=[
  (1,0,12),
  (2,1,11),
  (2,2,10),
  (3,1,10),
  (3,2,8),
]

datadic={
  'dim':[],
  'lag':[],
  'lyapunov point':[],
  'l ci':[],
  'u ci':[],
  'pval':[],
  'msd':[],
  'msd stderr':[],
}

for (dim,lag,maxtime) in dimslags:
  # reconstruct phase space
  
  #
  #sfreq=np.array(df1['S freq'])
  df2=copy.deepcopy(df[df['time point']<=maxtime])
  df2['y1']=df2['S freq']
  if dim==2 or dim==3:
    df1=df.copy()
    df1['time point']-=lag
    df1=df1[(df1['time point']<=maxtime) & (df1['time point']>=5)]
    print(len(df1),len(df2))
    df2['y2']=np.array(df1['S freq'])
  if dim==3:
    df1=df.copy()
    df1['time point']-=lag*2
    df1=df1[(df1['time point']<=maxtime) & (df1['time point']>=5)]
    df2['y3']=np.array(df1['S freq'])
  df2.to_csv('{}_{}_timedelayembedding.csv'.format(dim,lag))
  # find nearest neighbor
  neighbors=[]
  for i,group0 in df2[df2['time point']==5].groupby(by='time'):
    for j,row in group0.iterrows():
      group=group0.copy()
      dd0=np.vstack([np.array((row['y'+str(nn)]-group['y'+str(nn)])**2) for nn in range(1,dim+1)])
      group['dist']=np.sqrt(np.sum(dd0,axis=0))
      group[group['dist']==0]=np.inf
      neighbors.append((str(row['well']),group.loc[group['dist']==np.min(group['dist']), 'well'].item()))
  
  ts=[]
  ds=[] #log distance
  for (i,j) in neighbors:
    for ti in range(5,maxtime+1):
      row = df2[df2['time point']==ti]
      
      ts.append(np.float(row[row['well']==i]['time']))
      d00=np.sqrt(np.sum([(np.float(row[row['well']==i]['y'+str(1)]) - np.float(row[row['well']==j]['y'+str(1)]))**2 for nn in range(1,dim+1)]))
      ds.append(np.log(d00))
   # k-fold cross-validation
  #print(ts)
  #print(ds)
  ts=np.array(ts)
  ds=np.array(ds)
  #print(len(ts))
  kf = ShuffleSplit(n_splits=1000,train_size=30)
  kf.get_n_splits(ts)
  msd=[]
  for (train_index, test_index) in kf.split(ts):
    #print(len(train_index))
    p=np.polyfit(ts[train_index],ds[train_index],1)
    msd+=list((ts[test_index]*p[0] + p[1] - ds[test_index])**2)
  print(dim,lag,'msd',np.mean(msd),np.std(msd)/np.sqrt(len(msd)))
  mean_msd=np.mean(msd)
  stderr_msd=np.std(msd)/np.sqrt(len(msd))
  lya=np.polyfit(ts,ds,1)[0]
  #get pvals and cis
  indices=np.array(list(range(len(ts))))
  boot_lya=[]
  for i in range(100000):
    bi=np.random.choice(indices,size=len(ts))
    boot_lya.append(np.polyfit(ts[bi],ds[bi],1)[0])
  boot_lya=np.array(boot_lya)
  lci=np.quantile(boot_lya,0.025)
  uci=np.quantile(boot_lya,0.975)
  pval=len(boot_lya[boot_lya<0])/len(boot_lya)
  datadic['dim'].append(dim)
  datadic['lag'].append(lag)
  datadic['lyapunov point'].append(lya)
  datadic['l ci'].append(lci)
  datadic['u ci'].append(uci)
  datadic['pval'].append(pval)
  datadic['msd'].append(mean_msd)
  datadic['msd stderr'].append(stderr_msd)

pd.DataFrame(datadic).to_csv('rosenstein_cv.csv',index=False)
    


