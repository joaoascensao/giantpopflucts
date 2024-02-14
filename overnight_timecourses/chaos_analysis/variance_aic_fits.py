import pandas as pd
import scipy as sp 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize
from scipy.optimize import NonlinearConstraint
import aic
import random



sns.set(style="white",font_scale=1.5) #whitegrid

df=pd.read_csv('../ON_freqs.csv')

df=df[df['time']>6.5]



def fit_exp(theta,t,x):
  lam,bb = theta
  return np.sum(((lam*t + bb - x)**2))

def fit_linear1(theta,t,x):
  lam,bb = theta
  y=np.log(lam*t + bb)
  if np.all(np.isfinite(y)):
    return np.sum(((y - x)**2))
  else:
    return 1e20

def fit_quad1(theta,t,x):
  lam,b1,b2 = theta
  y=np.log(lam*t**2 + b1*t + b2)
  if np.all(np.isfinite(y)):
    return np.sum(((y - x)**2))
  else:
    return 1e20

def fit_power(theta,t,x):
  lam,b1,b2= theta
  y=np.log(lam*t**b2 + b1)
  if np.all(np.isfinite(y)):
    return np.sum(((y - x)**2))
  else:
    return 1e20

def CI(varss,n, alpha=0.32):
    #n = 24
    us=[]
    ls=[]
    for var in varss:
      a = sp.stats.chi2.isf(1-alpha/2, n-1)
      b = sp.stats.chi2.isf(alpha/2, n-1)
      u = ((n-1)*var)/a - var
      l = var - ((n-1)*var)/b
      us.append(u)
      ls.append(l)
    return np.vstack([np.array(ls),np.array(us)])

def CI_boot(group,n=100):
  samplingdist=[]
  for i in range(n):
    group2=group.sample(frac=1,replace=True)
    v=np.var(group2['S freq']) - np.mean(group2['S freq']*(1-group2['S freq'])/group2['total count0'])
    if v<0:
      v=0
    samplingdist.append(v)
  return np.array(samplingdist)

def compute_CI(ci,alpha):
  mean = np.mean(ci)
  l=mean - np.quantile(ci,alpha)
  u=np.quantile(ci,1-alpha) - mean
  return l,u

def varovertime(df, alpha=0.32):
  tt=[]
  vv=[]
  ls=[]
  us=[]
  ci0=0
  c=0
  t0=0
  v0=0
  

  for i,group in df.groupby(by='time'):
    v00=np.var(group['S freq']) - np.mean(group['S freq']*(1-group['S freq'])/group['total count0'])
    if v00<0:
      v00=0
    v0+=v00
    ci0+=CI_boot(group)
    t0+=i
    c+=1
    if c==2:
      vv.append(v0/2)
      tt.append(t0/2)
      l,u=compute_CI(ci0/2,alpha)
      ls.append(l)
      us.append(u)
      c=0
      t0=0
      v0=0

  return np.array(tt),np.array(vv), np.vstack([np.array(ls),np.array(us)])

def fits_boot(df,alpha=0.32,nboot=500):
  wells=list(set(list(df['well'])))
  aic1=[]
  aic2=[]
  aic3=[]
  aic4=[]
  lams=[]
  inter=[]
  for i in range(nboot):
    wb=list(random.choices(wells,k=len(wells)))
    dd2=pd.concat([df[df['well']==wi] for wi in wb])
    tps=[]
    vs=[]
    ts=[]
    for (t,tp),group in dd2.groupby(by=['time','time point']):
      #group=group0.sample(frac=1,replace=True)
      v00=np.var(group['S freq']) - np.mean(group['S freq']*(1-group['S freq'])/group['total count0'])
      if v00<0:
        v00=0
      tps.append(tp)
      vs.append(v00)
      ts.append(t)
    
    dd3=pd.DataFrame({'time point':tps,'time':ts,'var':vs}).groupby(by='time point').mean().reset_index()
    tt=np.array(dd3['time'])
    vl=np.log(np.array(dd3['var']))
    res1=sp.optimize.minimize(lambda theta: fit_exp(theta,tt,vl),[0,0]).x
    res2=sp.optimize.brute(lambda theta: fit_linear1(theta,tt,vl),[(-1e-4,1e-4),(-1e-4,1e-4)],Ns=30)
    res3=sp.optimize.brute(lambda theta: fit_quad1(theta,tt,vl),[(-1e-4,1e-4),(-1e-4,1e-4),(-1e-4,1e-4)],Ns=30)
    res4=sp.optimize.brute(lambda theta: fit_power(theta,tt,vl),[(-1e-4,1e-4),(-1e-4,1e-4),(1,5)],Ns=30)
    y1=np.exp(res1[0]*tt + res1[1])
    y2=res2[0]*tt + res2[1]
    y3=res3[0]*tt**2 + res3[1]*tt + res3[2]
    y4=res4[0]*tt**res4[2] + res4[1]
    aic1.append(aic.aic(np.log(y1),vl,2))
    aic2.append(aic.aic(np.log(y2),vl,2))
    aic3.append(aic.aic(np.log(y3),vl,3))
    aic4.append(aic.aic(np.log(y4),vl,3))
    lams.append(res1[0])
    inter.append(np.exp(res1[1]))
  return aic1,aic2,aic3,aic4,lams,inter





tt,vv,ci0=varovertime(df)
print(tt)

vl=np.log(vv)

res1=sp.optimize.minimize(lambda theta: fit_exp(theta,tt,vl),[0,0]).x
res2=sp.optimize.brute(lambda theta: fit_linear1(theta,tt,vl),[(-1e-4,1e-4),(-1e-4,1e-4)],Ns=40)
res3=sp.optimize.brute(lambda theta: fit_quad1(theta,tt,vl),[(-1e-4,1e-4),(-1e-4,1e-4),(-1e-4,1e-4)],Ns=40)
res4=sp.optimize.brute(lambda theta: fit_power(theta,tt,vl),[(-1e-4,1e-4),(-1e-4,1e-4),(1,5)],Ns=40)

print(res1,'\n',res2,'\n',res3,'\n',res4)#,res3,res4

plt.figure()

plt.errorbar(tt,vv,yerr=ci0,fmt='ko-',label='experimental',alpha=0.75)

y1=np.exp(res1[0]*tt + res1[1])
y2=res2[0]*tt + res2[1]
y3=res3[0]*tt**2 + res3[1]*tt + res3[2]
y4=res4[0]*tt**res4[2] + res4[1]

plt.plot(tt,y1,label='exponential fit',alpha=0.75)
plt.plot(tt,y2,label='linear fit',alpha=0.75)
plt.plot(tt,y3,label='quadratic fit',alpha=0.75)
plt.plot(tt,y4,label='power law fit',alpha=0.75)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel(r'var$(f)$')
plt.xlabel('Time, hours')
#plt.yscale('log')
#plt.xscale('log')
plt.xticks([5,10,15,20])
plt.xlim((5,25))
sns.despine()
plt.legend()
plt.tight_layout()
plt.savefig('fits_all.png',dpi=600)



# now I just need confidence intervals :)
aic1=aic.aic(np.log(y1),vl,2)
aic2=aic.aic(np.log(y2),vl,2)
aic3=aic.aic(np.log(y3),vl,3)
aic4=aic.aic(np.log(y4),vl,3)

pd.DataFrame({'model':['exponential','linear','quadratic','power law'],'aic':[aic1,aic2,aic3,aic4]}).to_csv('aic_point.csv')

aic1,aic2,aic3,aic4,lams,inter=fits_boot(df)
models=['exponential']*len(aic1) + ['linear']*len(aic2) + ['quadratic']*len(aic3) + ['power law']*len(aic4)
allaic=aic1+aic2+aic3+aic4
pd.DataFrame({'model':models,'aic':allaic,'sample':list(range(len(aic1)))*4}).to_csv('aic_boot.csv')

pd.DataFrame({'sample':['point']+list(range(len(aic1))),'lambda':[res1[0]]+lams, 'intercept':[np.exp(res1[1])]+inter}).to_csv('lambda_var_boot.csv')
