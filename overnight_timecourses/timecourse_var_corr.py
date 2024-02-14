import pandas as pd
import scipy as sp 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize
import random
import scipy.stats
import itertools
import copy
sns.set(style="ticks",font_scale=1.5) #whitegrid

df=pd.read_csv('ON_freqs.csv')
df8=pd.read_csv('eighthourtc_freqs.csv')
df8.rename(columns={'total count':'total count0'},inplace=True)


pb='GnBu'
pr='BuPu'
palette_blue = sns.color_palette(pb,desat=0.95) #itertools.cycle(
palette_red = sns.color_palette(pr,desat=0.95)

#df['time']>6.5


plt.figure()

c=0
for i,group0 in df.groupby(by='well'):
  group=copy.deepcopy(group0[group0['time point']>=4])
  plt.errorbar(group['time'],group['S freq'],marker='o',yerr=np.sqrt(group['S freq']*(1-group['S freq'])/group['total count0']),
    alpha=0.8,color=palette_blue[c],markersize=4)
  
  group=copy.deepcopy(group0[(group0['time point']==4) | (group0['time point']==1)])
  plt.errorbar(group['time'],group['S freq'],fmt='o:',yerr=np.sqrt(group['S freq']*(1-group['S freq'])/group['total count0']),
    alpha=0.5,color=palette_blue[c],markersize=4)
  c+=1
  if c>=len(palette_blue):
    c=0

c=0
for i,group in df8.groupby(by='well'):
  plt.errorbar(group['time'],group['S freq'],marker='s',yerr=np.sqrt(group['S freq']*(1-group['S freq'])/group['total count0']),
    alpha=0.8,color=palette_red[c],markersize=4)
  c+=1
  if c>=len(palette_red):
    c=0

plt.xlabel('Time, hours')
plt.ylabel(r'S frequency, $f_S$')

plt.tight_layout()
sns.despine()
plt.xticks([0,5,10,15,20])
plt.xlim((0,25))
#plt.show()
plt.savefig('tfhours.png',dpi=600)


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

def CI_boot(group,n=500):
  samplingdist=[]
  for i in range(n):
    group2=group.sample(frac=1,replace=True)
    v=np.var(group2['S freq']) - np.mean(group2['S freq']*(1-group2['S freq'])/group2['total count0'])
    if v<0:
      v=0
    samplingdist.append(np.float(v))
  return np.array(samplingdist)

def compute_CI(ci,alpha):
  mean = np.median(ci)
  l=mean - np.quantile(ci,alpha/2)
  u=np.quantile(ci,1-alpha/2) - mean
  return l,u

def varovertime(df, alpha=0.34):
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

  return tt,vv, np.vstack([np.array(ls),np.array(us)])

def varovertime2(df,alpha=0.32):
  tt=[]
  vv=[]
  ls=[]
  us=[]
  ci0=0
  c=0
  t0=0
  v0=0

  for i,group in df.groupby(by='time'):
    v0=np.var(group['S freq']) - np.mean(group['S freq']*(1-group['S freq'])/group['total count0'])
    if v0<0:
      v0=0
    t0=i
    ci0=CI_boot(group)
    l,u=compute_CI(ci0,alpha)
    ls.append(l)
    us.append(u)

    vv.append(v0)
    tt.append(t0)
    

  return tt, vv, np.vstack([np.array(ls),np.array(us)])

df=df[df['time']>6.5]

plt.figure()

tt,vv,ci0=varovertime(df)
plt.errorbar(tt,vv,yerr=ci0,fmt='o-',color=sns.color_palette(pb,desat=0.95)[-2],alpha=0.8)

tt8,vv8,ci8=varovertime2(df8[df8['time point']>0])
plt.errorbar(tt8,vv8,yerr=ci8,fmt='s-',color=sns.color_palette(pr,desat=0.95)[-2],alpha=0.8)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.ylabel(r'var$(f_S)$')
plt.xlabel('Time, hours')
#plt.yscale('log')
plt.xticks([0,5,10,15,20])
plt.xlim((0,25))
sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig('varovertime.png',dpi=600)

tt=np.array(tt)
pp=np.polyfit(tt,np.log(vv),1)

def bootstrap_exp_fit(t,v,pp0,n=500,alpha=0.05):
  df=pd.DataFrame({'t':t,'v':v})
  boot=[]
  ti=np.linspace(np.min(t)-0.35,np.max(t)+0.35,300)
  for i in range(n):
    ss=df.sample(frac=1,replace=True)
    pp=np.polyfit(ss['t'],np.log(ss['v']),1)
    boot.append(np.exp(pp[0]*ti + pp[1]))
  boot=np.vstack(boot)
  l=np.quantile(boot,alpha/2,axis=0)
  u=np.quantile(boot,1-alpha/2,axis=0)
  return ti,l,u

tint,ci_l,ci_u=bootstrap_exp_fit(tt,vv,pp)


plt.figure()
plt.errorbar(tt,vv,yerr=ci0,fmt='o-',color=sns.color_palette(pb,desat=0.95)[-2],alpha=0.8)
plt.plot(tint,np.exp(pp[0]*tint + pp[1]),'k:',alpha=0.85,dash_capstyle='round') #,linestyle=(0,(0.1,2))
plt.fill_between(tint,ci_l,ci_u,color='k',alpha=0.12)

#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#sns.regplot(x=tt,y=np.log(vv),scatter=False,color='k',logy=True)
plt.ylabel(r'var$(f_S)$')
plt.xlabel('Time, hours')
plt.yscale('log')
plt.xticks([10,15,20])
plt.xlim((7,25))
plt.ylim((1e-6,1.3e-4))
sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig('varovertime_log.png',dpi=600)



fs=[]
for i,group in df.groupby(by='well'):
  fs.append(list(np.interp(tt,np.array(group['time']),np.array(group['S freq']))))

fs=np.vstack(fs)
print(fs)
print(fs[:,-1])

order = fs[:,-1].argsort()
ranks_last = order.argsort()



corrs=[]
errs_u=[]
errs_l=[]
for i in range(len(tt)-1):
  order = fs[:,i].argsort()
  ranks = order.argsort()
  r,p=sp.stats.spearmanr(ranks,ranks_last)
  
  lower = r - np.tanh(np.arctanh(r) - 1/np.sqrt(len(order) - 3))
  upper = np.tanh(np.arctanh(r) + 1/np.sqrt(len(order) - 3)) - r

  errs_u.append(upper)
  errs_l.append(lower)
  corrs.append(r)


plt.figure()
plt.errorbar(tt[:-1],corrs,fmt='o-',yerr=np.vstack((errs_l,errs_u)),color=sns.color_palette(pb,desat=0.95)[-2],alpha=0.8)
plt.ylabel('Rank correlation with final time')
plt.xlabel('Time, hours')
plt.xticks([0,5,10,15,20])
plt.xlim((0,25))
sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig('corrtime2.png',dpi=600)