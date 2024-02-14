import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
sns.set(style="ticks",font_scale=1.5)

grey = '#949aa1'
day_var = list(pd.read_csv('posterior_alpha_var.csv')['alpha_var'])
delta_var = list(pd.read_csv('posterior_delta.csv')['delta'])
memvar=list(pd.read_csv('posterior_kappa_var.csv')['kappa_var'])

varl=delta_var+day_var+memvar
names=['Intrinsic\n'+r'($\delta_I$)']*len(delta_var) + ['Extrinsic\n'+r'($\delta_E$)']*len(day_var) + ['Shared\nmothers']*len(memvar) 

print(np.median(day_var),np.median(delta_var),np.median(memvar))

df=pd.DataFrame({'variance':varl,'name':names})

plt.figure(figsize=[4.7, 5])
#sns.boxplot(data=df,x='name',y='variance')
g=sns.barplot(data=df,x='name',y='variance',ci=None,estimator=lambda x: np.median(x),palette='Blues_r')
x_coords = [p.get_x() + 0.5*p.get_width() for p in g.patches]
y_coords = [p.get_height() for p in g.patches]

#dfm=df.groupby(by='names').median().reset_index()
mms=[]
ll=[]
uu=[]
for i,group in df.groupby(by='name'):
	mi=np.median(group['variance'])
	mms.append(mi)
	ll.append(mi-np.quantile(group['variance'],0.025))
	uu.append(np.quantile(group['variance'],0.975)-mi)
dfm=pd.DataFrame({'med':mms,'l':ll,'u':uu})

errors = pd.DataFrame({'x':x_coords,'med':y_coords}).merge(dfm,on='med',how='inner')
grey = '#949aa1'
red='#d91e3a'
blue='#2086e6'
lightgrey='#cbd1d6'
darkgrey='#40464d'
plt.errorbar(errors['x'], errors['med'], yerr=np.vstack((np.array(errors['l']),np.array(errors['u']))), fmt=' ',zorder=200,ecolor=grey)
plt.xlabel('')
plt.ylabel('Variance between replicates')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
sns.despine()
plt.tight_layout()
plt.savefig('variance_components.png',dpi=600)
