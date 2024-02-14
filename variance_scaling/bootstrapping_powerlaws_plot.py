import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
import scipy.stats
import random
sns.set(style="whitegrid",font_scale=1.2)

df=pd.read_csv('powerlawexps_boot.csv')

names0=['f','S','L','cov']
names=[r'var $f_S$',r'var $N_S$',r'var $N_L$',r'cov$\,(N_S,N_L)$']
means=[np.float(df[df['boot']=='point'][xi]) for xi in names0]

grey = '#949aa1'
red='#d91e3a'
blue='#2086e6'
lightgrey='#cbd1d6'
darkgrey='#40464d'

plt.figure()
plt.bar(names,means,color=sns.color_palette('viridis',4).as_hex(),alpha=0.9)
ls=[]
us=[]
for i,xi in enumerate(names0):
	v=np.array(df[df['boot']!='point'][xi])
	l=means[i]-np.quantile(v,0.025)
	u=np.quantile(v,0.975)-means[i]
	print(names0[i],means[i],np.vstack([l,u]))
	ls.append(l)
	us.append(u)
plt.errorbar(names,means,yerr=np.vstack([ls,us]),fmt='',zorder=100,c=darkgrey, ls='none')

plt.ylabel('Inferred power-law exponent,\n'+r' with respect to $\langle f_S \rangle$')
plt.xlabel('Quantity')
plt.tight_layout()
plt.savefig('powerlawexps.png',dpi=600)