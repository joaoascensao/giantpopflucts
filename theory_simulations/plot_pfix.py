import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from matplotlib.legend_handler import HandlerTuple
import matplotlib as mpl
import pandas as pd
sns.set(style="ticks",font_scale=1.2)

rho = 0.5
kappaA = 1e-3
kappaB = 1e-3
c1A = 0.4
c1B = 0.4
s=0.1
f0=1e-3

grey = '#949aa1'
red='#d91e3a'
blue='#2086e6'
lightgrey='#cbd1d6'
darkgrey='#40464d'

dfsims=pd.read_csv('pfix_sims.csv')

def get_pfixA(kappaA,kappaB,c1A,c1B,rho,s,f):

      delta = c1A + c1B - 2*np.sqrt(c1A*c1B)*rho

      beta = np.sqrt((kappaA - kappaB)**2 + delta*(delta + 2*(kappaA + kappaB)))
      gamma = -c1A + c1B + kappaA - kappaB + 2*s
      num = delta*(1-2*f) - kappaA + kappaB + beta
      dem = -delta*(1-2*f) + kappaA - kappaB + beta

      return (num/dem)**(gamma/beta)

def get_pfix(kappaA,kappaB,c1A,c1B,rho,s,f):
      pfix = lambda f: get_pfixA(kappaA,kappaB,c1A,c1B,rho,s,f)
      return (pfix(f) - pfix(0))/(pfix(1) - pfix(0))

def get_transition(kappaA,kappaB,c1A,c1B,rho,f,svec,pfix):
      delta = c1A + c1B - 2*np.sqrt(c1A*c1B)*rho
      beta = np.sqrt((kappaA - kappaB)**2 + delta*(delta + 2*(kappaA + kappaB)))
      A0 = (beta + delta - kappaA + kappaB)/(beta - delta + kappaA - kappaB)
      A1 = (beta - delta - kappaA + kappaB)/(beta + delta + kappaA - kappaB)
      #ses=beta/(np.log(A0) - np.log(A1))
      ses=0.5*(delta + kappaA*2)/np.log(delta/kappaA + 1)
      return ses, np.interp(ses, svec, pfix)



s=np.logspace(-5,0,100)

c1A_vector=list(np.logspace(-2,0.4,5)[:4])
#c1A_vector=[0.01,0.039810717,0.158489319,0.630957344]
print(c1A_vector)

#pb="blend:#B2D0E1,#3D71A8"
pb='crest'
palette_blue = sns.color_palette(pb,4,desat=1)
print(palette_blue)

plt.figure(figsize=(6.4*0.9,4.8))

for i,c1A in enumerate(c1A_vector):
      pfix=get_pfix(kappaA,kappaB,c1A,c1A,rho,s,f0)
      plt.plot(s,pfix,color=palette_blue[i],alpha=0.8,linewidth=1.8)
      ses,pfix_ses=get_transition(kappaA,kappaB,c1A,c1A,rho,f0,s,pfix)
      plt.plot(ses,pfix_ses,color=palette_blue[i],marker='*',markersize=12,zorder=1000,alpha=0.9,mec=darkgrey)
      plt.plot(s[0]*0.8,ses*f0/kappaA,color=palette_blue[i],marker='X',markersize=7,zorder=1000,alpha=0.9,mec=darkgrey)
      dd0=dfsims[(dfsims['c1A']>c1A*0.9) & (dfsims['c1A']<c1A*1.1) ]
      plt.errorbar(dd0['s'],dd0['pfix'],fmt='o',color=palette_blue[i],yerr=2*np.sqrt(dd0['pfix']*(1-dd0['pfix'])/dd0['n sims']),alpha=0.7,markersize=5)

      

plt.plot(s,(1-np.exp(-2*s*f0/kappaA))/(1-np.exp(-2*s/kappaA)),':',linewidth=2.5,alpha=0.8,color='k')
plt.plot(s,kappaA*np.ones_like(s),'--',linewidth=2.1,alpha=0.8,color='#d43b4d')
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'Fixation probability, $p_{fix}$')
plt.xlabel(r'Fitness effect, $s$')
sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig('pfix.pdf',format='pdf')


