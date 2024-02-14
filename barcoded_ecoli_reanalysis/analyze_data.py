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
from brokenaxes import brokenaxes
sns.set(style="ticks",font_scale=1.5) #whitegrid
dire='/Users/joaoascensao/Dropbox/Desktop/HallatschekLab/EcoEvo_Experiments/BarSeq/experiments/E_F/data/bc_counts'

tv=['0_1', '0_2', '1', '2', '3', '4']
tv2=['0','1', '2', '3', '4']
tvi=[0,1,2,3,4]

def getfreqs(sl):
  cc=pd.read_csv(dire+'/F{}1_counts.csv'.format(sl))
  rtot1 = pd.read_csv(dire+'/F{}1_Rtot.csv'.format(sl))
  if sl=='L':
    sl2='S'
  else:
    sl2='L'
  rtot2 = pd.read_csv(dire+'/F{}1_Rtot.csv'.format(sl2))
  cc[tv]=cc[tv]/(np.array(rtot1[tv]) + np.array(rtot2[tv]))
  cc['0']=(cc['0_1']+cc['0_2'])/2
  return cc

def getfreqs_solo(sl):
  cc=pd.read_csv(dire+'/F{}1_counts.csv'.format(sl))
  rtot1 = pd.read_csv(dire+'/F{}1_Rtot.csv'.format(sl))
  if sl=='L':
    sl2='S'
  else:
    sl2='L'
  rtot2 = pd.read_csv(dire+'/F{}1_Rtot.csv'.format(sl2))
  cc[tv]=cc[tv]/(np.array(rtot1[tv]))
  cc['0']=(cc['0_1']+cc['0_2'])/2
  return cc

Sfreq_tot=getfreqs('S')
Lfreq_tot=getfreqs('L')
Sfreq_solo=getfreqs_solo('S')
Lfreq_solo=getfreqs_solo('L')

sfreq=Sfreq_tot[tv2].sum(axis=0)

outliers=pd.read_csv(dire+'/../outliers/FS1_outliers.csv')
outliers=list(outliers[(outliers['RD']>4) | (outliers['RD']==1)]['barcode'])
print(outliers[0])

bcs=copy.deepcopy(Sfreq_tot[(Sfreq_tot['0']>5e-6)]) #  & (Sfreq_tot['0']>2e-4) & (Sfreq_tot['0']<1e-3)  (pd.notnull(Sfreq_tot['pos'])) & (pd.isnull(Sfreq_tot['gene_ID'])) & 
#bcs=bcs[~bcs['barcode'].isin(outliers)]
print(bcs)



#bcs2=random.sample(bcs,k=4)
#print(bcs2)


blue='#0052c8'
red='#d91e3a'

plt.figure(figsize=(4.9,4.79))
#plt.fill_between(tvi,[0]*5,sfreq,facecolor=blue,alpha=0.3)
#plt.fill_between(tvi,sfreq,[1]*5,facecolor=red,alpha=0.6)
plt.plot(tvi,sfreq,'k')

thr=[0.014,0.005,0.01,0.003,0.008,0.011]

c=0
z=0
hf_bcs=[]
for i,bci in bcs.iterrows():
  if c==0:
    bcc=np.array(bci[tv2])
  else:
    bcc+=np.array(bci[tv2])
  c+=1
  if bcc[0]>thr[z]:
    #plt.plot(tvi,bcc) #,color=cc[i]
    hf_bcs.append(bcc)
    c=0
    z+=1
  if z>5:
    break

print('done')
colors=list(sns.color_palette('Blues_r'))
for i,h in enumerate(hf_bcs):
  if i==0:
    l=np.array(sfreq) - h
    u=np.array(sfreq)
  else:
    #l0=l
    u=l
    l=l-h
    
  print(l,u)
  plt.fill_between(tvi,list(l),list(u),facecolor=colors[i],alpha=0.6) #
plt.fill_between(tvi,[0]*len(tvi),list(l),facecolor=colors[0],alpha=0.4)
plt.ylim((0.26,0.42))
sns.despine()
plt.xlim((0,4))
plt.xticks([0,1,2,3,4])
plt.yticks([0.3,0.35,0.4])
plt.ylabel('Genotype frequency')
plt.xlabel('Day')
#plt.yscale('log')
plt.tight_layout()
#plt.show()
plt.savefig('fig1A.png',dpi=600)

sfreq=np.array(sfreq)
plt.figure(figsize=(4.9,4.79))
for i,h in enumerate(hf_bcs):
  if i==0:
    l=sfreq - h
    u=np.array(sfreq)
  else:
    #l0=l
    u=l
    l=l-h
    
  print(l,u)
  plt.fill_between(tvi,list(l/sfreq),list(u/sfreq),facecolor=colors[i],alpha=0.6) #
plt.fill_between(tvi,[0]*len(tvi),list(l/sfreq),facecolor=colors[0],alpha=0.4)
sns.despine()
plt.xlim((0,4))
plt.xticks([0,1,2,3,4])
plt.yticks([0.85,0.9,0.95,1])
plt.ylim((0.835,1))
plt.xlabel('Day')
plt.ylabel(r'Within-$S$ genotype frequency')
plt.tight_layout()
plt.savefig('fig1B.png',dpi=600)

'''
fig,ax=plt.subplots(2,1,sharex=True,gridspec_kw={'height_ratios': [1,0.5]})

ax[0].fill_between(tvi,[0]*5,sfreq,facecolor=blue,alpha=0.3)
ax[0].fill_between(tvi,sfreq,[1]*5,facecolor=red,alpha=0.3)


ax[1].fill_between(tvi,[0]*5,sfreq,facecolor=blue,alpha=0.3)

ax[0].plot(tvi,sfreq,'k')
cc=list(sns.color_palette('Blues_r'))
for i,bci in enumerate(bcs):
  ax[1].plot(tvi,np.array(Sfreq_tot[Sfreq_tot['barcode']==bci][tv2]).T,color=cc[i])
#plt.yscale('log')
ax[0].set_ylim((0.30,0.45))
#ax[0].set_yticks([0.2,0.4,0.6])
ax[1].set_ylim((1e-4,8e-4))
ax[1].set_yticks([2e-4,6e-4])

ax[1].set_yticklabels([r'$2\cdot 10^{-4}$',r'$6\cdot 10^{-4}$'])

ax[0].set_xticks([])
ax[1].set_xticks([0,1,2,3,4])
#ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax[1].set_xlim((0,4))
ax[0].get_xaxis().set_visible(False)
sns.despine(ax=ax[1])
sns.despine(ax=ax[0],bottom=True)
ax[0].set_ylabel('Frequency')
ax[1].set_xlabel('Day')
plt.tight_layout()
plt.show()
'''
