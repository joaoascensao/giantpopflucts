import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
import matplotlib
sns.set(style="ticks",font_scale=1.5)

sns.set_palette('flare')
df=pd.read_csv('annodata.csv')

def CI_frequency(var,n,alpha=0.05):
    var=np.array(var)
    a = sp.stats.chi2.isf(1-alpha/2, n-1)
    b = sp.stats.chi2.isf(alpha/2, n-1)
    u = ((n-1)*var)/a - var
    l = var - ((n-1)*var)/b
    return l,u

containers = {
	'A':'glass',
	'C':'glass',
	'E':'plastic',
	'G':'plastic',
}
strains={
	'A':'SL',
	'C':'RP',
	'E':'SL',
	'G':'RP',
}

ylabels={
	'A':r'$S$ Frequency, $f_S$',
	'C':r'$\Delta pykF$ Mutant Frequency, $f$',
}

ylabels2={
	'A':r'var $f_S$',
	'C':r'var $f$',
}

df['col']=df['well'].apply(lambda x: np.int(x[1:]))
df['logit f']=np.log(df['RFP freq']) - np.log(1-df['RFP freq'])

grey = '#949aa1'
red='#d91e3a'
blue='#2086e6'
lightgrey='#cbd1d6'
darkgrey='#40464d'
green='#59b332'
markersize=6

for row,group in df.groupby(by='row'):
	if containers[row]=='plastic':
		continue
	plt.figure()
	for i,row2 in group.groupby(by='well'):
		#sns.lineplot(data=group,hue='col',x='day',y='RFP freq',palette='flare',legend=None)
		plt.errorbar(np.array(row2['day']),np.array(row2['RFP freq']),fmt='.-',yerr=2*np.sqrt(row2['RFP freq']*(1-row2['RFP freq'])/row2['total count']))
	plt.ylabel(ylabels[row])
	plt.xlabel('Day')
	sns.despine()
	plt.tight_layout()
	#plt.show()
	plt.savefig('traj_{}.png'.format(row),dpi=600)

	#varss=group.groupby(by='day').var().reset_index()
	days=[]
	varss=[]
	ls=[]
	us=[]
	pred=[]
	Ne=1e5
	for i,group2 in group.groupby(by='day'):
		if i==0:
			continue
		fm=np.mean(group2['RFP freq'])
		if i==1:
			pred.append(fm*(1-fm)/Ne)
		else:
			pred.append(pred[-1] + fm*(1-fm)/Ne)
		days.append(i)
		mad=np.median(np.abs(np.median(group2['RFP freq']) - group2['RFP freq']))/0.67449
		varss.append(mad**2)
		l,u=CI_frequency(mad**2,len(group2))
		ls.append(l)
		us.append(u)

	plt.figure()
	plt.errorbar(days,varss,fmt='s-',yerr=np.vstack((ls,us)),color=red,label='Experimental data',zorder=100)
	plt.plot(days,pred,'k--',label='Genetic drift prediction')
	
	plt.yscale('log')
	plt.ylabel(ylabels2[row])
	plt.xlabel('Day')
	handles,labels=plt.gca().get_legend_handles_labels()
	order=[1,0]
	plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],frameon=False,loc='upper left')
	sns.despine()
	 #loc=(1.15, 0.8),
	plt.tight_layout()
	plt.savefig('varrs_{}.png'.format(row),dpi=600)
