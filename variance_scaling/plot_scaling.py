import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
import scipy.stats
import random
sns.set(style="ticks",font_scale=1.5)
grs=['1-2','5-6','3-4','7-8','9-10','11-12']

rowmap={
	'A':'I',
	'B':'J',
	'C':'K',
	'D':'L',
	'E':'M',
	'F':'N',
	'G':'O',
	'H':'P'
}

vol={
	'1-2':50,
	'3-4':30,
	'5-6':20,
	'7-8':10,
	'9-10':5,
	'11-12':5,
}

f0well={
	'1-2':'B7',
	'3-4':'D7',
	'5-6':'E7',
	'7-8':'F7',
	'9-10':'G7',
	'11-12':'H7',
}

def repmap(x):
	col=np.int(x[1:])
	row=x[0]
	if col in [3,7,11]:
		return rowmap[row]
	else:
		return row

fm=[]
fvar=[]
Sm=[]
Svar=[]
Lm=[]
Lvar=[]
tot=[]
cov=[]
f0=[]
allS=[]
allL=[]

df_day0=pd.read_csv('data_d0.csv')
counts=[]
for gr in grs:
	f0.append(np.float(df_day0[df_day0['well']==f0well[gr]]['BFP freq']))
	df=pd.read_csv('data_d1_{}.csv'.format(gr))
	df['rep']=df['well'].apply(lambda x: repmap(x))
	df['S']=df['corrected total count']*df['BFP freq']*1000/vol[gr]
	df['L']=df['corrected total count']*(1-df['BFP freq'])*1000/vol[gr]
	print(df)
	dfm=df.groupby('rep').mean().reset_index()
	dfvar=df.groupby('rep').var().reset_index()
	print(dfm)
	print(np.mean(dfm['BFP freq']),np.log10(np.mean(dfvar['S'])/3),np.log10(np.mean(dfvar['L'])/3))
	fm.append(np.mean(dfm['BFP freq']))
	fvar.append(np.var(dfm['BFP freq']))

	Sm.append(np.mean(dfm['S']))
	Svar.append(np.var(dfm['S']))

	Lm.append(np.mean(dfm['L']))
	Lvar.append(np.var(dfm['L']))

	allS.append(list(np.array(dfm['S'])))
	allL.append(list(np.array(dfm['L'])))

	tot.append(np.mean(dfm['corrected total count']*1000/vol[gr]))
	#cov.append(np.cov(dfm['S'],dfm['L'])[0,1])
	#cov.append(sp.stats.pearsonr(dfm['S'],dfm['L'])[0])
	cc=np.cov(dfm['S'],dfm['L'])[0,1]#  - np.sqrt(np.mean(dfvar['S'])/3)*np.sqrt(np.mean(dfvar['L'])/3)
	cov.append(cc)

	counts.append(np.mean(dfm['corrected total count']))
	'''
	plt.figure()
	plt.title(np.mean(dfm['BFP freq']))
	plt.plot(dfm['S'],dfm['L'],'o')
	plt.show()
	'''

print(cov)
print(np.polyfit(np.log(fm),np.log(fvar),1))
print(np.polyfit(np.log(fm[1:]),np.log(Svar[1:]),1))
print(np.polyfit(np.log(fm[1:]),np.log(Lvar[1:]),1))
print(np.polyfit(np.log(fm[1:]),np.log(cov[1:]),1))
print(np.polyfit(np.log(fm[3:]),np.log(np.array(Svar[3:])/np.array(Lvar[3:])),1))

covfit=np.polyfit(np.log(fm),np.log(cov),1)
Sfit=np.polyfit(np.log(fm)[1:],np.log(Svar)[1:],1)


def CI_frequency(var,alpha=0.05):
    var=np.array(var)
    n = 16
    a = sp.stats.chi2.isf(1-alpha/2, n-1)
    b = sp.stats.chi2.isf(alpha/2, n-1)
    u = ((n-1)*var)/a - var
    l = var - ((n-1)*var)/b
    return l,u

plt.figure()
plt.plot(fm,np.array(cov)/np.sqrt(np.array(Svar)*np.array(Lvar)),'.-')
plt.xscale('log')
plt.show()


fm=np.array(fm)
fvar=np.array(fvar)
red='#cc0245'
blue='#0224cc'
green='#00721F'
plt.figure()
plt.errorbar(fm,fvar,yerr=CI_frequency(fvar),fmt='ks-',label='Observed')
plt.plot(fm,0.003*fm**2,'--',label=r'var$(f) \propto \langle f\rangle^2$',c=red)
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'var$(f_S)$')
plt.xlabel(r'Frequency of S, $\langle f_S\rangle$')
#plt.legend()
sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig('varf2_poster.png',dpi=600)

plt.figure()
sns.set_palette('husl',n_colors=3)
plt.errorbar(fm,Svar,yerr=CI_frequency(Svar),fmt='o-',label=r'var$(N_S)$')
plt.errorbar(fm,Lvar,yerr=CI_frequency(Lvar),fmt='o-',label=r'var$(N_L)$')
plt.errorbar(fm,cov,yerr=CI_frequency(cov),fmt='o-',label=r'cov$(N_S,N_L)$')

plt.plot(fm,2.7*np.exp(Sfit[1])*(fm**2),'--',c=red)
plt.plot(fm,np.exp(covfit[1])*fm,'--',c=blue)
plt.plot(fm,np.mean(Lvar)*np.ones_like(fm),'--',c=green)

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'Frequency of S, $\langle f_S\rangle$')
plt.legend()
sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig('varN_poster.png',dpi=600)

def ratCI(allS,allL,fm,rats):
	nboot=500
	l=[]
	u=[]
	for i,fi in enumerate(fm):
		bootsample=[]
		for j in range(nboot):
			varS=np.var(random.choices(allS[i],k=len(allS[i])))
			varL=np.var(random.choices(allL[i],k=len(allL[i])))
			bootsample.append(np.array(varS)/((fi**2)*np.array(varL)))
		l.append(rats[i]-np.quantile(bootsample,0.025))
		u.append(np.quantile(bootsample,0.975)-rats[i])
	return np.vstack((l,u))


plt.figure()
rats=np.array(Svar)/((fm**2)*np.array(Lvar))
plt.errorbar(fm,rats,fmt='^-',yerr=ratCI(allS,allL,fm,rats),c='k')
#plt.plot(fm[1:],Lvar[1:],'o-',label=r'var$(N_L)$')
#plt.plot(fm,cov,'o-',label=r'$cov(N_S,N_L)$')
plt.yscale('log')
plt.ylim((8e-2,17))
plt.xscale('log')
plt.xlabel(r'Frequency of S, $\langle f_S\rangle$')
plt.ylabel(r'$\frac{var(N_S)}{\langle f_S\rangle^2 var(N_L)}$')
#plt.legend()
sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig('varratio_poster.png',dpi=600)


