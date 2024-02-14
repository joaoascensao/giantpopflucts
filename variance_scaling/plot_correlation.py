import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
import scipy.stats
import random
import scipy.optimize
sns.set(style="ticks",font_scale=1.3)
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
	Svar.append(np.var(dfm['S']))#- np.mean(dfvar['S'])/3)

	Lm.append(np.mean(dfm['L']))
	Lvar.append(np.var(dfm['L']))#- np.mean(dfvar['L'])/3)

	allS.append(list(np.array(dfm['S'])))
	allL.append(list(np.array(dfm['L'])))

	tot.append(np.mean(dfm['corrected total count']*1000/vol[gr]))
	#cov.append(np.cov(dfm['S'],dfm['L'])[0,1])
	#cov.append(sp.stats.pearsonr(dfm['S'],dfm['L'])[0])
	cc=np.cov(dfm['S'],dfm['L'])[0,1]#  - np.sqrt(np.mean(dfvar['S'])/3)*np.sqrt(np.mean(dfvar['L'])/3)
	cov.append(cc)

	counts.append(np.mean(dfm['corrected total count']))

def fit_definedpowerlawslope(x,y,slope,x0):
	x=np.array(x)
	y=np.array(y)
	obj = lambda a: np.sum((y - a*(x**slope))**2)
	res=sp.optimize.minimize_scalar(obj)
	return res.x

print(cov)
print(np.polyfit(np.log(fm),np.log(fvar),1))
print(np.polyfit(np.log(fm[1:]),np.log(Svar[1:]),1))
print(np.polyfit(np.log(fm[1:]),np.log(Lvar[1:]),1))
print(np.polyfit(np.log(fm[1:]),np.log(cov[1:]),1))
print(np.polyfit(np.log(fm[3:]),np.log(np.array(Svar[3:])/np.array(Lvar[3:])),1))

covfit=np.polyfit(np.log(fm),np.log(cov),1)
Sfit=np.polyfit(np.log(fm)[1:],np.log(Svar)[1:],1)
Lfit=np.polyfit(np.log(fm)[1:],np.log(Lvar)[1:],1)

Ntot=np.mean(np.array(Lm)+np.array(Sm))

cAB = np.exp(covfit[1])
cS1 = np.exp(Sfit[1])
cL1 = np.exp(Lfit[1])

cAB=fit_definedpowerlawslope(fm,cov,1,cAB)*(Ntot**-2)
cS1=fit_definedpowerlawslope(fm[1:],Svar[1:],2,cS1)*(Ntot**-2)
cL1=fit_definedpowerlawslope(fm[1:],Lvar[1:],0,cL1)*(Ntot**-2)

print('delta',fit_definedpowerlawslope(fm[1:],fvar[1:],2,cL1))
cS0=5
cL0=5

print('params',cAB,cS1,cL1)
print(cS1+cS1-2*cAB)


#theory
fm2=np.linspace(np.min(fm),np.max(fm),1000)

def corrtheory_f(fm2,Ntot,corremp,theta0):
	theta=[np.exp(x) for x in theta0]
	_,cS0,cS1 = theta
	rho,_,_ = theta0
	cL1=cS1
	cAB=np.sqrt(cL1*cS1)*rho
	cL0=cS0
	
	corrtheory=cAB*fm2*(1-fm2)/( np.sqrt( (cS0)*fm2 + cS1*(fm2**2) )*np.sqrt( (cL0)*(1-fm2) + cL1*(((1-fm2))**2) ) )
	return np.sum((corrtheory-corremp)**2)

def corrtheory_fit(fm2,Ntot,theta0):
	theta=[np.exp(x) for x in theta0]
	_,cS0,cS1 = theta
	rho,_,_ = theta0
	cL1=cS1
	cAB=np.sqrt(cL1*cS1)*rho
	cL0=cS0
	corrtheory=cAB*fm2*(1-fm2)/( np.sqrt( (cS0)*fm2 + cS1*(fm2**2) )*np.sqrt( (cL0)*(1-fm2) + cL1*(((1-fm2))**2) ) )
	return corrtheory

def fit_corrtheory(Ntot,fm,corremp):
	obj = lambda theta: corrtheory_f(fm,Ntot,corremp,theta)
	#res=sp.optimize.minimize(obj,np.log(np.array([5,5,5,5])))
	res=sp.optimize.dual_annealing(obj,[(-1,0.99)]+[(-1,500)]*2)
	return res

grey = '#949aa1'
red='#d91e3a'
blue='#2086e6'
lightgrey='#cbd1d6'
darkgrey='#40464d'

fm=np.array(fm)
corremp=np.array(cov)/np.sqrt(np.array(Svar)*np.array(Lvar))
print('Ntot',Ntot)
res=fit_corrtheory(Ntot,fm,corremp)
print(res)


plt.figure()
plt.errorbar(fm,corremp,fmt='ko-',yerr=np.sqrt((1-corremp**2)/(16-2)),label='Empirical')
corrtheory=corrtheory_fit(fm2,Ntot,res.x)
plt.errorbar(fm2, corrtheory,color=red,label='Theory fit')
plt.ylabel(r'corr$\,(N_S,N_L)$')
plt.xlabel(r'Frequency of S, $\langle f_S\rangle$')
plt.xscale('log')
plt.legend()
sns.despine()
plt.tight_layout()
plt.savefig('corrfit.png',dpi=600)

#problems: N refers to Nb 


