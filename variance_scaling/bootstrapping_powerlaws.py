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
dfs={}
counts=[]
for gr in grs:
	f0.append(np.float(df_day0[df_day0['well']==f0well[gr]]['BFP freq']))
	df=pd.read_csv('data_d1_{}.csv'.format(gr))
	dfs[gr]=df
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
	cc=np.cov(dfm['S'],dfm['L'])[0,1]
	cov.append(cc)

	counts.append(np.mean(dfm['corrected total count']))


fa=[np.polyfit(np.log(fm),np.log(fvar),1)[0]]
sa=[np.polyfit(np.log(fm[1:]),np.log(Svar[1:]),1)[0]]
la=[np.polyfit(np.log(fm[1:]),np.log(Lvar[1:]),1)[0]]
ca=[np.polyfit(np.log(fm[1:]),np.log(cov[1:]),1)[0]]


fa_l=[]
sa_l=[]
la_l=[]
ca_l=[]
booti=[]
for j in range(1000):
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

	counts=[]
	for gr in grs:
		df=dfs[gr]
		df['rep']=df['well'].apply(lambda x: repmap(x))
		df['S']=df['corrected total count']*df['BFP freq']*1000/vol[gr]
		df['L']=df['corrected total count']*(1-df['BFP freq'])*1000/vol[gr]
		#print(df)
		dfm=df.groupby('rep').mean().reset_index().sample(frac=1,replace=True)
		#print(dfm)
		#print(np.mean(dfm['BFP freq']),np.log10(np.mean(dfvar['S'])/3),np.log10(np.mean(dfvar['L'])/3))
		fm.append(np.mean(dfm['BFP freq']))
		fvar.append(np.var(dfm['BFP freq']))

		Sm.append(np.mean(dfm['S']))
		Svar.append(np.var(dfm['S']))

		Lm.append(np.mean(dfm['L']))
		Lvar.append(np.var(dfm['L']))

		allS.append(list(np.array(dfm['S'])))
		allL.append(list(np.array(dfm['L'])))

		tot.append(np.mean(dfm['corrected total count']*1000/vol[gr]))
		cc=np.cov(dfm['S'],dfm['L'])[0,1]
		cov.append(cc)

		counts.append(np.mean(dfm['corrected total count']))


	fa_l.append(np.polyfit(np.log(fm),np.log(fvar),1)[0])
	sa_l.append(np.polyfit(np.log(fm[1:]),np.log(Svar[1:]),1)[0])
	la_l.append(np.polyfit(np.log(fm[1:]),np.log(Lvar[1:]),1)[0])
	ca_l.append(np.polyfit(np.log(fm[1:]),np.log(cov[1:]),1)[0])
	booti.append(j)


pd.DataFrame({'boot':['point']+booti,'f':fa+fa_l,'S':sa+sa_l,'L':la+la_l,'cov':ca+ca_l}).to_csv('powerlawexps_boot.csv')


