import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
sns.set(style="whitegrid",font_scale=1.5)


df = pd.read_csv('tc_data.csv').fillna(value=0)

plt.figure()
sns.lineplot(data=df,y='S freq',x='day',hue='group',legend=None)
plt.show()

def fitness(f1,f0):
	return np.log(f1/(1-f1)) - np.log(f0/(1-f0))

delts=[]
days=[]
anc1=[]
anc2=[]
anc3=[]
f0s=[]
var_s=[]
for day2 in range(1,13):
	for j,df0 in df.groupby(by='group'):
		d1=df0[df0['day']==day2]
		d0=df0[df0['day']==day2-1]
		f1=np.float(d1['S freq'])
		f0=np.float(d0['S freq'])
		delt = fitness(f1,f0)
		#anc = int(np.float(d1['anc1'] + d1['anc2'] + d1['anc3']))
		delts.append(delt)
		days.append(day2)
		anc1.append(int(np.float(d1['anc1'])))
		anc2.append(int(np.float(d1['anc2'])))
		anc3.append(int(np.float(d1['anc3'])))
		f0s.append(f0)
		var_s.append( 1/(np.float(d1['total count'])*f1*(1-f1)) + 1/(np.float(d0['total count'])*f0*(1-f0)))
df_plt=pd.DataFrame({
	'Fitness':delts,
	'Day':days,
	'anc1':anc1,
	'anc2':anc2,
	'anc3':anc3,
	'f0':f0s,
	'measurement var':var_s,
	}).drop_duplicates(subset=['Fitness','Day'])

#df_plt

strintfloat=lambda x: str(int(np.float(x)))

def set_ancestors(d1):
	if int(np.float(d1['Day']))==1:
		return 1#int(strintfloat(d1['anc1']) + strintfloat(d1['anc2']) + strintfloat(d1['anc3']))
	elif int(np.float(d1['Day']))==4:
		return int(d1['anc1'])+1
	elif int(np.float(d1['Day']))==8:
		return int(strintfloat(d1['anc1']) + strintfloat(d1['anc2']))
	else:
		return 0

def set_ancestors2(d1):
	if int(np.float(d1['Day']))==1:
		return np.int(d1['anc1']*10)
	else:
		return int(strintfloat(d1['anc1']) + strintfloat(d1['anc2']))

df_plt.fillna(value=0,inplace=True)
df_plt['Anc']=df_plt.apply(lambda d1: set_ancestors(d1), axis=1)
df_plt['Anc2']=df_plt.apply(lambda d1: set_ancestors2(d1), axis=1)

aa=list(np.sort(list(set(list(df_plt['Anc'])))))
aadic={aa[i]:i for i in range(len(aa))}
df_plt['anc num']=df_plt['Anc'].map(aadic)
df_plt.to_csv('fitness.csv')


'''
plt.figure()
sns.boxplot(data=df_plt,x='Day',y='Fitness',color='#c2c2c2')
sns.swarmplot(data=df_plt,x='Day',y='Fitness',hue='Anc')
plt.legend().set_visible(False)
plt.ylabel(r'$\Delta$logit $f_S$')
plt.tight_layout()
plt.show()
'''

