import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
sns.set(style="whitegrid",font_scale=1.2)


df=pd.read_csv('annodata.csv')

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
logit=lambda x: np.log(x) - np.log(1-x)
dayl = []
wells = []
fit = []
rows=[]
cols=[]
for (well,row),group in df.groupby(by=['well','row']):
	group=group.set_index('day')
	for day0 in range(6):
		day1=day0+1
		dayl.append(day1)
		#print(group.loc[day1])
		fit.append((logit(group.loc[day1]['RFP freq']) - logit(group.loc[day0]['RFP freq']))/6.64)
		wells.append(well)
		rows.append(row)
		cols.append(np.int(well[1:]))
		print(fit[-1],day1,well,row)

fitdf=pd.DataFrame({'day':dayl,'fitness':fit,'well':wells,'row':rows,'col':cols})
fitdf.to_csv('fitness.csv')

for row,group in fitdf.groupby(by='row'):
	plt.figure()
	sns.lineplot(data=group,hue='col',x='day',y='fitness',palette='viridis')
	plt.title(strains[row]+', '+containers[row])
	plt.savefig('fit_{}.png'.format(row),dpi=600)


'''
df = df[df['day']==1]
mvar = np.mean(df['RFP freq']*(1-df['RFP freq'])/df['total count'])
varvar=np.var(df['RFP freq'])
exp = np.mean(df['RFP freq'])*np.mean(1-df['RFP freq'])*1e-5

print(mvar,varvar,varvar-mvar,exp)
'''