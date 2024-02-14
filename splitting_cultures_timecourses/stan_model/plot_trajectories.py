import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
import matplotlib
sns.set(style="ticks",font_scale=1.5)


df = pd.read_csv('../tc_data1.csv').fillna(value=0)

#plt.figure()
#sns.lineplot(data=df,y='S freq',x='day',hue='group',legend=None)
#plt.show()


set1=[0,1,2,3]
set2=[3,4,5,6,7]
set3=[7,8,9,10,11,12]

colors1={
	3:sns.color_palette('Blues',n_colors=10,desat=0.95).as_hex()[7],
	2:sns.color_palette('Reds',n_colors=10,desat=0.95).as_hex()[7],
	1:sns.color_palette('Purples',n_colors=10,desat=0.95).as_hex()[7],
	4:sns.color_palette('Greens',n_colors=10,desat=0.95).as_hex()[7],
}


def change_cmap(color,n):
	min_val, max_val = 0.35,0.85
	orig_cmap=sns.color_palette(color,as_cmap=True)
	colors = orig_cmap(np.linspace(min_val, max_val, n))
	cmap=matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)
	matplotlib.cm.register_cmap("mycmap", cmap)
	cpal = sns.color_palette("mycmap", n_colors=n,desat=0.95) #, desat=0.2
	return cpal


colors2={
	3:change_cmap('Blues',3),
	2:change_cmap('Reds',3),
	1:change_cmap('Purples',3),
	4:change_cmap('Greens',3),
}
colors3={
	3:change_cmap('Blues',6),
	2:change_cmap('Reds',6),
	1:change_cmap('Purples',6),
	4:change_cmap('Greens',6),
}

plt.figure(figsize=[8, 5])
for i,group in df[df['day'].isin(set1)].drop_duplicates(subset=['anc1','day']).groupby(by='anc1'):
	#plt.plot(group['day'],group['S freq'],colors1[i])
	day=[]
	sfreq=[]
	for j,row in group.iterrows():
		stdi=2*np.float(np.sqrt(row['S freq']*(1-row['S freq'])/row['corrected total count']))
		meani=np.float(row['S freq'])
		xi=np.random.normal(loc=meani,scale=stdi,size=1000)
		day+=[np.int(row['day'])]*1000
		sfreq+=list(xi)
	group2=pd.DataFrame({'day':day,'S freq':sfreq})
	g=sns.lineplot(data=group2,x='day',y='S freq',color=colors1[i],alpha=0.8,marker='.',markeredgewidth=0,err_style='bars',ci='sd')


for i,group in df[df['day'].isin(set2)].drop_duplicates(subset=['anc1','anc2','day']).groupby(by=['anc1']):
	day=[]
	sfreq=[]
	groups=[]
	for j,row in group.iterrows():
		stdi=2*np.float(np.sqrt(row['S freq']*(1-row['S freq'])/row['corrected total count']))
		meani=np.float(row['S freq'])
		xi=np.random.normal(loc=meani,scale=stdi,size=1000)
		day+=[np.int(row['day'])]*1000
		sfreq+=list(xi)
		groups+=[np.int(row['anc2'])]*1000
	group2=pd.DataFrame({'day':day,'S freq':sfreq,'anc2':groups})
	sns.lineplot(data=group2,x='day',y='S freq',hue='anc2',
		palette=colors2[i],legend=None,alpha=0.8,marker='.',markeredgewidth=0,err_style='bars',ci='sd') #palette=sns.cubehelix_palette(n_colors=3, start=colors2[i], rot=0.18, gamma=0.93, hue=0.8, light=0.42, dark=0.43)


for i,group in df[df['day'].isin(set3)].groupby(by=['anc1']):
	day=[]
	sfreq=[]
	groups=[]
	for j,row in group.iterrows():
		stdi=2*np.float(np.sqrt(row['S freq']*(1-row['S freq'])/row['corrected total count']))
		meani=np.float(row['S freq'])
		xi=np.random.normal(loc=meani,scale=stdi,size=1000)
		day+=[np.int(row['day'])]*1000
		sfreq+=list(xi)
		groups+=[str(row['group'])]*1000
	group2=pd.DataFrame({'day':day,'S freq':sfreq,'group':groups})
	sns.lineplot(data=group2,x='day',y='S freq',hue='group',
		palette=colors3[i],legend=None,alpha=0.8,marker='.',markeredgewidth=0,err_style='bars',ci='sd')

grey = '#949aa1'
plt.axvline(0, color=grey,alpha=0.8)
plt.axvline(3, color=grey,alpha=0.8)
plt.axvline(7, color=grey,alpha=0.8)
sns.despine()
plt.ylabel(r'$S$ Frequency, $f_S$')
plt.xlabel(r'Day, $t$')
plt.yticks([0.05,0.1,0.15,0.2])
plt.tight_layout()
plt.savefig('traj.png',dpi=600)
