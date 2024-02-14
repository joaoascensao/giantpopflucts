import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
import matplotlib
sns.set(style="ticks",font_scale=1.5)

colors_o={
	30:sns.color_palette('Blues',n_colors=10,desat=0.95).as_hex()[7],
	20:sns.color_palette('Reds',n_colors=10,desat=0.95).as_hex()[7],
	10:sns.color_palette('Purples',n_colors=10,desat=0.95).as_hex()[7],
	40:sns.color_palette('Greens',n_colors=10,desat=0.95).as_hex()[7],
}

def change_cmap(color,n):
	min_val, max_val = 0.45,0.75
	orig_cmap=sns.color_palette(color,as_cmap=True)
	colors = orig_cmap(np.linspace(min_val, max_val, n))
	cmap=matplotlib.colors.LinearSegmentedColormap.from_list("mycmap", colors)
	matplotlib.cm.register_cmap("mycmap", cmap)
	cpal = sns.color_palette("mycmap", n_colors=n,desat=0.95) #, desat=0.2
	return cpal.as_hex()


children={
	1:[1,2,3],
	2:[4,5,6],
	3:[7,8,9],
	4:[10,11,12]
}

colors2={
	3:change_cmap('Blues',3),
	2:change_cmap('Reds',3),
	1:change_cmap('Purples',3),
	4:change_cmap('Greens',3),
}
strintfloat=lambda x: str(int(np.float(x)))
for anc1 in children:
	for i,child in enumerate(children[anc1]):
		colors_o[int(strintfloat(anc1)+strintfloat(child))]=colors2[anc1][i]



df = pd.read_csv('posterior_alphas.csv')
alpha_var=pd.read_csv('posterior_alpha_var.csv')
df2 = pd.read_csv('fitness.csv').fillna(value=0)
days=np.array(list(range(1,13)))
y=[]
lerr=[]
uerr=[]
err=[]
for i in days:
	xi=np.array(df[str(i)]*np.sqrt(np.array(alpha_var['alpha_var'])))
	mm=np.mean(xi)
	y.append(mm)
	err.append(np.std(xi))
	lerr.append(mm-np.quantile(xi,0.025))
	uerr.append(np.quantile(xi,0.975)-mm)

#dd2=df.melt(value_vars=[str(i) for i in days])
grey = '#949aa1'

fig, ax = plt.subplots(figsize=(12,5))
plt.axhline(0, color=grey,zorder=0.7)
alphadic=dict(alpha=.3)
sns.boxplot(data=df2,x='Day',y='Fitness',color='#c2c2c2',fliersize=0,boxprops=alphadic,whiskerprops=alphadic,medianprops=alphadic,capprops=alphadic)
sns.swarmplot(data=df2,x='Day',y='Fitness',label='Replicate populations',palette=colors_o,hue='Anc2',alpha=0.9) #,hue='Anc'
#sns.pointplot(data=df2,x='Day',y='Fitness')
plt.errorbar(days-1,y,yerr=np.vstack((lerr,uerr)),fmt='ks',zorder=100,alpha=0.7,label='Inferred marginal extrinsic effect',capsize=3)
#plt.legend().set_visible(False)
plt.ylabel(r'logit $f_t-$ logit $f_{t-1}$')
plt.xlabel(r'Second Day, $t$')

handles, labels = ax.get_legend_handles_labels()

ax.legend([(handles[1],handles[0],handles[2]),handles[-1]], [labels[0],labels[-1]],handler_map={tuple: HandlerTuple(ndivide=None)})

sns.despine()
plt.tight_layout()
#plt.show()
plt.savefig('displacement_swarm.png',dpi=600)



