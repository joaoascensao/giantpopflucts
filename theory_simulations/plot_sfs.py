import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from matplotlib.legend_handler import HandlerTuple
import matplotlib as mpl
import pandas as pd
import random
sns.set(style="ticks",font_scale=1.2)

rho = 0.5
kappaA = 1e-3
kappaB = 1e-3
c1A = 10
c1B = 10
s=0.02


def getsfs(kappaA,kappaB,c1A,c1B,rho,s,f):

      delta = c1A + c1B - 2*np.sqrt(c1A*c1B)*rho

      beta = np.sqrt((kappaA - kappaB)**2 + delta*(delta + 2*(kappaA + kappaB))) #(delta + kappaA)**2 + 2*(delta - kappaA)*kappaB + kappaB**2
      gamma = -c1A + c1B + kappaA - kappaB + 2*s

      B = (beta + delta - kappaA + kappaB)*(beta - delta*(1-2*f) + kappaA - kappaB)/(beta - delta + kappaA - kappaB)/(beta + delta*(1-2*f) - kappaA + kappaB)
      A = (beta + delta + kappaA - kappaB)*(beta + delta - kappaA + kappaB)/(beta - delta - kappaA + kappaB)/(beta - delta + kappaA - kappaB)
      #print('A',(beta - delta*(1-2*f) + kappaA - kappaB))
      #print('B',A)
      B = np.exp(np.log(B)*gamma/beta)
      A = np.exp(np.log(A)*gamma/beta)

      return (B-A)/((1-A)*(1-f)*f)

def getsfs_null(kappaA,s,f):
      return np.exp(2*s/kappaA)*(1-np.exp(-2*s*(1-f)/kappaA))/((np.exp(2*s/kappaA)-1)*f*(1-f))

#pb="blend:#B2D0E1,#3D71A8"
pb='crest'
#palette_blue = sns.color_palette(pb,8,desat=1)[3:]

palette_blue = sns.color_palette(pb,4)

print((kappaA - s)/kappaA)


f=np.linspace(0.01,1-1e-20,10**4)

print(np.log(f)[-3:-1],np.log(getsfs_null(kappaA,s,f))[-3:-1])

print(np.polyfit(np.log(f)[-3:-1],np.log(getsfs_null(kappaA,s,f))[-3:-1],1))

fig, ax1 = plt.subplots(1, 1)
fig.set_figheight(4.8)
fig.set_figwidth(6.4*1.2)
p1=[]
#c1A_vector=list(np.logspace(-1.5,1,5)[:4])+[60]
c1A_vector=list(np.logspace(-2,0.4,5)[:4])

dfsims=pd.read_csv('sfs_sims.csv')

print(dfsims.groupby(by='c1A').count())

logit = lambda x: np.log(x) - np.log(1-x)

print()

for i,c1A in enumerate(c1A_vector):
      sfs_calc=getsfs(kappaA,kappaB,c1A,c1A,rho,s,f)
      p10,=ax1.plot(f,sfs_calc,color=palette_blue[i],alpha=0.8,linewidth=1.8)
      if i==1 or i==2:
            p1.append(p10)
      fsims=np.array(dfsims[(dfsims['c1A']>c1A*0.9) & (dfsims['c1A']<c1A*1.1) ]['f'])
      
      logitbins=np.linspace(logit(1e-2)+i/15,logit(1-1e-2)-(3-i)/15,15)
      bins=1/(np.exp(-logitbins) + 1)
      hist,bine=np.histogram(fsims, bins=bins,density=True) #
      binm=[(bine[k]+bine[k+1])/2 for k in range(len(bine)-1)]
      #binl=np.array([(bine[k+1]-bine[k]) for k in range(len(bine)-1)])
      #yerr=np.sqrt(hist_un)*np.interp(binm[0],f,sfs_calc)/hist_un[0]/binl
      io=np.interp(binm[0],f,sfs_calc)
      histf=hist*io/hist[0]
      '''
      fsims=list(fsims)
      bbt=[]
      for l in range(10):
            fi=random.choices(fsims,k=len(fsims))
            histi,_=np.histogram(fi, bins=bins,density=True)
            bbt.append(histi*io/histi[0])
            print(l)
      yerr=np.std(np.vstack(bbt),axis=0)
      print(yerr)
      '''
      plt.errorbar(binm,histf,fmt='o',color=palette_blue[i],alpha=0.7,markersize=5,zorder=100)

p2,=ax1.plot(f,getsfs_null(kappaA,s,f),':',linewidth=2.5,alpha=0.8,color='k')
p3,=ax1.plot(f,1/f,'--',linewidth=2.1,alpha=0.8,color='#d43b4d') #012987




#plt.plot(f,(1-f)**-19)
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r'SFS, $p_{SFS}(f)/\theta $')
plt.xlabel(r'Frequency, $f$')



l = ax1.legend([tuple(p1), p2, p3], ['Decoupling\n'+r'$\delta \neq 0$', 'No Decoupling\n'+r'$\delta = 0$', 'Neutral\n'+r'$\delta = 0$ and $s=0$'],
               handler_map={tuple: HandlerTuple(ndivide=None)},frameon=False,bbox_to_anchor=(1.02,0.5))

#colorbar
cmap = sns.color_palette(pb,as_cmap=True)
norm = mpl.colors.Normalize(vmin=np.log10(c1A_vector[0]), vmax=np.log10(c1A_vector[-1]))

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
inset_axes = inset_axes(ax1,
                    width="15%", # width = 30% of parent_bbox
                    height="5%", # height : 1 inch
                    loc=3,
                    bbox_to_anchor=(1.15,0.8,1,1),
                    bbox_transform=ax1.transAxes,
                    )

cb1 = mpl.colorbar.ColorbarBase(inset_axes, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
cb1.set_label(r'log$_{10} \delta$')
ax1.set_ylim((1,130))
plt.tight_layout()
sns.despine()
#plt.show()
plt.savefig('sfs.pdf',format='pdf')



