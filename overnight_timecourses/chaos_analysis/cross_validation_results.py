import numpy as np 
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special
import seaborn as sns
import pandas as pd
import scipy.stats
import random
sns.set(style="ticks",font_scale=1.5)

df=pd.read_csv('rosenstein_cv.csv')
df = df.replace(r'\\n','\n', regex=True)



plt.figure()
plt.errorbar(list(df['name']),df['mse'],yerr=df['mse stderr']*2,fmt='ks')
plt.ylabel('MSE')
sns.despine()
plt.tight_layout()
plt.savefig('mse_xval.png',dpi=600)

ll=list(df['lyapunov point']-df['l ci'])
uu=list(df['u ci']-df['lyapunov point'])



plt.figure()
plt.errorbar(list(df['name']),df['lyapunov point'],yerr=np.vstack((ll,uu)),fmt='ks')
plt.ylabel(r'Lyapunov exponent, $\lambda$ (hr$^{-1}$)')
sns.despine()
plt.ylim((0,0.22))
plt.tight_layout()
plt.savefig('lya_xval.png',dpi=600)