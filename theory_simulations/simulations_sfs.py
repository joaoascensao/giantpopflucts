import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

K = 2e3
mu1 = 1
dt=0.1
rho = 0.5
c_vector=list(np.logspace(-2,0.4,5)[:4])

fs=[]
cs=[]
rep=[]
for j,c1 in enumerate(c_vector):
      print('c num:',j)
      c2=c1
      cAB = np.sqrt(c1*c2)*rho
      mu2 = 1 - 0.02
      for i in range(2*10**4):
            N1 = (1e-3)*K
            N2 = K
            t=0
            while True:
                  xi = np.random.multivariate_normal(mean=[0,0],cov=[[c1,cAB],[cAB,c2]])
                  Ntot=N1+N2
                  N1 += (mu1 - Ntot/K)*N1*dt + (xi[0]*N1 + np.sqrt(N1*(mu1 + Ntot/K))*np.random.normal(loc=0,scale=1))*np.sqrt(dt)
                  N2 += (mu2 - Ntot/K)*N2*dt + (xi[1]*N2 + np.sqrt(N2*(mu2 + Ntot/K))*np.random.normal(loc=0,scale=1))*np.sqrt(dt)
                  if N1<1e-1:
                        break
                  if N2<1e-1:
                        break
                  fs.append(N1/(N1+N2))
                  t+=1
            cs+=[c1]*t
            rep+=[i]*t

pd.DataFrame({'f':fs,'c1A':cs,'rep':rep}).to_csv('sfs_sims.csv',index=False)

