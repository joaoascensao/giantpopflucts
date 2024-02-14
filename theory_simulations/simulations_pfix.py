import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

K = 2e3
mu1 = 1
dt=0.1
rho = 0.5
c_vector=list(np.logspace(-2,0.4,5)[:4])

pfix=[]
tot=[]
cs=[]
sse=[]
for j,c1 in enumerate(c_vector):
      print('c num:',j)
      c2=c1
      cAB = np.sqrt(c1*c2)*rho
      svec=np.logspace(-5+j/8,0-(3-j)/8,10)
      for s in svec:
            print('s:',s)
            mu2 = 1 - s
            fix1=0
            fix2=0
            for i in range(10**5):
                  N1 = (1e-3)*K
                  N2 = K
                  while True:
                        xi = np.random.multivariate_normal(mean=[0,0],cov=[[c1,cAB],[cAB,c2]])
                        Ntot=N1+N2
                        N1 += (mu1 - Ntot/K)*N1*dt + (xi[0]*N1 + np.sqrt(N1*(mu1 + Ntot/K))*np.random.normal(loc=0,scale=1))*np.sqrt(dt)
                        N2 += (mu2 - Ntot/K)*N2*dt + (xi[1]*N2 + np.sqrt(N2*(mu2 + Ntot/K))*np.random.normal(loc=0,scale=1))*np.sqrt(dt)
                        if N1<1e-1:
                              fix2+=1
                              break
                        if N2<1e-1:
                              fix1+=1
                              break
            pfix.append(fix1/(fix1+fix2))
            tot.append(fix1+fix2)
            cs.append(c1)
            sse.append(s)

pd.DataFrame({'pfix':pfix,'n sims':tot,'c1A':cs,'s':sse}).to_csv('pfix_sims.csv',index=False)

