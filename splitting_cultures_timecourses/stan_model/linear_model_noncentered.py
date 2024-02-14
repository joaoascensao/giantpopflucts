import pandas as pd
import scipy as sp 
import numpy as np 
import matplotlib.pyplot as plt
import pystan
import sys


stancode = """

data {
  int<lower=0> T;           // # time points
  int<lower=0> G;           // # of different ancestry groups
  int<lower=0> N;           // # of different data point
  vector[N] s;        // fitnesses
  int d[N];            // day for each data point
  int g[N];            // group number for each data point
  vector[N] fo;             // f0
  vector[N] vm;           // variance from measurement error for each point 
}
parameters {
  vector<lower=-10,upper=10>[G] kappa;        // ancestry group effects
  vector<lower=-10,upper=10>[T] alpha;       // true bottleneck size at each time point
  real<lower=1e-20,upper=1> kappa_var; // variance of ancestry group effects
  real<lower=1e-10,upper=1> alpha_var; // variance of day effects
  real<lower=1e-10,upper=1> delta; // variance from decoupling flucts
  real<lower=-100,upper=100> beta; // frequency-dependent effect
  real<lower=-100,upper=100> interc; //intercept
}
model {
  kappa_var ~ pareto(1e-20, 1);
  alpha_var ~ pareto(1e-10, 1);
  delta ~ pareto(1e-10, 1);

  for (k in 1:G) {
    kappa[k] ~ normal(0, 1);
  }

  for (a in 1:T) {
    alpha[a] ~ normal(0, 1);
  }

  for (i in 1:N) {
    if (g[i]==0){
      s[i] ~ normal(interc + alpha[d[i]]*sqrt(alpha_var) + beta*fo[i], sqrt(delta + vm[i]));
    } else {
      s[i] ~ normal(interc + kappa[g[i]]*sqrt(kappa_var) + alpha[d[i]]*sqrt(alpha_var) + beta*fo[i], sqrt(delta + vm[i]));
    }
  }
}
"""


df=pd.read_csv('fitness.csv')
G=len(list(set(list(df['anc num']))))-1
T=12
data_dic ={
  'T':T,
  'G':G,
  'N':len(df),
  's':list(df['Fitness']),
  'd':list(df['Day']),
  'g':list(df['anc num']),
  'fo':list(df['f0']),
  'vm':list(df['measurement var']),
}


sm = pystan.StanModel(model_code=stancode)
fit = sm.sampling(data=data_dic, iter=1000, chains=4,control=dict(max_treedepth=20))


for pp in ['kappa_var','alpha_var','delta','beta','interc']:
  pd.DataFrame(fit.extract([pp])[pp],columns=[pp]).to_csv('posterior_{}.csv'.format(pp),index=False)

pd.DataFrame(fit.extract(['kappa'])['kappa'],columns=list(range(1,G+1))).to_csv('posterior_kappas.csv',index=False)
pd.DataFrame(fit.extract(['alpha'])['alpha'],columns=list(range(1,T+1))).to_csv('posterior_alphas.csv',index=False)



