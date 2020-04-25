data {
  int<lower=0> T; // Time horizon
  int<lower=0> T0; // Leave-one-out
  int<lower=0> P; // Population
  int C0[T]; // Cummulative infection
  int R0[T]; // Recovered
  int D0[T]; // Cummulative death

}
parameters {
  real<lower=0> init_inf;
  vector<lower=0>[T-1] b;
  vector<lower=0, upper=1>[T-1] q;
  vector<lower=0>[T-1] NI;
  real<lower=0, upper=1> a;
  real<lower=0, upper=1> d;
  real<lower=0, upper=1> p;
  }  
model {
    real C;
    real D;
    real I;
    real NR;
    real ND;
    real growth;
    
    a ~ beta(1, 1);
    d ~ beta(1, 1);
    p ~ beta(1, 1);
  
    init_inf ~ gamma(1, 1);
    
    I = init_inf;
    D = 0;
    C = init_inf;
    C0[1] ~ poisson(q[1] * init_inf);
    for (t in 1:T-1){
      if (t == 1){
        b[t] ~ gamma(1, 1);
        q[t] ~ beta(1, 1);
      } else {
        b[t] ~ gamma(b[t-1], 1);
        q[t] ~ beta(q[t-1], 1-q[t]);
      }
      growth = (1 - pow(1 - p, b[t] * I / (P - D))) * (P - C);
      NI[t] ~ normal(growth, sqrt(growth));
      NR = a*I;
      ND = d*I;
      I = I + NI[t] - NR - ND;
      D = D + ND;
      C = C + NI[t];
      if (t != T0){
        C0[t+1] - C0[t] ~ poisson(q[t] * NI[t]);
        D0[t+1] - D0[t] ~ poisson(d * (C0[t] - R0[t] - D0[t]));   
        R0[t+1] - R0[t] ~ poisson(a * (C0[t] - R0[t] - D0[t]));
      } 
    }
}
generated quantities {
  vector[T-1] log_lik;
  
  for (t in 1:T-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q[t] * NI[t]);
  }
}