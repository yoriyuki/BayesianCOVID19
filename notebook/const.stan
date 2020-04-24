data {
  int<lower=0> T; // Time horizon
  int<lower=0> T0; // Leave one out
  int<lower=0> P; // Population
  int C0[T]; // Cummulative infection
  int R0[T]; // Recovered
  int D0[T]; // Cummulative death

}
parameters {
  real<lower=0> init_inf;
  real<lower=0> b;
  real<lower=0, upper=1> p;
  real<lower=0, upper=1> q;
  vector<lower=0>[T-1] NI;

  real<lower=0, upper=1> a;
  real<lower=0, upper=1> d;
  }  model {
    real C;
    real R;
    real I;
    real NR;
    real ND;
    real D;
    real growth;
    
    a ~ beta(1, 1);
    d ~ beta(1, 1);
    p ~ beta(1, 1);

    b ~ gamma(1, 1);

    q ~ beta(1, 1);
  
    init_inf ~ gamma(1, 1);
    
    I = init_inf;
    R = 0;
    D = 0;
    C = 0;
    for (t in 1:T-1){
      growth = (1 - pow(1- p, b * I / (P - D))) * (P - C);
      if (t != T0){
        NI[t] ~ normal(growth, sqrt(growth));
      }
      NR = a * I;
      ND = d * I;
      D = D + ND;
      I = I + NI[t] - NR - ND;
      C = C + NI[t];
      R = R + NR;
    }

    C0[1] ~ poisson(q * init_inf);
    for (t in 1:T-1){
      if (t != T0){
        C0[t+1] - C0[t] ~ poisson(q * NI[t]);
        D0[t+1] - D0[t] ~ poisson(d * (C0[t] - R0[t] - D0[t]));   
        R0[t+1] - R0[t] ~ poisson(a * (C0[t] - R0[t] - D0[t]));
      }
    }
}
generated quantities {
  vector[T-1] log_lik;
  
  for (t in 1:T-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q * NI[t]);
  }
}