functions{
  vector ni(real a, real d, real p,
            real b,
            real init_inf, real P, int T){
    real C;
    real I;
    real NR;
    real ND;
    vector[T-1] NI;

    I = init_inf;
    C = 0;
    for (t in 1:T-1){
      NI[t] = (P - C) * b * (1 - pow(1 - p, I));
      C = C + NI[t];
      NR = a * I;
      ND = d * I;
      I = I + NI[t] - NR - ND;
    }
    return NI;
  } 
}
  data {
  int<lower=0> T; // Time horizon
  int<lower=0> T0;
  int<lower=0> P; // Population
  int C0[T]; // Cummulative infection
  int R0[T]; // Recovered
  int D0[T]; // Cummulative death

}
parameters {
  real<lower=0> init_inf;
  real<lower=0, upper=1> p;
  real<lower=0, upper=1> b;
  real<lower=0, upper=1> a;
  real<lower=0, upper=1> d;
  real<lower=0, upper=1> q;
  }
  transformed parameters {
  vector[T-1] NI;
  NI = ni(a, d, p, b, init_inf, P, T);
}
model {
    a ~ beta(1, 1);
    d ~ beta(1, 1);
    p ~ beta(1, 1);
    q ~ beta(1, 1);
    b ~ beta(1, 1);

    init_inf ~ gamma(1, 1);
    C0[1] ~ poisson(init_inf);
    for (t in 1:T0-1){
      C0[t+1] - C0[t] ~ poisson(q * NI[t]);
      D0[t+1] - D0[t] ~ poisson(d * (C0[t] - R0[t] - D0[t]));   
      R0[t+1] - R0[t] ~ poisson(a * (C0[t] - R0[t] - D0[t]));
    }
}
generated quantities {
  vector[T0-1] log_lik;
  real v_log_lik;
  
  for (t in 1:T0-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q * NI[t]);
  }
  
  v_log_lik=0;
  for(t in T0:T-1){
    v_log_lik += poisson_lpmf(C0[t+1] - C0[t] | q * NI[t]);
  }
}