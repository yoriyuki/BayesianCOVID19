functions{
  real logdiff(real x, real y){
    return x + log1p(-exp(-(x-y)));
  }

  // The Wilson-Hilferty approximation
  // real poisson_approx_rev_cum(real k, real lambda){
  //   real c;
  //   real mu;
  //   real sigma;
  //   c = cbrt(lambda/(1+k));
  //   mu = 1- 1/(9*k+9);
  //   sigma = 1/(3*sqrt(1+k));
  //   return logdiff(0, normal_lcdf(c | mu, sigma));
  // }

  real poisson_approx_cum(real k, real lambda){
    return normal_lcdf(k+0.5 | lambda, sqrt(lambda));
  }

  real poisson_approx_lpdf(real k, real lambda){
    return logdiff(poisson_approx_cum(k, lambda), poisson_approx_cum(k-1, lambda));
  }

} data {
  int<lower=0> T; // Time horizon
  int<lower=0> T0; // Leave-one-out
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
  } model {
    real C;
    real D;
    real I;
    real NR;
    real ND;
    real growth;
    
    a ~ beta(1, 1);
    d ~ beta(1, 1);
    p ~ beta(1, 1);
    b ~ gamma(1, 1);
    q ~ beta(1, 1);
  
    init_inf ~ gamma(1, 1);
    
    I = init_inf;
    D = 0;
    C = init_inf;
    C0[1] ~ poisson(q * init_inf);
    for (t in 1:T-1){
      growth = (1 - pow(1 - p, b * I / (P - D))) * (P - C);
      if (t != T0){
        NI[t] ~ normal(growth, sqrt(growth));
        NR = a*I;
        ND = d*I;
        C0[t+1] - C0[t] ~ poisson(q * NI[t]);
        D0[t+1] - D0[t] ~ poisson(d * (C0[t] - R0[t] - D0[t]));   
        R0[t+1] - R0[t] ~ poisson(a * (C0[t] - R0[t] - D0[t]));
        I = I + NI[t] - NR - ND;
        D = D + ND;
        C = C + NI[t];
      } else {
        I = I + NI[t-1] - NR - ND;
        D = D + ND;
        C = C + NI[t-1];
      } 
    }
}
generated quantities {
  vector[T-1] log_lik;
  
  for (t in 1:T-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q * NI[t]);
  }
}