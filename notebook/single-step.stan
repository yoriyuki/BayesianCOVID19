functions{
  real logdiff(real x, real y){
    return x + log1p(-exp(-(x-y)));
  }

  // The Wilson-Hilferty approximation
  real poisson_approx_rev_cum(real k, real lambda){
    real c;
    real mu;
    real sigma;
    c = cbrt(lambda/(1+k));
    mu = 1- 1/(9*k+9);
    sigma = 1/(3*sqrt(1+k));
    return normal_lcdf(c | mu, sigma);
  }

  real poisson_approx_pdf(real k, real lambda){
    return logdiff(poisson_approx_rev_cum(k-0.5, lambda), poisson_approx_rev_cum(k+0.5, lambda));
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
  real<lower=0> b0;
  real<lower=0> b1;
  real<lower=0> theta_b;
  real<lower=0, upper=T> b_date;
  real<lower=0, upper=1> p;
  real<lower=0, upper=1> q0;
  real<lower=0, upper=1> q1;
  real<lower=0> theta_q;
  real<lower=0, upper=T> q_date;
  vector<lower=0>[T-1] NI;
  vector<lower=0>[T-1] NR;
  vector<lower=0>[T-1] ND;
  real<lower=0, upper=1> a;
  real<lower=0, upper=1> d;
  }  
transformed parameters {
   vector[T-1] q;
   for (t in 1:T-1){
    q[t] = q0 + (q1 - q0) * inv_logit((t - q_date) / theta_q);
  }
}
model {
    real C;
    real D;
    real I;
    real b;
    real growth;
    
    a ~ beta(1, 1);
    d ~ beta(1, 1);
    p ~ beta(1, 1);

    b0 ~ gamma(1, 1);
    b1 ~ gamma(1, 1);
    theta_b ~ gamma(1, 1);
    b_date ~ uniform(0, T);

    q0 ~ beta(1, 1);
    q1 ~ beta(1, 1);
    theta_q ~ gamma(1, 1);
    q_date ~ uniform(0, T);
  
    init_inf ~ gamma(1, 1);
    
    I = init_inf;
    D = 0;
    C = init_inf;
    C0[1] ~ poisson(q[1] * init_inf);
    for (t in 1:T-1){
      b = b0 + (b1 - b0) * inv_logit((t - b_date) / theta_b);
      growth = (1 - pow(1 - p, b * I / (P - D))) * (P - C);
      if (t != T0){
        target += poisson_approx_pdf(NI[t], growth);
        target += poisson_approx_pdf(NR[t], a*I);
        target += poisson_approx_pdf(ND[t], d*I);
        C0[t+1] - C0[t] ~ poisson(q[t] * NI[t]);
        D0[t+1] - D0[t] ~ poisson(d * (C0[t] - R0[t] - D0[t]));   
        R0[t+1] - R0[t] ~ poisson(a * (C0[t] - R0[t] - D0[t]));
        I = I + NI[t] - NR[t] - ND[t];
        D = D + ND[t];
        C = C + NI[t];
      } else {
        I = I + NI[t-1] - NR[t-1] - ND[t-1];
        D = D + ND[t-1];
        C = C + NI[t-1];
      } 
    }
}
generated quantities {
  vector[T-1] log_lik;
  
  for (t in 1:T-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q[t] * NI[t]);
  }
}