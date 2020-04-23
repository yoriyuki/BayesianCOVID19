data {
  int<lower=0> T; // Time horizon
  int<lower=0> T0; // Used for learning
  int<lower=0> P; // Population
  int C0[T]; // Cummulative infection
  int R0[T]; // Recovered
  int D0[T]; // Cummulative death

}
parameters {
  real<lower=0> init_inf;
  real<lower=0, upper=1> b0;
  real<lower=0, upper=1> b1;
  real<lower=0> theta_b;
  real<lower=0, upper=T> b_date;
  real<lower=0, upper=1> p;
  real<lower=0, upper=1> q0;
  real<lower=0, upper=1> q1;
  real<lower=0> theta_q;
  real<lower=0, upper=T> q_date;
  vector<lower=0>[T-1] NI;

  real<lower=0, upper=1> a;
  real<lower=0, upper=1> d;
  }  
transformed parameters {
  vector[T-1] q;
   for (t in 1:T-1){
    q[t] = q0 + (q1 - q0) * inv_logit(theta_q * (t - q_date));
  }
}
model {
    real C;
    real R;
    real I;
    real NR;
    real ND;
    real D;
    real b;
    real growth;
    
    a ~ beta(1, 1);
    d ~ beta(1, 1);
    p ~ beta(1, 1);

    b0 ~ gamma(1, 1);
    b1 ~ gamma(1, 1);
    theta_b ~ gamma(1, 1);
    b_date ~ uniform(30, T);

    q0 ~ beta(1, 1);
    q1 ~ beta(1, 1);
    theta_q ~ gamma(1, 1);
    q_date ~ uniform(30, T);
  
    init_inf ~ gamma(1, 1);
    
    I = init_inf;
    R = 0;
    D = 0;
    C = 0;
    for (t in 1:T-1){
      b = b0 + (b1 - b0) * inv_logit(theta_b * (t - b_date));
      growth = (1 - pow(1- p, b * I / (P - D))) * (P - C);
      NI[t] ~ normal(growth, growth);
      NR = a * I;
      ND = d * I;
      D = D + ND;
      I = I + NI[t] - NR - ND;
      C = C + NI[t];
      R = R + NR;
    }

    C0[1] ~ poisson(q[1] * init_inf);
    for (t in 1:T0-1){
      C0[t+1] - C0[t] ~ poisson(q[t] * NI[t]);
      D0[t+1] - D0[t] ~ poisson(d * (C0[t] - R0[t] - D0[t]));   
      R0[t+1] - R0[t] ~ poisson(a * (C0[t] - R0[t] - D0[t]));
    }
}
generated quantities {
  vector[T0-1] log_lik;
  real v_log_lik;
  
  for (t in 1:T0-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q[t] * NI[t]);
  }
  
  v_log_lik=0;
  for(t in T0:T-1){
    v_log_lik += poisson_lpmf(C0[t+1] - C0[t] | q[t] * NI[t]);
  }
}