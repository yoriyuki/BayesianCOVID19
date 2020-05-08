data {
  int<lower=0> T; // Time horizon
  int<lower=0> T0; // Leave-one-out
  real<lower=0> P; // Population
  int C0[T]; // Cummulative infection
  // int D0[T]; // Cummulative death
  real b_beta_0;
}
parameters {
  real<lower=0> init_inf;
  real<lower=0> b_beta;
  vector<lower=0>[T-1] b;
  vector<lower=0>[T-1] NI;
  // real<lower=0, upper=1> d;
  vector<lower=0, upper=1>[T-1] q;
  } transformed parameters {
    vector<lower=0>[T] I;
    vector<lower=0>[T] C;
    // vector<lower=0>[T] R;
    // vector<lower=0>[T] D;
  
    I[1] = init_inf;
    C[1] = init_inf;
    // R[1] = 0;
    // D[1] = 0;
    for(t in 1:T-1){
        I[t+1] = I[t] + NI[t] - 0.04 * I[t];
        C[t+1] = C[t] + NI[t];
        // R[t+1] = R[t] + 0.04 * I[t];
        // D[t+1] = D[t] + d * I[t];
    }
  } model {
    real growth;
    real R0;
    
    // d ~ beta(1, 1);
    b_beta ~ gamma(b_beta_0*b_beta, b_beta);

    init_inf ~ student_t(3, 0, 1);
    C0[1] ~ poisson(q[1] * init_inf);
    R0=0;
    for (t in 1:T-1){
      if (t == 1){
        b[t] ~ student_t(3, 0, b_beta);
      } else {
        b[t] ~ student_t(3, b[t-1], b_beta);
      }
      growth = b[t] * I[t] * (1 - C[t]/P);
      NI[t] ~ normal(growth, sqrt(growth));
      if (t != T0){
        C0[t+1] - C0[t] ~ poisson(q[t] * NI[t]);
        // D0[t+1] - D0[t] ~ neg_binomial(d*(C0[t]-R0-D0[t])*phi_D, phi_D);   
      } 
      // R0 = R0 + 0.04*(C0[t]-R0-D0[t]);
    }
}
generated quantities {
  vector[T-1] log_lik;
  
   for (t in 1:T-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q[t] * NI[t]);
  }
}