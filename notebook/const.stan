data {
  int<lower=0> T; // Time horizon
  int<lower=0> T0; // Leave one out
  real<lower=0> P; // Population
  int C0[T]; // Cummulative infection
  // int D0[T]; // Cummulative death

}
parameters {
  real<lower=0> init_inf;
  real<lower=0> b;
  real<lower=0, upper=1> q;
  vector<lower=0>[T-1] NI;

  // real<lower=0, upper=1> a;
  // real<lower=0, upper=1> d;
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
      //  R[t+1] = R[t] + 0.04 * I[t];
      //  D[t+1] = D[t] + d * I[t];
    }
  } model {
    real growth;
    real R0;
    
    // d ~ beta(1, 1);
    b ~ student_t(3, 0, 1);
    q ~ beta(1, 1);
    C0[1] ~ poisson(q * init_inf);
    // R0 = 0;
    for (t in 1:T-1){
      growth = b * I[t] * (1 - C[t]/P);
      NI[t] ~ normal(growth, sqrt(growth));
      if (t != T0){
        C0[t+1] - C0[t] ~ poisson(q * NI[t]);
        // D0[t+1] - D0[t] ~ poisson(d*(C0[t]-R0-D0[t]));   
      }
      // R0 = R0 + 0.04*(C0[t]-R0-D0[t]);
    }
}
generated quantities {
  vector[T-1] log_lik;
  
  for (t in 1:T-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q * NI[t]);
  }
}