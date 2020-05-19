data {
  int<lower=0> T; // Time horizon
  int<lower=0> T0; // Leave-one-out
  real<lower=0> P; // Population
  int C0[T]; // Cummulative infection
  real b_sigma_0;
}
parameters {
  real<lower=0> init_inf;
  real<lower=0> b_beta;
  vector<lower=0>[T-1] b;
  vector<lower=0>[T-1] NI;
  vector<lower=0, upper=1>[T-1] q;
  } transformed parameters {
    vector<lower=0>[T] I;
    vector<lower=0>[T] C;
  
    I[1] = init_inf;
    C[1] = init_inf;
    for(t in 1:T-1){
        I[t+1] = I[t] + NI[t] - 0.04 * I[t];
        C[t+1] = C[t] + NI[t];
    }
  } model {
    real growth;
    real R0;
    
    b_sigma ~ exponential(b_sigma_0);

    init_inf ~ student_t(3, 0, 1);
    C0[1] ~ poisson(q[1] * init_inf);
    R0=0;
    for (t in 1:T-1){
      if (t == 1){
        b[t] ~ student_t(3, 0, b_sigma);
      } else {
        b[t] ~ student_t(3, b[t-1], b_sigma);
      }
      growth = b[t] * I[t] * (1 - C[t]/P);
      NI[t] ~ normal(growth, sqrt(growth));
      if (t != T0){
        C0[t+1] - C0[t] ~ poisson(q[t] * NI[t]);
      } 
    }
}
generated quantities {
  vector[T-1] log_lik;
  
   for (t in 1:T-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q[t] * NI[t]);
  }
}