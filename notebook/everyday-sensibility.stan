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
  real<lower=0> b_beta;
  vector<lower=0>[T-1] b;
  vector<lower=0, upper=1>[T-1] q;
  real<lower=0> q_factor;
  vector<lower=0>[T-1] NI;
  real<lower=0, upper=1> a;
  real<lower=0, upper=1> d;
  } transformed parameters {
    vector<lower=0>[T] I;
    vector<lower=0>[T] C;
    vector<lower=0>[T] R;
    vector<lower=0>[T] D;

    I[1] = init_inf;
    C[1] = init_inf;
    R[1] = 0;
    D[1] = 0;
    for(t in 1:T-1){
        I[t+1] = I[t] + NI[t] - (a+d) * I[t];
        C[t+1] = C[t] + NI[t];
        R[t+1] = R[t] + a * I[t];
        D[t+1] = D[t] + d * I[t];
    }
  } model {
    real growth;
    
    a ~ beta(1, 1);
    d ~ beta(1, 1);
    b_beta ~ gamma(10, 10);
    q_factor ~ gamma(10, 10);
    init_inf ~ student_t(3, 1, 1);
    C0[1] ~ poisson(q[1] * init_inf);
    for (t in 1:T-1){
      if (t == 1){
        b[t] ~ student_t(3, 0, 10);
        q[t] ~ beta(1, 1);
      } else {
        b[t] ~ student_t(3, b[t-1], b_beta);
        q[t] ~ beta(q[t-1]*q_factor, (1-q[t])*q_factor);
      }
      growth = b[t] * I[t] * (1 - C[t]/P);
      NI[t] ~ normal(growth, sqrt(growth));
      if (t != T0){
        C0[t+1] - C0[t] ~ poisson(q[t] * NI[t]);
        D0[t+1] - D0[t] ~ poisson(d*(C0[t] - R0[t] - D0[t]));   
        R0[t+1] - R0[t] ~ poisson(a*(C0[t] - R0[t] - D0[t]));
      } 
    }
}
generated quantities {
  vector[T-1] log_lik;
  
  for (t in 1:T-1) {
    log_lik[t] = poisson_lpmf(C0[t+1] - C0[t] | q[t] * NI[t]);
  }
}