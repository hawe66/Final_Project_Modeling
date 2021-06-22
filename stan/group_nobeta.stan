data {
  int<lower=1> N; // participants
  int<lower=1> T; // trials (=180)
  int<lower=1, upper=T> Tsubj[N];
  int<lower=1, upper=15> rounds[N,T];  // 15 rounds per PGG, total 12 PGG
  int<lower=0, upper=1> decision[N,T]; // decision for contribution
  int<lower=0, upper=1> result[N,T];   // result
  int<lower=2, upper=4> threshold[N,T];// k=2 or k=4
  real<lower=2, upper=4> thres[N,T];   // k=2 or k=4
  real<lower=0,upper=1> ratioC[N,T];   // ratio of contributors
  real<lower=0,upper=1> prior2[N,T];     // prior belief of #cont in k=2
  real<lower=0,upper=1> prior4[N,T];     // prior belief of #cont in k=4
}

parameters {
  real mu_w;
  real mu_p;
  real mu_a;
  real mu_l;
  real<lower=0> sigma_w;
  real<lower=0> sigma_p;
  real<lower=0> sigma_a;
  real<lower=0> sigma_l;
  //Matt-trick
  vector[N] w_pr;
  vector[N] pi_pr;
  vector[N] alpha_pr;
  vector[N] lambda_pr;
  //vector[N] theta_pr;
}

transformed parameters{
  vector<lower=0,upper=1>[N] w;
  vector<lower=-1,upper=1>[N] pi;
  vector<lower=-5,upper=0>[N] lambda;
  vector<lower=0,upper=1>[N] alpha; 
  //vector<lower=0,upper=1>[N] theta;
  
  for (i in 1:N) {
    w[i]      = Phi_approx( mu_w  + sigma_w * w_pr[i] );
    pi[i]     = Phi_approx( mu_p  + sigma_p * pi_pr[i] ) * 2 - 1;
    alpha[i]  = Phi_approx( mu_a  + sigma_a * alpha_pr[i] );
    lambda[i] = Phi_approx( mu_l  + sigma_l * lambda_pr[i] ) * (-5);
  }
}

model {
  //hyperparameters
  mu_w  ~ normal(0, 0.5);
  mu_p  ~ normal(0, 0.5);
  mu_a  ~ normal(0, 1.0);
  mu_l  ~ normal(0, 1.0);
  sigma_w ~ cauchy(0, 1.0);
  sigma_p ~ cauchy(0, 1.0);
  sigma_a ~ normal(0, 1.0);
  sigma_l ~ normal(0, 1.0);
  //individual parameters
  w_pr ~ cauchy(0, 1.0);
  pi_pr ~ cauchy(0, 1.0);
  alpha_pr ~ normal(0, 1.0);
  lambda_pr ~ normal(0, 1.0);
  //theta_pr ~ normal(0, 1.0);
  
  for (i in 1:N) {
    real gamma[T+1]; //belief of other's choice
    for (t in 1:T) {
      //real pCont;  //overall probability for contribution
      real I; real G; real Q;  //individual, group, total utility
      real PEs;                //social prediction error
      real K; real eG;
      real eI;
      int free;
      free = 5-threshold[i, t];   //desired limit of free-riders
      //initial belief of other's choice
      if (rounds[i, t] == 1){
        if (threshold[i, t] == 2) gamma[t] = 1 - prior2[i, t];
        else                      gamma[t] = 1 - prior4[i, t];
      }

      //compute Individual Utility
      eI = choose(4, free) * gamma[t]^(free) * (1-gamma[t])^(4-free);
      I = lambda[i] + eI * 2 + pi[i] * eI * 2 * 4;
        
      //compute Group Utility
      K = thres[i,t] / 5;
      eG = binomial_cdf(free, 4, gamma[t]);
      G = ((1-K^(16-rounds[i,t]))/(1-K)) * 2 * eG;
        
      //compute Total Utility
      Q = w[i] * I + (1-w[i]) * G;
      decision[i,t] ~ bernoulli_logit(Q);
      
      //update belief via social reinforcement learning
      PEs = (1 - ratioC[i, t]) - gamma[t]; //social prediction error
      gamma[t+1] = gamma[t] + alpha[i] * PEs;    //update belief by PEs
      //gamma is the probability, it needs boundary condition [0,1]
      //gamma[t+1] = fmin(gamma[t+1], 1-1e-2);
      //gamma[t+1] = fmax(gamma[t+1], 1e-2);
    }
  }
}

generated quantities {
  vector[N] log_lik;
  matrix[N,T] LL;
  for (i in 1:N){
    real gamma[T+1]; //belief of other's choice
    for (t in 1:T) {
      real I; real G; real Q;  //individual, group, total utility
      real PEs;                //social prediction error
      real K; real eG;
      real eI;
      int free;
      free = 5-threshold[i, t];   //desired limit of free-riders
      //initial belief of other's choice
      if (rounds[i, t] == 1){
        if (threshold[i, t] == 2) gamma[t] = 1 - prior2[i, t];
        else                      gamma[t] = 1 - prior4[i, t];
      }

      //compute Individual Utility
      eI = choose(4, free) * gamma[t]^(free) * (1-gamma[t])^(4-free);
      I = lambda[i] + eI * 2 + pi[i] * eI * 2 * 4;
        
      //compute Group Utility
      K = thres[i,t] / 5;
      eG = binomial_cdf(free, 4, gamma[t]);
      G = ((1-K^(16-rounds[i,t]))/(1-K)) * 2 * eG;
        
      //compute Total Utility
      Q = w[i] * I + (1-w[i]) * G;
      LL[i,t] = bernoulli_logit_lpmf(decision[i,t] | Q);
      
      //update belief via social reinforcement learning
      PEs = (1 - ratioC[i, t]) - gamma[t]; //social prediction error
      gamma[t+1] = gamma[t] + alpha[i] * PEs;    //update belief by PEs
      //gamma is the probability, it needs boundary condition [0,1]
      //gamma[t+1] = fmin(gamma[t+1], 1-1e-2);
      //gamma[t+1] = fmax(gamma[t+1], 1e-2);
    }
    log_lik[i] = sum(LL[i]);
  }
}