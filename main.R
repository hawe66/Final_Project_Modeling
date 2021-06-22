rm(list=ls())  # clear workspace
graphics.off() # close all figures


library(rstan)

# read the data file
dat = read.table("PGG_data.txt", header=T, sep="\t")

allSubjs = unique(dat$subjID)  # all subject IDs
N = length(allSubjs)      # number of subjects
T = table(dat$subjID)[1]  # number of trials per subject 

rounds  <- array(0, c(N, T))
decision<- array(-1, c(N, T))
result  <- array(-1, c(N, T))
threshold<-array(0, c(N, T))
thres   <- array(0, c(N, T))
ratioC  <- array(-1, c(N, T))
prior2  <- array(0, c(N, T))
prior4  <- array(0, c(N, T))

for (i in 1:N) {
  curSubj = allSubjs[i]
  tmp     = subset(dat, subjID == curSubj)
  rounds[i, 1:T] <- tmp$rounds
  decision[i, 1:T] <- tmp$decision
  result[i, 1:T]   <- tmp$result
  threshold[i, 1:T]<- tmp$threshold
  thres[i, 1:T]    <- tmp$thres
  ratioC[i, 1:T]   <- tmp$ratioC
  prior2[i, 1:T]   <- tmp$prior2
  prior4[i, 1:T]   <- tmp$prior4
}

dataList <- list(
  N       = N,
  T       = T,
  Tsubj   = rep(T, N),
  rounds  = rounds,
  decision= decision,
  result  = result,
  threshold  = threshold,
  thres   = thres,
  ratioC  = ratioC,
  prior2  = prior2,
  prior4  = prior4
)

# run!
output = stan("group_nobeta.stan", data = dataList, 
              control = list(adapt_delta = 0.8, max_treedepth = 12),
              pars = c("w", "pi", "lambda", "alpha", "log_lik"), 
              init_r=5, iter = 4000, warmup=1000, thin=2, chains=4, cores=4)

# convergence check
pairs(output, pars = c("w[24]", "pi[24]", "lambda[24]", "alpha[24]", 
                       "lp__"), las = 1)

# traceplot
traceplot(output, pars="pi", inc_warmup = TRUE)

# print summary

print(output)

# extract Stan fit object (parameters)
parameters <- rstan::extract(output)

# plot posterior distribution
stan_plot(output, pars="w", show_density = TRUE)
stan_plot(output, pars="pi", show_density = TRUE)
stan_plot(output, pars="lambda", show_density = TRUE)
stan_plot(output, pars="alpha", show_density = TRUE)
#stan_plot(output, pars="beta", show_density = TRUE)

# compute log likelihood
library(loo)

LL_2 <- extract_log_lik(output, parameter_name = "log_lik", merge_chains = TRUE)
r_eff <- relative_eff(exp(LL_2), chain_id = rep(1:4, each = 1500), cores = 4)
loo_2 <- loo(LL_2, r_eff = r_eff, save_psis = TRUE, cores = 4)
#loo_compare(loo_2, loo_3, criterion = c("loo", "kfold", "waic"), detail = FALSE)
#loo_compare(loo_1, loo_3, loo_4, loo_5, loo_6)
