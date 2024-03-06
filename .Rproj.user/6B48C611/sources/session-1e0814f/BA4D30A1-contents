data {
  int<lower=1> n;              // Number of trials
  int<lower=0, upper=1>[n] outcome; // Outcome (0 for loss, 1 for win)
}

parameters {
  real<lower=0, upper=1> p;    // Probability of staying after a win
}

model {
  // Prior for probability of staying
  p ~ beta(1, 1);              
  
  // Likelihood
  for (trial in 2:N) {
    
    if (outcome[trial - 1] == 1) {
      outcome[trial] ~ bernoulli(p);  // If won, stay with probability p
    } else {
      outcome[trial] ~ bernoulli(1);  // If lost, shift
    }
  }
}
