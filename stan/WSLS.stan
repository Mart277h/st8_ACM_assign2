//
// This Stan program defines a model for playing the Matching Pennies game with
// an asymmetrical Win-Stay-Lose-Shift strategy.

data {
  int<lower=1> trials; //trials - defines length of following data
  array[trials] int <lower=0> choices; //array of choices made by the agent
  vector[trials] feedback; //feedback from previous choices
}

// For our transformed data, we want to have a measure of which hand we win or 
// loose with and the inverse for the other hand. Such that we can use this 
// for our lineary model, and thereby indicating which probability the model
// should estimate

transformed data{
  vector[trials] winhand; //this vector codes the hand you chose if you won
  vector[trials] losshand; //this vector codes the hand you chose if you lost
  
  winhand[1] = 0; //0 gives us no probability in first trial
  losshand[1] = 0; //0 gives us no probability in first trial
  
  for(t in 2:trials) // we code for the 4 different scenarios that can happen 
                      //at a given trial. 0 means that it is irrelevant
    if (feedback[t-1] == 1 && choices[t-1] == 0){ //winning on left hand -1(log-odds space) 
      winhand[t]= -1;
      losshand[t]= 0; 
    }
    else if (feedback[t-1] == 1 && choices[t-1] == 1) { //winning on right hand 1 
      winhand[t]= 1;
      losshand[t]= 0;
    }
    else if (feedback[t-1] == 0 && choices[t-1] == 1){ //loosing on left hand -1
      winhand[t]= 0;
      losshand[t]= -1;
    }
    else if (feedback[t-1] == 0 && choices[t-1] == 0){ //loosing on right hand 1
      winhand[t]= 0;
      losshand[t]= 1;
    }
}

// Our parameters to be estimated are the win and loss probabilities 
parameters {
  real winprob; // win probability parameter 
  real lossprob; // loss probability parameter
}

//Our linear model that predicts choices by our estimated probabilities.
model {
  target += normal_lpdf(winprob| 0,1); //prior for win probability
  target += normal_lpdf(lossprob| 0,1); //prior for loss probability
  
  // Our model predicts choices from the win probability 
  // or loss probability weighted against which hand had a loss and which hand 
  // had a win (0 = left, 1 = right)
  target += bernoulli_logit_lpmf(choices | winprob * winhand + lossprob * losshand);
}

// The generated quantities gives us access to prior and posterior predictive
// checks. 
generated quantities{
  real winprob_prior; //creating the prior parameters
  real lossprob_prior;
  
  //Prior predictive parameters
  int <lower=0, upper= trials> prior_preds_wp1_l0; //win on right hand
  int <lower=0, upper= trials> prior_preds_wp0_l0; // win on left hand
  int <lower=0, upper= trials> prior_preds_w0_lp1; // loss on right hand
  int <lower=0, upper= trials> prior_preds_w0_lp0; // loss on left hand
  
  //Posterior predictive parameters
  int <lower=0, upper= trials> post_preds_wp1_l0; //win on right hand
  int <lower=0, upper= trials> post_preds_wp0_l0; // win on left hand
  int <lower=0, upper= trials> post_preds_w0_lp1; // loss on right hand
  int <lower=0, upper= trials> post_preds_w0_lp0; // loss on left hand
  
  //Setting the priors
  winprob_prior = normal_rng(0,1); // specifying the priors
  lossprob_prior = normal_rng(0,1);
  
  //Using binomial random number generater and inverse logit scale it such that we go to probability scale when choosing
  prior_preds_wp1_l0 = binomial_rng(trials, inv_logit(winprob_prior * 1 + lossprob_prior * 0));
  prior_preds_wp0_l0 = binomial_rng(trials, inv_logit(winprob_prior * -1 + lossprob_prior * 0));
  prior_preds_w0_lp1 = binomial_rng(trials, inv_logit(winprob_prior * 0 + lossprob_prior * 1));
  prior_preds_w0_lp0 = binomial_rng(trials, inv_logit(winprob_prior * 0 + lossprob_prior * -1));
  
  post_preds_wp1_l0 = binomial_rng(trials, inv_logit(winprob * 1 + lossprob * 0));
  post_preds_wp0_l0 = binomial_rng(trials, inv_logit(winprob * -1 + lossprob * 0));
  post_preds_w0_lp1 = binomial_rng(trials, inv_logit(winprob * 0 + lossprob * 1));
  post_preds_w0_lp0 = binomial_rng(trials, inv_logit(winprob * 0 + lossprob * -1));
  
}
