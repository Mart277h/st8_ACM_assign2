//
// This Stan program defines a model for playing the Matching Pennies game with
// an asymmetrical Win-Stay-Lose-Shift strategy.

data {
  int<lower=1> trials; //trials - defines length of following data
  array[trials] int choices; //array of choices made by the agent
  vector[trials] feedback; //feedback from previous choices
}

transformed data{
  vector[trials] winhand; //this vector codes the hand you chose if you won
  vector[trials] losshand; //this vector codes the hand you chose if you lost
  
  winhand[1] = 0; //0 gives us no probability 
  losshand[1] = 0; 
  
  for(t in 2:trials)
    if (feedback[t-1] == 1 && choices[t-1] == 0){
      winhand[t]= -1;
      losshand[t]= 0;
    }
    else if (feedback[t-1] == 1 && choices[t-1] == 1) {
      winhand[t]= 1;
      losshand[t]= 0;
    }
    else if (feedback[t-1] == 0 && choices[t-1] == 1){
      winhand[t]= 0;
      losshand[t]= -1;
    }
    else if (feedback[t-1] == 0 && choices[t-1] == 0){
      winhand[t]= 0;
      losshand[t]= 1;
    }
}

parameters {
  real winprob; // this is our beta for our linear model that estimated our win probability parameter
  real lossprob; // inverse of ^^
}

model {
  target += normal_lpdf(winprob| 0,1); //prior
  target += normal_lpdf(lossprob| 0,1); //prior
  
  // model
  target += bernoulli_logit_lpmf(choices | 1 + winprob * winhand + lossprob * losshand);
}

generated quantities{
  real winprob_prior;
  real lossprob_prior;
  int<lower=0, upper=trials> prior_preds;
  int<lower=0, upper=trials> post_preds;
  
  winprob_prior = inv_logit(normal_rng(0,1));
  lossprob_prior = inv_logit(normal_rng(0,1));
  prior_preds = binomial_rng(trials, inv_logit(1 + winprob_prior * winhand + lossprob_prior * losshand));
  post_preds = binomial_rng(trials, inv_logit(1 + winprob * winhand + lossprob * losshand));
}