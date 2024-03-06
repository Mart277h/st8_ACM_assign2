//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> trials; //trials
  array[trials] int choices; //this is the array of choices 
  vector[trials] feedback; //feedback from previous choices
}

transformed data{
  vector[trials] winhand; //this vector will give us the hand you chose if you won
  vector[trials] losshand; //this vector will give us the hand you chose if you los
  
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

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  real winprob; // this is our beta for our linear model that estimated our win probability parameter
  real lossprob; // inverse of ^^
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  target += normal_lpdf(winprob| 0,1); //prior
  target += normal_lpdf(lossprob| 0,1); //prior
  
  // model
  target += bernoulli_logit_lpmf(choices | 1 + winprob * winhand + lossprob * losshand);
}

