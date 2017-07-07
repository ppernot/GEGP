// Gaussian Process with optional use of gradients 
// (a.k.a. GEK: Gradient-Enganced Kriging)
// This version is only for sampling by the GP (no calibration)
//
// Author: P. Pernot
// Affiliation: CNRS
//
// Notes:
// * The main code is adapted from examples in the Stan manual (2.14)
// * The gradients covariances are adapted from 
//   Ulagnathan et al. (2016) doi:10.1007/s00366-015-0397-y
// * 2017/04/10: The code is designed for legibility, not efficiency...

data {
  
  // "Standard" GE info
  int<lower=1> M;
  int<lower=1> N1;
  matrix[N1,M] x1;
  vector[N1]   y1;
  vector[N1]   uy1;
  int<lower=1> N2;
  matrix[N2,M] x2;
  
  // Gradients info
  int<lower=0,upper=1> use_gradients;
  matrix[N1,M] dy1;
  matrix[N1,M] udy1;

  // GP parameters
  real<lower=0> eta_sq;
  vector<lower=0>[M] corlen;
}
transformed data {
  vector[M] rho2;
  real eta2; 
  real uy_m2;
  matrix[N1+N2,M] x0;
  matrix[N1+N2,M] x;
  vector[M] scale_x;
  matrix[N1,M] dys;
  matrix[N1,M] udys;
  vector[N1*(1+use_gradients*M)+N2]   mu;
  real rd2;
  row_vector[M] d;
  cov_matrix[N1*(1+use_gradients*M)+N2] V;
  matrix[N1*(1+use_gradients*M)+N2,
         N1*(1+use_gradients*M)+N2] L;
  
  for (i in 1:N1) x0[i   ,] = x1[i,];
  for (i in 1:N2) x0[N1+i,] = x2[i,];

  // Rescale x0 in x1-based unit hypercube
  for (k in 1:M) {
    scale_x[k] = 1. / (max(x1[,k])-min(x1[,k]))  ;
    x[,k] = (x0[,k] - min(x1[,k])) * scale_x[k];
  }

  dys  = dy1;
  udys = udy1;
  if(use_gradients == 1)
    for (k in 1:M)
      for (i in 1:N1) {
        dys[i,k]  = dy1[i,k]/scale_x[k];    
        udys[i,k] = udy1[i,k]/scale_x[k];    
      } 
      
  // Nugget or Mean variance for prediction
  // uy_m2 = mean(uy1 .* uy1);
  uy_m2 = (0.1*min(uy1))^2; // Nugget
  
  // Trend
  for (i in 1:(N1+N2)) mu[i] = mean(y1);
    
  if (use_gradients == 1)  
    for (k in 1:M)
      for (i in 1:N1)   
        mu[k*N1+N2+i] = mean(dys[,k]);

  rho2 = 0.5 * inv(corlen .* corlen);
  eta2 = eta_sq; 

  // Build cov matrix 
  # <y,y>
  for (i in   1:(N1+N2)) 
    for (j in i:(N1+N2)) {
      d = x[i,]-x[j,];
      rd2 = sum(rho2 .* (d .* d)'); 
      V[i,j] = eta2 * exp(-rd2); 
      V[j,i] = V[i,j];
    }
    
  // Augment diagonal with data variances
  for (i in 1:N1)                
    V[i,i] = V[i,i] + uy1[i]^2;          // Data var.
  for (i in 1:N2)
    V[N1+i,N1+i] = V[N1+i,N1+i] + uy_m2; // Pred. mean var.

  if(use_gradients == 1) {
    real cc;
    real vexp;
    # <y,dy>
    for (i in   1:(N1+N2)) 
      for (j in 1:N1) {
        d = x[i,]-x[j,];
        rd2 = sum(rho2 .* (d .* d)'); 
        vexp = eta2 * exp(-rd2);
        for (k in 1:M) {
          cc = 2 * rho2[k]*d[k] * vexp;
          V[i,k*N1+N2+j] = cc; 
          V[k*N1+N2+j,i] = cc; 
        }
      }

    # <dy,dy>
    for (i in   1:N1) 
      for (j in i:N1) {
        d = x[i,]-x[j,];
        rd2 = sum(rho2 .* (d .* d)'); 
        vexp = eta2 * exp(-rd2);
        for (k1 in 1:M) {
          cc = 2 * rho2[k1] * (1-2*rho2[k1]* d[k1]^2) * vexp;
          V[k1*N1+N2+i,k1*N1+N2+j] = cc; 
          V[k1*N1+N2+j,k1*N1+N2+i] = cc; 
        }
        for (k1 in    1:(M-1) ) {
          for (k2 in  (k1+1):M) {
            cc = -4 * rho2[k1]*d[k1] * rho2[k2]*d[k2] * vexp;
            V[k1*N1+N2+i,k2*N1+N2+j] = cc; 
            V[k2*N1+N2+j,k1*N1+N2+i] = cc; 
            V[k1*N1+N2+j,k2*N1+N2+i] = cc; 
            V[k2*N1+N2+i,k1*N1+N2+j] = cc; 
          }
        }
      }
   
    // Augment diagonal with data variances
    for (k in 1:M)
      for (i in 1:N1) 
        V[k*N1+N2+i,k*N1+N2+i] = V[k*N1+N2+i,k*N1+N2+i] 
                               + udys[i,k]^2; // Grad var.
  }  

  L = cholesky_decompose(V);
}
parameters {
  vector[N2] y2;
}
model {
  vector[N1*(1+use_gradients*M)+N2] y;

  for (i in 1:N1) y[i   ] = y1[i];
  for (i in 1:N2) y[N1+i] = y2[i];
    
  if(use_gradients == 1)
    for (k in 1:M)
      for (i in 1:N1) 
         y[k*N1+N2+i] = dys[i,k]; 

  y ~ multi_normal_cholesky(mu, L);
}
  

