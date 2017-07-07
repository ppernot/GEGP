// Gaussian Process with optional use of gradients 
// (a.k.a. GEK: Gradient-Enganced Kriging)
// This version only calibrates  the parameters of the GP (no prediction)
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
  
  // Gradients info
  int<lower=0,upper=1> use_gradients;
  matrix[N1,M] dy1;
  matrix[N1,M] udy1;
  
}
transformed data {
  vector[N1*(1+use_gradients*M)]   mu;
  matrix[N1,M] x;
  vector[M] scale_x;
  matrix[N1,M] dys;
  matrix[N1,M] udys;
  real avd;
  real cl_min;
  real cl_max;

  // Rescale x1 in unit hypercube
  for (i in 1:M) {
    scale_x[i] = 1. / (max(x1[,i])-min(x1[,i]))  ;
    x[,i] = (x1[,i] -min(x1[,i])) * scale_x[i];
  }
  
  dys  = dy1;
  udys = udy1;
  if(use_gradients == 1)
    for (k in 1:M)
      for (i in 1:N1) {
        dys[i,k]  = dy1[i,k]/scale_x[k];    
        udys[i,k] = udy1[i,k]/scale_x[k];    
      } 
  
  // Range for correlation length (Dalbey2013)
  avd = pow(1./N1,1./M); // Average distance between 2 points in unit cube  
  cl_min = avd/4;
  cl_max = avd*8;

  // Trend
  for (i in 1:N1)               
    mu[i] = mean(y1);
    
  if (use_gradients == 1)  
    for (k in 1:M)
      for (i in 1:N1)   
        mu[k*N1+i] = mean(dys[,k]);

}
parameters {
  real<lower=0> eta_sq;
  vector<lower=cl_min, upper=cl_max>[M] corlen;
}
transformed parameters {
  vector[M] rho2;
  real eta2; 
  rho2 = 0.5 * inv(corlen .* corlen);
  eta2 = eta_sq; 
}
model {
  real rd2;
  row_vector[M] d;
  matrix[N1*(1+use_gradients*M),
         N1*(1+use_gradients*M)] V;
  vector[N1*(1+use_gradients*M)] y;
  // vector[N1*(1+use_gradients*M)] eig;
  
  for (i in 1:N1) 
    y[i] = y1[i];

  if(use_gradients == 1)
    for (k in 1:M)
      for (i in 1:N1) 
         y[k*N1+i] = dys[i,k];

  // Build cov matrix 
  # <y,y>
  for (i in   1:N1) 
    for (j in i:N1) {
      d = x[i,]-x[j,];
      rd2 = sum(rho2 .* (d .* d)'); 
      V[i,j] = eta2 * exp(-rd2); 
      V[j,i] = V[i,j];
    }
  // Augment diagonal with data variances
  for (i in 1:N1)                
    V[i,i] = V[i,i] + uy1[i]^2; // Data var.

  if(use_gradients == 1) {
    real cc;
    real vexp;
    # <y,dy>
    for (i in   1:N1) 
      for (j in 1:N1) {
        d = x[i,]-x[j,];
        rd2 = sum(rho2 .* (d .* d)'); 
        vexp = eta2 * exp(-rd2);
        for (k in 1:M) {
          cc = 2 * rho2[k]*d[k] * vexp ;
          V[i,k*N1+j] = cc; 
          V[k*N1+j,i] = cc; 
        }
      }

    # <dy,dy>
    for (i in   1:N1) 
      for (j in i:N1) {
        d = x[i,]-x[j,];
        rd2 = sum(rho2 .* (d .* d)'); 
        vexp = eta2 * exp(-rd2);
        for (k1 in 1:M) {
          cc = 2 * rho2[k1] * (1-2*rho2[k1]* d[k1]^2) * vexp ;
          V[k1*N1+i,k1*N1+j] = cc; 
          V[k1*N1+j,k1*N1+i] = cc; 
        }
        for (k1 in    1:(M-1) ) {
          for (k2 in  (k1+1):M) {
            cc = -4 * rho2[k1]*d[k1] * rho2[k2]*d[k2] * vexp;
            V[k1*N1+i,k2*N1+j] = cc; 
            V[k2*N1+j,k1*N1+i] = cc; 
            V[k1*N1+j,k2*N1+i] = cc; 
            V[k2*N1+i,k1*N1+j] = cc; 
          }
        }
      }
      
    // Augment diagonal with data variances
    for (k in 1:M)
      for (i in 1:N1) 
        V[k*N1+i,k*N1+i] = V[k*N1+i,k*N1+i] 
                         + udys[i,k]^2; // Grad var.
  }  
  
  // Check condition number of V
  // eig = eigenvalues_sym(V);
  // print( eig[N1*(1+use_gradients*M)]/eig[1] );
  
  eta_sq ~ cauchy(0,5);

  y ~ multi_normal(mu, V);
}
  


