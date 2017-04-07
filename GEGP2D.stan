data {
  int<lower=1> M;
  int<lower=1> N1;
  matrix[N1,M] x1;
  vector[N1]   y1;
  vector[N1]   uy1;
  matrix[N1,M] dy1;
  matrix[N1,M] udy1;
  int<lower=1> N2;
  matrix[N2,M] x2;
}
transformed data {
  real uy_m2;
  matrix[N1+N2,M] x;
  vector[N1*(1+M)+N2]   mu;
  
  for (i in 1:N1)          
    x[i,] = x1[i,];
  for (i in 1:N2) 
    x[N1+i,] = x2[i,];
  
  // Trend
  for (i in 1:(N1+N2))                
    mu[i] = mean(y1);
  for (k in 1:M)
    for (i in 1:N1)   
      mu[k*N1+N2+i] = mean(dy1[,k]);

  // Mean variance for prediction
  uy_m2 = mean(uy1 .* uy1);

}
parameters {
  real<lower=0> eta_sq;
  real<lower=0> inv_rho_sq[M];
  vector[N2] y2;
}
transformed parameters {
  real cc;
  real rho2[M];
  real eta2; 
  real vexp;
  real rd2;
  row_vector[M] d;
  cov_matrix[N1*(1+M)+N2] V;
 
  rho2 = inv(inv_rho_sq);
  eta2 = eta_sq; 

  // Build cov matrix 
  # <y,y>
  for (i in   1:(N1+N2)) 
    for (j in i:(N1+N2)) {
      d = x[i,]-x[j,];
      vexp = exp(-rho2[1]* d[1]^2 
                 -rho2[2]* d[2]^2);
      V[i,j] = eta2 * vexp; 
      V[j,i] = V[i,j];
    }
  
  # <y,dy>
  for (i in   1:(N1+N2)) 
    for (j in 1:N1) {
      d = x[i,]-x[j,];
      vexp = exp(-rho2[1]* d[1]^2 
                 -rho2[2]* d[2]^2);
      for (k in 1:M) {
        V[i,k*N1+N2+j] = eta2 * 2*rho2[k]*d[k] * vexp; 
        V[k*N1+N2+j,i] = V[i,k*N1+N2+j]; 
      }
    }

  # <dy,dy>
  for (i in   1:N1) 
    for (j in 1:N1) {
      d = x[i,]-x[j,];
      rd2 = rho2[1]* d[1]^2 
          + rho2[2]* d[2]^2;
      vexp = exp(-rd2);
      for (k1 in    1:M) {
        for (k2 in k1:M) {
          cc = k1==k2 ? 2*rho2[k1]*(1-2*rho2[k1]* d[k1]^2) 
                      : -4*rho2[k1]*d[k1]*rho2[k2]*d[k2] ;
          V[k1*N1+N2+i,k2*N1+N2+j] = eta2 * cc * vexp; 
          V[k2*N1+N2+j,k1*N1+N2+i] = V[k1*N1+N2+i,k2*N1+N2+j]; 
        }
      }
    }

  // Augment diagonal with uncertainties
  for (i in 1:N1)                
    V[i,i] = V[i,i] + uy1[i]^2; // Data var.
  for (i in 1:N2)      
    V[N1+i,N1+i] = V[N1+i,N1+i] + uy_m2; // Pred. mean var.
  for (k in 1:M)
    for (i in 1:N1) 
      V[k*N1+N2+i,k*N1+N2+i] = V[k*N1+N2+i,k*N1+N2+i] + udy1[i,k]^2; // Grad var.
  
  
}
model {
  vector[N1*(1+M)+N2] y;

  for (i in 1:N1) 
    y[i] = y1[i];
  for (i in 1:N2) 
    y[N1+i] = y2[i];
  for (k in 1:M)
    for (i in 1:N1) 
       y[k*N1+N2+i] = dy1[i,k]; 
 
  eta_sq     ~ cauchy(0,1);
  inv_rho_sq ~ cauchy(0,1);

  y ~ multi_normal(mu, V);
}
 
