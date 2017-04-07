data {
  int<lower=1> N1;
  matrix[N1,2] x1;
  vector[N1]   y1;
  vector[N1]   uy1;
  int<lower=1> N2;
  matrix[N2,2] x2;
}
transformed data {
  real uy_m2;
  matrix[N1+N2,2] x;
  vector[N1+N2]   mu;
  
  for (i in 1:N1)     
    x[i,] = x1[i,];
  for (i in 1:N2) 
    x[N1+i,] = x2[i,];
  
  // Constant Trend
  for (i in 1:(N1+N2))  mu[i] = mean(y1);

  // Mean variance for prediction
  uy_m2 = mean(uy1 .* uy1);

}
parameters {
  real<lower=0> eta_sq;
  real<lower=0> inv_rho_sq[2];
  vector[N2] y2;
}
transformed parameters {
  real rho2[2];
  real eta2;  
  cov_matrix[N1+N2] V;
 
  rho2 = inv(inv_rho_sq);
  eta2 = eta_sq; 

  for (i in 1:(N1+N2)) 
    for (j in i:(N1+N2)) {
      V[i,j] = eta2 * exp(-rho2[1] * (x[i,1]-x[j,1])^2 
                          -rho2[2] * (x[i,2]-x[j,2])^2); 
      V[j,i] = V[i,j];
    } 
    
  for (i in 1:N1)           
    V[i,i] = V[i,i] + uy1[i]^2;
  for (i in 1:N2) 
    V[N1+i,N1+i] = V[N1+i,N1+i] + uy_m2;
  
}
model {
  vector[N1+N2] y;

  for (i in 1:N1)     
    y[i] = y1[i];
  for (i in 1:N2) 
    y[N1+i] = y2[i];

  eta_sq     ~ cauchy(0,1);
  inv_rho_sq ~ cauchy(0,1);

  y ~ multi_normal(mu, V);
}

