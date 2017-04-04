data {
  int<lower=1> N1;
  vector[N1] x1;
  vector[N1] y1;
  vector[N1] uy1;
  int<lower=1> N2;
  vector[N2] x2;
}
transformed data {
  real x_scale;
  real v_scale;
  real uy_m2;
  int<lower=2> N;
  vector[N1+N2] x;
  vector[N1+N2] mu;
  
  N = N1 + N2;

  for (i in 1:N1)     x[i] = x1[i];
  for (i in (N1+1):N) x[i] = x2[i-N1];
  
  // Center y
  for (i in 1:N)  mu[i] = mean(y1);

  // Scale x and variance
  x_scale = 1/(max(x)-min(x));
  v_scale = (max(y1)-min(y1))^2;
  
  // Mean variance for prediction
  uy_m2 = mean(uy1 .* uy1);

}
parameters {
  real<lower=0> eta_sq;
  real<lower=0> inv_rho_sq;
  vector[N2] y2;
}
transformed parameters {
  real rho2;
  real eta2;  
  cov_matrix[N] V;
 
  rho2 = inv(inv_rho_sq) * x_scale^2 ;
  eta2 = eta_sq * v_scale; 

  for (i in 1:N) 
    for (j in i:N) {
      V[i,j] = eta2 * exp(-rho2* (x[i]-x[j])^2 ); 
      V[j,i] = V[i,j];
    } 
    
  for (i in 1:N1)           V[i,i] = V[i,i] + uy1[i]^2;
  for (i in (N1+1):(N1+N2)) V[i,i] = V[i,i] + uy_m2;
  
}
model {
  vector[N] y;

  for (i in 1:N1)     y[i] = y1[i];
  for (i in (N1+1):N) y[i] = y2[i-N1];

  eta_sq     ~ normal(0,0.5);
  inv_rho_sq ~ normal(0,0.1);

  y ~ multi_normal(mu, V);
}
generated quantities{
}
 
 
