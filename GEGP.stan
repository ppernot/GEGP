// Trial version of GEGP0 with rescaling of data
data {
  int<lower=1> N1;
  vector[N1] x1;
  vector[N1] y1;
  vector[N1] uy1;
  vector[N1] dy1;
  vector[N1] udy1;
  int<lower=1> N2;
  vector[N2] x2;
}
transformed data {
  real x_scale;
  real y_scale;
  real uy_m2;
  int<lower=2> N;
  vector[N1+N2] x;
  vector[N1+N2] xs;
  vector[N1] ys;
  vector[N1] uys;
  vector[N1] dys;
  vector[N1] udys;
  vector[2*N1+N2] mu;
  
  N = N1 + N2;

  for (i in 1:N1)      x[i] = x1[i];
  for (i in (N1+1):N)  x[i] = x2[i-N1];
  
  
  // Rescale x and GP variance for easier prior definition
  x_scale = 1/(max(x)-min(x));
  y_scale = 1/(max(append_row(y1,dy1))-min(append_row(y1,dy1)));
  xs      = x    * x_scale;
  ys      = y1   * y_scale;
  uys     = uy1  * y_scale;
  dys     = dy1  / x_scale * y_scale;
  udys    = udy1 / x_scale * y_scale;

  // Mean variance for prediction
  uy_m2 = mean(uys .* uys);

  // Center y & dy
  for (i in 1:N)           mu[i] = mean(ys);
  for (i in (N+1):(N+N1))  mu[i] = mean(dys);

}
parameters {
  real<lower=0> eta_sq;
  real<lower=0> inv_rho_sq;
  vector[N2] y2s;
}
transformed parameters {
  real rho2;
  real eta2;
  real d;
  real rd2;
  cov_matrix[N+N1] V;
 
  rho2 = inv(inv_rho_sq);
  eta2 = eta_sq;

  // Build cov matrix 
  // _Cov(y,y)
  for (i in   1:N) 
    for (j in i:N) {
      V[i,j] = eta2 * exp(-rho2 * pow(xs[i] - xs[j],2)); 
      V[j,i] = V[i,j]; 
    }

  // _Cov(y,dy)
  for (i in 1:N) 
    for (j in (N+1):(N+N1)) {
      d = xs[i] - xs[j-N];
      V[i,j] = eta2 * 2*rho2*d * exp(-rho2*d^2); 
      V[j,i] = V[i,j]; 
    } 

  // _Cov(dy,dy)
  for (i in (N+1):(N+N1)) 
    for (j in   i:(N+N1)) {
      rd2 = rho2 * (xs[i-N]-xs[j-N])^2;
      V[i,j] = eta2 * 2*rho2*(1-2*rd2) * exp(-rd2); 
      V[j,i] = V[i,j]; 
    } 

  // Augment diagonal with uncertainties
  for (i in 1:N1)         V[i,i] = V[i,i] + uys[i]^2;    // Data var.
  for (i in (N1+1):N)     V[i,i] = V[i,i] + uy_m2;       // Pred. mean var.
  for (i in (N+1):(N+N1)) V[i,i] = V[i,i] + udys[i-N]^2; // Grad var.
  
}
model {
  vector[N+N1] y;

  for (i in 1:N1)         y[i] = ys[i];
  for (i in (N1+1):N)     y[i] = y2s[i-N1];
  for (i in (N+1):(N+N1)) y[i] = dys[i-N]; 

  eta_sq     ~ normal(0,1);
  inv_rho_sq ~ normal(0,1);

  y ~ multi_normal(mu, V);
}
generated quantities{
  vector[N2] y2;
  y2 = y2s / y_scale;
}
 
