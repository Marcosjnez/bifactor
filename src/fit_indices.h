#include "bifactor.h"

double log_lik_ratio(arma::mat R, arma::mat R_hat, int n) {
  // for maximum_likelihood (pag.58 Bartholomew 2011)

  int p = R.n_rows;
  arma::mat R_hat_inv_R = arma::inv_sympd(R_hat)*R;
  double log_lik_ratio = n * (arma::trace(R_hat_inv_R) - log(arma::det(R_hat_inv_R)) - p);

  return log_lik_ratio;

}

double log_lik_ratio_improved(arma::mat R, arma::mat R_hat, int n, int q) {

  // for maximum_likelihood (pag.58 Bartholomew 2011)

  int p = R.n_rows;
  double new_n = n - 1 - 1/6*(2*p + 5) - 2/3*q;
  arma::mat R_hat_inv_R = arma::inv_sympd(R_hat)*R;
  double log_lik_ratio = new_n * (arma::trace(R_hat_inv_R) - log(arma::det(R_hat_inv_R)) - p);

  return log_lik_ratio;

}

int log_lik_ratio_df(int p, int q) {

  // for maximum_likelihood (pag.58 Bartholomew 2011)

  return 0.5*((p-q)*(p-q) - p - q);

}

double AIC(double log_lik, int df) {

  // for maximum_likelihood (pag.58 Bartholomew 2011)

  return 2*(df - log_lik);

}

double rmsr(Rcpp::List fit) {

  Rcpp::List efa = fit["efa"];
  arma::mat loadings = efa["loadings"];
  double delta = efa["f"];
  int p = loadings.n_rows;
  double rmsr = sqrt(delta / (0.5*p*(p-1)));

  return rmsr;

}

double rmsea(Rcpp::List fit, int n) {

  Rcpp::List efa = fit["efa"];
  arma::mat loadings = efa["loadings"];
  double delta = efa["f"];
  int p = loadings.n_rows;
  int q = loadings.n_cols;
  double df = p*(p-1)/2 - p*q + (q-1)*q/2;

  double d = delta*(n-1) - df;
  double DIS1 = 0;
  if(d > 0) DIS1 = d;
  double rmsea = sqrt(DIS1 / (df*(n-1)));

  return rmsea;

}
