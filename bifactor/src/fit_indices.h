#include "twoTier.h"

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
  double log_lik_ratio = n * (arma::trace(R_hat_inv_R) - log(arma::det(R_hat_inv_R)) - p);

  return log_lik_ratio;

}

int log_lik_ratio_df(int p, int q) {

  // for maximum_likelihood (pag.58 Bartholomew 2011)

  return 0.5*(pow(p-q, 2) - p - q);

}

double AIC(double log_lik, int df) {

  // for maximum_likelihood (pag.58 Bartholomew 2011)

  return 2*(df - log_lik);

}

double RMSR(Rcpp::List fit) {

  arma::mat residuals = fit["residuals"];
  double rmsr = sqrt(arma::mean(arma::mean(residuals % residuals)));

  return rmsr;

  }
