#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector calculate_fourtuple_tetrads(NumericMatrix X) {
  NumericVector cors(6);
  NumericVector cors_prods(3);
  NumericVector tau(3);
  
  cors[0] = X(1, 0);
  cors[1] = X(2, 0);
  cors[2] = X(3, 0);
  cors[3] = X(2, 1);
  cors[4] = X(3, 1);
  cors[5] = X(3, 2);
  
  cors_prods[0] = cors[0] * cors[5]; // rho_12 * rho_34
  cors_prods[1] = cors[1] * cors[4]; // rho_13 * rho_24
  cors_prods[2] = cors[2] * cors[3]; // rho_14 * rho_23
  
  tau[0] = cors_prods[0] - cors_prods[1]; // tau1
  tau[1] = cors_prods[1] - cors_prods[2]; // tau2
  tau[2] = cors_prods[2] - cors_prods[0]; // tau3
  
  return tau;
}
