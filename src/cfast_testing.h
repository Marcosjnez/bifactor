#include <RcppArmadillo.h>
#include <Rcpp/Benchmark/Timer.h>
#include "structures.h"
#include "cfa_criteria.h"

Rcpp::List cfa_testing(arma::mat R, int q, arma::mat W, arma::vec lambda_parameters,
                       arma::vec phi_parameters, arma::vec psi_parameters,
                       arma::uvec lambda_indexes, arma::uvec phi_indexes, arma::uvec psi_indexes) {

  arguments_cfa xcfa;
  xcfa.estimator = "uls";
  xcfa.projection = "raw";
  xcfa.R = R;
  xcfa.p = xcfa.R.n_cols;
  xcfa.q = q;
  xcfa.Ip.set_size(xcfa.p, xcfa.p); xcfa.Ip.eye();
  xcfa.Iq.set_size(xcfa.q, xcfa.q); xcfa.Iq.eye();
  xcfa.W = W;
  xcfa.lambda_indexes = lambda_indexes-1;
  xcfa.phi_indexes = phi_indexes-1;
  xcfa.psi_indexes = psi_indexes-1;

  int n_lambda = lambda_indexes.size();
  int n_phi = phi_indexes.size();
  int n_psi = psi_indexes.size();
  xcfa.lambda.set_size(xcfa.p, xcfa.q); xcfa.lambda.zeros();
  xcfa.phi.set_size(xcfa.q, xcfa.q); xcfa.phi.zeros();
  xcfa.psi.set_size(xcfa.p, xcfa.p); xcfa.psi.zeros();
  xcfa.lambda.elem(xcfa.lambda_indexes) = lambda_parameters;
  xcfa.phi.elem(xcfa.phi_indexes) = phi_parameters; xcfa.phi = arma::symmatl(xcfa.phi);
  xcfa.psi.elem(xcfa.psi_indexes) = psi_parameters; xcfa.psi = arma::symmatl(xcfa.psi);

  cfa_criterion* cfa_criterion = choose_cfa_criterion(xcfa.estimator);
  // cfa_manifold* cfa_manifold = choose_cfa_manifold(xcfa.projection);
  cfa_criterion->F(xcfa);

  Rcpp::List result;
  result["lambda"] = xcfa.lambda;
  result["phi"] = xcfa.phi;
  result["psi"] = xcfa.psi;
  result["Rhat"] = xcfa.Rhat;
  result["residuals"] = xcfa.residuals;
  result["f"] = xcfa.f;

  return result;

};
