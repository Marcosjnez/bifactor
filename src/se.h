/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

// #include "structures.h"
// #include "checks.h"
// #include "manifolds.h"
// #include "criteria.h"
// #include "multiple_rotations.h"
// #include "method_derivatives.h"

Rcpp::List se(Rcpp::List fit,
              Rcpp::Nullable<int> nullable_nobs,
              Rcpp::Nullable<arma::mat> nullable_X,
              std::string type, double eta) {

  Rcpp::List modelInfo = fit["modelInfo"];
  Rcpp::Nullable<int> modelInfo_nobs = modelInfo["nobs"];
  int nobs;

  if(modelInfo_nobs.isNotNull()) {
    nobs = Rcpp::as<int>(modelInfo_nobs);
  } else if(nullable_nobs.isNotNull()) {
    nobs = Rcpp::as<int>(nullable_nobs);
  } else {
    Rcpp::stop("Please, specify the sample size");
  }

  arguments_rotate x;
  int random_starts = 1L, cores = 1L;

  // Overwrite x according to modelInfo:

  Rcpp::List rot = fit["rotation"];
  arma::mat L_ = rot["loadings"]; x.L = L_;
  Rcpp::List efa = fit["efa"];
  std::string projection_ = modelInfo["projection"]; x.projection = projection_;
  std::vector<std::string> rotation_ = modelInfo["rotation"]; x.rotations = rotation_;

  arma::mat lambda_ = efa["loadings"]; x.lambda = lambda_;
  x.p = x.lambda.n_rows;
  x.q = x.lambda.n_cols;
  x.lambda.set_size(x.p, x.q);
  x.Phi.set_size(x.q, x.q); x.Phi.eye();
  arma::mat T_ = rot["T"]; x.T = T_;
  arma::mat Phi_ = rot["Phi"]; x.Phi = Phi_;
  arma::vec gamma_ = modelInfo["gamma"]; x.gamma = gamma_;
  arma::vec k_ = modelInfo["k"]; x.k = k_;
  arma::vec epsilon_ = modelInfo["epsilon"]; x.epsilon = epsilon_;
  double w_ = modelInfo["w"]; x.w = w_;
  double alpha_ = modelInfo["alpha"]; x.alpha = alpha_;
  double a_ = modelInfo["a"]; x.a = a_;
  double b_ = modelInfo["b"]; x.b = b_;
  std::string between_blocks_ = modelInfo["between_blocks"]; x.between_blocks = between_blocks_;

  arma::mat S_ = modelInfo["R"]; x.S = S_;
  std::string method = modelInfo["method"];
  Rcpp::Nullable<arma::mat> Target_ = modelInfo["Target"];
  x.nullable_Target = Target_;
  Rcpp::Nullable<arma::mat> Weight_ = modelInfo["Weight"];
  x.nullable_Weight = Weight_;
  Rcpp::Nullable<arma::mat> PhiTarget_ = modelInfo["PhiTarget"];
  x.nullable_PhiTarget = PhiTarget_;
  Rcpp::Nullable<arma::mat> PhiWeight_ = modelInfo["PhiWeight"];
  x.nullable_PhiWeight = PhiWeight_;
  Rcpp::Nullable<arma::uvec> blocks_ = modelInfo["blocks"];
  x.nullable_blocks = blocks_;
  Rcpp::Nullable<std::vector<arma::uvec>> blocks_list_ = modelInfo["blocks_list"];
  x.nullable_blocks_list = blocks_list_;
  Rcpp::Nullable<arma::vec> block_weights_ = modelInfo["block_weights"];
  x.nullable_block_weights = block_weights_;
  Rcpp::Nullable<arma::uvec> oblq_blocks_ = modelInfo["oblq_blocks"];
  x.nullable_oblq_blocks = oblq_blocks_;
  Rcpp::Nullable<arma::uvec> rot_control_ = modelInfo["oblq_blocks"];
  x.nullable_rot_control = R_NilValue;

  check_rotate(x, random_starts, cores);

  rotation_manifold* manifold = choose_manifold(x.projection);
  rotation_criterion* criterion = choose_criterion(x.rotations, x.projection, x.blocks_list);

  // Compute the rotation constraints:

  manifold->param(x);
  criterion->F(x);
  criterion->gLP(x);
  criterion->hLP(x);
  manifold->g_constraints(x); // update x.d_constr

  // Rcpp::List xx;
  // xx["constr"] = x.d_constr_temp;
  // return xx;

  // Compute the Hessian:

  arma::mat H;
  if(method == "minres") {
    H = hessian_minres(x.S, x.L, x.Phi, x.projection);
  } else if(method == "ml") {
    H = hessian_ml(x.S, x.L, x.Phi, x.projection);
  } else {
    Rcpp::stop("Standard errors are not implemented yet for this extraction method");
  }

  // Add the constraints to the hessian matrix:
  int kk = x.d_constr.n_rows;
  H = arma::join_rows(H, x.d_constr.t());
  // Add zeros to make H a square matrix:
  arma::mat zeros(kk, kk);
  arma::mat C = arma::join_rows(x.d_constr, zeros);
  H = arma::join_cols(H, C);
  arma::mat H_inv = arma::inv(H);

  int pq = x.p*x.q;
  int q_cor = x.q*(x.q-1)/2;
  int m;
  arma::uvec indexes;

  if(x.projection == "orth") {
    m = x.p*x.q + x.p;
  } else {
    m = x.p*x.q + q_cor + x.p;
    indexes = arma::trimatl_ind(arma::size(x.Phi), -1);
  }

  // Find A^{-1}BA^{-1}:
  arma::mat A_inv = H_inv(arma::span(0, m-1), arma::span(0, m-1));
  arma::mat BB = B(x.S, x.L, x.Phi, nullable_X, method, x.projection,
                   type, eta); // Variance of the correlation matrix
  arma::mat VAR = A_inv * BB * A_inv;
  arma::vec se = sqrt(arma::diagvec(VAR)/(nobs-1));

  arma::mat Lambda_se(x.p, x.q);
  arma::mat Phi_se(x.q, x.q);
  Phi_se.diag().zeros();
  arma::vec psi_se(x.p);

  for(int i=0; i < pq; ++i) Lambda_se[i] = se[i];
  if(x.projection == "orth") {
    int ij = 0;
    for(int i=pq; i < m; ++i) {
      psi_se[ij] = se[i];
      ++ij;
    }
  } else {
    int ij = 0;
    for(int i=pq; i < (pq+q_cor); ++i) {
      Phi_se[indexes(ij)] = se[i];
      ++ij;
    }
    Phi_se = arma::symmatl(Phi_se);
    ij = 0;
    for(int i=pq+q_cor; i < m; ++i) {
      psi_se[ij] = se[i];
      ++ij;
    }
  }

  Rcpp::List result;
  result["Lambda_se"] = Lambda_se;
  result["Phi_se"] = Phi_se;
  result["psi_se"] = psi_se;

  return result;

}
