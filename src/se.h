#include "method_derivatives.h"

Rcpp::List se(int n, Rcpp::Nullable<Rcpp::List> nullable_fit,
              Rcpp::Nullable<arma::mat> nullable_S,
              Rcpp::Nullable<arma::mat> nullable_Lambda,
              Rcpp::Nullable<arma::mat> nullable_Phi,
              Rcpp::Nullable<arma::mat> nullable_X,
              std::string method,
              std::string projection,
              Rcpp::CharacterVector char_rotation,
              Rcpp::Nullable<arma::mat> nullable_Target,
              Rcpp::Nullable<arma::mat> nullable_Weight,
              Rcpp::Nullable<arma::mat> nullable_PhiTarget,
              Rcpp::Nullable<arma::mat> nullable_PhiWeight,
              Rcpp::Nullable<arma::uvec> nullable_blocks,
              Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list,
              Rcpp::Nullable<arma::vec> nullable_block_weights,
              Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
              double gamma, double k, double epsilon, double w, double alpha,
              bool normalize, std::string penalization,
              std::string type, double eta) {

  std::vector<std::string> rotation = Rcpp::as<std::vector<std::string>>(char_rotation);

  arguments x;
  x.rotations = rotation;
  x.projection = projection;
  int rot_maxit = 1L, random_starts = 1L, cores = 1L;
  double rot_eps = 0.05;

  if(nullable_fit.isNotNull()) {

    Rcpp::List fit = nullable_fit.get();
    Rcpp::List modelInfo = fit["modelInfo"];
    Rcpp::List rot = fit["rotation"];
    Rcpp::List efa = fit["efa"];

    std::string projection_ = modelInfo["projection"]; x.projection = projection_;
    std::vector<std::string> rotation_ = modelInfo["rotation"]; x.rotations = rotation_;

    // Overwrite x according to modelInfo

    arma::mat lambda_ = efa["loadings"]; x.lambda = lambda_;
    arma::mat T_ = rot["T"]; x.T = T_;
    arma::mat Phi_ = rot["Phi"]; x.Phi = Phi_;
    double gamma_ = modelInfo["gamma"]; x.gamma = gamma_;
    double k_ = modelInfo["k"]; x.k = k_;
    double epsilon_ = modelInfo["epsilon"]; x.epsilon = epsilon_;
    double w_ = modelInfo["w"]; x.w = w_;
    double alpha_ = modelInfo["alpha"]; x.a = alpha_;
    std::string penalization_ = modelInfo["penalization"];
    x.penalization = penalization_;

    x.p = x.lambda.n_rows;
    x.q = x.lambda.n_cols;

    arma::mat S_ = modelInfo["R"]; x.S = S_;
    std::string method_ = modelInfo["method"]; method = method_;
    Rcpp::Nullable<arma::mat> Target_ = modelInfo["Target"];
    Rcpp::Nullable<arma::mat> Weight_ = modelInfo["Weight"];
    Rcpp::Nullable<arma::mat> PhiTarget_ = modelInfo["PhiTarget"];
    Rcpp::Nullable<arma::mat> PhiWeight_ = modelInfo["PhiWeight"];
    Rcpp::Nullable<arma::uvec> blocks_ = modelInfo["blocks"];
    Rcpp::Nullable<std::vector<arma::uvec>> blocks_list_ = modelInfo["blocks_list"];
    Rcpp::Nullable<arma::vec> block_weights_ = modelInfo["block_weights"];
    Rcpp::Nullable<arma::uvec> oblq_blocks_ = modelInfo["oblq_blocks"];

    check_rotate(x,
                 Target_, Weight_, PhiTarget_, PhiWeight_,
                 blocks_, blocks_list_, block_weights_, oblq_blocks_,
                 R_NilValue, rot_maxit, rot_eps,
                 random_starts, cores);

  } else {

    // Overwrite x according to inputs

    x.gamma = gamma, x.k = k, x.epsilon = epsilon, x.w = w, x.a = alpha;
    x.penalization = penalization;

    if(nullable_S.isNotNull()) {
      x.S = Rcpp::as<arma::mat>(nullable_S);
    } else {
      Rcpp::stop("Either provide the sample correlation R or a model fit via the fit argument");
    }
    if(nullable_Lambda.isNotNull()) {
      x.L = Rcpp::as<arma::mat>(nullable_Lambda);
      x.lambda = x.L; // FIX
    } else {
      Rcpp::stop("Either provide loadings or a model fit via the fit argument");
    }
    if(nullable_Phi.isNotNull()) {
      x.Phi = Rcpp::as<arma::mat>(nullable_Phi);
    } else {
      if(x.projection == "oblq") {
        Rcpp::stop("Either provide Phi or a model fit via the fit argument");
      }
    }

    x.p = x.L.n_rows;
    x.q = x.L.n_cols;

    check_rotate(x,
                 nullable_Target, nullable_Weight,
                 nullable_PhiTarget, nullable_PhiWeight,
                 nullable_blocks, nullable_blocks_list, nullable_block_weights,
                 nullable_oblq_blocks,
                 R_NilValue, rot_maxit, rot_eps, random_starts, cores);

  }

  base_manifold* manifold = choose_manifold(x.projection);
  base_criterion* criterion = choose_criterion(x.rotations, x.projection, x.blocks_list);

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
    Rcpp::stop("Standard errors are not implemented yet for this method");
  }

  // Add constraints to the hessian matrix:
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
                   type, eta);
  arma::mat VAR = A_inv * BB * A_inv;
  arma::vec se = sqrt(arma::diagvec(VAR)/(n-1));

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
