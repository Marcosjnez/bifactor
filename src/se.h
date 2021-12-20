#include "method_derivatives.h"

Rcpp::List se(int n, Rcpp::Nullable<Rcpp::List> nullable_fit,
              Rcpp::Nullable<arma::mat> nullable_S,
              Rcpp::Nullable<arma::mat> nullable_Lambda,
              Rcpp::Nullable<arma::mat> nullable_Phi,
              Rcpp::Nullable<arma::mat> nullable_X,
              std::string method,
              std::string projection,
              std::string rotation,
              Rcpp::Nullable<arma::mat> nullable_Target,
              Rcpp::Nullable<arma::mat> nullable_Weight,
              Rcpp::Nullable<arma::mat> nullable_PhiTarget,
              Rcpp::Nullable<arma::mat> nullable_PhiWeight,
              double gamma, double k, double epsilon, double w,
              std::string type, double eta) {

  arma::mat S, Lambda, Phi, X, Target, Weight, PhiTarget, PhiWeight;

  if(nullable_fit.isNotNull()) {

    Rcpp::List fit = nullable_fit.get();
    Rcpp::List modelInfo = fit["modelInfo"];
    Rcpp::List rot = fit["rotation"];

    arma::mat S_ = modelInfo["R"]; S = S_;
    arma::mat Lambda_ = rot["loadings"]; Lambda = Lambda_;
    arma::mat Phi_ = rot["Phi"]; Phi = Phi_;
    std::string method_ = modelInfo["method"]; method = method_;
    std::string projection_ = modelInfo["projection"]; projection = projection_;
    std::string rotation_ = modelInfo["rotation"]; rotation = rotation_;
    double gamma_ = modelInfo["gamma"]; gamma = gamma_;
    double k_ = modelInfo["k"]; k = k_;
    double epsilon_ = modelInfo["epsilon"]; epsilon = epsilon_;
    double w_ = modelInfo["w"]; w = w_;
    arma::mat Target_ = modelInfo["Target"]; Target = Target_;
    arma::mat Weight_ = modelInfo["Weight"]; Weight = Weight_;
    arma::mat PhiTarget_ = modelInfo["PhiTarget"]; PhiTarget = PhiTarget_;
    arma::mat PhiWeight_ = modelInfo["PhiWeight"]; PhiWeight = PhiWeight_;

  } else {

    if(nullable_S.isNotNull()) {
      S = Rcpp::as<arma::mat>(nullable_S);
    } else {
      Rcpp::stop("Either provide a model fit via the fit argument or the sample correlation R");
    }
    if(nullable_Lambda.isNotNull()) {
      Lambda = Rcpp::as<arma::mat>(nullable_Lambda);
    } else {
      Rcpp::stop("Either provide a model fit via the fit argument or loadings");
    }
    if(nullable_Phi.isNotNull()) {
      Phi = Rcpp::as<arma::mat>(nullable_Phi);
    } else {
      if(projection == "oblq") {
        Rcpp::stop("Either provide a model fit via the fit argument or Phi");
      }
    }
    if(nullable_Target.isNotNull()) {
      Target = Rcpp::as<arma::mat>(nullable_Target);
    }
    if(nullable_Weight.isNotNull()) {
      Weight = Rcpp::as<arma::mat>(nullable_Weight);
    }
    if(nullable_PhiTarget.isNotNull()) {
      PhiTarget = Rcpp::as<arma::mat>(nullable_PhiTarget);
    }
    if(nullable_PhiWeight.isNotNull()) {
      PhiWeight = Rcpp::as<arma::mat>(nullable_PhiWeight);
    }

  }

  int p = Lambda.n_rows;
  int q = Phi.n_rows;
  int pq = p*q;
  int q_cor = q*(q-1)/2;
  int m;
  arma::uvec indexes;

  if(projection == "orth") {
    m = p*q + p;
  } else {
    m = p*q + q_cor + p;
    indexes = arma::trimatl_ind(arma::size(Phi), -1);
  }

  arma::mat H;
  if(method == "minres") {
    H = hessian_minres(S, Lambda, Phi, projection);
  } else if(method == "ml"){
    H = hessian_ml(S, Lambda, Phi, projection);
  } else {
    Rcpp::stop("Standard errors are not implemented yet for this method");
  }

  base_manifold* manifold = choose_manifold(projection);
  base_criterion *criterion = choose_criterion(rotation, projection, R_NilValue);

  // Compute the rotation constraints:
  arma::mat constraints_temp, gL, constraints;
  criterion->d_constraint(constraints_temp, gL, Lambda, Phi, Target, Weight,
                          PhiTarget, PhiWeight, gamma, k, epsilon, w);
  manifold->g_constraints(constraints, constraints_temp, Lambda, Phi, gL);
  // Add constraints to the hessian matrix:
  int kk = constraints.n_rows;
  H = arma::join_rows(H, constraints.t());
  // Add zeros to make H a square matrix:
  arma::mat zeros(kk, kk);
  arma::mat C = arma::join_rows(constraints, zeros);
  H = arma::join_cols(H, C);
  arma::mat H_inv = arma::inv(H);

  // Find A^{-1}BA^{-1}:
  arma::mat A_inv = H_inv(arma::span(0, m-1), arma::span(0, m-1));
  arma::mat BB = B(S, Lambda, Phi, nullable_X, method, projection,
                   type, eta);
  arma::mat VAR = A_inv * BB * A_inv;
  arma::vec se = sqrt(arma::diagvec(VAR)/(n-1));

  Rcpp::List result;
  arma::mat Lambda_se(p, q);
  arma::mat Phi_se(q, q);
  Phi_se.diag().zeros();
  arma::vec psi_se(p);

  for(int i=0; i < pq; ++i) Lambda_se[i] = se[i];
  if(projection == "orth") {
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

  result["Lambda_se"] = Lambda_se;
  result["Phi_se"] = Phi_se;
  result["psi_se"] = psi_se;

  return result;

}
