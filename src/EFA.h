#include "multiple_rotations.h"
#include "EFA_fit.h"

Rcpp::List efa(arma::vec psy, arma::mat R, int n_factors, std::string method,
               int efa_max_iter, double efa_factr, int m) {

  Rcpp::List result;

  arma::mat w, R_hat;
  arma::vec uniquenesses;

  int iteration = 0;

  if (method == "minres" || method == "ml") {

    Rcpp::List optim_result = optim_rcpp(psy, R, n_factors, method, efa_max_iter, efa_factr, m);

    arma::vec psy_temp = optim_result["par"];
    psy = psy_temp;
    arma::mat reduced_R = R - diagmat(psy);

    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, reduced_R);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, n_factors-1));
    arma::vec eigenvalues = eigval2(arma::span(0, n_factors-1));
    for(int i=0; i < n_factors; ++i) {
      if(eigenvalues(i) < 0) eigenvalues(i) = 0;
    }
    arma::mat D = diagmat(sqrt(eigenvalues));

    w = A * D;
    arma::mat ww = w * w.t();

    uniquenesses = 1 - diagvec(ww);

    R_hat = ww;
    R_hat.diag().ones();

    bool convergence = false;
    int convergence_result = optim_result["convergence"];

    if(convergence_result == 0) convergence = true;

    result["f"] = optim_result["value"];
    result["convergence"] = convergence;

  } else if (method == "pa") {

    Rcpp::List pa_result = principal_axis(psy, R, n_factors, 1e-03, efa_max_iter);

    arma::mat w_temp = pa_result["loadings"];
    arma::vec uniquenesses_temp = pa_result["uniquenesses"];
    arma::mat R_hat_temp = pa_result["R_hat"];

    w = w_temp;
    uniquenesses = uniquenesses_temp;
    R_hat= R_hat_temp;

    result["iterations"] = pa_result["iterations"];

  } else if (method == "minrank") {

    arma::vec communalities = sdp_cpp(R);

    psy = 1 - communalities;

    arma::mat reduced_R = R - diagmat(psy);

    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, reduced_R);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, n_factors-1));
    arma::vec eigenvalues = eigval2(arma::span(0, n_factors-1));
    for(int i=0; i < n_factors; ++i) {
      if(eigenvalues(i) < 0) eigenvalues(i) = 0;
    }
    arma::mat D = arma::diagmat(sqrt(eigenvalues));

    w = A * D;
    arma::mat ww = w * w.t();
    uniquenesses = 1 - arma::diagvec(ww);

    R_hat = ww;
    R_hat.diag().ones();

  }

  bool heywood = arma::any( uniquenesses < 0 );

  result["loadings"] = w;
  result["uniquenesses"] = uniquenesses;
  result["R_hat"] = R_hat;
  result["residuals"] = R - R_hat;
  result["Heywood"] = heywood;
  result["method"] = method;

  return result;
}

Rcpp::List efast(arma::mat R, int n_factors, std::string method, std::string rotation,
                 Rcpp::Nullable<Rcpp::NumericVector> init,
                 Rcpp::Nullable<Rcpp::NumericMatrix> LTarget,
                 Rcpp::Nullable<Rcpp::NumericMatrix> LWeight,
                 Rcpp::Nullable<Rcpp::NumericMatrix> PhiTarget,
                 Rcpp::Nullable<Rcpp::NumericMatrix> PhiWeight,
                 Rcpp::Nullable<Rcpp::List> oblique_indexes,
                 bool normalize, double gamma, double epsilon, double k, double w,
                 int random_starts, int cores,
                 int efa_max_iter, double efa_factr, int m,
                 int rot_max_iter, double rot_eps) {

  arma::vec psy;

  if (init.isNotNull()) {
    psy = Rcpp::as<arma::vec>(init);
  } else {
    psy = 1/arma::diagvec(arma::inv_sympd(R));
  }

  int n_items = R.n_rows;
  arma::mat Target(n_items, n_factors);
  arma::mat Weight(n_items, n_factors, arma::fill::ones);
  arma::mat Phi_Target(n_factors, n_factors);
  arma::mat Phi_Weight(n_factors, n_factors, arma::fill::zeros);
  std::vector<arma::uvec> indexes;

  if (LTarget.isNotNull()) {
    Target = Rcpp::as<arma::mat>(LTarget);
  }
  if (LWeight.isNotNull()) {
    Weight = Rcpp::as<arma::mat>(LWeight);
  }
  if (PhiTarget.isNotNull()) {
    Phi_Target = Rcpp::as<arma::mat>(PhiTarget);
  }
  if (PhiWeight.isNotNull()) {
    Phi_Weight = Rcpp::as<arma::mat>(PhiWeight);
  }
  if (oblique_indexes.isNotNull()) {
    indexes = Rcpp::as<std::vector<arma::uvec>>(oblique_indexes);
  }

  Rcpp::List result;

  Rcpp::List efa_result = efa(psy, R, n_factors, method, efa_max_iter, efa_factr, m);

  bool heywood = efa_result["Heywood"];

  if(heywood) {

    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::warning("Heywood case detected /n Using minimum rank factor analysis");

    efa_result = efa(psy, R, n_factors, "minrank", efa_max_iter, efa_factr, m);

  }

  efa_result["Heywood"] = heywood;
  arma::mat loadings = efa_result["loadings"];

  Rcpp::List rotation_result;

  arma::vec weigths;
  if (normalize) {

    weigths = sqrt(sum(pow(loadings, 2), 1));
    loadings.each_col() /= weigths;

  }

  if(rotation == "none" || random_starts < 1) {

    return efa_result;

  } else {

    rotation_result = multiple_rotations(loadings, rotation, Target, Weight, Phi_Target, Phi_Weight, indexes,
                                         gamma, epsilon, k, w, random_starts, cores, rot_eps, rot_max_iter);

  }

  arma::mat L = rotation_result["loadings"];
  arma::mat Phi = rotation_result["Phi"];

  if (normalize) {

    L.each_col() %= weigths;

  }

  arma::mat R_hat = L * Phi * L.t();
  rotation_result["loadings"] = L;
  rotation_result["Phi"] = Phi;

  arma::vec uniquenesses = 1 - diagvec(R_hat);

  rotation_result["uniquenesses"] = uniquenesses;

  R_hat.diag().ones();
  rotation_result["R_hat"] = R_hat;
  rotation_result["residuals"] = R - R_hat;

  result["efa"] = efa_result;
  result["rotation"] = rotation_result;

  result.attr("class") = "efa";
  return result;
}




