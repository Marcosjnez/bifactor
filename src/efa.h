/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 28/05/2022
 *
 */

// #include <Rcpp/Benchmark/Timer.h>
// #include "structures.h"
// #include "manifolds.h"
// #include "multiple_rotations.h"
// #include "NPF.h"
// #include "criteria.h"
// #include "EFA_fit.h"
// #include "checks.h"

Rcpp::List efa(arma::vec psi, arma::mat R, int nfactors, std::string method,
               int efa_max_iter, double efa_factr, int lmm) {

  Rcpp::List result;

  arma::mat w, Rhat;
  arma::vec uniquenesses;

  int iteration = 0;

  if (method == "minres") {

    Rcpp::List optim_result = optim_rcpp(psi, R, nfactors, method, efa_max_iter, efa_factr, lmm);

    arma::vec psi_temp = optim_result["par"];
    psi = psi_temp;
    arma::mat reduced_R = R - diagmat(psi);

    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, reduced_R);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, nfactors-1));
    arma::vec eigenvalues = eigval2(arma::span(0, nfactors-1));
    for(int i=0; i < nfactors; ++i) {
      if(eigenvalues(i) < 0) eigenvalues(i) = 0;
    }
    arma::mat D = diagmat(sqrt(eigenvalues));

    w = A * D;
    arma::mat ww = w * w.t();

    uniquenesses = 1 - diagvec(ww);

    Rhat = ww;
    Rhat.diag() = R.diag();

    bool convergence = false;
    int convergence_result = optim_result["convergence"];

    if(convergence_result == 0) convergence = true;

    result["f"] = optim_result["value"];
    result["convergence"] = convergence;

  } else if (method == "ml") {

    Rcpp::List optim_result = optim_rcpp(psi, R, nfactors, method, efa_max_iter, efa_factr, lmm);
    arma::vec psi_temp = optim_result["par"];
    psi = psi_temp;

    arma::vec sqrt_psi = sqrt(psi);
    arma::mat sc = diagmat(1/sqrt_psi);
    arma::mat Sstar = sc * R * sc;

    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, Sstar);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, nfactors-1));
    arma::vec eigenvalues = eigval2(arma::span(0, nfactors-1)) - 1;
    for(int i=0; i < nfactors; ++i) {
      if(eigenvalues[i] < 0) eigenvalues[i] = 0;
    }
    arma::mat D = diagmat(sqrt(eigenvalues));

    w = A * D;
    w = diagmat(sqrt_psi) * w;
    arma::mat ww = w * w.t();
    uniquenesses = 1 - diagvec(ww);

    Rhat = ww;
    Rhat.diag() = R.diag();

    bool convergence = false;
    int convergence_result = optim_result["convergence"];

    if(convergence_result == 0) convergence = true;

    result["f"] = optim_result["value"];
    result["convergence"] = convergence;

  } else if (method == "pa") {

    Rcpp::List pa_result = principal_axis(psi, R, nfactors, 1e-03, efa_max_iter);

    arma::mat w_temp = pa_result["loadings"];
    arma::vec uniquenesses_temp = pa_result["uniquenesses"];
    arma::mat Rhat_temp = pa_result["Rhat"];

    result["f"] = pa_result["f"];
    result["convergence"] = pa_result["convergence"];
    w = w_temp;
    uniquenesses = uniquenesses_temp;
    Rhat = Rhat_temp;

    result["iterations"] = pa_result["iterations"];

  } else if (method == "minrank") {

    arma::vec communalities = sdp_cpp(R);

    psi = 1 - communalities;

    arma::mat reduced_R = R - diagmat(psi);

    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, reduced_R);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, nfactors-1));
    arma::vec eigenvalues = eigval2(arma::span(0, nfactors-1));
    for(int i=0; i < nfactors; ++i) {
      if(eigenvalues(i) < 0) eigenvalues(i) = 0;
    }
    arma::mat D = arma::diagmat(sqrt(eigenvalues));

    w = A * D;
    arma::mat ww = w * w.t();
    uniquenesses = 1 - arma::diagvec(ww);

    Rhat = ww;
    Rhat.diag() = R.diag();

  } else {

    Rcpp::stop("Unkown method");

  }

  bool heywood = arma::any( uniquenesses < 0 );

  result["loadings"] = w;
  result["uniquenesses"] = uniquenesses;
  result["Rhat"] = Rhat;
  result["residuals"] = R - Rhat;
  result["Heywood"] = heywood;
  result["method"] = method;

  return result;
}



