#include "GPF.h"
#include "NPF.h"

arma::mat random_orthogonal(int p, int q) {

  arma::mat X(p, q, arma::fill::randn);
  arma::mat Q;
  arma::mat R;
  qr_econ(Q, R, X);

  return Q;

}

arma::mat random_poblq(int p, int q, std::vector<arma::uvec> indexes) {

  arma::mat X(p, q, arma::fill::randn);
  X = retraction_poblq(X, indexes);

  return X;

}

Rcpp::List multiple_rotations(arma::mat loadings, std::string rotation, arma::mat Target, arma::mat Weight, arma::mat Phi_Target, arma::mat Phi_Weight, std::vector<arma::uvec> indexes,
                              double gamma, double epsilon, double k, double w, int random_starts, int cores,
                              double eps, int max_iter) {

  int n_factors = loadings.n_cols;

  arma::mat Phi(n_factors, n_factors);
  arma::uvec indexes_1, indexes_2;

  if(rotation == "xtarget" || rotation == "target" || rotation == "targetQ" || rotation == "poblq_target") {

    if(arma::size(Target) != arma::size(loadings) ||
       arma::size(Weight) != arma::size(loadings) ||
       arma::size(Phi_Target) != arma::size(Phi) ||
       arma::size(Phi_Weight) != arma::size(Phi)) {

      Rcpp::stop("Incompatible Target dimensions");

    }

    if(rotation == "poblq_target") {

      for(int i=0; i < indexes.size(); i++) indexes[i] -= 1;
      arma::mat X(n_factors, n_factors, arma::fill::randu);
      arma::mat Q = zeros(X, indexes);
      indexes_1 = arma::find(Q == 0);
      // indexes_2 = arma::find(Q != 0);
      indexes_2 = arma::find(arma::trimatl(Q) != 0);

    }

  }

  arma::vec xf(random_starts);
  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> x;
  std::vector<std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool>> x2(random_starts);

  omp_set_num_threads(cores);
#pragma omp parallel for
  for (int i=0; i < random_starts; ++i) {

    arma::mat T = random_orthogonal(n_factors, n_factors);

    if (rotation == "xtarget") {
      x2[i] = NPF_xtarget(T, loadings, Target, Weight, Phi_Target, Phi_Weight, w, eps, max_iter);
    } else if (rotation == "poblq_target") {
      x2[i] = NPF_poblqtarget(T, loadings, Target, Weight, indexes, indexes_1, indexes_2, eps, max_iter);
    } else if (rotation == "target") {
      x2[i] = GPF_target(T, loadings, Target, Weight, eps, max_iter);
    } else if (rotation == "targetQ") {
      x2[i] = NPF_targetQ(T, loadings, Target, Weight, eps, max_iter);
    } else if (rotation == "cfT") {
      x2[i] = GPF_cfT(T, loadings, k, eps, max_iter);
    } else if (rotation == "cfQ") {
      x2[i] = GPF_cfQ(T, loadings, k, eps, max_iter);
    } else if (rotation == "varimax") {
      x2[i] = GPF_varimax_2(T, loadings, eps, max_iter);
    } else if (rotation == "oblimin") {
      x2[i] = NPF_oblimin(T, loadings, gamma, eps, max_iter);
    } else if (rotation == "geominQ") {
      x2[i] = NPF_geominQ(T, loadings, epsilon, eps, max_iter);
    } else if (rotation == "geominT") {
      x2[i] = GPF_geominT(T, loadings, epsilon, eps, max_iter);
    }

    xf[i] = std::get<3>(x2[i]);

  }

  arma::uword index_minimum = index_min(xf);
  x = x2[index_minimum];

  arma::mat L = std::get<0>(x);
  Phi = std::get<1>(x);
  arma::mat T = std::get<2>(x);
  double f = std::get<3>(x);
  int iterations = std::get<4>(x);
  bool convergence = std::get<5>(x);

  for (int j=0; j < n_factors; ++j) {
    if (sum(L.col(j)) < 0) {
      L.col(j)   *= -1;
      Phi.col(j) *= -1;
      Phi.row(j) *= -1;
    }
  }

  if(!convergence) {

    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::warning("Failed rotation convergence");

  }

  Rcpp::List result;
  result["loadings"] = L;
  result["Phi"] = Phi;
  result["T"] = T;
  result["f"] = f;
  result["iterations"] = iterations;
  result["convergence"] = convergence;

  result.attr("class") = "rotation";
  return result;

}
