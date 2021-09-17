#include "se.h"

arma::mat boot_sample(arma::mat X, bool replace) {

  int p = X.n_rows;
  int q = X.n_cols;
  arma::uvec indexes(p);
  arma::mat X_boot(p, q);

  if(replace) {

    for(int i=0; i < q; ++i) {
      indexes = arma::randi<arma::uvec>(p, arma::distr_param(0, p-1));
      arma::vec v = X.col(i);
      X_boot.col(i) = v(indexes);
    }

  } else {

    for(int i=0; i < q; ++i) {
      indexes = arma::randperm(p, p);
      arma::vec v = X.col(i);
      X_boot.col(i) = v(indexes);
      // X_boot.col(i) = arma::shuffle(X.col(i));
    }

  }

  return X_boot;

}

Rcpp::List PA(arma::mat X, int n_boot, double quant, bool replace,
              bool second_PA, Rcpp::Nullable<Rcpp::List> nullable_efa,
              int cores){

  int n = X.n_rows;
  int p = X.n_cols;

  arma::mat S = arma::cor(X);
  arma::vec eigval = eig_sym(S);

  arma::mat eigval_boot(n_boot, p);

  omp_set_num_threads(cores);
#pragma omp parallel for
  for(int i=0; i < n_boot; ++i) {

    arma::mat X_boot = boot_sample(X, replace);
    arma::mat S_boot = arma::cor(X_boot);
    eigval_boot.row(i) = eig_sym(S_boot);

  }

  arma::vec qquant(1);
  qquant[0] = quant;
  arma::mat cutoff = arma::quantile(eigval_boot, qquant);
  double PA_dim = arma::accu(eigval > cutoff);

  Rcpp::List result;
  result["eigval_boot"] = eigval_boot;
  result["PA_dim"] = PA_dim;

  if(PA_dim == 1 || !second_PA) return result;

  Rcpp::List efa;

  if(nullable_efa.isNotNull()) {
    efa = nullable_efa;
  }

  // Arguments to pass to efa:

  std::string method, rotation, projection;
  Rcpp::Nullable<arma::vec> nullable_init;
  Rcpp::Nullable<arma::mat> nullable_Target, nullable_Weight,
  nullable_PhiTarget, nullable_PhiWeight;
  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks;
  bool normalize;
  double gamma, epsilon, k, w;
  int random_starts, cores_2 = 0;
  Rcpp::Nullable<Rcpp::List> nullable_efa_control, nullable_rot_control;

  pass_to_efast(efa,
                method, rotation, projection,
                nullable_Target, nullable_Weight,
                nullable_PhiTarget, nullable_PhiWeight,
                nullable_oblq_blocks, normalize,
                gamma, epsilon, k, w,
                random_starts, cores_2,
                nullable_init,
                nullable_efa_control,
                nullable_rot_control);

  // efa:

  Rcpp::List fit = efast(S, PA_dim, method, rotation, projection,
                         nullable_Target, nullable_Weight,
                         nullable_PhiTarget, nullable_PhiWeight,
                         nullable_oblq_blocks,
                         normalize, gamma, epsilon, k, w,
                         random_starts, cores_2,
                         nullable_init,
                         nullable_efa_control, nullable_rot_control);

  Rcpp::List rot = fit["rotation"];
  arma::mat Phi = rot["Phi"];
  arma::mat loadings = rot["loadings"];
  arma::mat L = loadings * Phi;
  arma::mat W = arma::solve(S, L);
  arma::mat fs = X * W;

  arma::mat S2 = arma::cor(fs);
  arma::vec eigval2 = eig_sym(S2);
  arma::mat eigval2_boot(n_boot, PA_dim);

  omp_set_num_threads(cores);
#pragma omp parallel for
  for(int i=0; i < n_boot; ++i) {

    arma::mat X_boot = boot_sample(fs, replace);
    arma::mat S_boot = arma::cor(X_boot);
    eigval2_boot.row(i) = eig_sym(S_boot);

  }

  arma::mat cutoff2 = arma::quantile(eigval2_boot, qquant);
  int PA2_dim = arma::accu(eigval2 > cutoff2);

  result["eigval2_boot"] = eigval2_boot;
  result["PA2_dim"] = PA2_dim;

  return result;

}
