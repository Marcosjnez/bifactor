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

Rcpp::List parallel(arma::mat X, int n_boot, double quant,
                    bool mean, bool replace,
                    bool hierarchical, Rcpp::Nullable<Rcpp::List> nullable_efa,
                    int cores){

  int n = X.n_rows;
  int p = X.n_cols;
  arma::cube X_boots(n, p, n_boot);

  arma::mat S = arma::cor(X);
  arma::vec eigval = eig_sym(S);

  arma::mat eigval_boot(n_boot, p);

  omp_set_num_threads(cores);
#pragma omp parallel for
  for(int i=0; i < n_boot; ++i) {

    X_boots.slice(i) = boot_sample(X, replace);
    arma::mat S_boot = arma::cor(X_boots.slice(i));
    eigval_boot.row(i) = eig_sym(S_boot);

  }

  arma::vec cutoff;

  if(mean) {

    cutoff = arma::mean(eigval_boot);

  } else {

    arma::vec qquant(1);
    qquant[0] = quant;
    cutoff = arma::quantile(eigval_boot, qquant);

  }

  arma::umat booleans = arma::reverse(eigval > cutoff);
  arma::uvec ones = arma::find(booleans == 0);
  int groups = ones[0];

  Rcpp::List result;
  result["eigval_boot"] = eigval_boot;
  result["groups"] = groups;

  if(groups <= 1 || !hierarchical) return result;

  Rcpp::List efa;

  if(nullable_efa.isNotNull()) {
    efa = nullable_efa;
  }

  // Arguments to pass to efa:

  std::string method, rotation, projection;
  Rcpp::Nullable<arma::vec> nullable_init;
  Rcpp::Nullable<arma::mat> nullable_Target, nullable_Weight,
  nullable_PhiTarget, nullable_PhiWeight;
  Rcpp::Nullable<arma::uvec> nullable_blocks;
  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks;
  bool normalize;
  double gamma, epsilon, k, w;
  int random_starts, cores_2 = 0;
  Rcpp::Nullable<Rcpp::List> nullable_efa_control, nullable_rot_control;

  pass_to_efast(efa,
                method, rotation, projection,
                nullable_Target, nullable_Weight,
                nullable_PhiTarget, nullable_PhiWeight,
                nullable_blocks, nullable_oblq_blocks, normalize,
                gamma, epsilon, k, w,
                random_starts, cores_2,
                nullable_init,
                nullable_efa_control,
                nullable_rot_control);

  // efa:

  Rcpp::List fit = efast(S, groups, method, rotation, projection,
                         nullable_Target, nullable_Weight,
                         nullable_PhiTarget, nullable_PhiWeight,
                         nullable_blocks, nullable_oblq_blocks,
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
  arma::mat eigval2_boot(n_boot, groups);
  arma::mat eigval2_W_boot(n_boot, groups);

  omp_set_num_threads(cores);
#pragma omp parallel for
  for(int i=0; i < n_boot; ++i) {

    arma::mat X_boot = boot_sample(fs, replace);
    arma::mat S_boot = arma::cor(X_boot);
    eigval2_boot.row(i) = eig_sym(S_boot);

    X_boot = X_boots.slice(i) * W;
    S_boot = arma::cor(X_boot);
    eigval2_W_boot.row(i) = eig_sym(S_boot);

  }

  arma::mat cutoff2, cutoff2_W;

  if(mean) {

    cutoff2 = arma::mean(eigval2_boot);
    cutoff2_W = arma::mean(eigval2_W_boot);

  } else {

    arma::vec qquant(1);
    qquant[0] = quant;
    cutoff2 = arma::quantile(eigval2_boot, qquant);
    cutoff2_W = arma::quantile(eigval2_W_boot, qquant);

  }

  booleans = arma::reverse(eigval2 > cutoff2);
  ones = arma::find(booleans == 0);
  int generals = ones[0];

  booleans = arma::reverse(eigval2 > cutoff2_W);
  ones = arma::find(booleans == 0);
  int generalsW = ones[0];

  result["eigval2_boot"] = eigval2_boot;
  result["eigval2_W_boot"] = eigval2_W_boot;
  result["generals"] = generals;
  result["generalsW"] = generalsW;
  result["fit"] = fit;
  result["fs"] = fs;

  return result;

}

// Te same but only first order:
Rcpp::List pa(arma::mat X, int n_boot, arma::vec quant, bool replace,
              int cores){

  int n = X.n_rows;
  int p = X.n_cols;
  arma::cube X_boots(n, p, n_boot);

  arma::mat S = arma::cor(X);
  arma::vec eigval = eig_sym(S);

  arma::mat eigval_boot(p, n_boot);

  omp_set_num_threads(cores);
#pragma omp parallel for
  for(int i=0; i < n_boot; ++i) {

    X_boots.slice(i) = boot_sample(X, replace);
    arma::mat S_boot = arma::cor(X_boots.slice(i));
    eigval_boot.col(i) = eig_sym(S_boot);

  }

  arma::mat cutoff = arma::quantile(eigval_boot, quant, 1);
  int n_quant = quant.size();
  arma::vec groups(n_quant);

  for(int i=0; i < n_quant; ++i) {

    arma::umat booleans = arma::reverse(eigval > cutoff.col(i));
    arma::uvec ones = arma::find(booleans == 0);
    groups[i] = ones[0];

  }

  Rcpp::List result;
  result["eigval_sample"] = eigval;
  result["eigval_boot"] = eigval_boot;
  result["groups"] = groups;

  return result;

}

Rcpp::List cv_eigen(arma::mat X, int N, bool hierarchical,
                    Rcpp::Nullable<Rcpp::List> nullable_efa,
                    int cores) {

  arma::mat S = arma::cor(X);
  int q = S.n_cols;
  arma::mat CV_eigvals(N, q);
  int p = X.n_rows;
  arma::uvec indexes = consecutive(0, p-1);
  int half = p/2;

  omp_set_num_threads(cores);
#pragma omp parallel for
  for(int i=0; i < N; ++i) {

    arma::uvec selected = arma::randperm(p, half);
    arma::mat A = X.rows(selected);
    arma::mat B = X;
    B.shed_rows(selected);
    arma::mat cor_A = arma::cor(A);
    arma::mat cor_B = arma::cor(B);
    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, cor_A);

    arma::vec cv_values = eigvec.t() * cor_B * eigvec;
    CV_eigvals.row(i) = arma::diagvec(cv_values);

  }

  arma::vec avg_CV_eigvals = arma::mean(CV_eigvals, 0);
  arma::uvec which = arma::find(avg_CV_eigvals > 1);
  int dim = which.size();

  Rcpp::List result;
  result["CV_eigvals"] = avg_CV_eigvals;
  result["dim"] = dim;

  if(dim <= 1 || !hierarchical) return result;

  Rcpp::List efa;

  if(nullable_efa.isNotNull()) {
    efa = nullable_efa;
  }

  // Arguments to pass to efa:

  std::string method, rotation, projection;
  Rcpp::Nullable<arma::vec> nullable_init;
  Rcpp::Nullable<arma::mat> nullable_Target, nullable_Weight,
  nullable_PhiTarget, nullable_PhiWeight;
  Rcpp::Nullable<arma::uvec> nullable_blocks;
  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks;
  bool normalize;
  double gamma, epsilon, k, w;
  int random_starts, cores_2 = 0;
  Rcpp::Nullable<Rcpp::List> nullable_efa_control, nullable_rot_control;

  pass_to_efast(efa,
                method, rotation, projection,
                nullable_Target, nullable_Weight,
                nullable_PhiTarget, nullable_PhiWeight,
                nullable_blocks, nullable_oblq_blocks, normalize,
                gamma, epsilon, k, w,
                random_starts, cores_2,
                nullable_init,
                nullable_efa_control,
                nullable_rot_control);

  // efa:

  Rcpp::List fit = efast(S, dim, method, rotation, projection,
                         nullable_Target, nullable_Weight,
                         nullable_PhiTarget, nullable_PhiWeight,
                         nullable_blocks, nullable_oblq_blocks,
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

  arma::mat CV_eigvals2(N, dim);

  omp_set_num_threads(cores);
#pragma omp parallel for
  for(int i=0; i < N; ++i) {

    arma::uvec selected = arma::randperm(p, half);
    arma::mat A = fs.rows(selected);
    arma::mat B = fs;
    B.shed_rows(selected);
    arma::mat cor_A = arma::cor(A);
    arma::mat cor_B = arma::cor(B);
    arma::vec eigval;
    arma::mat eigvec;
    eig_sym(eigval, eigvec, cor_A);

    arma::vec cv_values = eigvec.t() * cor_B * eigvec;
    CV_eigvals2.row(i) = arma::diagvec(cv_values);

  }

  arma::vec avg_CV_eigvals2 = arma::mean(CV_eigvals2, 0);
  // arma::uvec which2 = arma::find(avg_CV_eigvals2 > 1);
  // int dim2 = which2.size();
  arma::uvec ones = arma::find((avg_CV_eigvals2 < 1) == 1);
  int dim2 = ones[0];

  result["CV_eigvals2"] = avg_CV_eigvals2;
  result["dim2"] = dim2;
  result["fit"] = fit;
  result["fs"] = fs;

  return result;

}
