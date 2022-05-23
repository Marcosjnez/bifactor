/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
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
    Rhat.diag().ones();

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
    Rhat.diag().ones();

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
    Rhat.diag().ones();

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

Rcpp::List rotate_efa(arguments_rotate x, base_manifold *manifold, base_criterion *criterion,
                      int random_starts, int cores, double eps, int maxit) {

  arma::vec xf(random_starts);
  TRN x1;
  std::vector<TRN> x2(random_starts);

  // Perform multiple rotations with random starting values:

  omp_set_num_threads(cores);
#pragma omp parallel for
  for (int i=0; i < random_starts; ++i) {

    arguments_rotate args = x;

    args.T = random_orth(args.q, args.q);
    // args.T = arma::mat(args.q, args.q, arma::fill::eye);

    x2[i] = NPF(args, manifold, criterion, x.rot_eps, x.rot_maxit);

    xf[i] = std::get<3>(x2[i]);

  }

  // Choose the rotation with the smallest objective value:

  arma::uword index_minimum = index_min(xf);
  x1 = x2[index_minimum];

  arma::mat L = std::get<0>(x1);
  arma::mat Phi = std::get<1>(x1);
  if(Phi.is_empty()) {Phi.set_size(x.q, x.q); Phi.eye();}
  arma::mat T = std::get<2>(x1);
  double f = std::get<3>(x1);
  int iterations = std::get<4>(x1);
  bool convergence = std::get<5>(x1);

  // Force average positive loadings in all factors:

  // arma::vec v = arma::sign(arma::sum(L, 0));
  // L.each_row() /= v;

  for (int j=0; j < x.q; ++j) {
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

Rcpp::List efast(arma::mat R, int nfactors, std::string method,
                 Rcpp::CharacterVector char_rotation,
                 std::string projection,
                 Rcpp::Nullable<int> nullable_nobs,
                 Rcpp::Nullable<arma::mat> nullable_Target,
                 Rcpp::Nullable<arma::mat> nullable_Weight,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                 Rcpp::Nullable<arma::uvec> nullable_blocks,
                 Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list,
                 Rcpp::Nullable<arma::vec> nullable_block_weights,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                 std::string normalization, std::string between_blocks,
                 arma::vec gamma, arma::vec epsilon, arma::vec k, double w,
                 double alpha, double a, double b,
                 int random_starts, int cores,
                 Rcpp::Nullable<arma::vec> nullable_init,
                 Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                 Rcpp::Nullable<Rcpp::List> nullable_rot_control) {

  Rcpp::Timer timer;

  std::vector<std::string> rotation = Rcpp::as<std::vector<std::string>>(char_rotation);

  // Create defaults:

  int efa_maxit, lmm;
  double efa_eps, efa_factr;
  arma::vec init;

  // Check EFA inputs:

  check_efa(R, nfactors, nullable_init, init,
            nullable_efa_control,
            efa_maxit, lmm, efa_factr);

  Rcpp::List result;
  Rcpp::List efa_result = efa(init, R, nfactors, method, efa_maxit, efa_factr, lmm);

  bool heywood = efa_result["Heywood"];

  if(heywood) {

    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::warning("Heywood case detected /n Using minimum rank factor analysis");

    efa_result = efa(init, R, nfactors, "minrank", efa_maxit, efa_factr, lmm);

  }

  int p = R.n_cols;
  int q = nfactors;
  double df_null = p*(p-1)/2;
  double df = p*(p+1)/2 - (p*q + p - q*(q-1)/2);

  double f_null;
  if(method == "minres" || method == "pa") {
    f_null = 0.5*(arma::accu(R % R) - R.n_cols);
  } else if(method == "ml") {
    f_null = -arma::log_det_sympd(R);
  } else if(method == "minrank") {
    f_null = 0;
  }

  Rcpp::List modelInfo;
  modelInfo["method"] = method;
  modelInfo["projection"] = projection;
  modelInfo["rotation"] = rotation;
  modelInfo["n_vars"] = R.n_cols;
  modelInfo["nfactors"] = nfactors;
  modelInfo["nobs"] = nullable_nobs;
  modelInfo["df"] = df;
  modelInfo["df_null"] = df_null;
  modelInfo["f_null"] = f_null;
  modelInfo["k"] = k;
  modelInfo["gamma"] = gamma;
  modelInfo["epsilon"] = epsilon;
  modelInfo["w"] = w;
  modelInfo["alpha"] = alpha;
  modelInfo["a"] = a;
  modelInfo["b"] = b;
  modelInfo["normalization"] = normalization;
  modelInfo["between_blocks"] = between_blocks;
  modelInfo["R"] = R;
  modelInfo["Target"] = nullable_Target;
  modelInfo["Weight"] = nullable_Weight;
  modelInfo["PhiTarget"] = nullable_PhiTarget;
  modelInfo["PhiWeight"] = nullable_PhiWeight;
  modelInfo["blocks"] = nullable_blocks;
  modelInfo["blocks_list"] = nullable_blocks_list;
  modelInfo["block_weights"] = nullable_block_weights;
  modelInfo["oblq_blocks"] = nullable_oblq_blocks;

  arma::mat loadings = efa_result["loadings"];

  // Structure of rotation arguments:

  arguments_rotate x;
  x.lambda = loadings;
  x.p = R.n_rows, x.q = nfactors;
  // x.lambda.set_size(x.p, x.q);
  x.Phi.set_size(x.q, x.q); x.Phi.eye();
  x.gamma = gamma, x.epsilon = epsilon, x.k = k, x.w = w, x.alpha = alpha,
  x.a = a, x.b = b;
  x.between_blocks = between_blocks;
  x.rotations = rotation;
  x.projection = projection;
  x.nullable_Target = nullable_Target;
  x.nullable_Weight = nullable_Weight;
  x.nullable_PhiTarget = nullable_PhiTarget;
  x.nullable_PhiWeight = nullable_PhiWeight;
  x.nullable_blocks = nullable_blocks;
  x.nullable_oblq_blocks = nullable_oblq_blocks;
  x.nullable_block_weights = nullable_block_weights;
  x.nullable_blocks_list = nullable_blocks_list;
  x.nullable_rot_control = nullable_rot_control;

  // Check rotation inputs and compute constants for rotation criteria:

  check_rotate(x, random_starts, cores);

  // Select one manifold:
  base_manifold* manifold = choose_manifold(x.projection);
  // Select one specific criteria or mixed criteria:
  base_criterion* criterion = choose_criterion(x.rotations, x.projection, x.blocks_list);

  Rcpp::List rotation_result;

  arma::vec weigths;
  if (normalization == "kaiser") {

    weigths = sqrt(sum(x.lambda % x.lambda, 1));
    x.lambda.each_col() /= weigths;

  }

  if(rotation.size() == 1 && rotation[0] == "none" || random_starts < 1) {

    result["efa"] = efa_result;
    result["modelInfo"] = modelInfo;
    timer.step("elapsed");
    result["elapsed"] = timer;

    return result;

  } else {

    rotation_result = rotate_efa(x, manifold, criterion,
                                 random_starts, cores, x.rot_eps, x.rot_maxit);

  }

  arma::mat L = rotation_result["loadings"];
  arma::mat Phi = rotation_result["Phi"];

  if (normalization == "kaiser") {

    L.each_col() %= weigths;

  }

  rotation_result["loadings"] = L;
  rotation_result["Phi"] = Phi;
  rotation_result["Rhat"] = efa_result["Rhat"];
  rotation_result["uniquenesses"] = efa_result["uniquenesses"];
  rotation_result["Rhat"] = efa_result["Rhat"];
  rotation_result["residuals"] = efa_result["residuals"];

  result["efa"] = efa_result;
  result["rotation"] = rotation_result;
  result["modelInfo"] = modelInfo;

  timer.step("elapsed");
  result["elapsed"] = timer;

  result.attr("class") = "efa";
  return result;
}

// Do not export this (overloaded to support std::vector<std::string> rotation):
Rcpp::List efast(arma::mat R, int nfactors, std::string method,
                 std::vector<std::string> rotation,
                 std::string projection,
                 Rcpp::Nullable<int> nullable_nobs,
                 Rcpp::Nullable<arma::mat> nullable_Target,
                 Rcpp::Nullable<arma::mat> nullable_Weight,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                 Rcpp::Nullable<arma::uvec> nullable_blocks,
                 Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list,
                 Rcpp::Nullable<arma::vec> nullable_block_weights,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                 std::string normalization, std::string between_blocks,
                 arma::vec gamma, arma::vec epsilon, arma::vec k, double w,
                 double alpha, double a, double b,
                 int random_starts, int cores,
                 Rcpp::Nullable<arma::vec> nullable_init,
                 Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                 Rcpp::Nullable<Rcpp::List> nullable_rot_control) {

  Rcpp::Timer timer;

  // Create defaults:

  int efa_maxit, lmm;
  double efa_eps, efa_factr;
  arma::vec init;

// Check EFA inputs:

  check_efa(R, nfactors, nullable_init, init,
            nullable_efa_control,
            efa_maxit, lmm, efa_factr);

  Rcpp::List result;
  Rcpp::List efa_result = efa(init, R, nfactors, method, efa_maxit, efa_factr, lmm);

  bool heywood = efa_result["Heywood"];

  if(heywood) {

    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::warning("Heywood case detected /n Using minimum rank factor analysis");

    efa_result = efa(init, R, nfactors, "minrank", efa_maxit, efa_factr, lmm);

  }

  int p = R.n_cols;
  int q = nfactors;
  double df_null = p*(p-1)/2;
  double df = p*(p+1)/2 - (p*q + p - q*(q-1)/2);

  double f_null;
  if(method == "minres" || method == "pa") {
    f_null = 0.5*(arma::accu(R % R) - R.n_cols);
  } else if(method == "ml") {
    f_null = -arma::log_det_sympd(R);
  } else if(method == "minrank") {
    f_null = 0;
  }

  Rcpp::List modelInfo;
  modelInfo["method"] = method;
  modelInfo["projection"] = projection;
  modelInfo["rotation"] = rotation;
  modelInfo["n_vars"] = R.n_cols;
  modelInfo["nfactors"] = nfactors;
  modelInfo["nobs"] = nullable_nobs;
  modelInfo["df"] = df;
  modelInfo["df_null"] = df_null;
  modelInfo["f_null"] = f_null;
  modelInfo["k"] = k;
  modelInfo["gamma"] = gamma;
  modelInfo["epsilon"] = epsilon;
  modelInfo["w"] = w;
  modelInfo["alpha"] = alpha;
  modelInfo["a"] = a;
  modelInfo["b"] = b;
  modelInfo["normalization"] = normalization;
  modelInfo["between_blocks"] = between_blocks;
  modelInfo["R"] = R;
  modelInfo["Target"] = nullable_Target;
  modelInfo["Weight"] = nullable_Weight;
  modelInfo["PhiTarget"] = nullable_PhiTarget;
  modelInfo["PhiWeight"] = nullable_PhiWeight;
  modelInfo["blocks"] = nullable_blocks;
  modelInfo["blocks_list"] = nullable_blocks_list;
  modelInfo["block_weights"] = nullable_block_weights;
  modelInfo["oblq_blocks"] = nullable_oblq_blocks;

  arma::mat loadings = efa_result["loadings"];

  // Structure of rotation arguments:

  arguments_rotate x;
  x.lambda = loadings;
  x.p = R.n_rows, x.q = nfactors;
  // x.lambda.set_size(x.p, x.q);
  x.Phi.set_size(x.q, x.q); x.Phi.eye();
  x.gamma = gamma, x.epsilon = epsilon, x.k = k, x.w = w, x.alpha = alpha,
    x.a = a, x.b = b;
  x.between_blocks = between_blocks;
  x.rotations = rotation;
  x.projection = projection;
  x.nullable_Target = nullable_Target;
  x.nullable_Weight = nullable_Weight;
  x.nullable_PhiTarget = nullable_PhiTarget;
  x.nullable_PhiWeight = nullable_PhiWeight;
  x.nullable_blocks = nullable_blocks;
  x.nullable_oblq_blocks = nullable_oblq_blocks;
  x.nullable_block_weights = nullable_block_weights;
  x.nullable_blocks_list = nullable_blocks_list;
  x.nullable_rot_control = nullable_rot_control;

  // Check inputs and compute constants for rotation criteria:

  check_rotate(x, random_starts, cores);

  // Select one manifold:
  base_manifold* manifold = choose_manifold(x.projection);
  // Select one specific criteria or mixed criteria:
  base_criterion* criterion = choose_criterion(x.rotations, x.projection, x.blocks_list);

  Rcpp::List rotation_result;

  arma::vec weigths;
  if (normalization == "kaiser") {

    weigths = sqrt(sum(x.lambda % x.lambda, 1));
    x.lambda.each_col() /= weigths;

  }

  if(rotation.size() == 1 && rotation[0] == "none" || random_starts < 1) {

    result["efa"] = efa_result;
    result["modelInfo"] = modelInfo;
    timer.step("elapsed");
    result["elapsed"] = timer;

    return result;

  } else {

    rotation_result = rotate_efa(x, manifold, criterion,
                                 random_starts, cores, x.rot_eps, x.rot_maxit);

  }

  arma::mat L = rotation_result["loadings"];
  arma::mat Phi = rotation_result["Phi"];

  if (normalization == "kaiser") {

    L.each_col() %= weigths;

  }

  rotation_result["loadings"] = L;
  rotation_result["Phi"] = Phi;
  rotation_result["Rhat"] = efa_result["Rhat"];
  rotation_result["uniquenesses"] = efa_result["uniquenesses"];
  rotation_result["Rhat"] = efa_result["Rhat"];
  rotation_result["residuals"] = efa_result["residuals"];

  result["efa"] = efa_result;
  result["rotation"] = rotation_result;
  result["modelInfo"] = modelInfo;

  timer.step("elapsed");
  result["elapsed"] = timer;

  result.attr("class") = "efa";
  return result;
}


