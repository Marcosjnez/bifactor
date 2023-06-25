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

Rcpp::List rotate_efa(arguments_rotate x, rotation_manifold *manifold, rotation_criterion *criterion,
                      int random_starts, int cores) {

  arma::vec xf(random_starts);
  NTR x1;
  std::vector<NTR> x2(random_starts);
  // Select the optimization rutine:
  rotation_optim* algorithm = choose_optim(x.optim);

  // Perform multiple rotations with random starting values:

#ifdef _OPENMP
  omp_set_num_threads(cores);
#pragma omp parallel for
#endif
  for (int i=0; i < random_starts; ++i) {

    arguments_rotate args = x;

    args.T = random_orth(args.q, args.q);
    // args.T = arma::mat(args.q, args.q, arma::fill::eye);

    x2[i] = algorithm->optim(args, manifold, criterion);
    // x2[i] = ntr(args, manifold, criterion);
    // x2[i] = gd(args, manifold, criterion);

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

Rcpp::List efast(arma::mat X, int nfactors, std::string cor, std::string method,
                 Rcpp::CharacterVector char_rotation,
                 std::string projection,
                 Rcpp::Nullable<int> nullable_nobs,
                 Rcpp::Nullable<arma::mat> nullable_Target,
                 Rcpp::Nullable<arma::mat> nullable_Weight,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                 Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> nullable_blocks,
                 Rcpp::Nullable<arma::vec> nullable_block_weights,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_factors,
                 arma::vec gamma, arma::vec epsilon, arma::vec k, double w,
                 int random_starts, int cores,
                 Rcpp::Nullable<arma::vec> nullable_init,
                 Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                 Rcpp::Nullable<Rcpp::List> nullable_rot_control) {

  Rcpp::Timer timer;
  Rcpp::List result;

  Rcpp::List correlation_result;
  arma::mat R;

  if(X.is_square()) {

    R = X;

  } else {

    if(cor == "poly") {
      correlation_result = polyfast(X, "none", false, 0L, cores);
      arma::mat polys = correlation_result["polychorics"];
      R = polys;
    } else if(cor == "pearson") {
      R = arma::cor(X);
      correlation_result["type"] = "pearson";
      correlation_result["correlation"] = R;
    }

  }

  std::vector<std::string> rotation = Rcpp::as<std::vector<std::string>>(char_rotation);

  // Create defaults:

  int efa_maxit, lmm;
  double efa_eps, efa_factr;
  arma::vec init;

  // Check EFA inputs:

  // check_efa(R, nfactors, nullable_init, init,
  //           nullable_efa_control,
  //           efa_maxit, lmm, efa_factr);
  //
  // Rcpp::List efa_result = efa(init, R, nfactors, method, efa_maxit, efa_factr, lmm);

  // Structure of efa arguments:

  arguments_efa xefa;
  xefa.method = method;
  xefa.R = R;
  xefa.p = R.n_cols;
  xefa.q = nfactors;
  xefa.upper = arma::diagvec(xefa.R);
  if (nullable_init.isNotNull()) {
    xefa.psi = Rcpp::as<arma::vec>(nullable_init);
  } else if(method == "dwls") {
    xefa.manifold = "orth";
    xefa.psi = random_orth(xefa.p, xefa.q+1);
  } else {
    xefa.psi = 1/arma::diagvec(arma::inv_sympd(xefa.R));
  }

  check_efa(xefa, nullable_efa_control);

  // Select one manifold:
  efa_manifold* efa_manifold = choose_efa_manifold(xefa.manifold);
  // Select the factor extraction method:
  efa_criterion* efa_criterion = choose_efa_criterion(xefa.method);

  Rcpp::List efa_result = efa(xefa, efa_manifold, efa_criterion,
                              xefa.random_starts, xefa.cores);

  xefa.heywood = efa_result["heywood"];

  // if(xefa.heywood) {
  //
  //   Rcpp::Rcout << "\n" << std::endl;
  //   Rcpp::warning("Heywood case detected /n Using minimum rank factor analysis");
  //
  //   efa_result = efa(xefa.psi, xefa.R, xefa.q, "minrank", efa_maxit, efa_factr, lmm);
  //
  // }

  double df_null = xefa.p*(xefa.p-1)/2;
  double df = xefa.p*(xefa.p+1)/2 - (xefa.p*xefa.q + xefa.p - xefa.q*(xefa.q-1)/2);

  double f_null;
  if(method == "minres" || method == "pa") {
    f_null = arma::accu(R % R) - xefa.p;
  } else if(method == "ml") {
    f_null = -arma::log_det_sympd(R);
  } else if(method == "minrank") {
    f_null = 0;
  }

  Rcpp::List modelInfo;
  modelInfo["R"] = R;
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
  arma::vec clf_epsilon = {0.01};
  modelInfo["clf_epsilon"] = clf_epsilon;
  modelInfo["w"] = w;
  modelInfo["normalization"] = xefa.normalization;
  modelInfo["Target"] = nullable_Target;
  modelInfo["Weight"] = nullable_Weight;
  modelInfo["PhiTarget"] = nullable_PhiTarget;
  modelInfo["PhiWeight"] = nullable_PhiWeight;
  modelInfo["blocks"] = nullable_blocks;
  modelInfo["block_weights"] = nullable_block_weights;
  modelInfo["oblq_factors"] = nullable_oblq_factors;
  modelInfo["lower"] = xefa.lower;
  modelInfo["upper"] = xefa.upper;

  arma::mat loadings = efa_result["loadings"];

  // Structure of rotation arguments:

  arguments_rotate x;
  x.lambda = loadings;
  x.p = R.n_rows, x.q = nfactors;
  // x.lambda.set_size(x.p, x.q);
  x.Phi.set_size(x.q, x.q); x.Phi.eye();
  x.gamma = gamma, x.epsilon = epsilon, x.k = k, x.w = w, x.clf_epsilon = clf_epsilon;
  x.rotations = rotation;
  x.projection = projection;
  x.nullable_Target = nullable_Target;
  x.nullable_Weight = nullable_Weight;
  x.nullable_PhiTarget = nullable_PhiTarget;
  x.nullable_PhiWeight = nullable_PhiWeight;
  x.nullable_oblq_factors = nullable_oblq_factors;
  x.nullable_block_weights = nullable_block_weights;
  x.nullable_blocks = nullable_blocks;
  x.nullable_rot_control = nullable_rot_control;

  // Check rotation inputs and compute constants for rotation criteria:

  check_rotate(x, random_starts, cores);

  // Select one manifold:
  rotation_manifold* manifold = choose_manifold(x.projection);
  // Select one specific criteria or mixed criteria:
  rotation_criterion* criterion = choose_criterion(x.rotations, x.projection, x.cols_list);

  Rcpp::List rotation_result;

  arma::vec weigths;
  if (xefa.normalization == "kaiser") {

    weigths = sqrt(arma::sum(x.lambda % x.lambda, 1));
    x.lambda.each_col() /= weigths;

  } else if(xefa.normalization == "cureton") {

  } else if(xefa.normalization == "none") {

  } else {

    Rcpp::stop("Unkown normalization");

  }

  if(rotation.size() == 1 && rotation[0] == "none" || random_starts < 1) {

    arma::vec propVar = arma::diagvec(x.lambda.t() * x.lambda)/x.p;
    efa_result["loadings"] = x.lambda;
    efa_result["propVar"] = propVar;

    result["efa"] = efa_result;
    result["modelInfo"] = modelInfo;
    timer.step("elapsed");
    result["elapsed"] = timer;

    result.attr("class") = "efa";
    return result;

  } else {

    rotation_result = rotate_efa(x, manifold, criterion,
                                 random_starts, cores);

  }

  arma::mat L = rotation_result["loadings"];
  arma::mat Phi = rotation_result["Phi"];

  if (xefa.normalization == "kaiser") {

    L.each_col() %= weigths;

  } else if(xefa.normalization == "cureton") {

  }

  // Force average positive loadings in all factors:

  // arma::vec v = arma::sign(arma::sum(L, 0));
  // L.each_row() /= v;

  for (int j=0; j < x.q; ++j) {
    if (arma::sum(L.col(j)) < 0) {
      L.col(j)   *= -1;
      Phi.col(j) *= -1;
      Phi.row(j) *= -1;
    }
  }

  arma::vec propVar = arma::diagvec(Phi * L.t() * L)/x.p;

  rotation_result["loadings"] = L;
  rotation_result["Phi"] = Phi;
  rotation_result["Rhat"] = efa_result["Rhat"];
  rotation_result["uniquenesses"] = efa_result["uniquenesses"];
  rotation_result["Rhat"] = efa_result["Rhat"];
  rotation_result["residuals"] = efa_result["residuals"];
  rotation_result["propVar"] = propVar;

  result["correlation"] = correlation_result;
  result["efa"] = efa_result;
  result["rotation"] = rotation_result;
  result["modelInfo"] = modelInfo;

  timer.step("elapsed");
  result["elapsed"] = timer;

  result.attr("class") = "efa";
  return result;
}

// Do not export this (overloaded to support std::vector<std::string> rotation):
Rcpp::List efast(arma::mat X, int nfactors, std::string cor, std::string method,
                 std::vector<std::string> rotation,
                 std::string projection,
                 Rcpp::Nullable<int> nullable_nobs,
                 Rcpp::Nullable<arma::mat> nullable_Target,
                 Rcpp::Nullable<arma::mat> nullable_Weight,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                 Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> nullable_blocks,
                 Rcpp::Nullable<arma::vec> nullable_block_weights,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_factors,
                 arma::vec gamma, arma::vec epsilon, arma::vec k, double w,
                 int random_starts, int cores,
                 Rcpp::Nullable<arma::vec> nullable_init,
                 Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                 Rcpp::Nullable<Rcpp::List> nullable_rot_control) {

  Rcpp::Timer timer;
  Rcpp::List result;

  Rcpp::List correlation_result;
  arma::mat R;

  if(X.is_square()) {

    R = X;

  } else {

    if(cor == "poly") {
      correlation_result = polyfast(X, "none", false, 0L, cores);
      arma::mat polys = correlation_result["polychorics"];
      R = polys;
    } else if(cor == "pearson") {
      R = arma::cor(X);
      correlation_result["type"] = "pearson";
      correlation_result["correlation"] = R;
    }

  }

  // Create defaults:

  int efa_maxit, lmm;
  double efa_eps, efa_factr;
  arma::vec init;

  // Check EFA inputs:

  // check_efa(R, nfactors, nullable_init, init,
  //           nullable_efa_control,
  //           efa_maxit, lmm, efa_factr);
  //
  // Rcpp::List efa_result = efa(init, R, nfactors, method, efa_maxit, efa_factr, lmm);

  arguments_efa xefa;
  xefa.method = method;
  xefa.R = R;
  xefa.p = R.n_cols;
  xefa.q = nfactors;
  xefa.upper = arma::diagvec(xefa.R);
  if (nullable_init.isNotNull()) {
    xefa.psi = Rcpp::as<arma::vec>(nullable_init);
  } else if(method == "dwls") {
    xefa.psi = random_orth(xefa.p, xefa.q+1);
  } else {
    xefa.psi = 1/arma::diagvec(arma::inv_sympd(xefa.R));
  }

  check_efa(xefa, nullable_efa_control);

  // Select one manifold:
  efa_manifold* efa_manifold = choose_efa_manifold(xefa.manifold);
  // Select the factor extraction method:
  efa_criterion* efa_criterion = choose_efa_criterion(xefa.method);

  Rcpp::List efa_result = efa(xefa, efa_manifold, efa_criterion,
                              xefa.random_starts, xefa.cores);

  xefa.heywood = efa_result["heywood"];

  // if(xefa.heywood) {
  //
  //   Rcpp::Rcout << "\n" << std::endl;
  //   Rcpp::warning("Heywood case detected /n Using minimum rank factor analysis");
  //
  //   efa_result = efa(xefa.psi, xefa.R, xefa.q, "minrank", efa_maxit, efa_factr, lmm);
  //
  // }

  arma::mat loadings = efa_result["loadings"];

  // Structure of rotation arguments:

  arguments_rotate x;
  x.lambda = loadings;
  x.p = R.n_rows, x.q = nfactors;
  // x.lambda.set_size(x.p, x.q);
  x.Phi.set_size(x.q, x.q); x.Phi.eye();
  x.gamma = gamma, x.epsilon = epsilon, x.k = k, x.w = w, x.clf_epsilon = {0.01};
  x.rotations = rotation;
  x.projection = projection;
  x.nullable_Target = nullable_Target;
  x.nullable_Weight = nullable_Weight;
  x.nullable_PhiTarget = nullable_PhiTarget;
  x.nullable_PhiWeight = nullable_PhiWeight;
  x.nullable_oblq_factors = nullable_oblq_factors;
  x.nullable_block_weights = nullable_block_weights;
  x.nullable_blocks = nullable_blocks;
  x.nullable_rot_control = nullable_rot_control;

  // Check inputs and compute constants for rotation criteria:

  check_rotate(x, random_starts, cores);

  // Model Info:

  double df_null = x.p*(x.p-1)/2;
  double df = x.p*(x.p+1)/2 - (x.p*x.q + x.p - x.q*(x.q-1)/2);

  double f_null;
  if(method == "minres" || method == "pa") {
    f_null = arma::accu(R % R) - x.p;
  } else if(method == "ml") {
    f_null = -arma::log_det_sympd(R);
  } else if(method == "minrank") {
    f_null = 0;
  }

  Rcpp::List modelInfo;
  modelInfo["R"] = R;
  modelInfo["method"] = method;
  modelInfo["projection"] = x.projection;
  modelInfo["rotation"] = x.rotations;
  modelInfo["n_vars"] = R.n_cols;
  modelInfo["nfactors"] = nfactors;
  modelInfo["nobs"] = nullable_nobs;
  modelInfo["df"] = df;
  modelInfo["df_null"] = df_null;
  modelInfo["f_null"] = f_null;
  modelInfo["k"] = x.k;
  modelInfo["gamma"] = x.gamma;
  modelInfo["epsilon"] = x.epsilon;
  modelInfo["clf_epsilon"] = x.clf_epsilon;
  modelInfo["w"] = x.w;
  modelInfo["normalization"] = x.normalization;
  modelInfo["Target"] = x.nullable_Target;
  modelInfo["Weight"] = x.nullable_Weight;
  modelInfo["PhiTarget"] = x.nullable_PhiTarget;
  modelInfo["PhiWeight"] = x.nullable_PhiWeight;
  modelInfo["blocks"] = x.nullable_blocks;
  modelInfo["block_weights"] = x.nullable_block_weights;
  modelInfo["oblq_factors"] = x.nullable_oblq_factors;
  modelInfo["lower"] = xefa.lower;
  modelInfo["upper"] = xefa.upper;

  // Select one manifold:
  rotation_manifold* manifold = choose_manifold(x.projection);
  // Select one specific criteria or mixed criteria:
  rotation_criterion* criterion = choose_criterion(x.rotations, x.projection, x.cols_list);

  Rcpp::List rotation_result;

  arma::vec weigths;
  if (xefa.normalization == "kaiser") {

    weigths = sqrt(arma::sum(x.lambda % x.lambda, 1));
    x.lambda.each_col() /= weigths;

  } else if(xefa.normalization == "cureton") {

  } else if(xefa.normalization == "none") {

  } else {

    Rcpp::stop("Unkown normalization");

  }

  if(rotation.size() == 1 && rotation[0] == "none" || random_starts < 1) {

    arma::vec propVar = arma::diagvec(x.lambda.t() * x.lambda)/x.p;
    efa_result["loadings"] = x.lambda;
    efa_result["propVar"] = propVar;

    result["efa"] = efa_result;
    result["modelInfo"] = modelInfo;
    timer.step("elapsed");
    result["elapsed"] = timer;

    result.attr("class") = "efa";
    return result;

  } else {

    rotation_result = rotate_efa(x, manifold, criterion,
                                 random_starts, cores);

  }

  arma::mat L = rotation_result["loadings"];
  arma::mat Phi = rotation_result["Phi"];

  if (xefa.normalization == "kaiser") {

    L.each_col() %= weigths;

  } else if(xefa.normalization == "cureton") {

  }

  // Force average positive loadings in all factors:

  // arma::vec v = arma::sign(arma::sum(L, 0));
  // L.each_row() /= v;

  for (int j=0; j < x.q; ++j) {
    if (arma::sum(L.col(j)) < 0) {
      L.col(j)   *= -1;
      Phi.col(j) *= -1;
      Phi.row(j) *= -1;
    }
  }

  arma::vec propVar = arma::diagvec(Phi * L.t() * L)/x.p;

  rotation_result["loadings"] = L;
  rotation_result["Phi"] = Phi;
  rotation_result["Rhat"] = efa_result["Rhat"];
  rotation_result["uniquenesses"] = efa_result["uniquenesses"];
  rotation_result["Rhat"] = efa_result["Rhat"];
  rotation_result["residuals"] = efa_result["residuals"];
  rotation_result["propVar"] = propVar;

  result["correlation"] = correlation_result;
  result["efa"] = efa_result;
  result["rotation"] = rotation_result;
  result["modelInfo"] = modelInfo;

  timer.step("elapsed");
  result["elapsed"] = timer;

  result.attr("class") = "efa";
  return result;
}
