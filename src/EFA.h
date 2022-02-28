#include "multiple_rotations.h"
#include "EFA_fit.h"

void check_efa(arma::mat R, int n_factors, Rcpp::Nullable<arma::vec> nullable_init,
               arma::vec& init, Rcpp::Nullable<Rcpp::List> nullable_efa_control,
               int& efa_maxit, int& lmm, double& efa_factr) {

  if(R.n_rows < n_factors) Rcpp::stop("Too many factors");

  if (nullable_init.isNotNull()) {
    init = Rcpp::as<arma::vec>(nullable_init);
  } else {
    init = 1/arma::diagvec(arma::inv_sympd(R));
  }

  Rcpp::List efa_control;

  if (nullable_efa_control.isNotNull()) {
    efa_control = Rcpp::as<Rcpp::List>(efa_control);
  }
  if(efa_control.containsElementNamed("maxit") ){
    efa_maxit = efa_control["maxit"];
  } else {
    efa_maxit = 1e3;
  }
  if(efa_control.containsElementNamed("factr")) {
    efa_factr = efa_control["factr"];
  } else {
    efa_factr = 1e07;
  }
  if(efa_control.containsElementNamed("lmm")) {
    lmm = efa_control["lmm"];
  } else {
    lmm = 5;
  }

}

Rcpp::List efa(arma::vec psi, arma::mat R, int n_factors, std::string method,
               int efa_max_iter, double efa_factr, int lmm) {

  Rcpp::List result;

  arma::mat w, Rhat;
  arma::vec uniquenesses;

  int iteration = 0;

  if (method == "minres") {

    Rcpp::List optim_result = optim_rcpp(psi, R, n_factors, method, efa_max_iter, efa_factr, lmm);

    arma::vec psi_temp = optim_result["par"];
    psi = psi_temp;
    arma::mat reduced_R = R - diagmat(psi);

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

    Rhat = ww;
    Rhat.diag().ones();

    bool convergence = false;
    int convergence_result = optim_result["convergence"];

    if(convergence_result == 0) convergence = true;

    result["f"] = optim_result["value"];
    result["convergence"] = convergence;

  } else if (method == "ml") {

    Rcpp::List optim_result = optim_rcpp(psi, R, n_factors, method, efa_max_iter, efa_factr, lmm);
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

    arma::mat A = eigvec2(arma::span::all, arma::span(0, n_factors-1));
    arma::vec eigenvalues = eigval2(arma::span(0, n_factors-1)) - 1;
    for(int i=0; i < n_factors; ++i) {
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

    Rcpp::List pa_result = principal_axis(psi, R, n_factors, 1e-03, efa_max_iter);

    arma::mat w_temp = pa_result["loadings"];
    arma::vec uniquenesses_temp = pa_result["uniquenesses"];
    arma::mat Rhat_temp = pa_result["Rhat"];

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

    arma::mat A = eigvec2(arma::span::all, arma::span(0, n_factors-1));
    arma::vec eigenvalues = eigval2(arma::span(0, n_factors-1));
    for(int i=0; i < n_factors; ++i) {
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

Rcpp::List rotate_efa(arguments x, base_manifold *manifold, base_criterion *criterion,
                      int random_starts, int cores, double eps, int maxit) {

  arma::vec xf(random_starts);
  TRN x1;
  std::vector<TRN> x2(random_starts);

  // Perform multiple rotations with random starting values:

  omp_set_num_threads(cores);
#pragma omp parallel for
  for (int i=0; i < random_starts; ++i) {

    arguments args = x;
    args.T = random_orth(args.q, args.q);

    x2[i] = NPF(args, manifold, criterion, eps, maxit);

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

Rcpp::List efast(arma::mat R, int n_factors, std::string method,
                 Rcpp::CharacterVector char_rotation,
                 std::string projection,
                 Rcpp::Nullable<arma::mat> nullable_Target,
                 Rcpp::Nullable<arma::mat> nullable_Weight,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                 Rcpp::Nullable<arma::uvec> nullable_blocks,
                 Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list,
                 Rcpp::Nullable<arma::vec> nullable_block_weights,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                 bool normalize, std::string penalization,
                 double gamma, double epsilon, double k, double w, double alpha,
                 int random_starts, int cores,
                 Rcpp::Nullable<arma::vec> nullable_init,
                 Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                 Rcpp::Nullable<Rcpp::List> nullable_rot_control) {

  Rcpp::Timer timer;

  std::vector<std::string> rotation = Rcpp::as<std::vector<std::string>>(char_rotation);

  // Create defaults:

  int rot_maxit, efa_maxit, lmm;
  double rot_eps, efa_eps, efa_factr;
  arma::vec init;

  // Structure of arguments:

  arguments x;
  x.p = R.n_rows, x.q = n_factors;
  x.lambda.set_size(x.p, x.q);
  x.Phi.set_size(x.q, x.q); x.Phi.eye();
  x.gamma = gamma, x.epsilon = epsilon, x.k = k, x.w = w, x.a = alpha;
  x.penalization = penalization;
  x.rotations = rotation;
  x.projection = projection;

  // Check inputs and compute constants for rotation criteria:

  check_rotate(x,
               nullable_Target, nullable_Weight,
               nullable_PhiTarget, nullable_PhiWeight,
               nullable_blocks,
               nullable_blocks_list,
               nullable_block_weights,
               nullable_oblq_blocks,
               nullable_rot_control,
               rot_maxit, rot_eps,
               random_starts, cores);

  check_efa(R, n_factors, nullable_init, init,
            nullable_efa_control,
            efa_maxit, lmm, efa_factr);

  // Select one manifold:
  base_manifold* manifold = choose_manifold(x.projection);
  // Select one specific criteria or mixed criteria:
  base_criterion* criterion = choose_criterion(x.rotations, x.projection, x.blocks_list);

  Rcpp::List result;

  Rcpp::List efa_result = efa(init, R, n_factors, method, efa_maxit, efa_factr, lmm);

  bool heywood = efa_result["Heywood"];

  if(heywood) {

    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::warning("Heywood case detected /n Using minimum rank factor analysis");

    efa_result = efa(init, R, n_factors, "minrank", efa_maxit, efa_factr, lmm);

  }

  efa_result["Heywood"] = heywood;
  arma::mat loadings = efa_result["loadings"];
  x.lambda = loadings;

  Rcpp::List rotation_result;

  arma::vec weigths;
  if (normalize) {

    weigths = sqrt(sum(loadings % loadings, 1));
    loadings.each_col() /= weigths;

  }

  if(rotation.size() == 1 && rotation[0] == "none" || random_starts < 1) {

    return efa_result;

  } else {

    rotation_result = rotate_efa(x, manifold, criterion,
                                 random_starts, cores, rot_eps, rot_maxit);

  }

  arma::mat L = rotation_result["loadings"];
  arma::mat Phi = rotation_result["Phi"];

  if (normalize) {

    L.each_col() %= weigths;

  }

  arma::mat Rhat = L * Phi * L.t();
  rotation_result["loadings"] = L;
  rotation_result["Phi"] = Phi;

  arma::vec uniquenesses = 1 - arma::diagvec(Rhat);

  rotation_result["uniquenesses"] = uniquenesses;

  Rhat.diag().ones();
  rotation_result["Rhat"] = Rhat;
  rotation_result["residuals"] = R - Rhat;

  result["efa"] = efa_result;
  result["rotation"] = rotation_result;

  Rcpp::List modelInfo;
  modelInfo["method"] = method;
  modelInfo["projection"] = projection;
  modelInfo["rotation"] = rotation;
  modelInfo["k"] = k;
  modelInfo["gamma"] = gamma;
  modelInfo["epsilon"] = epsilon;
  modelInfo["w"] = w;
  modelInfo["alpha"] = alpha;
  modelInfo["normalize"] = normalize;
  modelInfo["penalization"] = x.penalization;
  modelInfo["R"] = R;
  modelInfo["Target"] = nullable_Target;
  modelInfo["Weight"] = nullable_Weight;
  modelInfo["PhiTarget"] = nullable_PhiTarget;
  modelInfo["PhiWeight"] = nullable_PhiWeight;
  modelInfo["blocks"] = nullable_blocks;
  modelInfo["blocks_list"] = nullable_blocks_list;
  modelInfo["block_weights"] = nullable_block_weights;
  modelInfo["oblq_blocks"] = nullable_oblq_blocks;
  result["modelInfo"] = modelInfo;

  timer.step("elapsed");

  result["elapsed"] = timer;

  result.attr("class") = "efa";
  return result;
}

// Do not export this (overloaded to support std::vector<std::string> rotation):
Rcpp::List efast(arma::mat R, int n_factors, std::string method,
                 std::vector<std::string> rotation,
                 std::string projection,
                 Rcpp::Nullable<arma::mat> nullable_Target,
                 Rcpp::Nullable<arma::mat> nullable_Weight,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                 Rcpp::Nullable<arma::uvec> nullable_blocks,
                 Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list,
                 Rcpp::Nullable<arma::vec> nullable_block_weights,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                 bool normalize, std::string penalization,
                 double gamma, double epsilon, double k, double w, double alpha,
                 int random_starts, int cores,
                 Rcpp::Nullable<arma::vec> nullable_init,
                 Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                 Rcpp::Nullable<Rcpp::List> nullable_rot_control) {

  Rcpp::Timer timer;

  // Create defaults:

  int rot_maxit, efa_maxit, lmm;
  double rot_eps, efa_eps, efa_factr;
  arma::vec init;

  // Structure of arguments:

  arguments x;
  x.p = R.n_rows, x.q = n_factors;
  x.lambda.set_size(x.p, x.q);
  x.Phi.set_size(x.q, x.q); x.Phi.eye();
  x.gamma = gamma, x.epsilon = epsilon, x.k = k, x.w = w, x.a = alpha;
  x.penalization = penalization;
  x.rotations = rotation;
  x.projection = projection;

  // Check inputs and compute constants for rotation criteria:

  check_rotate(x,
               nullable_Target, nullable_Weight,
               nullable_PhiTarget, nullable_PhiWeight,
               nullable_blocks,
               nullable_blocks_list,
               nullable_block_weights,
               nullable_oblq_blocks,
               nullable_rot_control,
               rot_maxit, rot_eps,
               random_starts, cores);

  check_efa(R, n_factors, nullable_init, init,
            nullable_efa_control,
            efa_maxit, lmm, efa_factr);

  // Select one manifold:
  base_manifold* manifold = choose_manifold(x.projection);
  // Select one specific criteria or mixed criteria:
  base_criterion* criterion = choose_criterion(x.rotations, x.projection, x.blocks_list);

  Rcpp::List result;

  Rcpp::List efa_result = efa(init, R, n_factors, method, efa_maxit, efa_factr, lmm);

  bool heywood = efa_result["Heywood"];

  if(heywood) {

    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::warning("Heywood case detected /n Using minimum rank factor analysis");

    efa_result = efa(init, R, n_factors, "minrank", efa_maxit, efa_factr, lmm);

  }

  efa_result["Heywood"] = heywood;
  arma::mat loadings = efa_result["loadings"];
  x.lambda = loadings;

  Rcpp::List rotation_result;

  arma::vec weigths;
  if (normalize) {

    weigths = sqrt(sum(loadings % loadings, 1));
    loadings.each_col() /= weigths;

  }

  if(rotation.size() == 1 && rotation[0] == "none" || random_starts < 1) {

    return efa_result;

  } else {

    rotation_result = rotate_efa(x, manifold, criterion,
                                 random_starts, cores, rot_eps, rot_maxit);

  }

  arma::mat L = rotation_result["loadings"];
  arma::mat Phi = rotation_result["Phi"];

  if (normalize) {

    L.each_col() %= weigths;

  }

  arma::mat Rhat = L * Phi * L.t();
  rotation_result["loadings"] = L;
  rotation_result["Phi"] = Phi;

  arma::vec uniquenesses = 1 - arma::diagvec(Rhat);

  rotation_result["uniquenesses"] = uniquenesses;

  Rhat.diag().ones();
  rotation_result["Rhat"] = Rhat;
  rotation_result["residuals"] = R - Rhat;

  result["efa"] = efa_result;
  result["rotation"] = rotation_result;

  Rcpp::List modelInfo;
  modelInfo["method"] = method;
  modelInfo["projection"] = projection;
  modelInfo["rotation"] = rotation;
  modelInfo["k"] = k;
  modelInfo["gamma"] = gamma;
  modelInfo["epsilon"] = epsilon;
  modelInfo["w"] = w;
  modelInfo["alpha"] = alpha;
  modelInfo["normalize"] = normalize;
  modelInfo["penalization"] = x.penalization;
  modelInfo["R"] = R;
  modelInfo["Target"] = nullable_Target;
  modelInfo["Weight"] = nullable_Weight;
  modelInfo["PhiTarget"] = nullable_PhiTarget;
  modelInfo["PhiWeight"] = nullable_PhiWeight;
  modelInfo["blocks"] = nullable_blocks;
  modelInfo["blocks_list"] = nullable_blocks_list;
  modelInfo["block_weights"] = nullable_block_weights;
  modelInfo["oblq_blocks"] = nullable_oblq_blocks;
  result["modelInfo"] = modelInfo;

  timer.step("elapsed");

  result["elapsed"] = timer;

  result.attr("class") = "efa";
  return result;
}


