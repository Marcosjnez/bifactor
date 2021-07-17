#include <Rcpp/Benchmark/Timer.h>
#include "GPF.h"
#include "NPF.h"

base_manifold* choose_manifold(std::string projection) {

  base_manifold* manifold;
  if(projection == "orth") {
    manifold = new orth();
  } else if(projection == "oblq") {
    manifold = new oblq();
  } else if(projection == "poblq") {
    manifold = new poblq();
  } else if(projection == "none") {

  } else {

    Rcpp::stop("Available projections: \n orth, oblq, poblq");

  }

  return manifold;

}

base_criterion* choose_criterion(std::string rotation, std::string projection) {

  base_criterion *criterion;

  if (rotation == "target") {

    if(projection == "orth") {

      criterion = new targetT();

    } else {

      criterion = new targetQ();

    }

  }  else if(rotation == "xtarget") {

    if(projection == "orth") {

      criterion = new xtargetT();

    } else {

      criterion = new xtargetQ();

    }

  } else if(rotation == "cf") {

    if(projection == "orth") {

      criterion = new cfT();

    } else {

      criterion = new cfQ();

    }

  } else if(rotation == "oblimin") {

    if(projection == "orth") {

      criterion = new obliminT();

    } else {

      criterion = new obliminQ();

    }

  } else if(rotation == "geomin") {

    if(projection == "orth") {

      criterion = new geominT();

    } else {

      criterion = new geominQ();

    }

  } else if(rotation == "none") {

  } else {

    Rcpp::stop("Available rotations: \n target, xtarget, cf, oblimin, geomin");

  }

  return criterion;

}

arma::mat zeros(arma::mat X, std::vector<arma::uvec> indexes) {

  int I = indexes.size();
  arma::mat X0 = X;

  for(int i=0; i < I; ++i) {

    X0(indexes[i], indexes[i]).zeros();

  }

  X0.diag() = X.diag();

  return X0;

}

void check_rotate(std::string rotation, std::string projection,
                  int n, int n_factors,
                  Rcpp::Nullable<arma::mat> nullable_Target,
                  Rcpp::Nullable<arma::mat> nullable_Weight,
                  Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                  Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                  arma::mat loadings, arma::mat Phi,
                  arma::mat& Target, arma::mat& Weight,
                  arma::mat& PhiTarget, arma::mat& PhiWeight,
                  arma::mat& Weight2, arma::mat& PhiWeight2,
                  double gamma, double epsilon, double k, double w,
                  arma::mat& I_gamma_C, arma::mat& N, arma::mat& M, double& p2,
                  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                  std::vector<arma::uvec>& list_oblq_blocks, arma::uvec& oblq_blocks,
                  Rcpp::Nullable<Rcpp::List> nullable_rot_control,
                  int& rot_maxit, double& rot_eps,
                  int random_starts, int cores) {

  // Check partially oblique projection:

  if(projection == "poblq") {

    if(nullable_oblq_blocks.isNull()) {

      Rcpp::stop("Please, provide a vector with the number of factors in each oblique block via the oblq_blocks argument");

    } else {

      oblq_blocks = Rcpp::as<arma::uvec>(nullable_oblq_blocks);
      if(arma::accu(oblq_blocks) > n_factors) Rcpp::stop("To many factors declared in oblq_blocks");
      list_oblq_blocks = vector_to_list(oblq_blocks);

      for(int i=0; i < list_oblq_blocks.size(); i++) list_oblq_blocks[i] -= 1;
      arma::mat X(n_factors, n_factors, arma::fill::ones);
      arma::mat Q = zeros(X, list_oblq_blocks);
      oblq_blocks = arma::find(Q == 0);

    }

  }

  // Check criteria:

  if(rotation == "target") {

    if (nullable_Target.isNotNull()) {
      Target = Rcpp::as<arma::mat>(nullable_Target);
    } else {
      Rcpp::stop("Provide a Target for target rotation");
    }
    if (nullable_Weight.isNotNull()) {
      Weight = Rcpp::as<arma::mat>(nullable_Weight);
    } else {
      Weight = 1 - Target;
    }

    if(arma::size(Target) != arma::size(loadings) ||
       arma::size(Weight) != arma::size(loadings)) {

      Rcpp::stop("Incompatible Target or Weight dimensions");

    }

    Weight2 = Weight % Weight;

  } else if(rotation == "xtarget") {

    if(w < 0) Rcpp::stop("w must be nonnegative");

    if (nullable_Target.isNotNull()) {
      Target = Rcpp::as<arma::mat>(nullable_Target);
    } else {
      Rcpp::stop("Provide a Target for xtarget rotation");
    }
    if (nullable_PhiTarget.isNotNull()) {
      PhiTarget = Rcpp::as<arma::mat>(nullable_PhiTarget);
    } else {
      Rcpp::stop("Provide a PhiTarget for xtarget rotation");
    }
    if (nullable_Weight.isNotNull()) {
      Weight = Rcpp::as<arma::mat>(nullable_Weight);
    } else {
      Weight = 1 - Target;
    }
    if (nullable_PhiWeight.isNotNull()) {
      PhiWeight = Rcpp::as<arma::mat>(nullable_PhiWeight);
    } else {
      PhiWeight = 1 - PhiTarget;
    }

    if(arma::size(Target) != arma::size(loadings) ||
       arma::size(Weight) != arma::size(loadings) ||
       arma::size(PhiTarget) != arma::size(Phi) ||
       arma::size(PhiWeight) != arma::size(Phi)) {

      Rcpp::stop("Incompatible Target, PhiTarget, Weight or PhiWeight dimensions");

    }

    Weight2 = Weight % Weight;
    PhiWeight2 = PhiWeight % PhiWeight;

  } else if(rotation == "geomin") {

    if(epsilon <= 0) {

      Rcpp::stop("epsilon must be greater than 0");

    }

    p2 = 2/(n_factors + 0.0);

  } else if(rotation == "oblimin") {

    if(gamma < 0) {

      Rcpp::stop("gamma must be nonnegative");

    }

    N.set_size(n_factors, n_factors); N.ones();
    N.diag(0).zeros();
    arma::mat I(n, n, arma::fill::eye), gamma_C(n, n, arma::fill::ones);
    double gamma_n = gamma/n;
    gamma_C *= gamma_n;
    I_gamma_C = (I - gamma_C);

  } else if(rotation == "cf") {

    if(k < 0 || k > 1) {

      Rcpp::stop("k must be a scalar between 0 and 1");

    }

    N.set_size(n_factors, n_factors); N.ones();
    N.diag(0).zeros();
    M.set_size(n, n); M.ones();
    M.diag(0).zeros();

  } else if(rotation == "none") {

  } else {

    Rcpp::stop("Available rotations: \n target, xtarget, cf, oblimin, geomin, varimax");

  }

  // Check rotation parameters:

  Rcpp::List rot_control;

  if (nullable_rot_control.isNotNull()) {

    rot_control = Rcpp::as<Rcpp::List>(nullable_rot_control);

  }
  if(rot_control.containsElementNamed("maxit")) {

    rot_maxit = rot_control["maxit"];

  } else {

    rot_maxit = 1e04;

  }
  if(rot_control.containsElementNamed("eps")) {

    rot_eps = rot_control["eps"];

  } else {

    rot_eps = 1e-05;

  }

  if(random_starts < 0) Rcpp::stop("random_starts must be nonnegative");
  if(cores < 1) Rcpp::stop("The number of cores must be a positive integer");

}

Rcpp::List rotate(arma::mat loadings, std::string rotation, std::string projection,
                  Rcpp::Nullable<arma::mat> nullable_Target,
                  Rcpp::Nullable<arma::mat> nullable_Weight,
                  Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                  Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                  double gamma, double epsilon, double k, double w,
                  int random_starts, int cores,
                  Rcpp::Nullable<Rcpp::List> rot_control) {

  Rcpp::Timer timer;

  int n = loadings.n_rows;
  int n_factors = loadings.n_cols;

  // Create defaults:

  arma::mat Target, Weight, PhiTarget, PhiWeight;
  std::vector<arma::uvec> list_oblq_blocks;
  arma::uvec oblq_blocks;

  int maxit;
  double eps;

  // Constants for rotation criteria:

  arma::mat empty_loadings(n, n_factors), empty_Phi(n_factors, n_factors),
  Weight2, PhiWeight2, I_gamma_C, N, M;

  double p2;

  // Check inputs and compute constants for rotation criteria:

  check_rotate(rotation, projection,
               n, n_factors,
               nullable_Target, nullable_Weight,
               nullable_PhiTarget, nullable_PhiWeight,
               empty_loadings, empty_Phi,
               Target, Weight, PhiTarget, PhiWeight,
               Weight2, PhiWeight2,
               gamma, epsilon, k, w,
               I_gamma_C, N, M, p2, // Constants
               nullable_oblq_blocks, list_oblq_blocks, oblq_blocks,
               rot_control, maxit, eps,
               random_starts, cores);

  base_manifold* manifold = choose_manifold(projection);
  base_criterion *criterion = choose_criterion(rotation, projection);

  arma::vec xf(random_starts);
  TRN x;
  std::vector<TRN> x2(random_starts);

  // Perform multiple rotations with random starting values:

  omp_set_num_threads(cores);
#pragma omp parallel for
  for (int i=0; i < random_starts; ++i) {

    arma::mat T = random_orth(n_factors, n_factors);

    x2[i] = NPF(manifold, criterion, T, loadings,
                Target, Weight,
                PhiTarget, PhiWeight,
                list_oblq_blocks, oblq_blocks,
                w, gamma, epsilon,
                eps, maxit,
                Weight2, PhiWeight2, I_gamma_C, N, M, p2, k);

    xf[i] = std::get<3>(x2[i]);

  }

  // Choose the rotation with the smallest objective value:

  arma::uword index_minimum = index_min(xf);
  x = x2[index_minimum];

  arma::mat L = std::get<0>(x);
  arma::mat Phi = std::get<1>(x);
  if(Phi.is_empty()) {Phi.set_size(n_factors, n_factors); Phi.eye();}
  arma::mat T = std::get<2>(x);
  double f = std::get<3>(x);
  int iterations = std::get<4>(x);
  bool convergence = std::get<5>(x);

  // Force average positive loadings in all factors:

  // arma::vec v = arma::sign(arma::sum(L, 0));
  // L.each_row() /= v;

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

  timer.step("elapsed");

  result["elapsed"] = timer;

  result.attr("class") = "rotation";
  return result;

}
