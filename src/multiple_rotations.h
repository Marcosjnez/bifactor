#include <Rcpp/Benchmark/Timer.h>
#include "NPF.h"

// Choose the manifold:

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

// Choose the rotation criteria:

base_criterion* choose_criterion(std::vector<std::string> rotations, std::string projection,
                                 std::vector<arma::uvec> blocks_list) {

  base_criterion *criterion;

  if(!blocks_list.empty()) {

    // Rcpp::stop("Mixed rotation criteria not supported yet");
    criterion = new mixed();

  } else if (rotations[0] == "target") {

    criterion = new target();

  }  else if(rotations[0] == "xtarget") {

    if(projection == "orth") {
      criterion = new target();
    } else {
      criterion = new xtarget();
    }

  } else if(rotations[0] == "cf") {

      criterion = new cf();

  } else if(rotations[0] == "oblimin") {

      criterion = new oblimin();

  } else if(rotations[0] == "geomin") {

      criterion = new geomin();

  } else if(rotations[0] == "varimax") {

      criterion = new varimax();

  } else if(rotations[0] == "varimin") {

      criterion = new varimin();

  } else if(rotations[0] == "none") {

  } else {

    Rcpp::stop("Available rotations: \n cf, oblimin, geomin, varimax, varimin, target, xtarget");

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

arma::vec new_k(std::vector<std::string> x, std::string y, arma::vec k) {

  // Resize gamma, k, epsilon and w

  int k_size = k.size();
  int x_size = x.size();

  std::vector<int> id; // Setup storage for found IDs

  for(int i =0; i < x.size(); i++) // Loop through input
    if(x[i] == y) {// check if input matches target
      id.push_back(i);
    }

    arma::uvec indexes = arma::conv_to<arma::uvec>::from(id);
    std::vector<double> ks = arma::conv_to<std::vector<double>>::from(k);

    if(k_size < x_size) {
      ks.resize(x_size, k.back());
      k = arma::conv_to<arma::vec>::from(ks);
      k(indexes) = k(arma::span(0, indexes.size()-1));
    }

    return k; // send locations to R (c++ index shift!)
}

void check_rotate(arguments& x,
                  Rcpp::Nullable<arma::mat> nullable_Target,
                  Rcpp::Nullable<arma::mat> nullable_Weight,
                  Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                  Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                  Rcpp::Nullable<arma::uvec> nullable_blocks,
                  Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list,
                  Rcpp::Nullable<arma::vec> nullable_block_weights,
                  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                  Rcpp::Nullable<Rcpp::List> nullable_rot_control,
                  int& rot_maxit, double& rot_eps,
                  int random_starts, int cores) {

  x.n_rotations = x.rotations.size();

  // Create a list of column indexes for each block of factors:

  if(nullable_blocks.isNotNull() || nullable_blocks_list.isNotNull()) {

    if(nullable_blocks_list.isNull()) {
      x.blocks_vector = Rcpp::as<arma::uvec>(nullable_blocks);
      if(arma::accu(x.blocks_vector) > x.q) Rcpp::stop("To many factors declared in blocks");

      x.blocks_list = vector_to_list2(x.blocks_vector);
    } else {
      x.blocks_list = Rcpp::as<std::vector<arma::uvec>>(nullable_blocks_list);
      arma::uvec v_blocks = list_to_vector(x.blocks_list);
      if(v_blocks.max() > x.q || v_blocks.min() < 1) {
        Rcpp::stop("blocks_list must contain values between 1 and the number of factors");
      }
      for(int i=0; i < x.blocks_list.size(); ++i) x.blocks_list[i] -= 1;
    }

    x.n_blocks = x.blocks_list.size();

    if(nullable_block_weights.isNull()) {

      x.block_weights.set_size(x.n_blocks);
      x.block_weights.ones();

    } else {

      x.block_weights = Rcpp::as<arma::vec>(nullable_block_weights);
      if(x.block_weights.size() != x.n_blocks) {
        Rcpp::stop("The vector of weights must equal the number of blocks");
      }

    }

    x.Li.resize(x.n_blocks), x.Li2.resize(x.n_blocks), x.Ni.resize(x.n_blocks),
    x.HLi2.resize(x.n_blocks), x.LoLi2.resize(x.n_blocks), x.termi.resize(x.n_blocks),
    x.IgCL2Ni.resize(x.n_blocks), x.f1i.resize(x.n_blocks), x.Weighti.resize(x.n_blocks),
    x.Targeti.resize(x.n_blocks);

  // Resize rotation to match the length of blocks:

    if(x.n_rotations == 1) {

      x.rotations.resize(x.n_blocks, x.rotations[0]);

    } else {

      if(x.n_rotations != x.n_blocks) Rcpp::stop("The number of rotation criteria and blocks does not match. \n Provide either one rotation criteria for all the blocks or one for each block");

    }

  } else {

    if(x.n_rotations > 1) Rcpp::stop("Multiple rotation criteria but no blocks were specified \n Provide one block for each rotation criteria");

  }

  // Check inputs for partially oblique projection:

  if(x.projection == "poblq") {

    if(nullable_oblq_blocks.isNull()) {

      Rcpp::stop("Please, provide a vector with the number of factors in each oblique block in the oblq_blocks argument");

    } else {

      // Create indexes to fix to zero the entries for the nonconstrained correlations:

      x.oblq_indexes = Rcpp::as<arma::uvec>(nullable_oblq_blocks);
      if(arma::accu(x.oblq_indexes) > x.q) Rcpp::stop("To many factors declared in oblq_blocks");
      x.list_oblq_indexes = vector_to_list2(x.oblq_indexes);

      arma::mat X(x.q, x.q, arma::fill::ones);
      arma::mat Q = zeros(X, x.list_oblq_indexes);
      x.oblq_indexes = arma::find(Q == 0);

    }

  }

  // Check and build defaults for each rotation criteria:

  std::vector<std::string> all_rotations = {"cf", "oblimin", "geomin", "target",
                                            "xtarget", "varimax", "varimin", "none"};

  for (auto i: x.rotations) {
    if (std::find(all_rotations.begin(), all_rotations.end(), i) == all_rotations.end()) {
      Rcpp::stop("Available rotations: \n cf, oblimin, geomin, varimax, varimin, target, xtarget");
    }
  }

  if(std::find(x.rotations.begin(), x.rotations.end(), "target") != x.rotations.end()) {

    if (nullable_Target.isNotNull()) {
      x.Target = Rcpp::as<arma::mat>(nullable_Target);
    } else {
      Rcpp::stop("Provide a Target for target rotation");
    }
    if (nullable_Weight.isNotNull()) {
      x.Weight = Rcpp::as<arma::mat>(nullable_Weight);
    } else {
      x.Weight = 1 - x.Target;
    }

    if(arma::size(x.Target) != arma::size(x.lambda) ||
       arma::size(x.Weight) != arma::size(x.lambda)) {

      Rcpp::stop("Incompatible Target and Weight dimensions");

    }

    x.Weight2 = x.Weight % x.Weight;

  }

  if(std::find(x.rotations.begin(), x.rotations.end(), "xtarget") != x.rotations.end()) {

    if(x.w < 0) Rcpp::stop("w must be nonnegative");

    if (nullable_Target.isNotNull()) {
      x.Target = Rcpp::as<arma::mat>(nullable_Target);
    } else {
      Rcpp::stop("Provide a Target for xtarget rotation");
    }
    if (nullable_PhiTarget.isNotNull()) {
      x.Phi_Target = Rcpp::as<arma::mat>(nullable_PhiTarget);
    } else {
      Rcpp::stop("Provide a PhiTarget for xtarget rotation");
    }
    if (nullable_Weight.isNotNull()) {
      x.Weight = Rcpp::as<arma::mat>(nullable_Weight);
    } else {
      x.Weight = 1-x.Target;
    }
    if (nullable_PhiWeight.isNotNull()) {
      x.Phi_Weight = Rcpp::as<arma::mat>(nullable_PhiWeight);
    } else {
      x.Phi_Weight = 1-x.Phi_Target;
    }

    if(arma::size(x.Target) != arma::size(x.lambda) ||
       arma::size(x.Weight) != arma::size(x.lambda) ||
       arma::size(x.Phi_Target) != arma::size(x.Phi) ||
       arma::size(x.Phi_Weight) != arma::size(x.Phi)) {

      Rcpp::stop("Incompatible dimensions between Target, PhiTarget, Weight or PhiWeight");

    }

    x.Phi_Weight.diag().zeros();
    x.Weight2 = x.Weight % x.Weight;
    x.Phi_Weight2 = x.Phi_Weight % x.Phi_Weight;

  }
  if(std::find(x.rotations.begin(), x.rotations.end(), "geomin") != x.rotations.end()) {

    if(x.epsilon.min() <= 0) {

      Rcpp::stop("epsilon must be positive");

    }

    x.epsilon = new_k(x.rotations, "geomin", x.epsilon);

  }
  if(std::find(x.rotations.begin(), x.rotations.end(), "oblimin") != x.rotations.end()) {

    if(x.gamma < 0) {

      Rcpp::stop("gamma must be nonnegative");

    }

    x.N.set_size(x.q, x.q); x.N.ones();
    x.N.diag(0).zeros();
    arma::mat I(x.p, x.p, arma::fill::eye), gamma_C(x.p, x.p, arma::fill::ones);
    double gamma_n = x.gamma/x.p;
    gamma_C *= gamma_n;
    x.I_gamma_C = (I - gamma_C);

  }
  if(std::find(x.rotations.begin(), x.rotations.end(), "cf") != x.rotations.end()) {

    x.N.set_size(x.q, x.q); x.N.ones();
    x.N.diag(0).zeros();
    x.M.set_size(x.p, x.p); x.M.ones();
    x.M.diag(0).zeros();

    x.k = new_k(x.rotations, "cf", x.k);

  }
  if(std::find(x.rotations.begin(), x.rotations.end(), "varimax") != x.rotations.end()) {

    if(x.projection == "oblq" | x.projection == "poblq") {
      Rcpp::warning("Usually, the varimax criterion does not converge with (partially) oblique projection. \n Consider using cf with k = 1/(number of items), which is equivalent to varimax for orthogonal rotation but also converges with (partially) oblique projection.");
    }

    arma::vec v(x.p, arma::fill::ones);
    arma::mat I(x.p, x.p, arma::fill::eye);
    x.H = I - v * v.t() / (x.p + 0.0); // Centering matrix

  }
  if(std::find(x.rotations.begin(), x.rotations.end(), "varimin") != x.rotations.end()) {

    arma::vec v(x.p, arma::fill::ones);
    arma::mat I(x.p, x.p, arma::fill::eye);
    x.H = I - v * v.t() / (x.p + 0.0); // Centering matrix

  }
  if(x.rotations[0] == "none") {

    // do nothing

  }

  if(x.penalization == "TL" || x.penalization == "TLM") {

    x.penalize = true;

    if(!x.blocks_list.empty()) {

      if(x.n_blocks < 2) {
        Rcpp::stop("Penalization can only be used with more than one block of factors");
      }

    } else {
      Rcpp::stop("Penalization can only be used with more than one block of factors");
    }

  } else if(x.penalization != "none") {

    Rcpp::stop("Unkown penalization. Available penalizations: TL and TLM");

  }

  // Check rotation optimization parameters:

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

  // Check parallelization setup:

  if(random_starts < 1) Rcpp::stop("random_starts must be a positive integer");
  if(cores < 1) Rcpp::stop("The number of cores must be a positive integer");

}

Rcpp::List rotate(arma::mat loadings, Rcpp::CharacterVector char_rotation,
                  std::string projection,
                  double gamma, arma::vec epsilon, arma::vec k,
                  double w, double alpha,
                  Rcpp::Nullable<arma::mat> nullable_Target,
                  Rcpp::Nullable<arma::mat> nullable_Weight,
                  Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                  Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                  Rcpp::Nullable<arma::uvec> nullable_blocks,
                  Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list,
                  Rcpp::Nullable<arma::vec> nullable_block_weights,
                  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                  std::string penalization,
                  Rcpp::Nullable<Rcpp::List> nullable_rot_control,
                  int random_starts, int cores) {

  Rcpp::Timer timer;

  std::vector<std::string> rotation = Rcpp::as<std::vector<std::string>>(char_rotation);

  // Structure of arguments:

  arguments x;
  x.p = loadings.n_rows, x.q = loadings.n_cols;
  x.lambda = loadings;
  x.Phi.set_size(x.q, x.q); x.Phi.eye();
  x.gamma = gamma, x.epsilon = epsilon, x.k = k, x.w = w, x.a = alpha;
  x.penalization = penalization;
  x.rotations = rotation;
  x.projection = projection;

  int maxit;
  double eps;

  // Check inputs and compute constants for rotation criteria:
  check_rotate(x,
               nullable_Target, nullable_Weight,
               nullable_PhiTarget, nullable_PhiWeight,
               nullable_blocks,
               nullable_blocks_list,
               nullable_block_weights,
               nullable_oblq_blocks,
               nullable_rot_control, maxit, eps,
               random_starts, cores);

  // Select one manifold:
  base_manifold* manifold = choose_manifold(x.projection);
  // Select one specific criteria or mixed criteria:
  base_criterion* criterion = choose_criterion(x.rotations, x.projection, x.blocks_list);

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

  timer.step("elapsed");

  result["elapsed"] = timer;

  result.attr("class") = "rotation";
  return result;

}
