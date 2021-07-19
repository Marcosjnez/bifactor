#include "EFA.h"

void check_rotate_GSLiD(std::string rotation, std::string projection,
                        int n, int n_factors, double w,
                        arma::mat Target, arma::mat Weight,
                        Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                        Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                        arma::mat loadings, arma::mat Phi,
                        arma::mat& PhiTarget, arma::mat& PhiWeight,
                        arma::mat& Weight2, arma::mat& PhiWeight2,
                        Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                        std::vector<arma::uvec>& list_oblq_blocks,
                        arma::uvec& oblq_blocks,
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

    if(arma::size(Target) != arma::size(loadings) ||
       arma::size(Weight) != arma::size(loadings)) {

      Rcpp::stop("Incompatible Target or Weight dimensions");

    }

    Weight2 = Weight % Weight;

  } else if(rotation == "xtarget") {

    if(w < 0) Rcpp::stop("w must be nonnegative");

    if (nullable_PhiTarget.isNotNull()) {
      PhiTarget = Rcpp::as<arma::mat>(nullable_PhiTarget);
    } else {
      Rcpp::stop("Provide a PhiTarget for xtarget rotation");
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

  } else {

    Rcpp::stop("Invalid rotation criteria");

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

arma::vec tucker_congruence(arma::mat X, arma::mat Y) {

  arma::vec YX = diagvec(Y.t() * X);
  arma::vec YY = diagvec(Y.t() * Y);
  arma::vec XX = diagvec(X.t() * X);

  // arma::vec YX = arma::sum(Y % X, 0);
  // arma::vec YY = arma::sum(Y % Y, 0);
  // arma::vec XX = arma::sum(X % X, 0);

  arma::vec congruence = YX / arma::sqrt(YY % XX);

  return arma::abs(congruence);

}

bool is_duplicate(arma::cube Targets, arma::mat Target, int length) {

  for(int i=length; i > -1; --i) {

    if(arma::approx_equal(Targets.slice(i), Target, "absdiff", 0)) return true;

  }

  return false;

}

void pass_to_sl(Rcpp::List efa_args,
                std::string& method, std::string& rotation, std::string& projection,
                Rcpp::Nullable<arma::mat>& nullable_Target,
                Rcpp::Nullable<arma::mat>& nullable_Weight,
                Rcpp::Nullable<arma::mat>& nullable_PhiTarget,
                Rcpp::Nullable<arma::mat>& nullable_PhiWeight,
                Rcpp::Nullable<arma::uvec>& nullable_oblq_blocks,
                bool& normalize, double& gamma, double& epsilon,
                double& k, double& w,
                int& random_starts, int& cores,
                Rcpp::Nullable<arma::vec>& nullable_init,
                Rcpp::Nullable<Rcpp::List>& nullable_efa_control,
                Rcpp::Nullable<Rcpp::List>& nullable_rot_control) {

  if (efa_args.containsElementNamed("method")) {
    std::string method_ = efa_args["method"];
    method = method_;
  } else {
    method = "minres";
  }
  if(efa_args.containsElementNamed("rotation")) {
    std::string rotation_ = efa_args["rotation"];
    rotation = rotation_;
  } else{
    rotation = "oblimin";
  }
  if(efa_args.containsElementNamed("projection")) {
    std::string projection_ = efa_args["projection"];
    projection = projection_;
  } else{
    projection = "oblq";
  }
  if(efa_args.containsElementNamed("init")) {
    Rcpp::Nullable<arma::vec> init_ = efa_args["init"];
    nullable_init = init_;
  } else{
    nullable_init = R_NilValue;
  }
  if (efa_args.containsElementNamed("Target")) {
    Rcpp::Nullable<arma::mat> Target_ = efa_args["Target"];
    nullable_Target = Target_;
  } else{
    nullable_Target = R_NilValue;
  }
  if (efa_args.containsElementNamed("Weight")) {
    Rcpp::Nullable<arma::mat> Weight_ = efa_args["Weight"];
    nullable_Weight = Weight_;
  } else{
    nullable_Weight = R_NilValue;
  }
  if (efa_args.containsElementNamed("PhiTarget")) {
    Rcpp::Nullable<arma::mat> PhiTarget_ = efa_args["PhiTarget"];
    nullable_PhiTarget = PhiTarget_;
  } else{
    nullable_PhiTarget = R_NilValue;
  }
  if (efa_args.containsElementNamed("PhiWeight")) {
    Rcpp::Nullable<arma::mat> PhiWeight_ = efa_args["PhiWeight"];
    nullable_PhiWeight = PhiWeight_;
  } else{
    nullable_PhiWeight = R_NilValue;
  }
  if (efa_args.containsElementNamed("oblq_blocks")) {
    Rcpp::Nullable<arma::uvec> oblq_blocks_ = efa_args["oblq_blocks"];
    nullable_oblq_blocks = oblq_blocks_;
  } else{
    nullable_oblq_blocks = R_NilValue;
  }
  if (efa_args.containsElementNamed("normalize")) {
    bool normalize_ = efa_args["normalize"];
    normalize = normalize_;
  } else{
    normalize = false;
  }
  if (efa_args.containsElementNamed("gamma")) {
    double gamma_ = efa_args["gamma"];
    gamma = gamma_;
  } else{
    gamma = 0;
  }
  if (efa_args.containsElementNamed("epsilon")) {
    double epsilon_ = efa_args["epsilon"];
    epsilon = epsilon_;
  } else{
    epsilon = 0.01;
  }
  if (efa_args.containsElementNamed("k")) {
    double k_ = efa_args["k"];
    k = k_;
  } else{
    k = 0;
  }
  if (efa_args.containsElementNamed("w")) {
    double w_ = efa_args["w"];
    w = w_;
  } else{
    w = 1;
  }
  if (efa_args.containsElementNamed("random_starts")) {
    int random_starts_ = efa_args["random_starts"];
    random_starts = random_starts_;
  } else{
    random_starts = 10;
  }
  if (efa_args.containsElementNamed("cores")) {
    int cores_ = efa_args["cores"];
    cores = cores_;
  } else{
    cores = 1;
  }
  if (efa_args.containsElementNamed("efa_control")) {
    Rcpp::Nullable<Rcpp::List> efa_control_ = efa_args["efa_control"];
    nullable_efa_control = efa_control_;
  } else{
    nullable_efa_control = R_NilValue;
  }
  if (efa_args.containsElementNamed("rot_control")) {
    Rcpp::Nullable<Rcpp::List> rot_control_ = efa_args["rot_control"];
    nullable_rot_control = rot_control_;
  } else{
    nullable_rot_control = R_NilValue;
  }

}

Rcpp::List sl(arma::mat R, int n_generals, int n_groups,
              Rcpp::Nullable<Rcpp::List> first_efa,
              Rcpp::Nullable<Rcpp::List> second_efa) {

  Rcpp::List first, second;

  if(first_efa.isNotNull()) {
    first = first_efa;
  }

  if(second_efa.isNotNull()) {
    second = second_efa;
  }

  int n_items = R.n_rows;

  // Arguments to pass to first efa in SL:

  std::string method_1, rotation_1, projection_1;
  Rcpp::Nullable<arma::vec> nullable_init_1;
  Rcpp::Nullable<arma::mat> nullable_Target_1, nullable_Weight_1,
  nullable_PhiTarget_1, nullable_PhiWeight_1;
  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks_1;
  bool normalize_1;
  double gamma_1, epsilon_1, k_1, w_1;
  int random_starts_1, cores_1;
  Rcpp::Nullable<Rcpp::List> nullable_efa_control_1, nullable_rot_control_1;

  // Check inputs:

  pass_to_sl(first,
             method_1, rotation_1, projection_1,
             nullable_Target_1, nullable_Weight_1,
             nullable_PhiTarget_1, nullable_PhiWeight_1,
             nullable_oblq_blocks_1, normalize_1,
             gamma_1, epsilon_1, k_1, w_1,
             random_starts_1, cores_1,
             nullable_init_1,
             nullable_efa_control_1,
             nullable_rot_control_1);

  // Arguments to pass to second efa in SL:

  std::string method_2, rotation_2, projection_2;
  Rcpp::Nullable<arma::vec> nullable_init_2;
  Rcpp::Nullable<arma::mat> nullable_Target_2, nullable_Weight_2,
  nullable_PhiTarget_2, nullable_PhiWeight_2;
  Rcpp::Nullable<arma::uvec> nullable_oblq_blocks_2;
  bool normalize_2;
  double gamma_2, epsilon_2, k_2, w_2;
  int random_starts_2, cores_2;
  Rcpp::Nullable<Rcpp::List> nullable_efa_control_2, nullable_rot_control_2;

  // Check inputs:

  pass_to_sl(second,
             method_2, rotation_2, projection_2,
             nullable_Target_2, nullable_Weight_2,
             nullable_PhiTarget_2, nullable_PhiWeight_2,
             nullable_oblq_blocks_2, normalize_2,
             gamma_2, epsilon_2, k_2, w_2,
             random_starts_2, cores_2,
             nullable_init_2,
             nullable_efa_control_2,
             nullable_rot_control_2);

  // First efa:

  Rcpp::List first_order_efa = efast(R, n_groups, method_1, rotation_1, projection_1,
                                     nullable_Target_1, nullable_Weight_1,
                                     nullable_PhiTarget_1, nullable_PhiWeight_1,
                                     nullable_oblq_blocks_1,
                                     normalize_1, gamma_1, epsilon_1, k_1, w_1,
                                     random_starts_1, cores_1,
                                     nullable_init_1,
                                     nullable_efa_control_1, nullable_rot_control_1);

  Rcpp::List result;
  Rcpp::List efa_model_rotation = first_order_efa["rotation"];

  arma::mat loadings_1 = efa_model_rotation["loadings"];
  arma::mat Phi_1 = efa_model_rotation["Phi"];

  if ( n_generals == 1 ) {

    Rcpp::List efa_result = efast(Phi_1, n_generals, method_2, "none", "none",
                                  nullable_Target_2, nullable_Weight_2,
                                  nullable_PhiTarget_2, nullable_PhiWeight_2,
                                  nullable_oblq_blocks_2,
                                  normalize_2, gamma_2, epsilon_2, k_2, w_2,
                                  random_starts_2, cores_2,
                                  nullable_init_2,
                                  nullable_efa_control_2, nullable_rot_control_2);

    arma::mat loadings_2 = efa_result["loadings"];
    arma::vec uniquenesses_2 = efa_result["uniquenesses"];

    arma::mat L = join_rows(loadings_2, diagmat(sqrt(uniquenesses_2)));
    arma::mat SL_loadings = loadings_1 * L;

    for (int j=0; j < SL_loadings.n_cols; ++j) {
      if (sum(SL_loadings.col(j)) < 0) {
        SL_loadings.col(j) *= -1;
      }
    }

    arma::mat Hierarchical_Phi(1, 1, arma::fill::eye);
    efa_result["Phi"] = Hierarchical_Phi;

    arma::mat Rhat = SL_loadings * SL_loadings.t();
    arma::vec uniquenesses = 1 - diagvec(Rhat);
    Rhat.diag().ones();

    result["loadings"] = SL_loadings;
    result["first_order_solution"] = first_order_efa;
    result["second_order_solution"] = efa_result;
    result["uniquenesses"] = uniquenesses;
    result["Rhat"] = Rhat;

  } else {

    Rcpp::List efa_result = efast(Phi_1, n_generals, method_2, rotation_2, projection_2,
                                  nullable_Target_2, nullable_Weight_2,
                                  nullable_PhiTarget_2, nullable_PhiWeight_2,
                                  nullable_oblq_blocks_2,
                                  normalize_2, gamma_2, epsilon_2, k_2, w_2,
                                  random_starts_2, cores_2,
                                  nullable_init_2,
                                  nullable_efa_control_2, nullable_rot_control_2);

    Rcpp::List efa_result_rotation = efa_result["rotation"];
    arma::mat loadings_2 = efa_result_rotation["loadings"];

    arma::vec uniquenesses_2 = efa_result_rotation["uniquenesses"];

    arma::mat Hierarchical_Phi = efa_result_rotation["Phi"];
    arma::mat sqrt_Hierarchical_Phi = arma::sqrtmat_sympd(Hierarchical_Phi);

    arma::mat loadings_12 = loadings_1 * loadings_2;
    arma::mat sqrt_uniquenesses_2 = diagmat(sqrt(uniquenesses_2));
    arma::mat lu = loadings_1 * sqrt_uniquenesses_2;

    arma::mat A = join_rows(loadings_12 * sqrt_Hierarchical_Phi, lu);
    arma::mat SL_loadings = join_rows(loadings_12, lu);

    arma::mat Rhat = A * A.t();
    arma::vec uniquenesses = 1 - diagvec(Rhat);
    Rhat.diag().ones();

    result["loadings"] = SL_loadings;
    result["first_order_solution"] = first_order_efa;
    result["second_order_solution"] = efa_result;
    result["uniquenesses"] = uniquenesses;
    result["Rhat"] = Rhat;

  }

  return result;

}

arma::mat get_target(arma::mat L, arma::mat Phi, double cutoff) {

  int I = L.n_rows;
  int J = L.n_cols;

  L.elem( arma::find_nonfinite(L) ).zeros();
  arma::mat loadings = L;

  if(cutoff > 0) {

    arma::mat A(I, J, arma::fill::ones);
    A.elem( find(abs(L) <= cutoff) ).zeros();
    return A;

  }

  /*
   * Find the squared normalized loadings.
   */

  arma::vec sqrt_communalities = sqrt(diagvec(L * Phi * L.t()));
  arma::mat norm_loadings = loadings;
  norm_loadings.each_col() /= sqrt_communalities;
  norm_loadings = pow(norm_loadings, 2);

  /*
   * Sort the squared normalized loadings by column in increasing direction and
   * compute the mean of the adyacent differences (DIFFs)
   */

  arma::mat sorted_norm_loadings = sort(norm_loadings);
  arma::mat diff_sorted_norm_loadings = diff(sorted_norm_loadings);
  arma::mat diff_means = mean(diff_sorted_norm_loadings, 0);

  // return diff_means;
  /*
   * Sort the absolute loading values by column in increasing direction and
   * find the column loading cutpoints (the smallest loading which DIFF is above the average)
   */

  arma::mat sorted_loadings = sort(abs(loadings));
  arma::vec cuts(J);

  for(int j=0; j < J; ++j) {
    for(int i=0; i < I; ++i) {
      if (diff_sorted_norm_loadings(i, j) >= diff_means(j)) {
        cuts(j) = sorted_loadings(i, j);
        // cuts(j) = sorted_norm_loadings(i, j);
        break;
      }
    }
  }

  /*
   * Create a target matrix inserting ones where squared normalized loadings are
   *  above the cutpoint
   */

  arma::mat Target(I, J, arma::fill::zeros);
  for(int j=0; j < J; ++j) {
    for(int i=0; i < I; ++i) {

      if(norm_loadings(i, j) > cuts(j)) {
        Target(i, j) = 1;
      }

    }
  }

  // return Target;

  arma::mat Target2 = Target;

  /*
   * check conditions C1 C2 C3
   */

  /*
   * C2
   * Replicate the loading matrix but with overall positive factors
   * Create submatrices for each column where the rows are 0
   * Check the rank of these submatrices
   */

  arma::mat multiplier = L;
  arma::mat a(1, J);
  double full_rank = J-1;

  for (int j=0; j < J; ++j) {

    if (mean(L.col(j)) < 0) {
      multiplier.col(j) = -L.col(j);
    }

    int size = I - accu(Target2.col(j)); // Number of 0s in column j

    arma::mat m(size, J); // submatrix of 0s in column j

    int p = 0;
    for(int i=0; i < I; ++i) {
      if(Target2(i, j) == 0) {
        m.row(p) = Target2.row(i);
        p = p+1;
      }
    }
    m.shed_col(j);

    double r = arma::rank(m);

    a(0, j) = r;
  }

  double condition = accu(full_rank - a);

  if (condition == 0) { // if all submatrices are of full rank

    return Target;

  } else {

    // Rcpp::Rcout << "Solution might not be identified" << std::endl;

    // indices de a que indican que las filas de m no son linealmente independientes o el numero de filas de m es inferior a J-1:
    int size = 0;
    for(int j=0; j < J; ++j) {
      if (a(0, j) != full_rank) {
        size = size+1;
      }
    }

    arma::uvec c(size);

    int p = 0;
    for(int j=0; j < J; ++j) {
      if (a(0, j) != full_rank) {
        c(p) = j;
        p = p+1;
      }
    }

    int h = 1;
    // Targ2[Targ2 == 0] <- NA
    for(int i=0; i < I; ++i) {
      for(int j=0; j < J; ++j) {
        if (Target2(i, j) == 0) {
          Target2(i, j) = arma::datum::nan;
        }
      }
    }

    for (int i=0; i < c.size(); ++i) {

      int h = c(i);
      // Targ2[which.min(as.matrix(multiplier[, h + 1]) * Targ2[, h]),h] <- NA
      arma::uword min_index = arma::index_min(multiplier.col(h) % Target2.col(h));
      Target2(min_index, h) = arma::datum::nan;

      // m <- Targ2[which(is.na(Targ2[, h])), -h]
      arma::uvec indexes = arma::find_nonfinite(Target2.col(h));
      arma::mat m(indexes.size(), J);

      for(int k=0; k < indexes.size(); ++k) {
        m.row(k) = Target2.row(indexes(k));
      }
      m.shed_col(h);

      // m[which(is.na(m))] <- 0
      m.elem( arma::find_nonfinite(m) ).zeros();

    }

    // Targ2[is.na(Targ2)] <- 0
    Target2.elem( arma::find_nonfinite(Target2) ).zeros();
    // Targ2[Targ2 == 1] <- NA
    // Targ[, 2:ncol(Targ)] <- Targ2

    return Target2;
  }

}

arma::mat get_target(arma::mat loadings, Rcpp::Nullable<arma::mat> nullable_Phi, double cutoff) {

  int J = loadings.n_cols;

  arma::mat Phi(J, J);

  if(nullable_Phi.isNotNull()) {
    Phi = Rcpp::as<arma::mat>(nullable_Phi);
  } else {
    Phi.eye();
  }

  return get_target(loadings, Phi, cutoff);

}

void update_target(int n_generals, int n, int n_factors,
                   arma::mat loadings, arma::mat Phi, double cutoff,
                   arma::mat& new_Target) {

  if(n_generals == 1) {

    loadings.shed_col(0);
    Phi.shed_col(0);
    Phi.shed_row(0);
    new_Target = get_target(loadings, Phi, cutoff);
    arma::vec add(n, arma::fill::ones);

    new_Target.insert_cols(0, add);

  } else {

    arma::mat loadings_g = loadings(arma::span::all, arma::span(0, n_generals-1));
    arma::mat loadings_s = loadings(arma::span::all, arma::span(n_generals, n_factors-1));

    arma::mat Phi_g = Phi(arma::span(0, n_generals-1), arma::span(0, n_generals-1));
    arma::mat Phi_s = Phi(arma::span(n_generals, n_factors-1), arma::span(n_generals, n_factors-1));

    arma::mat new_Target_g = get_target(loadings_g, Phi_g, cutoff);
    arma::mat new_Target_s = get_target(loadings_s, Phi_s, cutoff);

    new_Target = join_rows(new_Target_g, new_Target_s);

  }

}

Rcpp::List GSLiD(std::string projection,
                 arma::mat R, int n_generals, int n_groups,
                 std::string method, arma::mat initialTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                 double w, int maxit, int random_starts, int cores,
                 Rcpp::Nullable<arma::vec> nullable_init,
                 Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                 Rcpp::Nullable<Rcpp::List> nullable_rot_control,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                 double cutoff, bool verbose) {

  // Rcpp::Timer timer;

  int n = R.n_rows;
  int n_factors = n_generals + n_groups;

  // Choose manifold and rotation criteria:

  std::string rotation;

  if(projection == "poblq" || projection == "orth") {
    rotation = "target";
  } else if(projection == "oblq") {
    rotation = "xtarget";
  } else {
    Rcpp::stop("Unkown projection method");
  }

  base_manifold* manifold = choose_manifold(projection);
  base_criterion *criterion = choose_criterion(rotation, projection);

  // Create defaults:

  arma::mat PhiTarget, PhiWeight;
  arma::mat Weight = 1 - initialTarget;
  std::vector<arma::uvec> list_oblq_blocks;
  arma::uvec oblq_blocks;
  arma::vec init;

  int efa_maxit, lmm, rot_maxit;
  double efa_factr, rot_eps;

  // Rcpp::stop("Hasta aquí bien");

  // Check inputs for efa:

  check_efa(R, n_factors, nullable_init, init,
            nullable_efa_control,
            efa_maxit, lmm, efa_factr);

  // Check inputs and compute constants for rotation criteria:

  arma::mat empty_loadings(n, n_factors), empty_Phi(n_factors, n_factors),
  Weight2, PhiWeight2, I_gamma_C, N, M;

  double p2;
  double epsilon = 0, k = 0, gamma = 0;

  check_rotate_GSLiD(rotation, projection,
                     n, n_factors, w,
                     initialTarget, Weight,
                     nullable_PhiTarget, nullable_PhiWeight,
                     empty_loadings, empty_Phi,
                     PhiTarget, PhiWeight,
                     Weight2, PhiWeight2,
                     nullable_oblq_blocks, list_oblq_blocks, oblq_blocks,
                     nullable_rot_control, rot_maxit, rot_eps,
                     random_starts, cores);

  // Unrotated efa:

  Rcpp::List efa_result = efa(init, R, n_factors, method, efa_maxit, efa_factr, lmm);
  arma::mat unrotated_loadings = efa_result["loadings"];
  arma::mat loadings = unrotated_loadings;

  Rcpp::List result, rotation_result;
  rotation_result["unrotated_loadings"] = unrotated_loadings;

  // Initialize stuff:

  arma::mat new_Target = initialTarget;
  Weight = 1 - new_Target;
  Weight2 = Weight % Weight;

  arma::vec congruence;
  arma::cube Targets(n, n_factors, maxit, arma::fill::zeros);
  Targets.slice(0) = new_Target;
  arma::vec max_abs_diffs(maxit), min_congruences(maxit);
  int Target_discrepancies;
  bool Target_convergence = true;
  arma::mat old_Target;

  if (verbose) Rcpp::Rcout << "Rotating..." << std::endl;

  int i = 0;

  do{

    old_Target = new_Target;

    rotation_result = rotate_efa(manifold, criterion,
                                 n, n_factors, unrotated_loadings,
                                 new_Target, Weight, PhiTarget, PhiWeight,
                                 list_oblq_blocks, oblq_blocks,
                                 gamma, epsilon, k, w,
                                 random_starts, cores, rot_eps, rot_maxit,
                                 Weight2, PhiWeight2, I_gamma_C, N, M, p2);

    arma::mat new_loadings = rotation_result["loadings"];

    congruence = tucker_congruence(loadings, new_loadings);

    min_congruences[i] = congruence.min();
    max_abs_diffs[i] = arma::abs(loadings - new_loadings).max();

    loadings = new_loadings;
    arma::mat new_Phi = rotation_result["Phi"];

    update_target(n_generals, n, n_factors, loadings, new_Phi, cutoff, new_Target);

    Weight = 1 - new_Target;
    Weight2 = Weight % Weight;

    Target_discrepancies = arma::accu(arma::abs(old_Target - new_Target));

    bool check = is_duplicate(Targets, new_Target, i);
    Targets.slice(i) = new_Target;

    ++i;

    if (verbose) Rcpp::Rcout << "\r" << "  Iteration " << i
                             << ":  Mean Tucker congruence = " << mean(congruence) <<
    "  Target discrepancies = " << Target_discrepancies << "   \r";

    if(check) break;

  } while (i < maxit);

  if(i == maxit && Target_discrepancies != 0) {

    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::warning("Maximum iteration reached without convergence");

    Target_convergence = false;

  } else if(Target_discrepancies != 0) {

    Rcpp::Rcout << "\n" << std::endl;
    Rcpp::warning("Recursive Target iterates. The last result of the iteration is returned");

    Target_convergence = false;

  }

  arma::mat Phi = rotation_result["Phi"];
  rotation_result["loadings"] = loadings;
  rotation_result["Phi"] = Phi;
  arma::mat R_hat = loadings * Phi * loadings.t();
  rotation_result["uniquenesses"] = 1 - diagvec(R_hat);
  R_hat.diag().ones();
  rotation_result["R_hat"] = R_hat;
  rotation_result["Target"] = new_Target;
  rotation_result["Weights"] = Weight;
  rotation_result["Target_iterations"] = i;
  rotation_result["Target_convergence"] = Target_convergence;
  rotation_result["min_congruences"] = min_congruences.head(i);
  rotation_result["max_abs_diffs"] = max_abs_diffs.head(i);
  // Targets  = Targets(arma::span::all, arma::span::all, arma::span(0, i-1));
  // rotation_result["Targets"] = Targets;

  // timer.step("elapsed");
  // rotation_result["elapsed"] = timer;

  result["efa"] = efa_result;
  result["twoTier"] = rotation_result;

  return result;

}

Rcpp::List twoTier(arma::mat R, int n_generals, int n_groups,
                   std::string twoTier_method, std::string projection,
                   Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                   Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                   Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                   Rcpp::Nullable<arma::mat> nullable_initialTarget,
                   std::string method, int maxit, double cutoff,
                   double w, int random_starts, int cores,
                   Rcpp::Nullable<arma::vec> nullable_init,
                   Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                   Rcpp::Nullable<Rcpp::List> nullable_rot_control,
                   Rcpp::Nullable<Rcpp::List> nullable_SL_first_efa,
                   Rcpp::Nullable<Rcpp::List> nullable_SL_second_efa,
                   bool verbose) {

  Rcpp::Timer timer;

  if(maxit < 1) Rcpp::stop("maxit must be an integer greater than 0");
  if(cutoff < 0) Rcpp::stop("cutoff must be nonnegative");

  int n = R.n_rows;
  int n_factors = n_generals + n_groups;

  Rcpp::List result, SL_result;

  if(twoTier_method == "SL") {

    SL_result = sl(R, n_generals, n_groups, nullable_SL_first_efa, nullable_SL_second_efa);
    result["SL"] = SL_result;

  } else if(twoTier_method == "GSLiD") {

    arma::mat initialTarget;

    // Create initial target with Schmid-Leiman (SL) if there is no custom initial target:

    if(nullable_initialTarget.isNull()) {

      SL_result = sl(R, n_generals, n_groups, nullable_SL_first_efa, nullable_SL_second_efa);

      // Create the factor correlation matrix for the SL solution:

      arma::mat new_Phi(n_factors, n_factors, arma::fill::eye);

      if(n_generals > 1) {

        Rcpp::List second_order_solution = SL_result["second_order_solution"];
        Rcpp::List second_order_solution_rotation = second_order_solution["rotation"];
        arma::mat Phi_generals = second_order_solution_rotation["Phi"];
        new_Phi(arma::span(0, n_generals-1), arma::span(0, n_generals-1)) = Phi_generals;

      }

      // SL loadings:

      arma::mat SL_loadings = SL_result["loadings"];
      arma::mat loadings = SL_loadings;

      // Create initial target:

      update_target(n_generals, n, n_factors, loadings, new_Phi, cutoff, initialTarget);

    } else {

      // Use a custom initial target:

      initialTarget = Rcpp::as<arma::mat>(nullable_initialTarget);

    }

    Rcpp::List GSLiD_result = GSLiD(projection,
                                    R, n_generals, n_groups,
                                    method, initialTarget,
                                    nullable_PhiTarget, nullable_PhiWeight,
                                    w, maxit, random_starts, cores,
                                    nullable_init, nullable_efa_control,
                                    nullable_rot_control, nullable_oblq_blocks,
                                    cutoff, verbose);

    result = GSLiD_result;
    result["SL"] = SL_result;

  } else {

    Rcpp::stop("Unkown twoTier method");

  }


  timer.step("elapsed");

  result["elapsed"] = timer;

  result.attr("class") = "twoTier";

  return result;

}
