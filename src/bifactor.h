/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

// #include <Rcpp/Benchmark/Timer.h>
// #include "structures.h"
// #include "manifolds.h"
// #include "criteria.h"
// #include "checks.h"
// #include "multiple_rotations.h"
// #include "EFA.h"

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

void pass_to_efast(Rcpp::List efa_args, arguments_efast& x) {

  if (efa_args.containsElementNamed("estimator")) {
    std::string estimator_ = efa_args["estimator"]; x.estimator = estimator_;
  }
  if(efa_args.containsElementNamed("rotation")) {
    std::vector<std::string> rotation_ = efa_args["rotation"]; x.rotation = rotation_;
  }
  if(efa_args.containsElementNamed("projection")) {
    std::string projection_ = efa_args["projection"]; x.projection = projection_;
  }
  if(efa_args.containsElementNamed("nobs")) {
    Rcpp::Nullable<int> nobs_ = efa_args["nobs"]; x.nullable_nobs = nobs_;
  }
  if(efa_args.containsElementNamed("init")) {
    Rcpp::Nullable<arma::vec> init_ = efa_args["init"]; x.nullable_init = init_;
  }
  if (efa_args.containsElementNamed("Target")) {
    Rcpp::Nullable<arma::mat> Target_ = efa_args["Target"]; x.nullable_Target = Target_;
  }
  if (efa_args.containsElementNamed("Weight")) {
    Rcpp::Nullable<arma::mat> Weight_ = efa_args["Weight"]; x.nullable_Weight = Weight_;
  }
  if (efa_args.containsElementNamed("PhiTarget")) {
    Rcpp::Nullable<arma::mat> PhiTarget_ = efa_args["PhiTarget"]; x.nullable_PhiTarget = PhiTarget_;
  }
  if (efa_args.containsElementNamed("PhiWeight")) {
    Rcpp::Nullable<arma::mat> PhiWeight_ = efa_args["PhiWeight"]; x.nullable_PhiWeight = PhiWeight_;
  }
  if (efa_args.containsElementNamed("blocks")) {
    Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> blocks_ = efa_args["blocks"]; x.nullable_blocks = blocks_;
  }
  if (efa_args.containsElementNamed("block_weights")) {
    Rcpp::Nullable<arma::vec> block_weights_ = efa_args["block_weights"]; x.nullable_block_weights = block_weights_;
  }
  if (efa_args.containsElementNamed("oblq_factors")) {
    Rcpp::Nullable<arma::uvec> oblq_factors_ = efa_args["oblq_factors"]; x.nullable_oblq_factors = oblq_factors_;
  }
  if (efa_args.containsElementNamed("gamma")) {
    arma::vec gamma_ = efa_args["gamma"]; x.gamma = gamma_;
  }
  if (efa_args.containsElementNamed("epsilon")) {
    arma::vec epsilon_ = efa_args["epsilon"]; x.epsilon = epsilon_;
  }
  if (efa_args.containsElementNamed("k")) {
    arma::vec k_ = efa_args["k"]; x.k = k_;
  }
  if (efa_args.containsElementNamed("w")) {
    double w_ = efa_args["w"]; x.w = w_;
  }
  if (efa_args.containsElementNamed("random_starts")) {
    int random_starts_ = efa_args["random_starts"]; x.random_starts = random_starts_;
  }
  if (efa_args.containsElementNamed("cores")) {
    int cores_ = efa_args["cores"]; x.cores = cores_;
  }
  if (efa_args.containsElementNamed("efa_control")) {
    Rcpp::Nullable<Rcpp::List> efa_control_ = efa_args["efa_control"]; x.nullable_efa_control = efa_control_;
  }
  if (efa_args.containsElementNamed("rot_control")) {
    Rcpp::Nullable<Rcpp::List> rot_control_ = efa_args["rot_control"]; x.nullable_rot_control = rot_control_;
  }

}

Rcpp::List sl(arma::mat X, int n_generals, int n_groups,
              std::string cor, Rcpp::Nullable<int> nullable_nobs,
              Rcpp::Nullable<Rcpp::List> first_efa,
              Rcpp::Nullable<Rcpp::List> second_efa, int cores) {

  Rcpp::Timer timer;
  Rcpp::List result;

  Rcpp::List correlation_result;
  arma::mat R;

  if(X.is_square()) {

    R = X;

  } else {

    if(cor == "poly") {
      correlation_result = polyfast(X, "none", "none", 0.00, 0L, false, cores);
      correlation_result["type"] = "polychorics";
      arma::mat polys = correlation_result["correlation"];
      R = polys;
    } else if(cor == "pearson") {
      R = arma::cor(X);
      correlation_result["type"] = "pearson";
      correlation_result["correlation"] = R;
    } else {
      Rcpp::stop("Unkown correlation method");
    }

  }

  result["correlation"] = correlation_result;

  Rcpp::List first, second;

  if(first_efa.isNotNull()) {
    first = first_efa;
  }

  if(second_efa.isNotNull()) {
    second = second_efa;
  }

  int nfactors = n_generals + n_groups;
  int n_items = R.n_rows;

  // Arguments to pass to first efa in SL:

  arguments_efast x1;

  // Check inputs:

  pass_to_efast(first, x1);

  // Arguments to pass to second efa in SL:

  arguments_efast x2;

  // Check inputs:

  pass_to_efast(second, x2);

  // First efa:

  Rcpp::List first_order_efa = efast(R, n_groups, x1.cor, x1.estimator, x1.rotation,
                                     x1.projection, nullable_nobs,
                                     x1.nullable_Target, x1.nullable_Weight,
                                     x1.nullable_PhiTarget, x1.nullable_PhiWeight,
                                     x1.nullable_blocks,
                                     x1.nullable_block_weights,
                                     x1.nullable_oblq_factors,
                                     x1.gamma, x1.epsilon, x1.k, x1.w,
                                     x1.random_starts, x1.cores,
                                     x1.nullable_init,
                                     x1.nullable_efa_control, x1.nullable_rot_control);

  Rcpp::List efa_model_rotation = first_order_efa["rotation"];

  arma::mat loadings_1 = efa_model_rotation["lambda"];
  arma::mat Phi_1 = efa_model_rotation["phi"];
  std::vector<std::string> rotation_none(1); rotation_none[0] = "none";

  if ( n_generals == 1 ) {

    Rcpp::List efa_result = efast(Phi_1, n_generals, x2.cor, x2.estimator, rotation_none,
                                  "none", nullable_nobs,
                                  x2.nullable_Target, x2.nullable_Weight,
                                  x2.nullable_PhiTarget, x2.nullable_PhiWeight,
                                  x2.nullable_blocks,
                                  x2.nullable_block_weights,
                                  x2.nullable_oblq_factors,
                                  x2.gamma, x2.epsilon, x2.k, x2.w,
                                  x2.random_starts, x2.cores,
                                  x2.nullable_init,
                                  x2.nullable_efa_control, x2.nullable_rot_control);

    Rcpp::List efa_result_efa = efa_result["efa"];
    arma::mat loadings_2 = efa_result_efa["lambda"];
    arma::vec uniquenesses_2 = efa_result_efa["uniquenesses"];

    arma::mat L = join_rows(loadings_2, diagmat(sqrt(uniquenesses_2)));
    arma::mat SL_loadings = loadings_1 * L;

    for (int j=0; j < SL_loadings.n_cols; ++j) {
      if (sum(SL_loadings.col(j)) < 0) {
        SL_loadings.col(j) *= -1;
      }
    }

    arma::mat Hierarchical_Phi(1, 1, arma::fill::eye);
    efa_result_efa["phi"] = Hierarchical_Phi;

    arma::mat Rhat = SL_loadings * SL_loadings.t();
    arma::vec uniquenesses = 1 - diagvec(Rhat);
    Rhat.diag().ones();

    result["lambda"] = SL_loadings;
    result["first_order_solution"] = first_order_efa;
    result["second_order_solution"] = efa_result_efa;
    result["uniquenesses"] = uniquenesses;
    result["Rhat"] = Rhat;

  } else {

    Rcpp::List efa_result = efast(Phi_1, n_generals, x2.cor, x2.estimator, x2.rotation,
                                  x2.projection, nullable_nobs,
                                  x2.nullable_Target, x2.nullable_Weight,
                                  x2.nullable_PhiTarget, x2.nullable_PhiWeight,
                                  x2.nullable_blocks,
                                  x2.nullable_block_weights,
                                  x2.nullable_oblq_factors,
                                  x2.gamma, x2.epsilon, x2.k, x2.w,
                                  x2.random_starts, x2.cores,
                                  x2.nullable_init,
                                  x2.nullable_efa_control, x2.nullable_rot_control);

    Rcpp::List efa_result_rotation = efa_result["rotation"];
    arma::mat loadings_2 = efa_result_rotation["lambda"];

    arma::vec uniquenesses_2 = efa_result_rotation["uniquenesses"];

    arma::mat Phi2 = efa_result_rotation["phi"];
    arma::mat SL_Phi(nfactors, nfactors, arma::fill::eye);
    SL_Phi(arma::span(0, n_generals-1), arma::span(0, n_generals-1)) = Phi2;
    // arma::mat sqrt_Phi2 = arma::sqrtmat_sympd(Phi2);

    arma::mat loadings_12 = loadings_1 * loadings_2;
    arma::mat sqrt_uniquenesses_2 = diagmat(sqrt(uniquenesses_2));
    arma::mat lu = loadings_1 * sqrt_uniquenesses_2;

    // arma::mat A = join_rows(loadings_12 * sqrt_Phi2, lu);
    arma::mat SL_loadings = join_rows(loadings_12, lu);
    arma::mat Rhat = SL_loadings * SL_Phi * SL_loadings.t();
    arma::vec uniquenesses = 1 - diagvec(Rhat);
    Rhat.diag().ones();

    result["lambda"] = SL_loadings;
    result["phi"] = SL_Phi;
    result["first_order_solution"] = first_order_efa;
    result["second_order_solution"] = efa_result;
    result["uniquenesses"] = uniquenesses;
    result["Rhat"] = Rhat;

  }

  Rcpp::List modelInfo;
  modelInfo["R"] = R;
  modelInfo["n_generals"] = n_generals;
  modelInfo["n_groups"] = n_groups;
  modelInfo["nullable_nobs"] = nullable_nobs;
  modelInfo["first_efa"] = first_efa;
  modelInfo["second_efa"] = second_efa;
  result["modelInfo"] = modelInfo;

  timer.step("elapsed");
  result["elapsed"] = timer;

  result.attr("class") = "SL";
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

void update_target(int n_generals, int n, int nfactors,
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
    arma::mat loadings_s = loadings(arma::span::all, arma::span(n_generals, nfactors-1));

    arma::mat Phi_g = Phi(arma::span(0, n_generals-1), arma::span(0, n_generals-1));
    arma::mat Phi_s = Phi(arma::span(n_generals, nfactors-1), arma::span(n_generals, nfactors-1));

    arma::mat new_Target_g = get_target(loadings_g, Phi_g, cutoff);
    arma::mat new_Target_s = get_target(loadings_s, Phi_s, cutoff);

    new_Target = join_rows(new_Target_g, new_Target_s);

  }

}

Rcpp::List bifad(arma::mat R, int n_generals, int n_groups,
                 std::string projection,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_factors,
                 double cutoff,
                 std::string normalization,
                 Rcpp::Nullable<int> nullable_nobs,
                 Rcpp::Nullable<Rcpp::List> first_efa,
                 Rcpp::Nullable<Rcpp::List> second_efa,
                 Rcpp::Nullable<Rcpp::List> nullable_rot_control,
                 int random_starts, int cores) {

  Rcpp::List first, second;

  if(first_efa.isNotNull()) {
    first = first_efa;
  }
  if(second_efa.isNotNull()) {
    second = second_efa;
  }

  int nfactors = n_generals + n_groups;

  // Arguments to pass to first efa:
  arguments_efast x1;
  // Check inputs:
  pass_to_efast(first, x1);

  // First EFA (group factors):

  Rcpp::List first_order_efa = efast(R, n_groups, x1.cor, x1.estimator, x1.rotation,
                                     x1.projection, nullable_nobs,
                                     x1.nullable_Target, x1.nullable_Weight,
                                     x1.nullable_PhiTarget, x1.nullable_PhiWeight,
                                     x1.nullable_blocks,
                                     x1.nullable_block_weights,
                                     x1.nullable_oblq_factors,
                                     x1.gamma, x1.epsilon, x1.k, x1.w,
                                     x1.random_starts, x1.cores,
                                     x1.nullable_init,
                                     x1.nullable_efa_control, x1.nullable_rot_control);

  Rcpp::List efa_model = first_order_efa["efa"];
  Rcpp::List efa_model_rotation = first_order_efa["rotation"];

  arma::mat unrotated = efa_model["lambda"];
  arma::mat loadings_1 = efa_model_rotation["lambda"];
  arma::mat Phi_1 = efa_model_rotation["phi"];

  arma::mat add;
  Rcpp::List second_order_efa;
  if(n_generals > 1) {

    // Arguments to pass to second efa:
    arguments_efast x2;
    // Check inputs:
    pass_to_efast(second, x2);

    // Second EFA (general factors):

    second_order_efa = efast(R, n_generals, x2.cor, x2.estimator, x2.rotation,
                             x2.projection, nullable_nobs,
                             x2.nullable_Target, x2.nullable_Weight,
                             x2.nullable_PhiTarget, x2.nullable_PhiWeight,
                             x2.nullable_blocks,
                             x2.nullable_block_weights,
                             x2.nullable_oblq_factors,
                             x2.gamma, x2.epsilon, x2.k, x2.w,
                             x2.random_starts, x2.cores,
                             x2.nullable_init,
                             x2.nullable_efa_control, x2.nullable_rot_control);

    Rcpp::List rot = second_order_efa["rotation"];

    arma::mat loadings_2 = rot["lambda"];
    arma::mat Phi_2 = rot["phi"];
    add = get_target(loadings_2, Phi_2, cutoff);

  } else {

    add.set_size(R.n_rows, 1);
    add.ones();

  }

  // Rotation (partially specified target with augmented loading matrix):

  unrotated.insert_cols(0, n_generals);

  // Create the augmented target matrix:
  arma::mat Target = get_target(loadings_1, Phi_1, cutoff);
  Target.insert_cols(0, add);
  SEXP Target_ = Rcpp::wrap(Target);
  Rcpp::Nullable<arma::mat> nullable_Target = Target_;
  // arma::mat Weight(Target.n_rows, Target.n_cols, arma::fill::eye);
  // SEXP Weight_ = Rcpp::wrap(Weight);
  // Rcpp::Nullable<arma::mat> nullable_Weight = Weight_;

  Rcpp::List rot = rotate(unrotated, "target", projection,
                          {0}, {0}, {0}, 0,
                          nullable_Target, R_NilValue, //nullable_Weight,
                          R_NilValue, R_NilValue,
                          R_NilValue, R_NilValue,
                          nullable_oblq_factors,
                          normalization,
                          nullable_rot_control,
                          random_starts, cores);

  Rcpp::List result;
  result["first_order"] = first_order_efa;
  result["second_order"] = second_order_efa;
  result["bifactor"] = rot;
  result["Target"] = Target;

  Rcpp::List modelInfo;
  modelInfo["R"] = R;
  modelInfo["n_generals"] = n_generals;
  modelInfo["n_groups"] = n_groups;
  modelInfo["projection"] = projection;
  modelInfo["cutoff"] = cutoff;
  modelInfo["normalization"] = normalization;
  modelInfo["nullable_nobs"] = nullable_nobs;
  modelInfo["first_efa"] = first_efa;
  modelInfo["second_efa"] = second_efa;
  modelInfo["nullable_rot_control"] = nullable_rot_control;

  result.attr("class") = "BIFAD";
  return result;

}

Rcpp::List GSLiD(arma::mat R, int n_generals, int n_groups,
                 std::string estimator, std::string projection,
                 Rcpp::Nullable<int> nullable_nobs,
                 Rcpp::Nullable<arma::mat> nullable_Target,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                 double w, int maxit, int random_starts, int cores,
                 Rcpp::Nullable<arma::vec> nullable_init,
                 Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                 Rcpp::Nullable<Rcpp::List> nullable_rot_control,
                 Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> nullable_blocks,
                 Rcpp::Nullable<arma::vec> nullable_block_weights,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_factors,
                 double cutoff, bool verbose) {

  std::vector<std::string> rotation;

  // Structure of rotation arguments:

  arguments_rotate x;
  x.p = R.n_rows, x.q = n_generals + n_groups;
  x.lambda.set_size(x.p, x.q);
  x.Phi.set_size(x.q, x.q); x.Phi.eye();
  x.w = w;
  x.rotations = rotation;
  x.projection = projection;
  x.nullable_Target = nullable_Target;
  x.nullable_Weight = R_NilValue;
  x.nullable_PhiTarget = nullable_PhiTarget;
  x.nullable_PhiWeight = nullable_PhiWeight;
  x.nullable_blocks = nullable_blocks;
  x.nullable_oblq_factors = nullable_oblq_factors;
  x.nullable_block_weights = nullable_block_weights;
  x.nullable_rot_control = nullable_rot_control;

  // Choose manifold and rotation criteria:

  if(x.projection == "poblq" || x.projection == "orth") {
    x.rotations = {"target"};
  } else if(x.projection == "oblq") {
    x.rotations = {"xtarget"};
  } else {
    Rcpp::stop("Unkown projection estimator");
  }

  // Model Info:

  int p = R.n_cols;
  int q = x.q;
  double df_null = p*(p-1)/2;
  double df = p*(p+1)/2 - (p*q + p - q*(q-1)/2);

  double f_null;
  if(estimator == "uls" || estimator == "pa") {
    f_null = arma::accu(R % R) - R.n_cols;
  } else if(estimator == "ml") {
    f_null = -arma::log_det_sympd(R);
  } else if(estimator == "minrank") {
    f_null = 0;
  }

  Rcpp::List modelInfo;
  modelInfo["R"] = R;
  modelInfo["estimator"] = estimator;
  modelInfo["projection"] = projection;
  modelInfo["rotation"] = x.rotations;
  modelInfo["n_vars"] = R.n_cols;
  modelInfo["nfactors"] = q;
  modelInfo["nobs"] = nullable_nobs;
  modelInfo["df"] = df;
  modelInfo["df_null"] = df_null;
  modelInfo["f_null"] = f_null;
  modelInfo["w"] = x.w;
  modelInfo["k"] = x.k;
  modelInfo["gamma"] = x.gamma;
  modelInfo["epsilon"] = x.epsilon;
  modelInfo["w"] = x.w;
  modelInfo["normalization"] = x.normalization;
  modelInfo["Target"] = x.nullable_Target;
  modelInfo["Weight"] = x.nullable_Weight;
  modelInfo["PhiTarget"] = x.nullable_PhiTarget;
  modelInfo["PhiWeight"] = x.nullable_PhiWeight;
  modelInfo["blocks"] = x.nullable_blocks;
  modelInfo["block_weights"] = x.nullable_block_weights;
  modelInfo["oblq_factors"] = x.nullable_oblq_factors;

  // Create defaults:

  int efa_maxit, lmm;
  double efa_factr;

  arma::vec init;

  // Check inputs for efa:

  check_efa(R, x.q, nullable_init, init,
            nullable_efa_control,
            efa_maxit, lmm, efa_factr);

  // Check inputs and compute constants for rotation criteria:

  Rcpp::Nullable<arma::mat> nullable_Weight = R_NilValue;

  check_rotate(x, random_starts, cores);

  rotation_manifold* manifold = choose_manifold(projection);
  rotation_criterion *criterion = choose_criterion(x.rotations, x.projection, x.cols_list);

  arma::mat new_Target = x.Target;

  // Unrotated efa:

  Rcpp::List efa_result = efa(init, R, x.q, estimator, efa_maxit, efa_factr, lmm);
  arma::mat unrotated_loadings = efa_result["lambda"];
  arma::mat loadings = unrotated_loadings;

  Rcpp::List result, rotation_result;
  rotation_result["unrotated_loadings"] = unrotated_loadings;

  // Initialize stuff:

  x.lambda = unrotated_loadings; // x.lambda will always correspond to the unrotated loadings

  arma::vec congruence;
  arma::cube Targets(x.p, x.q, maxit, arma::fill::zeros);
  Targets.slice(0) = new_Target;
  arma::vec max_abs_diffs(maxit), min_congruences(maxit);
  int Target_discrepancies;
  bool Target_convergence = true;
  arma::mat old_Target;

  if (verbose) Rcpp::Rcout << "Rotating..." << std::endl;

  int i = 0;

  do{

    Rcpp::checkUserInterrupt();

    old_Target = new_Target;

    rotation_result = rotate_efa(x, manifold, criterion, random_starts,
                                 cores);

    arma::mat new_loadings = rotation_result["lambda"];

    // congruence = tucker_congruence(loadings, new_loadings);

    // min_congruences[i] = congruence.min();
    // max_abs_diffs[i] = arma::abs(loadings - new_loadings).max();

    loadings = new_loadings;
    arma::mat new_Phi = rotation_result["phi"];

    // update target
    update_target(n_generals, x.p, x.q, loadings, new_Phi, cutoff, new_Target);

    x.Target = new_Target;
    x.Weight = 1 - new_Target;
    x.Weight2 = x.Weight % x.Weight;

    Target_discrepancies = arma::accu(arma::abs(old_Target - new_Target));

    bool check = is_duplicate(Targets, new_Target, i);
    Targets.slice(i) = new_Target;

    ++i;

    if (verbose) Rcpp::Rcout << "\r" << "  Iteration " << i
                             << "  Target discrepancies = " << Target_discrepancies << "   \r";

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

  arma::mat L = loadings;
  arma::mat Phi = rotation_result["phi"];

  for (int j=0; j < x.q; ++j) {
    if (sum(L.col(j)) < 0) {
      L.col(j)   *= -1;
      Phi.col(j) *= -1;
      Phi.row(j) *= -1;
    }
  }

  arma::vec propVar = arma::diagvec(Phi * L.t() * L)/x.p;

  rotation_result["lambda"] = L;
  rotation_result["phi"] = Phi;
  arma::mat R_hat = L * Phi * L.t();
  rotation_result["uniquenesses"] = 1 - diagvec(R_hat);
  R_hat.diag().ones();
  rotation_result["R_hat"] = R_hat;
  rotation_result["Target"] = x.Target;
  rotation_result["Weights"] = x.Weight;
  rotation_result["Target_iterations"] = i;
  rotation_result["Target_convergence"] = Target_convergence;
  // rotation_result["min_congruences"] = min_congruences.head(i);
  // rotation_result["max_abs_diffs"] = max_abs_diffs.head(i);
  // Targets  = Targets(arma::span::all, arma::span::all, arma::span(0, i-1));
  // rotation_result["Targets"] = Targets;
  result["efa"] = efa_result;
  result["bifactor"] = rotation_result;
  result["modelInfo"] = modelInfo;

  return result;

}

Rcpp::List botmin(arma::mat R, int n_generals, int n_groups,
                  std::string estimator, std::string projection,
                  Rcpp::Nullable<int> nullable_nobs,
                  Rcpp::Nullable<arma::uvec> nullable_oblq_factors,
                  double cutoff, int random_starts, int cores,
                  Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                  Rcpp::Nullable<Rcpp::List> nullable_rot_control) {

  if(cutoff < 0) {
    Rcpp::stop("The cutoff needs to be nonnegative.");
  }

  // Arguments to pass to the first efa:

  arguments_efast x1;
  x1.estimator = estimator;
  x1.projection = "oblq";
  x1.nullable_oblq_factors = R_NilValue;
  x1.nullable_efa_control = nullable_efa_control;
  x1.nullable_rot_control = nullable_rot_control;
  x1.rotation = {"oblimin"};
  // x1.normalization; x1.gamma

  // First-order efa:

  Rcpp::List first_order_efa = efast(R, n_groups, x1.cor, x1.estimator, x1.rotation,
                                     x1.projection, x1.nullable_nobs,
                                     x1.nullable_Target, x1.nullable_Weight,
                                     x1.nullable_PhiTarget, x1.nullable_PhiWeight,
                                     x1.nullable_blocks,
                                     x1.nullable_block_weights,
                                     x1.nullable_oblq_factors,
                                     x1.gamma, x1.epsilon, x1.k, x1.w,
                                     x1.random_starts, x1.cores,
                                     x1.nullable_init,
                                     x1.nullable_efa_control, x1.nullable_rot_control);

  Rcpp::List first_rot = first_order_efa["rotation"];
  arma::mat L = first_rot["lambda"];
  arma::mat Phi = first_rot["phi"];

  // Final efa:

  // Arguments to pass to the final efa:

  arguments_efast x2;
  x2.estimator = estimator;
  x2.projection = projection;
  x2.nullable_oblq_factors = nullable_oblq_factors;
  x2.nullable_efa_control = nullable_efa_control;
  x2.nullable_rot_control = nullable_rot_control;
  x2.rotation = {"oblimin", "target"};

  // Build the target using the cutoff:
  arma::mat Target = get_target(L, Phi, cutoff);
  SEXP Target_ = Rcpp::wrap(Target);
  x2.nullable_Target = Target_;
  // Set up the block list:
  unsigned g = n_generals; unsigned s = n_groups; unsigned p = R.n_rows;
  arma::uvec blocks_vector1 = {p, p};
  std::vector<arma::uvec> blocks_list1 = vector_to_list3(blocks_vector1);
  arma::uvec blocks_vector2 = {g, s};
  std::vector<arma::uvec> blocks_list2 = vector_to_list4(blocks_vector2);
  std::vector<std::vector<arma::uvec>> blocks_list(2);
  blocks_list[0] = blocks_list1;
  blocks_list[1] = blocks_list2;

  Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> blocks_list_ = Rcpp::wrap(blocks_list);
  // SEXP blocks_list_ = Rcpp::wrap(blocks_list);
  x2.nullable_blocks = blocks_list_;

  // conversion from R to C++
  // Rcpp::as<T>();
  // conversion from C++ to R
  // Rcpp::wrap();

  int nfactors = n_generals + n_groups;
  Rcpp::List final_efa = efast(R, nfactors, x2.cor, x2.estimator, x2.rotation,
                               x2.projection, x2.nullable_nobs,
                               x2.nullable_Target, x2.nullable_Weight,
                               x2.nullable_PhiTarget, x2.nullable_PhiWeight,
                               x2.nullable_blocks,
                               x2.nullable_block_weights,
                               x2.nullable_oblq_factors,
                               x2.gamma, x2.epsilon, x2.k, x2.w,
                               x2.random_starts, x2.cores,
                               x2.nullable_init,
                               x2.nullable_efa_control, x2.nullable_rot_control);

  Rcpp::List efa_result = final_efa["efa"];
  Rcpp::List rotation_result = final_efa["rotation"];
  Rcpp::List modelInfo = final_efa["modelInfo"];
  Rcpp::List results;
  results["efa"] = efa_result;
  results["bifactor"] = rotation_result;
  results["modelInfo"] = modelInfo;
  results["first_order_solution"] = first_order_efa;

  return results;

}

Rcpp::List bifactor(arma::mat X, int n_generals, int n_groups,
                    std::string method, std::string cor,
                    std::string estimator, std::string projection,
                    Rcpp::Nullable<int> nullable_nobs,
                    Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                    Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                    Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> nullable_blocks,
                    Rcpp::Nullable<arma::vec> nullable_block_weights,
                    Rcpp::Nullable<arma::uvec> nullable_oblq_factors,
                    Rcpp::Nullable<arma::mat> nullable_Target,
                    int maxit, double cutoff, std::string normalization,
                    double w, int random_starts, int cores,
                    Rcpp::Nullable<arma::vec> nullable_init,
                    Rcpp::Nullable<Rcpp::List> nullable_efa_control,
                    Rcpp::Nullable<Rcpp::List> nullable_rot_control,
                    Rcpp::Nullable<Rcpp::List> nullable_first_efa,
                    Rcpp::Nullable<Rcpp::List> nullable_second_efa,
                    bool verbose) {

  Rcpp::Timer timer;
  Rcpp::List result, SL_result;

  Rcpp::List correlation_result;
  arma::mat R;

  if(X.is_square()) {

    R = X;

  } else {

    if(cor == "poly") {
      correlation_result = polyfast(X, "none", "none", 0.00, 0L, false, cores);
      correlation_result["type"] = "polychorics";
      arma::mat polys = correlation_result["correlation"];
      R = polys;
    } else if(cor == "pearson") {
      R = arma::cor(X);
      correlation_result["type"] = "pearson";
      correlation_result["correlation"] = R;
    } else {
      Rcpp::stop("Unkown correlation method");
    }

  }

  result["correlation"] = correlation_result;

  if(maxit < 1) Rcpp::stop("maxit must be an integer greater than 0");
  if(cutoff < 0) Rcpp::stop("cutoff must be nonnegative");

  int n = R.n_rows;
  int nfactors = n_generals + n_groups;

  if(method == "botmin") {

    Rcpp::List botmin_result = botmin(R, n_generals, n_groups,
                                      estimator, projection,
                                      nullable_nobs,
                                      nullable_oblq_factors,
                                      cutoff, random_starts, cores,
                                      nullable_efa_control,
                                      nullable_rot_control);

    result = botmin_result;

  } else if(method == "bifad") {

    Rcpp::List bifad_result = bifad(R, n_generals, n_groups,
                                    projection,
                                    nullable_oblq_factors,
                                    cutoff, normalization,
                                    nullable_nobs,
                                    nullable_first_efa,
                                    nullable_second_efa,
                                    nullable_rot_control,
                                    random_starts, cores);

    result = bifad_result;

  } else if(method == "SL") {

    SL_result = sl(R, n_generals, n_groups, cor, nullable_nobs,
                   nullable_first_efa, nullable_second_efa, cores);
    result["SL"] = SL_result;

  } else if(method == "GSLiD") {

    // Create initial target with Schmid-Leiman (SL) if there is no custom initial target:

    if(nullable_Target.isNull()) {

      SL_result = sl(R, n_generals, n_groups, cor, nullable_nobs,
                     nullable_first_efa, nullable_second_efa, cores);

      // Create the factor correlation matrix for the SL solution:

      arma::mat new_Phi(nfactors, nfactors, arma::fill::eye);

      if(n_generals > 1) {

        Rcpp::List second_order_solution = SL_result["second_order_solution"];
        Rcpp::List second_order_solution_rotation = second_order_solution["rotation"];
        arma::mat Phi_generals = second_order_solution_rotation["phi"];
        new_Phi(arma::span(0, n_generals-1), arma::span(0, n_generals-1)) = Phi_generals;

      }

      // SL loadings:

      arma::mat SL_loadings = SL_result["lambda"];
      arma::mat loadings = SL_loadings;

      // Create initial target:

      arma::mat Target;
      update_target(n_generals, n, nfactors, loadings, new_Phi, cutoff, Target);
      SEXP Target_ = Rcpp::wrap(Target);
      nullable_Target = Target_;

    }

    Rcpp::List GSLiD_result = GSLiD(R, n_generals, n_groups,
                                    estimator, projection,
                                    nullable_nobs,
                                    nullable_Target,
                                    nullable_PhiTarget, nullable_PhiWeight,
                                    w, maxit, random_starts, cores,
                                    nullable_init, nullable_efa_control,
                                    nullable_rot_control,
                                    nullable_blocks,
                                    nullable_block_weights,
                                    nullable_oblq_factors,
                                    cutoff, verbose);

    result = GSLiD_result;
    result["SL"] = SL_result;

  } else {

    Rcpp::stop("Unkown bifactor method");

  }

  timer.step("elapsed");

  result["elapsed"] = timer;

  result.attr("class") = "bifactor";
  return result;

}
