#include <Rcpp/Benchmark/Timer.h>
#include "EFA.h"

arma::vec tucker_congruence(arma::mat X, arma::mat Y) {

  arma::vec YX = diagvec(Y.t() * X);
  arma::vec YY = diagvec(Y.t() * Y);
  arma::vec XX = diagvec(X.t() * X);

  arma::vec congruence = YX / arma::sqrt(YY % XX);

  return arma::abs(congruence);

}

bool is_duplicate(arma::cube Targets, arma::mat Target, int length) {

  for(int i=length; i > -1; --i) {

    if(arma::approx_equal(Targets.slice(i), Target, "absdiff", 0)) return true;

  }

  return false;

}

Rcpp::List SL(arma::mat R, int n_generals, int n_specifics, std::string method,
              std::string rotation, bool normalize, int random_starts, int cores,
              double gamma, double epsilon, double k, double w, int efa_max_iter,
              double efa_factr, int m, int rot_max_iter, double rot_eps) {

  Rcpp::List first_order_efa = efast(R, n_specifics, method, rotation, R_NilValue,
                                     R_NilValue, R_NilValue, R_NilValue, R_NilValue,
                                     normalize, gamma, epsilon, k, w, random_starts, cores,
                                     efa_max_iter, efa_factr, m, rot_max_iter, rot_eps);

  Rcpp::List result;
  Rcpp::List efa_model_rotation = first_order_efa["rotation"];

  arma::mat loadings_1 = efa_model_rotation["loadings"];
  arma::mat Phi_1 = efa_model_rotation["Phi"];

  if ( n_generals == 1 ) {

    arma::vec psy = 1/diagvec(inv_sympd(Phi_1));
    Rcpp::List efa_result = efast(Phi_1, n_generals, method, "none", R_NilValue,
                                  R_NilValue, R_NilValue, R_NilValue, R_NilValue,
                                  normalize, gamma, epsilon, k, w, random_starts, cores,
                                  efa_max_iter, efa_factr, m,
                                  rot_max_iter, rot_eps);

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

    arma::mat R_hat = SL_loadings * SL_loadings.t();
    arma::vec uniquenesses = 1 - diagvec(R_hat);
    R_hat.diag().ones();

    result["loadings"] = SL_loadings;
    result["first_order_solution"] = first_order_efa;
    result["second_order_solution"] = efa_result;
    result["uniquenesses"] = uniquenesses;
    result["R_hat"] = R_hat;

  } else {

    Rcpp::List efa_result = efast(Phi_1, n_generals, method, rotation, R_NilValue,
                                  R_NilValue, R_NilValue, R_NilValue, R_NilValue,
                                  normalize, gamma, epsilon, k, w, random_starts, cores,
                                  efa_max_iter, efa_factr, m,
                                  rot_max_iter, rot_eps);

    Rcpp::List efa_result_rotation = efa_result["rotation"];
    arma::mat loadings_2 = efa_result_rotation["loadings"];

    arma::vec uniquenesses_2 = efa_result_rotation["uniquenesses"];

    arma::mat Hierarchical_Phi = efa_result_rotation["Phi"];
    // arma::mat U;
    // arma::vec s;
    // arma::mat V;
    // svd(U, s, V, Hierarchical_Phi);
    //
    // arma::mat sqrt_Hierarchical_Phi = U * diagmat(sqrt(s)) * V.t();
    arma::mat sqrt_Hierarchical_Phi = arma::sqrtmat_sympd(Hierarchical_Phi);

    arma::mat loadings_12 = loadings_1 * loadings_2;
    arma::mat sqrt_uniquenesses_2 = diagmat(sqrt(uniquenesses_2));
    arma::mat lu = loadings_1 * sqrt_uniquenesses_2;

    arma::mat A = join_rows(loadings_12 * sqrt_Hierarchical_Phi, lu);
    arma::mat SL_loadings = join_rows(loadings_12, lu);

    arma::mat R_hat = A * A.t();
    arma::vec uniquenesses = 1 - diagvec(R_hat);
    R_hat.diag().ones();

    result["loadings"] = SL_loadings;
    result["first_order_solution"] = first_order_efa;
    result["second_order_solution"] = efa_result;
    result["uniquenesses"] = uniquenesses;
    result["R_hat"] = R_hat;

  }

  return result;

}

arma::mat get_target(arma::mat L, arma::mat Phi) {

  L.elem( arma::find_nonfinite(L) ).zeros();
  arma::mat loadings = L;
  int I = loadings.n_rows;
  int J = loadings.n_cols;

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

arma::mat get_target_with_cutoff(arma::mat L, double cutoff) {

  L.elem( arma::find_nonfinite(L) ).ones();
  int n = L.n_rows;
  int p  = L.n_cols;
  arma::mat A(n, p, arma::fill::ones);
  A.elem( find(abs(L) <= cutoff) ).zeros();

  return A;

}

Rcpp::List SLiD(Rcpp::List SL_result, arma::mat R, int n_generals, int n_specifics, std::string method,
                std::string rotation, arma::mat PhiTarget, arma::mat PhiWeight,
                double w, int random_starts, int cores,
                int efa_max_iter, double efa_factr, int m,
                int rot_max_iter, double rot_eps,
                int max_iter, bool verbose = true) {

  Rcpp::List second_order_solution = SL_result["second_order_solution"];
  Rcpp::List second_order_solution_rotation;

  arma::mat SL_loadings = SL_result["loadings"];

  int n_factors = SL_loadings.n_cols;
  int n_indicators = SL_loadings.n_rows;

  arma::mat Phi_generals, Phi_specifics(n_specifics, n_specifics, arma::fill::eye);

  arma::vec psy = 1/diagvec(inv_sympd(R));
  Rcpp::List efa_result = efa(psy, R, n_factors, method, efa_max_iter, efa_factr, m);

  arma::mat unrotated_loadings = efa_result["loadings"];

  Rcpp::List result, rotation_result;
  rotation_result["unrotated_loadings"] = unrotated_loadings;

  arma::mat old_Target, loadings_g, loadings_s, new_Target_g, new_Target_s, new_Target, Weight;
  arma::vec add(n_indicators, arma::fill::zeros);

  arma::mat loadings = SL_loadings;
  double gamma = 0;
  double epsilon = 1e-02;
  double k = 0;

  if (n_generals == 1) {

    arma::mat SL_specifics = SL_loadings;
    SL_specifics.shed_col(0);
    new_Target = get_target(SL_specifics, Phi_specifics);
    arma::vec add(n_indicators, arma::fill::ones);
    new_Target.insert_cols(0, add);

  } else {

    second_order_solution_rotation = second_order_solution["rotation"];
    arma::mat SL_Phi = second_order_solution_rotation["Phi"];
    Phi_generals = SL_Phi;

    loadings_g = SL_loadings(arma::span::all, arma::span(0, n_generals-1));
    loadings_s = SL_loadings(arma::span::all, arma::span(n_generals, n_factors-1));

    new_Target_g = get_target(loadings_g, Phi_generals);
    new_Target_s = get_target(loadings_s, Phi_specifics);

    new_Target = join_rows(new_Target_g, new_Target_s);

  }

  Weight = 1-new_Target;
  arma::vec congruence;
  arma::cube Targets(n_indicators, n_factors, max_iter, arma::fill::zeros);
  Targets.slice(0) = new_Target;
  arma::vec max_abs_diffs(max_iter), min_congruences(max_iter);
  int i = 0;
  int Target_discrepancies;
  bool Target_convergence = true;

  if (verbose) Rcpp::Rcout << "Rotating..." << std::endl;

  do{

    old_Target = new_Target;

    rotation_result = multiple_rotations(unrotated_loadings, rotation, new_Target, Weight,
                                         PhiTarget, PhiWeight, gamma, epsilon, k, w,
                                         random_starts, cores, rot_eps, rot_max_iter);

    arma::mat new_loadings = rotation_result["loadings"];

    congruence = tucker_congruence(loadings, new_loadings);
    min_congruences[i] = congruence.min();
    max_abs_diffs[i] = arma::abs(loadings - new_loadings).max();

    loadings = new_loadings;

    if(n_generals == 1) {

      arma::mat specifics = loadings;
      specifics.shed_col(0);
      new_Target = get_target(specifics, Phi_specifics);
      arma::vec add(n_indicators, arma::fill::ones);
      new_Target.insert_cols(0, add);

    } else {

      loadings_g = loadings(arma::span::all, arma::span(0, n_generals-1));
      loadings_s = loadings(arma::span::all, arma::span(n_generals, n_factors-1));

      arma::mat new_Phi = rotation_result["Phi"];
      Phi_generals = new_Phi(arma::span(0, n_generals-1), arma::span(0, n_generals-1));
      Phi_specifics = new_Phi(arma::span(n_generals, n_factors-1), arma::span(n_generals, n_factors-1));

      new_Target_g = get_target(loadings_g, Phi_generals);
      new_Target_s = get_target(loadings_s, Phi_specifics);

      new_Target = join_rows(new_Target_g, new_Target_s);

    }

    Weight = 1-new_Target;
    Target_discrepancies = accu(abs(old_Target - new_Target));

    bool check = is_duplicate(Targets, new_Target, i);
    Targets.slice(i) = new_Target;

    ++i;

    if (verbose) Rcpp::Rcout << "\r" << "  Iteration " << i << ":  Mean Tucker congruence = " << mean(congruence) <<
      "  Target discrepancies = " << Target_discrepancies << "   \r";

    if(check) break;

  } while (i < max_iter);

  if(i == max_iter && Target_discrepancies != 0) {

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

  result["efa"] = efa_result;
  result["Schmid_Leiman"] = SL_result;

  if( rotation == "xtarget" ) {
    result["GSLiD"] = rotation_result;
  } else if( rotation == "target" ) {
    result["SLiD"] = rotation_result;
  } else if( rotation == "targetQ" ) {
    result["SLiDQ"] = rotation_result;
  }

  return result;

}

Rcpp::List SLi(Rcpp::List SL_result, arma::mat R, int n_generals, int n_specifics, std::string method,
               std::string rotation, arma::mat PhiTarget, arma::mat PhiWeight,
               double w, int random_starts, int cores, double cutoff,
               int efa_max_iter, double efa_factr, int m,
               int rot_max_iter, double rot_eps,
               int max_iter, bool verbose) {

  arma::mat SL_loadings = SL_result["loadings"];

  int n_factors = SL_loadings.n_cols;
  int n_indicators = SL_loadings.n_rows;

  arma::vec psy = 1/diagvec(inv_sympd(R));
  Rcpp::List efa_result = efa(psy, R, n_factors, method, efa_max_iter, efa_factr, m);

  arma::mat unrotated_loadings = efa_result["loadings"];

  Rcpp::List result, rotation_result;
  rotation_result["unrotated_loadings"] = unrotated_loadings;

  arma::mat old_Target, loadings_g, loadings_s, new_Target_g, new_Target_s, new_Target, Weight;
  arma::vec add(n_indicators, arma::fill::zeros);

  arma::mat loadings = SL_loadings;
  double gamma = 0;
  double epsilon = 1e-02;
  double k = 0;

  if (n_generals == 1) {

    arma::mat SL_specifics = SL_loadings;
    SL_specifics.shed_col(0);
    new_Target = get_target_with_cutoff(SL_specifics, cutoff);
    arma::vec add(n_indicators, arma::fill::ones);
    new_Target.insert_cols(0, add);

  } else {

    loadings_g = SL_loadings(arma::span::all, arma::span(0, n_generals-1));
    loadings_s = SL_loadings(arma::span::all, arma::span(n_generals, n_factors-1));

    new_Target_g = get_target_with_cutoff(loadings_g, cutoff);
    new_Target_s = get_target_with_cutoff(loadings_s, cutoff);

    new_Target = join_rows(new_Target_g, new_Target_s);

  }

  Weight = 1-new_Target;
  arma::vec congruence;
  arma::cube Targets(n_indicators, n_factors, max_iter, arma::fill::zeros);
  Targets.slice(0) = new_Target;
  arma::vec max_abs_diffs(max_iter), min_congruences(max_iter);
  int i = 0;
  int Target_discrepancies;
  bool Target_convergence = true;

  if (verbose) Rcpp::Rcout << "Rotating..." << std::endl;

  do{

    old_Target = new_Target;

    rotation_result = multiple_rotations(unrotated_loadings, rotation, new_Target, Weight,
                                         PhiTarget, PhiWeight, gamma, epsilon, k, w,
                                         random_starts, cores, rot_eps, rot_max_iter);

    arma::mat new_loadings = rotation_result["loadings"];

    congruence = tucker_congruence(loadings, new_loadings);
    min_congruences[i] = congruence.min();
    max_abs_diffs[i] = arma::abs(loadings - new_loadings).max();

    loadings = new_loadings;

    if(n_generals == 1) {

      arma::mat specifics = loadings;
      specifics.shed_col(0);
      new_Target = get_target_with_cutoff(specifics, cutoff);
      arma::vec add(n_indicators, arma::fill::ones);
      new_Target.insert_cols(0, add);

    } else {

      loadings_g = loadings(arma::span::all, arma::span(0, n_generals-1));
      loadings_s = loadings(arma::span::all, arma::span(n_generals, n_factors-1));

      new_Target_g = get_target_with_cutoff(loadings_g, cutoff);
      new_Target_s = get_target_with_cutoff(loadings_s, cutoff);

      new_Target = join_rows(new_Target_g, new_Target_s);

    }

    Weight = 1-new_Target;
    Target_discrepancies = accu(abs(old_Target - new_Target));

    bool check = is_duplicate(Targets, new_Target, i);
    Targets.slice(i) = new_Target;

    ++i;

    if (verbose) Rcpp::Rcout << "\r" << "  Iteration " << i << ":  Mean Tucker congruence = " << mean(congruence) <<
      "  Target discrepancies = " << Target_discrepancies << "   \r";

    if(check) break;

  } while (i < max_iter);

  if(i == max_iter && Target_discrepancies != 0) {

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

  result["efa"] = efa_result;
  result["Schmid_Leiman"] = SL_result;
  if( rotation == "xtarget" ) {
    result["GSLi"] = rotation_result;
  } else if( rotation == "target" ) {
    result["SLi"] = rotation_result;
  } else if( rotation == "targetQ" ) {
    result["SLiQ"] = rotation_result;
  }

  return result;

}

Rcpp::List iD(arma::mat R, int n_generals, int n_specifics, std::string method,
              std::string rotation, arma::mat Target, arma::mat PhiTarget, arma::mat PhiWeight,
              double w, int random_starts, int cores,
              int efa_max_iter, double efa_factr, int m,
              int rot_max_iter, double rot_eps,
              int max_iter, bool verbose = true) {

  int n_factors = n_generals + n_specifics;
  int n_indicators = R.n_rows;

  arma::mat Phi_generals, Phi_specifics(n_specifics, n_specifics, arma::fill::eye);

  arma::vec psy = 1/diagvec(inv_sympd(R));
  Rcpp::List efa_result = efa(psy, R, n_factors, method, efa_max_iter, efa_factr, m);

  arma::mat unrotated_loadings = efa_result["loadings"];

  Rcpp::List result, rotation_result;
  rotation_result["unrotated_loadings"] = unrotated_loadings;

  arma::mat old_Target, loadings_g, loadings_s, new_Target_g, new_Target_s, new_Target, Weight;
  arma::vec add(n_indicators, arma::fill::zeros);

  arma::mat loadings = Target;
  double gamma = 0;
  double epsilon = 1e-02;
  double k = 0;

  new_Target = Target;
  Weight = 1-new_Target;
  arma::vec congruence;
  arma::cube Targets(n_indicators, n_factors, max_iter, arma::fill::zeros);
  Targets.slice(0) = new_Target;
  arma::vec max_abs_diffs(max_iter), min_congruences(max_iter);
  int i = 0;
  int Target_discrepancies;
  bool Target_convergence = true;

  if (verbose) Rcpp::Rcout << "Rotating..." << std::endl;

  do{

    old_Target = new_Target;

    rotation_result = multiple_rotations(unrotated_loadings, rotation, new_Target, Weight,
                                         PhiTarget, PhiWeight, gamma, epsilon, k, w,
                                         random_starts, cores, rot_eps, rot_max_iter);

    arma::mat new_loadings = rotation_result["loadings"];

    congruence = tucker_congruence(loadings, new_loadings);
    min_congruences[i] = congruence.min();
    max_abs_diffs[i] = arma::abs(loadings - new_loadings).max();

    loadings = new_loadings;

    if(n_generals == 1) {

      arma::mat specifics = loadings;
      specifics.shed_col(0);
      new_Target = get_target(specifics, Phi_specifics);
      arma::vec add(n_indicators, arma::fill::ones);
      new_Target.insert_cols(0, add);

    } else {

      loadings_g = loadings(arma::span::all, arma::span(0, n_generals-1));
      loadings_s = loadings(arma::span::all, arma::span(n_generals, n_factors-1));

      arma::mat new_Phi = rotation_result["Phi"];
      Phi_generals = new_Phi(arma::span(0, n_generals-1), arma::span(0, n_generals-1));
      Phi_specifics = new_Phi(arma::span(n_generals, n_factors-1), arma::span(n_generals, n_factors-1));

      new_Target_g = get_target(loadings_g, Phi_generals);
      new_Target_s = get_target(loadings_s, Phi_specifics);

      new_Target = join_rows(new_Target_g, new_Target_s);

    }

    Weight = 1-new_Target;
    Target_discrepancies = accu(abs(old_Target - new_Target));

    bool check = is_duplicate(Targets, new_Target, i);
    Targets.slice(i) = new_Target;

    ++i;

    if (verbose) Rcpp::Rcout << "\r" << "  Iteration " << i << ":  Mean Tucker congruence = " << mean(congruence) <<
      "  Target discrepancies = " << Target_discrepancies << "   \r";

    if(check) break;

  } while (i < max_iter);

  if(i == max_iter && Target_discrepancies != 0) {

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

  result["efa"] = efa_result;

  if( rotation == "xtarget" ) {
    result["GiD"] = rotation_result;
  } else if( rotation == "target" ) {
    result["iD"] = rotation_result;
  } else if( rotation == "targetQ" ) {
    result["iDQ"] = rotation_result;
  }

  return result;

}

Rcpp::List i(arma::mat R, int n_generals, int n_specifics, std::string method,
             std::string rotation, arma::mat Target, arma::mat PhiTarget, arma::mat PhiWeight,
             double w, int random_starts, int cores, double cutoff,
             int efa_max_iter, double efa_factr, int m,
             int rot_max_iter, double rot_eps,
             int max_iter, bool verbose) {

  int n_factors = n_generals + n_specifics;
  int n_indicators = R.n_rows;

  arma::vec psy = 1/diagvec(inv_sympd(R));
  Rcpp::List efa_result = efa(psy, R, n_factors, method, efa_max_iter, efa_factr, m);

  arma::mat unrotated_loadings = efa_result["loadings"];

  Rcpp::List result, rotation_result;
  rotation_result["unrotated_loadings"] = unrotated_loadings;

  arma::mat old_Target, loadings_g, loadings_s, new_Target_g, new_Target_s, new_Target, Weight;
  arma::vec add(n_indicators, arma::fill::zeros);

  arma::mat loadings = Target;
  double gamma = 0;
  double epsilon = 1e-02;
  double k = 0;

  new_Target = Target;
  Weight = 1-new_Target;
  arma::vec congruence;
  arma::cube Targets(n_indicators, n_factors, max_iter, arma::fill::zeros);
  Targets.slice(0) = new_Target;
  arma::vec max_abs_diffs(max_iter), min_congruences(max_iter);
  int i = 0;
  int Target_discrepancies;
  bool Target_convergence = true;

  if (verbose) Rcpp::Rcout << "Rotating..." << std::endl;

  do{

    old_Target = new_Target;

    rotation_result = multiple_rotations(unrotated_loadings, rotation, new_Target, Weight,
                                         PhiTarget, PhiWeight, gamma, epsilon, k, w,
                                         random_starts, cores, rot_eps, rot_max_iter);

    arma::mat new_loadings = rotation_result["loadings"];

    congruence = tucker_congruence(loadings, new_loadings);
    min_congruences[i] = congruence.min();
    max_abs_diffs[i] = arma::abs(loadings - new_loadings).max();

    loadings = new_loadings;

    if(n_generals == 1) {

      arma::mat specifics = loadings;
      specifics.shed_col(0);
      new_Target = get_target_with_cutoff(specifics, cutoff);
      arma::vec add(n_indicators, arma::fill::ones);
      new_Target.insert_cols(0, add);

    } else {

      loadings_g = loadings(arma::span::all, arma::span(0, n_generals-1));
      loadings_s = loadings(arma::span::all, arma::span(n_generals, n_factors-1));

      new_Target_g = get_target_with_cutoff(loadings_g, cutoff);
      new_Target_s = get_target_with_cutoff(loadings_s, cutoff);

      new_Target = join_rows(new_Target_g, new_Target_s);

    }

    Weight = 1-new_Target;
    Target_discrepancies = accu(abs(old_Target - new_Target));

    bool check = is_duplicate(Targets, new_Target, i);
    Targets.slice(i) = new_Target;

    ++i;

    if (verbose) Rcpp::Rcout << "\r" << "  Iteration " << i << ":  Mean Tucker congruence = " << mean(congruence) <<
      "  Target discrepancies = " << Target_discrepancies << "   \r";

    if(check) break;

  } while (i < max_iter);

  if(i == max_iter && Target_discrepancies != 0) {

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

  result["efa"] = efa_result;

  if( rotation == "xtarget" ) {
    result["Gi"] = rotation_result;
  } else if( rotation == "target" ) {
    result["i"] = rotation_result;
  } else if( rotation == "targetQ" ) {
    result["iQ"] = rotation_result;
  }

  return result;

}

Rcpp::List bifactor(arma::mat R, int n_generals, int n_specifics, std::string method,
                    std::string rotation, Rcpp::Nullable<Rcpp::NumericVector> init,
                    bool normalize,  double gamma, double epsilon, double k, double w,
                    std::string bifactor_method, int SLiD_max_iter, double cutoff,
                    Rcpp::Nullable<Rcpp::NumericMatrix> LTarget,
                    Rcpp::Nullable<Rcpp::NumericMatrix> PhiTarget,
                    Rcpp::Nullable<Rcpp::NumericMatrix> PhiWeight,
                    int random_starts, int cores,
                    int efa_max_iter, double efa_factr, int m,
                    int rot_max_iter, double rot_eps, bool verbose) {

  Rcpp::Timer timer;

  Rcpp::List result, SL_result, SLi_result, SLiD_result, iDx_result, ix_result;

  SL_result = SL(R, n_generals, n_specifics, method, rotation, normalize,
                 random_starts, cores, gamma, epsilon, k, w,
                 efa_max_iter, efa_factr, m, rot_max_iter, rot_eps);

  arma::mat Target, Phi_Target, Phi_Weight;

  if (LTarget.isNotNull()) {
    Target = Rcpp::as<arma::mat>(LTarget);
  }
  if (PhiTarget.isNotNull()) {
    Phi_Target = Rcpp::as<arma::mat>(PhiTarget);
  }
  if (PhiWeight.isNotNull()) {
    Phi_Weight = Rcpp::as<arma::mat>(PhiWeight);
  }

  if (bifactor_method == "GSLiD") {

    std::string rotation2 = "xtarget";
    SLiD_result = SLiD(SL_result, R, n_generals, n_specifics, method, rotation2,
                       Phi_Target, Phi_Weight, w, random_starts, cores,
                       efa_max_iter, efa_factr, m,
                       rot_max_iter, rot_eps,
                       SLiD_max_iter, verbose);
    result = SLiD_result;

  } else if (bifactor_method == "GSLi") {

    std::string rotation2 = "xtarget";
    SLi_result = SLi(SL_result, R, n_generals, n_specifics, method, rotation2,
                     Phi_Target, Phi_Weight, w, random_starts, cores,
                     cutoff, efa_max_iter, efa_factr, m, rot_max_iter, rot_eps,
                     SLiD_max_iter, verbose);
    result = SLi_result;

  } else if (bifactor_method == "SLiD") {

    std::string rotation2 = "target";
    SLiD_result = SLiD(SL_result, R, n_generals, n_specifics, method, rotation2,
                       Phi_Target, Phi_Weight, w, random_starts, cores,
                       efa_max_iter, efa_factr, m,
                       rot_max_iter, rot_eps,
                       SLiD_max_iter, verbose);
    result = SLiD_result;

  } else if (bifactor_method == "SLi") {

    std::string rotation2 = "target";
    SLi_result = SLi(SL_result, R, n_generals, n_specifics, method, rotation2,
                     Phi_Target, Phi_Weight, w, random_starts, cores,
                     cutoff, efa_max_iter, efa_factr, m, rot_max_iter, rot_eps,
                     SLiD_max_iter, verbose);
    result = SLi_result;

  } else if (bifactor_method == "SLiDQ") {

    std::string rotation2 = "targetQ";
    SLiD_result = SLiD(SL_result, R, n_generals, n_specifics, method, rotation2,
                       Phi_Target, Phi_Weight, w, random_starts, cores,
                       efa_max_iter, efa_factr, m, rot_max_iter, rot_eps,
                       SLiD_max_iter, verbose);
    result = SLiD_result;

  } else if (bifactor_method == "GiD") {

    std::string rotation2 = "xtarget";
    iDx_result = iD(R, n_generals, n_specifics, method, rotation2,
                    Target, Phi_Target, Phi_Weight, w, random_starts, cores,
                    efa_max_iter, efa_factr, m, rot_max_iter, rot_eps,
                    SLiD_max_iter, verbose);
    result = iDx_result;

  } else if (bifactor_method == "Gi") {

    std::string rotation2 = "xtarget";
    ix_result = i(R, n_generals, n_specifics, method, rotation2,
                  Target, Phi_Target, Phi_Weight, w, random_starts, cores,
                  cutoff, efa_max_iter, efa_factr, m, rot_max_iter, rot_eps,
                  SLiD_max_iter, verbose);
    result = ix_result;

  } else if (bifactor_method == "SL") {

    result = SL_result;

  }

  timer.step("elapsed");

  result["elapsed"] = timer;

  result.attr("class") = "bifactor";
  return result;

}
