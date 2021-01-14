// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>

#include "EFA.h"

Rcpp::List SL(arma::mat R, int n_generals, int n_specifics, std::string method, std::string rotation, bool normalize,
        int random_starts, int cores, double gamma, double epsilon,
        double k, double w, int efa_max_iter, double efa_factr, int m,
        int rot_max_iter, double rot_eps) {
  
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
    
    // Cambiamos el signo de los loadings que suman en negativo por factor:
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
    arma::mat U;
    arma::vec s;
    arma::mat V;
    svd(U, s, V, Hierarchical_Phi);
    
    arma::mat sqrt_Hierarchical_Phi = U * diagmat(sqrt(s)) * V.t();
    
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
  
  // Normalización:
  
  arma::vec sqrt_communalities = sqrt(diagvec(L * Phi * L.t()));
  arma::mat norm_loadings = loadings;
  norm_loadings.each_col() /= sqrt_communalities;
  norm_loadings = pow(norm_loadings, 2);
  
  arma::mat sorted_loadings = sort(abs(loadings));
  arma::mat sorted_norm_loadings = sort(norm_loadings);
  
  arma::mat diff_sorted_norm_loadings = diff(sorted_norm_loadings);
  arma::mat diff_means = mean(diff_sorted_norm_loadings, 0);
  arma::vec cuts(J);
  
  for(int j=0; j < J; ++j) {
    for(int i=0; i < I; ++i) {
      if (diff_sorted_norm_loadings(i, j) >= diff_means(j)) {
        cuts(j) = sorted_loadings(i, j);
        break;
      }
    }
  }
  
  arma::mat Target(I, J, arma::fill::zeros);
  for(int j=0; j < J; ++j) {
    for(int i=0; i < I; ++i) {
      
      if(norm_loadings(i, j) > cuts(j)) {
        Target(i, j) = 1;
      }
      
    }
  }
  
  arma::mat Target2 = Target;
  
  arma::mat multiplier = L;
  arma::mat a(1, J);
  
  // arma::mat vec_full_rank(1, J, arma::fill::ones);
  // vec_full_rank *= (J-1);
  double full_rank = J-1;
  
  for (int j=0; j < J; ++j) {
    
    if (mean(L.col(j)) < 0) {
      multiplier.col(j) = -L.col(j);
    }
    
    int size = I - accu(Target2.col(j)); // Cuantos 0s hay en la columna j?
    
    arma::mat m(size, J); // submatriz de 0s en la columna j
    
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
  
  if (condition == 0) {
    
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
  int n_indicadores = SL_loadings.n_rows;
  
  arma::mat SL_Phi, SL_Phi_specifics(n_specifics, n_specifics, arma::fill::eye);
  
  arma::vec psy = 1/diagvec(inv_sympd(R));
  Rcpp::List efa_result = efa(psy, R, n_factors, method, efa_max_iter, efa_factr, m);
  
  arma::mat unrotated_loadings = efa_result["loadings"];
  
  Rcpp::List result, rotation_result;
  rotation_result["unrotated_loadings"] = unrotated_loadings;
  
  arma::mat old_Target, loadings_g, loadings_s, new_Target_g, new_Target_s, new_Target, Weight;
  arma::vec add(n_indicadores, arma::fill::zeros);
  
  arma::mat loadings;
  double gamma = 0;
  double epsilon = 1e-02;
  double k = 0;
  
  if (n_generals == 1) {
    
    arma::mat SL_specifics = SL_loadings;
    SL_specifics.shed_col(0);
    new_Target = get_target(SL_specifics, SL_Phi_specifics);
    arma::vec add(n_indicadores, arma::fill::ones);
    new_Target.insert_cols(0, add);
    
  } else {
    
    second_order_solution_rotation = second_order_solution["rotation"];
    arma::mat SL_Phi_temp = second_order_solution_rotation["Phi"];
    SL_Phi = SL_Phi_temp;
    
    loadings_g = SL_loadings(arma::span::all, arma::span(0, n_generals-1));
    loadings_s = SL_loadings(arma::span::all, arma::span(n_generals, n_factors-1));
    
    new_Target_g = get_target(loadings_g, SL_Phi);
    new_Target_s = get_target(loadings_s, SL_Phi_specifics);
    
    new_Target = join_rows(new_Target_g, new_Target_s);
    
  }
  
  Weight = 1-new_Target;
  
  // arma::mat T = random_orthogonal(n_factors, n_factors);
  // Rcpp::List targetQ = NPF_targetQ(T, unrotated_loadings, new_Target, Weight, rot_eps, rot_max_iter);
  // arma::mat LL = targetQ["loadings"];
  
  // Rcpp::List targetQ = GPF_target(T, unrotated_loadings, new_Target, Weight, rot_eps, rot_max_iter);
  // arma::mat LL = targetQ["loadings"];
  // 
  // if (n_generals == 1) {
  // 
  //   arma::mat SL_specifics = LL;
  //   SL_specifics.shed_col(0);
  //   new_Target = get_target(SL_specifics, SL_Phi_specifics);
  //   arma::vec add(n_indicadores, arma::fill::ones);
  //   new_Target.insert_cols(0, add);
  // 
  // } else {
  // 
  //   arma::mat SL_Phi_temp = targetQ["Phi"];
  //   SL_Phi_temp = SL_Phi_temp(arma::span(0, n_generals-1), arma::span(0, n_generals-1));
  //   SL_Phi = SL_Phi_temp;
  // 
  //   loadings_g = LL(arma::span::all, arma::span(0, n_generals-1));
  //   loadings_s = LL(arma::span::all, arma::span(n_generals, n_factors-1));
  // 
  //   new_Target_g = get_target(loadings_g, SL_Phi);
  //   new_Target_s = get_target(loadings_s, SL_Phi_specifics);
  // 
  //   new_Target = join_rows(new_Target_g, new_Target_s);
  // 
  // }
  
  double i = 0;
  double Target_discrepancies;
  
  if (verbose) Rcpp::Rcout << "Rotating..." << std::endl;
  
  do{
    
    i = i + 1;
    old_Target = new_Target;
    
    rotation_result = multiple_rotations(unrotated_loadings, rotation, new_Target, Weight,
                                           PhiTarget, PhiWeight, gamma, epsilon, k, w, 
                                           random_starts, cores, rot_eps, rot_max_iter);
    
    double f = rotation_result["f"];
    arma::mat new_loadings = rotation_result["loadings"];
    
    loadings = new_loadings;
    
    if( n_generals == 1 ) {
      
      arma::mat specifics = loadings;
      specifics.shed_col(0);
      new_Target = get_target(specifics, SL_Phi_specifics);
      arma::vec add(n_indicadores, arma::fill::ones);
      new_Target.insert_cols(0, add);
      
    } else {
      
      loadings_g = loadings(arma::span::all, arma::span(0, n_generals-1));
      loadings_s = loadings(arma::span::all, arma::span(n_generals, n_factors-1));
      
      new_Target_g = get_target(loadings_g, SL_Phi);
      new_Target_s = get_target(loadings_s, SL_Phi_specifics);
      
      new_Target = join_rows(new_Target_g, new_Target_s);
      
      arma::mat new_Phi = rotation_result["Phi"];
      SL_Phi = new_Phi(arma::span(0, n_generals-1), arma::span(0, n_generals-1));
      SL_Phi_specifics = new_Phi(arma::span(n_generals, n_factors-1), arma::span(n_generals, n_factors-1));
      
    }
    
    Weight = 1-new_Target;
    
    Target_discrepancies = accu(abs(old_Target - new_Target));
    if (verbose) Rcpp::Rcout << "\r" << "  Iteration " << i << ":  f = " << f << 
      "  Target discrepancies = " << Target_discrepancies << "   \r";
    
  } while (Target_discrepancies != 0 && i < max_iter);
  
  arma::mat Phi = rotation_result["Phi"];
  
  rotation_result["loadings"] = loadings;
  rotation_result["Phi"] = Phi;
  arma::mat R_hat = loadings * Phi * loadings.t();
  rotation_result["uniquenesses"] = 1 - diagvec(R_hat);
  R_hat.diag().ones();
  rotation_result["R_hat"] = R_hat;
  rotation_result["Target"] = new_Target;
  rotation_result["Weights"] = Weight;
  rotation_result["Target_Iterations"] = i;
  
  result["efa"] = efa_result;
  result["Schmid_Leiman"] = SL_result;
  if( rotation == "xtarget" ) {
    result["SLiDx"] = rotation_result;
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
  int n_indicadores = SL_loadings.n_rows;
  
  arma::vec psy = 1/diagvec(inv_sympd(R));
  Rcpp::List efa_result = efa(psy, R, n_factors, method, efa_max_iter, efa_factr, m);
  
  arma::mat unrotated_loadings = efa_result["loadings"];
  
  Rcpp::List result, rotation_result;
  rotation_result["unrotated_loadings"] = unrotated_loadings;
  
  arma::mat old_Target, loadings_g, loadings_s, new_Target_g, new_Target_s, new_Target, Weight;
  arma::vec add(n_indicadores, arma::fill::zeros);
  
  arma::mat loadings;
  double gamma = 0;
  double epsilon = 1e-02;
  double k = 0;
  
  if (n_generals == 1) {
    
    arma::mat SL_specifics = SL_loadings;
    SL_specifics.shed_col(0);
    new_Target = get_target_with_cutoff(SL_specifics, cutoff);
    arma::vec add(n_indicadores, arma::fill::ones);
    new_Target.insert_cols(0, add);
    
  } else {
    
    loadings_g = SL_loadings(arma::span::all, arma::span(0, n_generals-1));
    loadings_s = SL_loadings(arma::span::all, arma::span(n_generals, n_factors-1));
    
    new_Target_g = get_target_with_cutoff(loadings_g, cutoff);
    new_Target_s = get_target_with_cutoff(loadings_s, cutoff);
    
    new_Target = join_rows(new_Target_g, new_Target_s);
    
  }
  
  Weight = 1-new_Target;
  double i = 0;
  double Target_discrepancies;
  
  if (verbose) Rcpp::Rcout << "Rotating..." << std::endl;
  
  do{
    
    i = i + 1;
    old_Target = new_Target;
    
    rotation_result = multiple_rotations(unrotated_loadings, rotation, new_Target, Weight,
                                           PhiTarget, PhiWeight, gamma, epsilon, k, w,
                                           random_starts, cores, rot_eps, rot_max_iter);
    
    double f = rotation_result["f"];
    arma::mat new_loadings = rotation_result["loadings"];
    
    loadings = new_loadings;
    
    if( n_generals == 1 ) {
      
      arma::mat specifics = loadings;
      specifics.shed_col(0);
      new_Target = get_target_with_cutoff(specifics, cutoff);
      arma::vec add(n_indicadores, arma::fill::ones);
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
    
    if (verbose) Rcpp::Rcout << "\r" << "  Iteration " << i << ":  f = " << f << 
      "  Target discrepancies = " << Target_discrepancies << "   \r";
    
  } while (Target_discrepancies != 0 && i < max_iter);
  
  arma::mat Phi = rotation_result["Phi"];
  
  rotation_result["loadings"] = loadings;
  rotation_result["Phi"] = Phi;
  arma::mat R_hat = loadings * Phi * loadings.t();
  rotation_result["uniquenesses"] = 1 - diagvec(R_hat);
  R_hat.diag().ones();
  rotation_result["R_hat"] = R_hat;
  rotation_result["Target"] = new_Target;
  rotation_result["Weights"] = Weight;
  rotation_result["Target_Iterations"] = i;
  
  result["efa"] = efa_result;
  result["Schmid_Leiman"] = SL_result;
  if( rotation == "xtarget" ) {
    result["SLix"] = rotation_result;
  } else if( rotation == "target" ) {
    result["SLi"] = rotation_result;
  } else if( rotation == "targetQ" ) {
    result["SLiQ"] = rotation_result;
  }
  
  return result;
  
}

Rcpp::List bifactor(arma::mat R, int n_generals, int n_specifics, std::string method, std::string rotation,
               Rcpp::Nullable<Rcpp::NumericVector> init, bool normalize,
               double gamma, double epsilon, double k, double w,
               std::string bifactor_method, int SLiD_max_iter, double cutoff,
               Rcpp::Nullable<Rcpp::NumericMatrix> PhiTarget, 
               Rcpp::Nullable<Rcpp::NumericMatrix> PhiWeight,
               int random_starts, int cores,
               int efa_max_iter, double efa_factr, int m,
               int rot_max_iter, double rot_eps, bool verbose) {
  
  Rcpp::List result, SL_result, SLi_result, SLiD_result;
  
  SL_result = SL(R, n_generals, n_specifics, method, rotation, normalize,
                 random_starts, cores, gamma, epsilon, k, w,
                 efa_max_iter, efa_factr, m, rot_max_iter, rot_eps);
  
  arma::mat Phi_Target, Phi_Weight;
  
  if (PhiTarget.isNotNull()) {
    Phi_Target = Rcpp::as<arma::mat>(PhiTarget);
  }
  if (PhiWeight.isNotNull()) {
    Phi_Weight = Rcpp::as<arma::mat>(PhiWeight);
  }
  
  if (bifactor_method == "SLiDx") {
    
    std::string rotation2 = "xtarget";
    SLiD_result = SLiD(SL_result, R, n_generals, n_specifics, method, rotation2,
                       Phi_Target, Phi_Weight, w, random_starts, cores,
                       efa_max_iter, efa_factr, m,
                       rot_max_iter, rot_eps,
                       SLiD_max_iter, verbose);
    result = SLiD_result;
    
  } else if (bifactor_method == "SLiD") {
    
    std::string rotation2 = "target";
    SLiD_result = SLiD(SL_result, R, n_generals, n_specifics, method, rotation2,
                       Phi_Target, Phi_Weight, w, random_starts, cores,
                       efa_max_iter, efa_factr, m, 
                       rot_max_iter, rot_eps,
                       SLiD_max_iter, verbose);
    result = SLiD_result;
    
  } else if (bifactor_method == "SLix") {
    
    std::string rotation2 = "xtarget";
    SLi_result = SLi(SL_result, R, n_generals, n_specifics, method, rotation2,
                     Phi_Target, Phi_Weight, w, random_starts, cores,
                     cutoff, efa_max_iter, efa_factr, m, rot_max_iter, rot_eps,
                     SLiD_max_iter, verbose);
    result = SLi_result;
    
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
    
  } else if (bifactor_method == "SL") {
    
    result = SL_result;
    
  }
  
  return result;
  
}
