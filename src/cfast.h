/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 14/09/2023
 *
 */

Rcpp::List cfast(arma::vec parameters, arma::mat X, int nfactors,
                 arma::uvec lambda_indexes,
                 arma::uvec phi_indexes,
                 arma::uvec psi_indexes,
                 std::string cor, std::string estimator, std::string missing,
                 Rcpp::Nullable<int> nullable_nobs,
                 Rcpp::Nullable<arma::mat> nullable_Target,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 int random_starts, int cores,
                 Rcpp::Nullable<arma::vec> nullable_init,
                 Rcpp::Nullable<Rcpp::List> nullable_control) {

  Rcpp::Timer timer;

  // cor structure:
  arguments_cor xcor;
  xcor.X = X;
  xcor.cor = cor;
  xcor.estimator = estimator;
  xcor.p = X.n_cols;
  xcor.q = nfactors;
  xcor.missing = missing;
  xcor.cores = cores;
  if(nullable_nobs.isNotNull()) {
    xcor.nobs = Rcpp::as<int>(nullable_nobs);
  }

  Rcpp::Rcout << "check_cor" << std::endl;
  check_cor(xcor);

  // cfa structure:
  arguments_cfa xcfa;
  xcfa.parameters = parameters;
  arma::vec dparameters(parameters.n_elem, arma::fill::zeros);
  xcfa.estimator = estimator;
  // xcfa.projection = projection;
  // xcfa.optim = optim; // FIX optim in check_cfa
  xcfa.X = xcor.X;
  xcfa.p = xcor.p;
  xcfa.q = xcor.q;
  xcfa.R = xcor.R;
  xcfa.cor = xcor.cor;
  xcfa.missing = xcor.missing;
  xcfa.cores = xcor.cores;
  xcfa.nobs = xcor.nobs;
  xcfa.random_starts = random_starts;
  xcfa.nullable_control = nullable_control;
  xcfa.nullable_init = nullable_init;
  xcfa.Ip.set_size(xcfa.p, xcfa.p); xcfa.Ip.eye();
  xcfa.Iq.set_size(xcfa.q, xcfa.q); xcfa.Iq.eye();
  xcfa.W = xcor.W;
  xcfa.lambda_indexes = lambda_indexes-1;
  xcfa.phi_indexes = phi_indexes-1;
  xcfa.psi_indexes = psi_indexes-1;
  xcfa.std_error = xcor.std_error;

  Rcpp::Rcout << "check_cfa" << std::endl;
  check_cfa(xcfa);

  // Select one manifold:
  cfa_manifold* cfa_manifold = choose_cfa_manifold(xcfa.projection);
  // Select the estimator:
  cfa_criterion* cfa_criterion = choose_cfa_criterion(xcfa.estimator);
  // Select the optimizer:
  cfa_optim* algorithm = choose_cfa_optim(xcfa.optim);

  Rcpp::Rcout << "optim" << std::endl;
  algorithm->optim(xcfa, cfa_manifold, cfa_criterion);
  Rcpp::Rcout << "outcomes" << std::endl;
  cfa_criterion->outcomes(xcfa);

  Rcpp::List result;
  result["correlation"] = xcor.correlation_result;
  result["estimates"] = xcfa.parameters;
  result["value"] = xcfa.f;
  result["iters"] = xcfa.iteration;
  result["convergence"] = xcfa.convergence;
  result["lambda"] = xcfa.lambda;
  result["phi"] = xcfa.phi;
  result["psi"] = xcfa.psi;
  result["Rhat"] = xcfa.Rhat;
  result["residuals"] = xcfa.residuals;

  // result["modelInfo"] = modelInfo;

  timer.step("elapsed");
  result["elapsed"] = timer;

  result.attr("class") = "cfa";
  return result;
}
