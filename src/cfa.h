/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 17/09/2023
 *
 */

Rcpp::List cfa(arma::vec parameters,
               std::vector<arma::mat> X,
               arma::ivec nfactors,
               arma::ivec nobs,
               std::vector<arma::mat> lambda,
               std::vector<arma::mat> phi,
               std::vector<arma::mat> psi,
               std::vector<arma::uvec> lambda_indexes,
               std::vector<arma::uvec> phi_indexes,
               std::vector<arma::uvec> psi_indexes,
               std::vector<arma::uvec> target_indexes,
               std::vector<arma::uvec> targetphi_indexes,
               std::vector<arma::uvec> targetpsi_indexes,
               Rcpp::CharacterVector char_cor,
               Rcpp::CharacterVector char_estimator,
               Rcpp::CharacterVector char_projection,
               Rcpp::CharacterVector char_missing,
               int random_starts, int cores,
               Rcpp::Nullable<Rcpp::List> nullable_control) {


  Rcpp::Timer timer;

  std::vector<std::string> cor = Rcpp::as<std::vector<std::string>>(char_cor);
  std::vector<std::string> estimator = Rcpp::as<std::vector<std::string>>(char_estimator);
  std::vector<std::string> projection = Rcpp::as<std::vector<std::string>>(char_projection);
  std::vector<std::string> missing = Rcpp::as<std::vector<std::string>>(char_missing);

  int nblocks = X.size();
  std::vector<arguments_cor> xcor(nblocks);
  std::vector<arguments_cfa> xcfa(nblocks);

  for(int i=0; i < nblocks; ++i) {

    // cor structure:
    xcor[i].X = X[i];
    xcor[i].cor = cor[i];
    xcor[i].estimator = estimator[i];
    xcor[i].p = X[i].n_cols;
    xcor[i].q = nfactors[i];
    xcor[i].missing = missing[i];
    xcor[i].cores = cores;
    xcor[i].nobs = nobs[i];

    check_cor(xcor[i]);

    // cfa structure:
    xcfa[i].parameters = parameters;
    xcfa[i].estimator = xcor[i].estimator;
    xcfa[i].projection = projection[i];
    // xcfa.optim = optim; // FIX optim in check_cfa
    // if(i == 1L) Rcpp::stop("4");
    xcfa[i].X = xcor[i].X;
    xcfa[i].p = xcor[i].p;
    xcfa[i].q = xcor[i].q;
    xcfa[i].R = xcor[i].R;
    xcfa[i].cor = xcor[i].cor;
    xcfa[i].missing = xcor[i].missing;
    xcfa[i].cores = xcor[i].cores;
    xcfa[i].nobs = xcor[i].nobs;
    xcfa[i].random_starts = random_starts;
    xcfa[i].nullable_control = nullable_control;
    xcfa[i].Ip.set_size(xcfa[i].p, xcfa[i].p); xcfa[i].Ip.eye();
    xcfa[i].Iq.set_size(xcfa[i].q, xcfa[i].q); xcfa[i].Iq.eye();
    xcfa[i].W = xcor[i].W;
    xcfa[i].std_error = xcor[i].std_error;
    xcfa[i].lambda = lambda[i];
    xcfa[i].phi = phi[i];
    xcfa[i].psi = psi[i];
    xcfa[i].lambda_indexes = lambda_indexes[i]-1;
    xcfa[i].phi_indexes = phi_indexes[i]-1;
    xcfa[i].psi_indexes = psi_indexes[i]-1;
    xcfa[i].target_indexes = target_indexes[i]-1;
    xcfa[i].targetphi_indexes = targetphi_indexes[i]-1;
    xcfa[i].targetpsi_indexes = targetpsi_indexes[i]-1;

    check_cfa(xcfa[i]);

  }

  // Select the optimizer:
  arguments_optim opt;
  opt.nblocks = nblocks;
  opt.parameters = parameters;
  opt.nullable_control = nullable_control;
  check_opt(opt);
  cfa_optim* algorithm = choose_cfa_optim(opt.optim);

  algorithm->optim(opt, xcfa);
  cfa_criterion2* cfa_criterion = new ultimate_criterion();
  cfa_criterion->outcomes(opt, xcfa);

  Rcpp::List cfa;
  cfa["f"] = opt.f;
  cfa["iterations"] = opt.iteration;
  cfa["convergence"] = opt.convergence;
  cfa["lambda"] = opt.lambda;
  cfa["phi"] = opt.phi;
  cfa["psi"] = opt.psi;
  cfa["fs"] = opt.fs;
  cfa["parameters"] = opt.parameters;
  cfa["gradient"] = opt.g;

  Rcpp::List modelInfo;
  // modelInfo["cor"] = xcor[i].cor;
  // modelInfo["estimator"] = xcfa[i].estimator;
  // modelInfo["projection"] = xcfa[i].projection;
  // modelInfo["nvars"] = xcfa[i].p;
  // modelInfo["nfactors"] = xcfa[i].q;
  // modelInfo["nobs"] = xcfa[i].nobs;
  // modelInfo["df"] = df;
  // modelInfo["df_null"] = df_null;
  // modelInfo["f_null"] = f_null;
  // modelInfo["lower"] = xefa.lower;
  // modelInfo["upper"] = xefa.upper;

  Rcpp::List result;
  result["correlation"] = opt.R;
  result["cfa"] = cfa;
  result["modelInfo"] = modelInfo;

  timer.step("elapsed");
  result["elapsed"] = timer;

  result.attr("class") = "cfa";
  return result;

}

// [[Rcpp::export]]
Rcpp::List cfa_test(arma::vec parameters,
               arma::vec dX,
               std::vector<arma::mat> X,
               arma::ivec nfactors,
               arma::ivec nobs,
               std::vector<arma::mat> lambda,
               std::vector<arma::mat> phi,
               std::vector<arma::mat> psi,
               std::vector<arma::uvec> lambda_indexes,
               std::vector<arma::uvec> phi_indexes,
               std::vector<arma::uvec> psi_indexes,
               std::vector<arma::uvec> target_indexes,
               std::vector<arma::uvec> targetphi_indexes,
               std::vector<arma::uvec> targetpsi_indexes,
               Rcpp::CharacterVector char_cor,
               Rcpp::CharacterVector char_estimator,
               Rcpp::CharacterVector char_projection,
               Rcpp::CharacterVector char_missing,
               int random_starts, int cores,
               Rcpp::Nullable<arma::vec> nullable_init,
               Rcpp::Nullable<Rcpp::List> nullable_control) {


  Rcpp::Timer timer;

  std::vector<std::string> cor = Rcpp::as<std::vector<std::string>>(char_cor);
  std::vector<std::string> estimator = Rcpp::as<std::vector<std::string>>(char_estimator);
  std::vector<std::string> projection = Rcpp::as<std::vector<std::string>>(char_projection);
  std::vector<std::string> missing = Rcpp::as<std::vector<std::string>>(char_missing);

  int nblocks = X.size();
  std::vector<arguments_cor> xcor(nblocks);
  std::vector<arguments_cfa> xcfa(nblocks);

  for(int i=0; i < nblocks; ++i) {

    // cor structure:
    xcor[i].X = X[i];
    xcor[i].cor = cor[i];
    xcor[i].estimator = estimator[i];
    xcor[i].p = X[i].n_cols;
    xcor[i].q = nfactors[i];
    xcor[i].missing = missing[i];
    xcor[i].cores = cores;
    xcor[i].nobs = nobs[i];

    check_cor(xcor[i]);

    // cfa structure:
    xcfa[i].parameters = parameters;
    xcfa[i].estimator = xcor[i].estimator;
    xcfa[i].projection = projection[i];
    // xcfa.optim = optim; // FIX optim in check_cfa
    // if(i == 1L) Rcpp::stop("4");
    xcfa[i].X = xcor[i].X;
    xcfa[i].p = xcor[i].p;
    xcfa[i].q = xcor[i].q;
    xcfa[i].R = xcor[i].R;
    xcfa[i].cor = xcor[i].cor;
    xcfa[i].missing = xcor[i].missing;
    xcfa[i].cores = xcor[i].cores;
    xcfa[i].nobs = xcor[i].nobs;
    xcfa[i].random_starts = random_starts;
    xcfa[i].nullable_control = nullable_control;
    xcfa[i].Ip.set_size(xcfa[i].p, xcfa[i].p); xcfa[i].Ip.eye();
    xcfa[i].Iq.set_size(xcfa[i].q, xcfa[i].q); xcfa[i].Iq.eye();
    xcfa[i].W = xcor[i].W;
    xcfa[i].std_error = xcor[i].std_error;
    xcfa[i].lambda = lambda[i];
    xcfa[i].phi = phi[i];
    xcfa[i].psi = psi[i];
    xcfa[i].lambda_indexes = lambda_indexes[i]-1;
    xcfa[i].phi_indexes = phi_indexes[i]-1;
    xcfa[i].psi_indexes = psi_indexes[i]-1;
    xcfa[i].target_indexes = target_indexes[i]-1;
    xcfa[i].targetphi_indexes = targetphi_indexes[i]-1;
    xcfa[i].targetpsi_indexes = targetpsi_indexes[i]-1;

    check_cfa(xcfa[i]);

  }

  // Select the optimizer:
  arguments_optim opt;
  opt.nblocks = nblocks;
  opt.parameters = parameters;
  opt.dparameters = dX;
  opt.nullable_control = nullable_control;
  check_opt(opt);

  cfa_criterion2* cfa_criterion = new ultimate_criterion();
  cfa_manifold2* cfa_manifold = new ultimate_manifold();
  cfa_manifold->param(opt, xcfa);
  cfa_manifold->dparam(opt, xcfa);
  cfa_criterion->F(opt, xcfa);
  cfa_criterion->G(opt, xcfa);
  cfa_criterion->dG(opt, xcfa);
  cfa_manifold->grad(opt, xcfa);
  cfa_manifold->dgrad(opt, xcfa);
  cfa_criterion->outcomes(opt, xcfa);

  Rcpp::List cfa;
  cfa["f"] = opt.f;
  cfa["iterations"] = opt.iteration;
  cfa["convergence"] = opt.convergence;
  cfa["lambda"] = opt.lambda;
  cfa["phi"] = opt.phi;
  cfa["psi"] = opt.psi;
  cfa["fs"] = opt.fs;
  cfa["parameters"] = opt.parameters;
  cfa["g"] = opt.g;
  cfa["dg"] = opt.dg;
  cfa["gradient"] = opt.gradient;
  cfa["dgradient"] = opt.dgradient;
  cfa["dlambda"] = xcfa[0].dlambda;
  cfa["dphi"] = xcfa[0].dphi;
  cfa["dpsi"] = xcfa[0].dpsi;
  cfa["lambda2"] = xcfa[0].lambda;
  cfa["phi2"] = xcfa[0].phi;
  cfa["psi2"] = xcfa[0].psi;

  Rcpp::List modelInfo;
  // modelInfo["cor"] = xcor[i].cor;
  // modelInfo["estimator"] = xcfa[i].estimator;
  // modelInfo["projection"] = xcfa[i].projection;
  // modelInfo["nvars"] = xcfa[i].p;
  // modelInfo["nfactors"] = xcfa[i].q;
  // modelInfo["nobs"] = xcfa[i].nobs;
  // modelInfo["df"] = df;
  // modelInfo["df_null"] = df_null;
  // modelInfo["f_null"] = f_null;
  // modelInfo["lower"] = xefa.lower;
  // modelInfo["upper"] = xefa.upper;

  Rcpp::List result;
  result["correlation"] = opt.R;
  result["cfa"] = cfa;
  result["modelInfo"] = modelInfo;

  timer.step("elapsed");
  result["elapsed"] = timer;

  result.attr("class") = "cfa";
  return result;

}
