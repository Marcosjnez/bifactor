void check_cfa(arguments_cfa& x) {

  Rcpp::List cfa_control;
  // Check control parameters:
  if (x.nullable_cfa_control.isNotNull()) {
    cfa_control = Rcpp::as<Rcpp::List>(x.nullable_cfa_control);
  }

  if(x.p < x.q) Rcpp::stop("Too many factors");

  // Choose custom weight matrix for the dwls estimator:
  if(cfa_control.containsElementNamed("W")) {
    arma::mat W = cfa_control["W"];
    x.W = W;
  } else if(x.W.is_empty() & x.estimator == "dwls") {
    Rcpp::stop("For the dwls estimator, please introduce the raw data or provide the reciprocal of the variance of the correlations in cfa_control = list(W = ...)");
  }

  if(x.estimator == "dwls") {
    if(x.cor == "poly") {
      Rcpp::stop("The dwls estimator is only available for cor = 'poly'");
    }
    if(x.optim == "gradient") {
      Rcpp::warning("To achive convergence with the dwls estimator and gradient optim algorithm, you may need to increase the number of maximum iterations: cfa_control = list(maxit = 100000)");
    }
  }

  // Check initial values:
  // if (x.nullable_init.isNotNull()) {
  //   Rcpp::warning("Initial values not available for the dwls estimator");
  //   x.psi = Rcpp::as<arma::vec>(x.nullable_init);
  // } else { // Check for positive definiteness only if custom init values are not specified
  //   if(x.R.is_sympd()) {
  //     x.psi = 1/arma::diagvec(arma::inv_sympd(x.R));
  //   } else {
  //     x.smoothed = smoothing(x.R, 0.001);
  //     x.psi = 1/arma::diagvec(arma::inv_sympd(x.smoothed));
  //   }
  // }

  if(cfa_control.containsElementNamed("projection")) {
    std::string manifold = cfa_control["projection"];
    x.manifold = manifold;
  }

  if(cfa_control.containsElementNamed("optim")) {
    std::string optim = cfa_control["optim"];
    x.optim = optim;
  }
  if(cfa_control.containsElementNamed("search")) {
    std::string search = cfa_control["search"];
    x.search = search;
    if(x.search != "back" & x.search != "wolfe") {
      Rcpp::stop("Unkown line-search method. Available methods: back and wolfe");
    }
  } else {
    x.search = "back";
  }
  if(cfa_control.containsElementNamed("maxit") ){
    x.maxit = cfa_control["maxit"];
  } else {
    x.maxit = 1e3;
  }

  // if(cfa_control.containsElementNamed("upper")) {
  //   arma::vec upper = cfa_control["upper"];
  //   x.upper = upper;
  // } else {
  //   x.upper = arma::diagvec(x.R);
  // }
  // if(cfa_control.containsElementNamed("lower")) {
  //   arma::vec lower = cfa_control["lower"];
  //   x.lower = lower;
  // } else {
  //   x.lower.set_size(x.p);
  //   for(int i=0; i < x.p; ++i) x.lower[i] = 0.005;
  // }

  if(cfa_control.containsElementNamed("c1")) {
    x.c1 = cfa_control["c1"];
    if(x.c1 <= 0 | x.c1 >= 1) Rcpp::stop("The parameter c1 must be a positive number lower than 1.");
  } else {
    if(x.optim == "gradient") {
      x.c1 = 0.5;
    } else {
      x.c1 = 10e-04;
    }
  }
  if(cfa_control.containsElementNamed("c2")) {
    x.c2 = cfa_control["c2"];
    if(x.c2 < x.c1 | x.c2 >= 1) Rcpp::stop("The parameter c2 must be a positive number between c1 and 1.");
  } else {
    x.c2 = std::max(x.c1, 0.5);
  }
  if(cfa_control.containsElementNamed("rho")) {
    x.rho = cfa_control["rho"];
    if(x.rho <= 0 | x.rho >= 1) Rcpp::stop("The parameter rho must be a positive number between 0 and 1.");
  }
  if(cfa_control.containsElementNamed("M")) {
    x.M = cfa_control["M"];
  }
  if(cfa_control.containsElementNamed("armijo_maxit")) {
    x.armijo_maxit = cfa_control["armijo_maxit"];
  }

  if(x.random_starts < 1) {
    Rcpp::stop("The number of random starts should be a positive integer");
  }
  if(x.cores < 1) {
    Rcpp::stop("The number of cores should be a positive integer");
  }

  // if(x.estimator == "dwls") {
  //   x.optim = "L-BFGS";
  //   x.lambda_parameters = x.p * x.q - 0.5*x.q*(x.q-1);
  //   x.manifold = "dwls";
  //   x.maxit = 10000;
  //   x.psi = arma::randu(x.lambda_parameters);
  //   // x.psi = arma::randu(x.lambda_parameters + x.p);
  //   x.lambda.set_size(x.p, x.q); x.lambda.zeros();
  //   x.lower_tri_ind = arma::trimatl_ind(arma::size(x.lambda));
  // }

  // Arrange stuff:
  x.Ip.set_size(x.p, x.p); x.Ip.eye();
  x.Iq.set_size(x.q, x.q); x.Iq.eye();
  x.w = arma::vectorise(x.W);
  x.indexes_diag_q.set_size(x.q);
  for(int i=0; i < x.q; ++i) x.indexes_diag_q[i] = i * x.q + i;
  x.indexes_diag_p.set_size(x.p);
  for(int i=0; i < x.p; ++i) x.indexes_diag_p[i] = i * x.p + i;
  x.indexes_diag_q2.set_size(x.q);
  x.indexes_diag_q2[0] = 0;
  for(int i=1; i < x.q; ++i) x.indexes_diag_q2[i] = x.indexes_diag_q2[i-1] + (x.q-i+2);
  x.indexes_p = arma::trimatl_ind(arma::size(x.psi));
  x.indexes_q = arma::trimatl_ind(arma::size(x.phi));
  x.S_indexes = x.indexes_p;

  x.n_lambda = x.lambda_indexes.size();
  x.n_phi = x.phi_indexes.size();
  x.n_psi = x.psi_indexes.size();

  x.lambda.set_size(x.p, x.q); x.lambda.zeros();
  x.phi.set_size(x.q, x.q); x.phi.zeros();
  x.psi.set_size(x.p, x.p); x.psi.zeros();

  x.dlambda.set_size(x.p, x.q); x.dlambda.zeros();
  x.dphi.set_size(x.q, x.q); x.dphi.zeros();
  x.dpsi.set_size(x.p, x.p); x.dpsi.zeros();

}
