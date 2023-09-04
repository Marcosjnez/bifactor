void check_efa(arma::mat R, int nfactors, Rcpp::Nullable<arma::vec> nullable_init,
               arma::vec& init, Rcpp::Nullable<Rcpp::List> nullable_efa_control,
               int& efa_maxit, int& lmm, double& efa_factr) {

  if(R.n_rows < nfactors) Rcpp::stop("Too many factors");

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

void check_efa(arguments_efa& x) {

  Rcpp::List efa_control;
  // Check control parameters:
  if (x.nullable_efa_control.isNotNull()) {
    efa_control = Rcpp::as<Rcpp::List>(x.nullable_efa_control);
  }

  if(x.p < x.q) Rcpp::stop("Too many factors");

  // Choose custom weight matrix for the dwls estimator:
  if(efa_control.containsElementNamed("Inv_W")) {
    arma::mat Inv_W = efa_control["Inv_W"];
    x.Inv_W = Inv_W;
  } else if(x.Inv_W.is_empty() & x.estimator == "dwls") {
    Rcpp::stop("For the dwls estimator, please introduce the raw data or provide the reciprocal of the variance of the correlations in efa_control = list(Inv_W = ...)");
  }

  if(x.estimator == "dwls") {
    if(x.cor == "poly") {
      Rcpp::stop("The dwls estimator is only available for cor = 'poly'");
    }
    if(x.optim == "gradient") {
      Rcpp::warning("To achive convergence with the dwls estimator and gradient optim algorithm, you may need to increase the number of maximum iterations: efa_control = list(maxit = 100000)");
    }
  }

  // Check initial values:
  if (x.nullable_init.isNotNull()) {
    Rcpp::warning("Initial values not available for the dwls estimator");
    x.psi = Rcpp::as<arma::vec>(x.nullable_init);
  } else { // Check for positive definiteness only if custom init values are not specified
    if(x.R.is_sympd()) {
      x.psi = 1/arma::diagvec(arma::inv_sympd(x.R));
    } else {
      x.smoothed = smoothing(x.R, 0.001);
      x.psi = 1/arma::diagvec(arma::inv_sympd(x.smoothed));
    }
  }

  if(efa_control.containsElementNamed("projection")) {
    std::string manifold = efa_control["projection"];
    x.manifold = manifold;
  }

  if(efa_control.containsElementNamed("optim")) {
    std::string optim = efa_control["optim"];
    x.optim = optim;
  }
  if(efa_control.containsElementNamed("search")) {
    std::string search = efa_control["search"];
    x.search = search;
    if(x.search != "back" & x.search != "wolfe") {
      Rcpp::stop("Unkown line-search method. Available methods: back and wolfe");
    }
  } else {
    x.search = "back";
  }
  if(efa_control.containsElementNamed("maxit") ){
    x.maxit = efa_control["maxit"];
  } else {
    x.maxit = 1e3;
  }

  if(efa_control.containsElementNamed("upper")) {
    arma::vec upper = efa_control["upper"];
    x.upper = upper;
  } else {
    x.upper = arma::diagvec(x.R);
  }
  if(efa_control.containsElementNamed("lower")) {
    arma::vec lower = efa_control["lower"];
    x.lower = lower;
  } else {
    x.lower.set_size(x.p);
    for(int i=0; i < x.p; ++i) x.lower[i] = 0.005;
  }
  if(efa_control.containsElementNamed("c1")) {
    x.c1 = efa_control["c1"];
    if(x.c1 <= 0 | x.c1 >= 1) Rcpp::stop("The parameter c1 must be a positive number lower than 1.");
  } else {
    if(x.optim == "gradient") {
      x.c1 = 0.5;
    } else {
      x.c1 = 10e-04;
    }
  }
  if(efa_control.containsElementNamed("c2")) {
    x.c2 = efa_control["c2"];
    if(x.c2 < x.c1 | x.c2 >= 1) Rcpp::stop("The parameter c2 must be a positive number between c1 and 1.");
  } else {
    x.c2 = std::max(x.c1, 0.5);
  }
  if(efa_control.containsElementNamed("rho")) {
    x.rho = efa_control["rho"];
    if(x.rho <= 0 | x.rho >= 1) Rcpp::stop("The parameter rho must be a positive number between 0 and 1.");
  }
  if(efa_control.containsElementNamed("M")) {
    x.M = efa_control["M"];
  }
  if(efa_control.containsElementNamed("armijo_maxit")) {
    x.armijo_maxit = efa_control["armijo_maxit"];
  }

  if(x.random_starts < 1) {
    Rcpp::stop("The number of random starts should be a positive integer");
  }
  if(x.cores < 1) {
    Rcpp::stop("The number of cores should be a positive integer");
  }

  if(x.estimator == "dwls") {
    x.optim = "L-BFGS";
    x.lambda_parameters = x.p * x.q - 0.5*x.q*(x.q-1);
    x.manifold = "dwls";
    x.maxit = 10000;
    x.psi = arma::randu(x.lambda_parameters);
    // x.psi = arma::randu(x.lambda_parameters + x.p);
    x.lambda.set_size(x.p, x.q); x.lambda.zeros();
    x.lower_tri_ind = arma::trimatl_ind(arma::size(x.lambda));
  }

}
