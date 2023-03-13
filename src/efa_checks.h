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

void check_efa(arguments_efa& x, Rcpp::Nullable<Rcpp::List> nullable_efa_control) {

  if(x.R.n_rows < x.q) Rcpp::stop("Too many factors");

  Rcpp::List efa_control;

  if (nullable_efa_control.isNotNull()) {
    efa_control = Rcpp::as<Rcpp::List>(nullable_efa_control);
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

}
