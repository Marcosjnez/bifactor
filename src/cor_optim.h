// Optimization algorithms for rotation

class cor_optim {

public:

  virtual cor_NTR optim(arguments_cor x, cor_manifold *manifold,
                        cor_criterion *criterion) = 0;

};

// Riemannian gradient descent:

class cor_RGD:public cor_optim {

public:

  cor_NTR optim(arguments_cor x, cor_manifold *manifold,
                cor_criterion *criterion) {

    return cor_gd(x, manifold, criterion);

  }

};

// Riemannian Newton Trust-Region:

class cor_RNTR:public cor_optim {

public:

  cor_NTR optim(arguments_cor x, cor_manifold *manifold,
                cor_criterion *criterion) {

    return cor_ntr(x, manifold, criterion);

  }

};

// Select the optimization algorithm:

cor_optim* choose_cor_optim(std::string cor) {

  cor_optim* algorithm;
  if(cor == "gradient") {
    algorithm = new cor_RGD();
  } else if(cor == "newtonTR") {
    algorithm = new cor_RNTR();
  } else if(cor == "BFGS") {
  } else if(cor == "L-BFGS") {
  } else {

    Rcpp::stop("Available optimization rutines for rotation: \n 'gradient', 'BFGS', 'L-BFGS', 'newtonTR'. The default method is 'newtonTR'.");

  }

  return algorithm;

}

// [[Rcpp::export]]
Rcpp::List poly2(const arma::mat& X, const int cores) {

  /*
   * Function to estimate the full polychoric correlation matrix
   */

  Rcpp::Timer timer;

  Rcpp::List result;
  const int n = X.n_rows;
  const int q = X.n_cols;

  arma::mat cor = arma::cor(X);
  std::vector<std::vector<int>> cols(q);
  std::vector<int> maxs(q);
  std::vector<std::vector<double>> taus(q);
  std::vector<size_t> s(q);
  std::vector<std::vector<double>> mvphi(q);

  omp_set_num_threads(cores);
#pragma omp parallel for
  for(size_t i = 0; i < q; ++i) {
    cols[i] = arma::conv_to<std::vector<int>>::from(X.col(i));
    maxs[i] = *max_element(cols[i].begin(), cols[i].end());
    std::vector<int> frequencies = count(cols[i], n, maxs[i]-1L);
    mvphi[i] = cumsum(frequencies);
    taus[i] = mvphi[i]; // Cumulative frequencies
    for (size_t j = 0; j < maxs[i]; ++j) {
      mvphi[i][j] /= n;
      taus[i][j] = Qnorm(mvphi[i][j]);
    }
    mvphi[i].push_back(1.0);
    mvphi[i].insert(mvphi[i].begin(), 0.0);
    taus[i].push_back(pos_inf);
    taus[i].insert(taus[i].begin(), neg_inf);
    s[i] = taus[i].size() -1L;
  }

  timer.step("thresholds");

  int K = 0.5*q*(q-1);
  std::vector<std::vector<std::vector<int>>> tabs(K);
  int k = 0;

#ifdef _OPENMP
  omp_set_num_threads(cores);
#pragma omp parallel for
#endif
  for(size_t i=0; i < (q-1L); ++i) {
    for(int j=(i+1L); j < q; ++j) {
      tabs[k] = joint_frequency_table(cols[i], n, maxs[i], cols[j], maxs[j]);
      ++k;
    }
  }

  timer.step("contingency_tables");

  arguments_cor x;
  x.nobs = n;
  x.q = q;
  x.taus = taus;
  x.mvphi = mvphi;
  x.s = s;
  x.n = tabs;
  x.gcor.set_size(q, q); x.gcor.zeros();
  x.dgcor.set_size(q, q); x.dgcor.zeros();
  x.n_pairs = 0.5*q*(q-1);

  // Select one manifold:
  cor_manifold* manifold = choose_cor_manifold("oblq");
  // Select one specific criteria:
  cor_criterion* criterion = choose_cor_criterion("poly");
  // Select the optimization routine:
  cor_optim* algorithm = choose_cor_optim("newtonTR");

  // x.T = init;
  // x.dT = dX;
  // manifold->param(x);
  // criterion->F(x);
  // criterion->gcor(x);
  // manifold->dcor(x);
  // manifold->grad(x);
  // criterion->dgcor(x);
  // manifold->grad(x);
  // manifold->dgrad(x);
  // result["cor"] = x.cor;
  // result["f"] = x.f;
  // result["gcor"] = x.gcor;
  // result["dcor"] = x.dcor;
  // result["grad"] = x.g;
  // result["dgcor"] = x.dgcor;
  // result["dgrad"] = x.dg;
  // result["taus"] = taus;
  // result["mvphi"] = mvphi;
  // result["tabs"] = tabs;
  // return result;

  x.T = random_oblq(q, q);
  cor_NTR x2 = algorithm->optim(x, manifold, criterion);

  timer.step("optimization");

  arma::mat COR = std::get<0>(x2);
  arma::mat T = std::get<1>(x2);
  double obj = std::get<2>(x2);
  int iters = std::get<3>(x2);
  bool converge = std::get<4>(x2);

  result["correlation"] = COR;
  result["X"] = T;
  result["f"] = obj;
  result["convergence"] = converge;
  result["iters"] = iters;
  result["elapsed"] = timer;

  return result;

}

typedef std::tuple<arma::mat,
                   std::vector<std::vector<double>>,
                   std::vector<std::vector<double>>,
                   std::vector<std::vector<std::vector<int>>>,
                   arma::mat,
                   double,
                   arma::mat,
                   bool> polyfast_object;

polyfast_object poly(const arma::mat& X, const bool PD, const bool fit,
                     const int cores) {

  /*
   * Function to estimate the full polychoric correlation matrix
   *                      (Multiple cores)
   */

  const int n = X.n_rows;
  const int q = X.n_cols;

  arma::mat cor = arma::cor(X);
  std::vector<std::vector<int>> cols(q);
  std::vector<int> maxs(q);
  std::vector<std::vector<double>> taus(q);
  std::vector<size_t> s(q);
  std::vector<std::vector<double>> mvphi(q);

  for(size_t i = 0; i < q; ++i) {
    cols[i] = arma::conv_to<std::vector<int>>::from(X.col(i));
    maxs[i] = *max_element(cols[i].begin(), cols[i].end());
    std::vector<int> frequencies = count(cols[i], n, maxs[i]-1L);
    mvphi[i] = cumsum(frequencies);
    taus[i] = mvphi[i]; // Cumulative frequencies
    for (size_t j = 0; j < maxs[i]; ++j) {
      mvphi[i][j] /= n;
      taus[i][j] = Qnorm(mvphi[i][j]);
    }
    mvphi[i].push_back(1.0);
    mvphi[i].insert(mvphi[i].begin(), 0.0);
    taus[i].push_back(pos_inf);
    taus[i].insert(taus[i].begin(), neg_inf);
    s[i] = taus[i].size() -1L;
  }

  arma::mat polys(q, q, arma::fill::eye);
  arma::mat iters(q, q, arma::fill::zeros);

  int d = 0.5*q*(q-1);
  std::vector<std::vector<std::vector<int>>> tabs(d);
  arma::vec seq = arma::linspace(0, q-1, q);
  arma::vec I = arma::cumsum(q - seq) - 2*q;

#ifdef _OPENMP
  omp_set_num_threads(cores);
#pragma omp parallel for
#endif
  for(size_t i=0; i < (q-1L); ++i) {
    for(size_t j=(i+1L); j < q; ++j) {
      int k = I[i+1] + j;
      tabs[k] = joint_frequency_table(cols[i], n, maxs[i], cols[j], maxs[j]);
      std::vector<double> rho = optimize(taus[i], taus[j], tabs[k], s[i], s[j], mvphi[i], mvphi[j], n, cor(i, j));
      polys(i, j) = polys(j, i) = rho[0];
      iters(i, j) = iters(j, i) = rho[1];
    }
  }

  arguments_cor x;
  x.nobs = n;
  x.q = q;
  x.taus = taus;
  x.mvphi = mvphi;
  x.s = s;
  x.n = tabs;
  x.n_pairs = d;

  if(PD) {

    if(!polys.is_sympd()) {

      x.gcor.set_size(q, q); x.gcor.zeros();
      x.dgcor.set_size(q, q); x.dgcor.zeros();

      // Select one manifold:
      cor_manifold* manifold = choose_cor_manifold("oblq");
      // Select one specific criteria:
      cor_criterion* criterion = choose_cor_criterion("poly");
      // Select the optimization routine:
      cor_optim* algorithm = choose_cor_optim("newtonTR");

      // x.T = random_oblq(q, q);
      x.T = arma::real(arma::sqrtmat(polys));
      cor_NTR x2 = algorithm->optim(x, manifold, criterion);

      polys = std::get<0>(x2);
      x.T = std::get<1>(x2);
      x.f = std::get<2>(x2);
      iters = std::get<3>(x2);
      x.convergence = std::get<4>(x2);

    }
  } else if(fit) {

    x.cor = polys;
    // Select one specific criteria:
    cor_criterion* criterion = choose_cor_criterion("poly");
    criterion->F(x);

  }

  polyfast_object result = std::make_tuple(polys, taus, mvphi, tabs, iters,
                                           x.f, x.T, x.convergence);
  return result;

}

polyfast_object poly_no_cores(const arma::mat& X, const bool PD) {

  /*
   * Function to estimate the full polychoric correlation matrix
   *              (Single core; for bootstrapping)
   */

  const int n = X.n_rows;
  const int q = X.n_cols;

  arma::mat cor = arma::cor(X);
  std::vector<std::vector<int>> cols(q);
  std::vector<int> maxs(q);
  std::vector<std::vector<double>> taus(q);
  std::vector<size_t> s(q);
  std::vector<std::vector<double>> mvphi(q);

  for(size_t i = 0; i < q; ++i) {
    cols[i] = arma::conv_to<std::vector<int>>::from(X.col(i));
    maxs[i] = *max_element(cols[i].begin(), cols[i].end());
    std::vector<int> frequencies = count(cols[i], n, maxs[i]-1L);
    mvphi[i] = cumsum(frequencies);
    taus[i] = mvphi[i]; // Cumulative frequencies
    for (size_t j = 0; j < maxs[i]; ++j) {
      mvphi[i][j] /= n;
      taus[i][j] = Qnorm(mvphi[i][j]);
    }
    mvphi[i].push_back(1.0);
    mvphi[i].insert(mvphi[i].begin(), 0.0);
    taus[i].push_back(pos_inf);
    taus[i].insert(taus[i].begin(), neg_inf);
    s[i] = taus[i].size() -1L;
  }

  arma::mat polys(q, q, arma::fill::eye);
  arma::mat iters(q, q, arma::fill::zeros);

  int d = 0.5*q*(q-1);
  std::vector<std::vector<std::vector<int>>> tabs(d);
  int k = 0;

  for(size_t i=0; i < (q-1L); ++i) {
    for(int j=(i+1L); j < q; ++j) {
      tabs[k] = joint_frequency_table(cols[i], n, maxs[i], cols[j], maxs[j]);
      std::vector<double> rho = optimize(taus[i], taus[j], tabs[k], s[i], s[j], mvphi[i], mvphi[j], n, cor(i, j));
      polys(i, j) = polys(j, i) = rho[0];
      iters(i, j) = iters(j, i) = rho[1];
      ++k;
    }
  }

  arguments_cor x;
  x.nobs = n;
  x.q = q;
  x.taus = taus;
  x.mvphi = mvphi;
  x.s = s;
  x.n = tabs;
  x.n_pairs = d;

  if(PD) {

    if(!polys.is_sympd()) {

      x.gcor.set_size(q, q); x.gcor.zeros();
      x.dgcor.set_size(q, q); x.dgcor.zeros();

      // Select one manifold:
      cor_manifold* manifold = choose_cor_manifold("oblq");
      // Select one specific criteria:
      cor_criterion* criterion = choose_cor_criterion("poly");
      // Select the optimization routine:
      cor_optim* algorithm = choose_cor_optim("newtonTR");

      // x.T = random_oblq(q, q);
      x.T = arma::real(arma::sqrtmat(polys));
      cor_NTR x2 = algorithm->optim(x, manifold, criterion);

      polys = std::get<0>(x2);
      x.T = std::get<1>(x2);
      x.f = std::get<2>(x2);
      iters.resize(1, 1);
      iters(0, 0) = std::get<3>(x2);
      x.convergence = std::get<4>(x2);

    }

  }

  polyfast_object result = std::make_tuple(polys, taus, mvphi, tabs, iters,
                                           x.f, x.T, x.convergence);
  return result;

}

Rcpp::List polyfast(const arma::mat& X, std::string acov, const bool PD,
                    const int nboot, const bool fit, const int cores) {

  /*
   * Function to estimate the full polychoric correlation matrix
   */

  Rcpp::Timer timer;

  polyfast_object x = poly(X, PD, false, cores);

  timer.step("polychorics");

  arma::mat polys = std::get<0>(x);
  std::vector<std::vector<double>> taus = std::get<1>(x);
  std::vector<std::vector<double>> mvphis = std::get<2>(x);
  std::vector<std::vector<std::vector<int>>> tabs = std::get<3>(x);
  arma::mat iters = std::get<4>(x);
  double f = std::get<5>(x);
  arma::mat T = std::get<6>(x);
  bool convergence = std::get<7>(x);

  int n = X.n_rows;
  int p = X.n_cols;
  arma::mat covariance;
  if(acov == "var") {
    covariance = DACOV2(n, polys, tabs, taus, mvphis);
  } else if(acov == "cov") {
    covariance = ACOV(X, polys, cores);
  } else if(acov == "bootstrap") {
    int d = 0.5*p*(p-1);
    arma::uvec lower_indices = arma::trimatl_ind(arma::size(polys), -1);
    arma::mat boot_correlations(nboot, d);
#ifdef _OPENMP
    omp_set_num_threads(cores);
#pragma omp parallel for
#endif
    for(int i=0; i < nboot; ++i) {
      arma::uvec indexes = arma::randi<arma::uvec>(n, arma::distr_param(0, n-1));
      polyfast_object polychor = poly_no_cores(X.rows(indexes), PD);
      arma::mat cor_boot = std::get<0>(polychor);
      boot_correlations.row(i) = cor_boot(lower_indices);
    }
    covariance = n*arma::cov(boot_correlations);
  } else if(acov == "none") {
  } else {
    Rcpp::stop("Unknown acov. Use acov = 'cov' to obtain the asymptotic covariance matrix and acov = 'var' to simply obtain the asymptotic variances");
  }

  timer.step("acov");

  Rcpp::List result;
  result["type"] = "polychorics";
  result["correlation"] = polys;
  result["thresholds"] = taus;
  result["contingency_tables"] = tabs;
  result["acov"] = covariance;
  result["iters"] = iters;
  result["f"] = f;
  result["X"] = T;
  result["convergence"] = convergence;
  result["elapsed"] = timer;

  return result;

}

// [[Rcpp::export]]
double fpoly(const arma::mat& X, const arma::mat& S) {

  /*
   * Function to estimate the full polychoric correlation matrix
   */

  const int n = X.n_rows;
  const int q = X.n_cols;

  arma::mat cor = arma::cor(X);
  std::vector<std::vector<int>> cols(q);
  std::vector<int> maxs(q);
  std::vector<std::vector<double>> taus(q);
  std::vector<size_t> s(q);
  std::vector<std::vector<double>> mvphi(q);

  for(size_t i = 0; i < q; ++i) {
    cols[i] = arma::conv_to<std::vector<int>>::from(X.col(i));
    maxs[i] = *max_element(cols[i].begin(), cols[i].end());
    std::vector<int> frequencies = count(cols[i], n, maxs[i]-1L);
    mvphi[i] = cumsum(frequencies);
    taus[i] = mvphi[i]; // Cumulative frequencies
    for (size_t j = 0; j < maxs[i]; ++j) {
      mvphi[i][j] /= n;
      taus[i][j] = Qnorm(mvphi[i][j]);
    }
    mvphi[i].push_back(1.0);
    mvphi[i].insert(mvphi[i].begin(), 0.0);
    taus[i].push_back(pos_inf);
    taus[i].insert(taus[i].begin(), neg_inf);
    s[i] = taus[i].size() -1L;
  }

  int K = 0.5*q*(q-1);
  std::vector<std::vector<std::vector<int>>> tabs(K);
  int k = 0;

  for(size_t i=0; i < (q-1L); ++i) {
    for(int j=(i+1L); j < q; ++j) {
      tabs[k] = joint_frequency_table(cols[i], n, maxs[i], cols[j], maxs[j]);
      ++k;
    }
  }

  arguments_cor x;
  x.nobs = n;
  x.q = q;
  x.taus = taus;
  x.mvphi = mvphi;
  x.s = s;
  x.n = tabs;
  x.n_pairs = 0.5*q*(q-1);
  x.cor = S;
  // Select one specific criteria:
  cor_criterion* criterion = choose_cor_criterion("poly");
  criterion->F(x);

  return x.f;

}

// [[Rcpp::export]]
Rcpp::List old_poly(const arma::mat& X, const int cores) {

  /*
   * Function to estimate the full polychoric correlation matrix (fast)
   */

  Rcpp::Timer timer;

  const int n = X.n_rows;
  const int q = X.n_cols;

  arma::mat cor = arma::cor(X);
  std::vector<std::vector<int>> cols(q);
  std::vector<int> maxs(q);
  std::vector<std::vector<double>> taus(q);
  std::vector<size_t> s(q);
  std::vector<std::vector<double>> mvphi(q);

  for(size_t i = 0; i < q; ++i) {
    cols[i] = arma::conv_to<std::vector<int>>::from(X.col(i));
    maxs[i] = *max_element(cols[i].begin(), cols[i].end());
    std::vector<int> frequencies = count(cols[i], n, maxs[i]-1L);
    mvphi[i] = cumsum(frequencies);
    taus[i] = mvphi[i]; // Cumulative frequencies
    for (size_t j = 0; j < maxs[i]; ++j) {
      mvphi[i][j] /= n;
      taus[i][j] = Qnorm(mvphi[i][j]);
    }
    mvphi[i].push_back(1.0);
    mvphi[i].insert(mvphi[i].begin(), 0.0);
    taus[i].push_back(pos_inf);
    taus[i].insert(taus[i].begin(), neg_inf);
    s[i] = taus[i].size() -1L;
  }

  timer.step("thresholds");

  arma::mat polys(q, q, arma::fill::eye);
  arma::mat iters(q, q, arma::fill::zeros);

#ifdef _OPENMP
  omp_set_num_threads(cores);
#pragma omp parallel for
#endif
  for(size_t i=0; i < (q-1L); ++i) {
    for(int j=(i+1L); j < q; ++j) {
      std::vector<std::vector<int>> tab = joint_frequency_table(cols[i], n, maxs[i], cols[j], maxs[j]);
      std::vector<double> rho = optimize(taus[i], taus[j], tab, s[i], s[j], mvphi[i], mvphi[j], n, cor(i, j));
      polys(i, j) = polys(j, i) = rho[0];
      iters(i, j) = iters(j, i) = rho[1];
    }
  }

  timer.step("polychorics");

  Rcpp::List result;
  result["polychorics"] = polys;
  result["thresholds"] = taus;
  result["iters"] = iters;
  result["elapsed"] = timer;

  return result;

}
