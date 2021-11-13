double f_cfa_uls(arma::vec x, arma::mat R, int n_factors, arma::uvec indexes) {

  int n_items = R.n_rows;
  int n_parameters = x.size();
  int n_loadings = n_parameters - n_items;

  arma::mat loadings(n_items, n_factors, arma::fill::zeros);
  loadings(indexes) = x(arma::span(0, n_loadings-1));
  arma::vec uniquenesses = x(arma::span(n_loadings, n_parameters));
  arma::mat Rhat = loadings * loadings.t() + arma::diagmat(uniquenesses);
  arma::mat residuals = R - Rhat;

  double objective = 0.5*arma::accu(residuals % residuals);

  return objective;

}

arma::vec g_cfa_uls(arma::vec x, arma::mat R, int n_factors, arma::uvec indexes) {

  int n_items = R.n_rows;
  int n_parameters = x.size();
  int n_loadings = n_parameters - n_items;

  arma::mat loadings(n_items, n_factors, arma::fill::zeros);
  loadings(indexes) = x(arma::span(0, n_loadings-1));
  arma::vec uniquenesses = x(arma::span(n_loadings, n_parameters));
  arma::mat Rhat = loadings * loadings.t() + arma::diagmat(uniquenesses);
  arma::mat residuals = R - Rhat;

  arma::mat g_loadings = -2*residuals * loadings;
  arma::vec gradient(n_parameters);
  gradient(arma::span(0, n_loadings-1)) = g_loadings(indexes);
  gradient(arma::span(n_loadings, n_parameters)) = -diagvec(residuals);

  return gradient;

}
