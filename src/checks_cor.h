void check_cor(arguments_efa& x) {

  if(x.X.is_square()) {

    x.R = x.X;
    x.correlation_result["correlation"] = x.R;

  } else {

    x.nobs = x.X.n_rows;
    missingness(x); // Handle missing values

    // ALLOW DWLS for continuous data
    if(x.cor == "poly") {
      if(x.estimator == "dwls") {
        x.correlation_result = polyfast(x.X, x.missing, "var", "none", 0.00, 0L, false, x.cores);
        arma::mat W = x.correlation_result["acov"];
        x.Inv_W = 1/W; x.Inv_W.diag().zeros();
      } else {
        x.correlation_result = polyfast(x.X, x.missing, "none", "none", 0.00, 0L, false, x.cores);
      }
      arma::mat polys = x.correlation_result["correlation"];
      x.R = polys;
    } else if(x.cor == "pearson") {
      if(x.X.has_nan()) {
        x.R = pairwise_cor(x.X);
      } else {
        x.R = arma::cor(x.X);
      }
      x.correlation_result["type"] = "pearson";
      x.correlation_result["correlation"] = x.R;
    } else {
      Rcpp::stop("Unkown correlation method");
    }

  }

}
