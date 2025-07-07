void check_cor(arguments_cor& x) {

  if(x.estimator == "dwls") { // Weighting

    if(x.X.is_square()) {

      x.R = x.X;
      x.correlation_result["correlation"] = x.R;

      arma::vec asymp_diag;
      asymp_diag = arma::diagvec(asymptotic_normal(x.R));
      x.correlation_result["std_error"] = "normal";
      arma::mat W = arma::reshape(asymp_diag, x.p, x.p);
      x.W = 1/W; x.W.diag().zeros();
      // Rcpp::Rcout << "Full data was not provided. The variance of the correlations were estimated assuming the items are normally distributed." << std::endl;

    } else {

      x.nobs = x.X.n_rows;
      missingness(x); // Handle missing values

      if(x.cor == "poly") {

        x.correlation_result = polyfast(x.X, x.missing, "var", "none", 0.00, 0L, false, x.cores);
        arma::mat W = x.correlation_result["acov"];
        x.W = 1/W; x.W.diag().zeros();
        arma::mat polys = x.correlation_result["correlation"];
        x.R = polys;
        x.correlation_result["type"] = "poly";

      } else if(x.cor == "pearson") {

        if(x.X.has_nan()) {
          x.R = pairwise_cor(x.X);
        } else {
          x.R = arma::cor(x.X);
        }

        arma::vec asymp_diag;
        asymp_diag = arma::diagvec(asymptotic_general(x.X));
        x.correlation_result["std_error"] = "general";
        arma::mat W = arma::reshape(asymp_diag, x.p, x.p);
        x.W = 1/W; x.W.diag().zeros();
        x.correlation_result["type"] = "pearson";

      } else {
        Rcpp::stop("Available correlations: 'pearson' and 'poly'.");
      }

    }

  } else { // No weighting

    if(x.X.is_square()) {

      x.R = x.X;
      x.correlation_result["correlation"] = x.R;

    } else {

      x.nobs = x.X.n_rows;
      missingness(x); // Handle missing values

      if(x.cor == "poly") {

        x.correlation_result = polyfast(x.X, x.missing, "none", "none", 0.00, 0L, false, x.cores);
        arma::mat polys = x.correlation_result["correlation"];
        x.R = polys;
        x.correlation_result["type"] = "poly";

      } else if(x.cor == "pearson") {

        if(x.X.has_nan()) {
          x.R = pairwise_cor(x.X);
        } else {
          x.R = arma::cor(x.X);
        }
        x.correlation_result["type"] = "pearson";

      }  else {
        Rcpp::stop("Available correlations: 'pearson' and 'poly'.");
      }

    }

    arma::mat W(x.p, x.p); W.ones(); W.diag().zeros();
    x.W = W;

  }

}



