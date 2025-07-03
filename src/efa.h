/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 28/05/2022
 *
 */

// #include <Rcpp/Benchmark/Timer.h>
// #include "structures.h"
// #include "manifolds.h"
// #include "multiple_rotations.h"
// #include "NPF.h"
// #include "criteria.h"
// #include "EFA_fit.h"
// #include "checks.h"

void extract_efa(arguments_efa& x, efa_manifold *manifold, efa_criterion *criterion) {

  efa_NTR x1;
  // Select the optimization routine:
  efa_optim* algorithm = choose_efa_optim(x.optim);

  arguments_efa args = x;
  // args.parameters = x.init;

  // Rprintf("26");
  x1 = algorithm->optim(args, manifold, criterion);
  // Rprintf("28");

  x.lambda = std::get<0>(x1);
  x.uniquenesses = std::get<1>(x1);
  x.Rhat = std::get<2>(x1);
  x.f = std::get<3>(x1);
  x.iterations = std::get<4>(x1);
  x.convergence = std::get<5>(x1);

  // if(!x.convergence) {
  //
  //   Rcpp::Rcout << "\n" << std::endl;
  //   Rcpp::warning("Failed rotation convergence");
  //
  // }

}

Rcpp::List efa(arguments_efa x, efa_manifold* manifold, efa_criterion* criterion,
               int random_starts, int cores) {

  Rcpp::List result;
  // result["R"] = x.R;
  // result["Rhat"] = x.Rhat;
  // result["W"] = x.W;
  // result["lambda"] = x.lambda;
  // result["init"] = x.init;
  // result["estimator"] = x.estimator;
  // result["p"] = x.p;
  // result["q"] = x.q;
  // return result;

  if(x.estimator == "ml" || x.estimator == "uls" || x.estimator == "dwls") {

    // Rprintf("%g ", 60.00);
    // Rprintf("R rows %zu ", x.R.n_rows);
    // Rprintf("R rows %zu ", x.R.n_rows);
    // Rprintf("lambda rows %zu ", x.lambda.n_rows);
    // Rprintf("lambda cols %zu ", x.lambda.n_cols);
    // Rprintf("W rows %zu ", x.W.n_rows);
    // Rprintf("W cols %zu ", x.W.n_cols);
    // Rprintf("p %zu ", x.p);
    // Rprintf("q %zu ", x.q);
    // Rprintf("lower_tri_ind %zu ", x.lower_tri_ind.n_elem);
    // Rprintf("parameters %zu ", x.parameters.n_elem);

    extract_efa(x, manifold, criterion);
    // Rprintf("%g ", 62.00);
    // result["R"] = x.R;
    // result["Rhat"] = x.Rhat;
    // result["W"] = x.W;
    // result["lambda"] = x.lambda;
    // result["init"] = x.init;
    // result["estimator"] = x.estimator;
    // result["p"] = x.p;
    // result["q"] = x.q;
    // return result;

  } else if(x.estimator == "pa") {

    Rcpp::List pa_result = principal_axis(x.parameters, x.R, x.q, x.eps, x.maxit);

    arma::mat w_temp = pa_result["lambda"];
    arma::vec uniquenesses_temp = pa_result["uniquenesses"];
    arma::mat Rhat_temp = pa_result["Rhat"];

    result["f"] = pa_result["f"];
    result["convergence"] = pa_result["convergence"];
    x.lambda = w_temp;
    x.uniquenesses = uniquenesses_temp;
    x.Rhat = Rhat_temp;
    x.iterations = pa_result["iterations"];

  } else if (x.estimator == "minrank") {

    arma::vec communalities = sdp_cpp(x.R);
    x.parameters = 1 - communalities;
    arma::mat reduced_R = x.R - diagmat(x.parameters);

    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, reduced_R);

    arma::vec eigval2 = reverse(eigval);
    arma::mat eigvec2 = reverse(eigvec, 1);

    arma::mat A = eigvec2(arma::span::all, arma::span(0, x.q-1));
    arma::vec eigenvalues = eigval2(arma::span(0, x.q-1));
    for(int i=0; i < x.q; ++i) {
      if(eigenvalues(i) < 0) eigenvalues(i) = 0;
    }
    arma::mat D = arma::diagmat(sqrt(eigenvalues));

    x.lambda = A * D;
    x.Rhat = x.lambda * x.lambda.t();
    x.uniquenesses = 1 - arma::diagvec(x.Rhat);
    x.Rhat.diag() = x.R.diag();
    x.iterations = 0;

  } else {

    Rcpp::stop("Unkown estimator");

  }

  x.heywood = arma::any( arma::vectorise(x.uniquenesses) <= 0 );

  result["lambda"] = x.lambda;
  result["uniquenesses"] = x.uniquenesses;
  result["Rhat"] = x.Rhat;
  result["residuals"] = x.R - x.Rhat;
  result["heywood"] = x.heywood;
  result["f"] = x.f;
  result["estimator"] = x.estimator;
  result["iterations"] = x.iterations;
  result["convergence"] = x.convergence;

  return result;

}
