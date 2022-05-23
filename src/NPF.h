/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

// #include "structures.h"
// #include "auxiliary_manifolds.h"
// #include "manifold.h"
// #include "auxiliary_criteria.h"
// #include "criteria.h"

std::vector<int> subvector(std::vector<int> v, int lower, int upper) {

  std::vector<int> subv(&v[lower], &v[upper]);

  return subv;

}

std::vector<std::vector<int>> subvectors(std::vector<std::vector<int>> v, int lower, int upper) {

  std::vector<std::vector<int>> subv(&v[lower], &v[upper]);

  return subv;

}

// Gram-Schmidt process:

arma::mat gram(arma::mat X) {

  int n = X.n_rows;
  int k = X.n_cols;
  X.col(0) /= sqrt(arma::accu(X.col(0) % X.col(0)));

  for(int i=1; i < k; ++i) {

    for(int j=0; j < i; ++j) {

      X.col(i) -= arma::accu(X.col(j) % X.col(i)) / arma::accu(X.col(j) % X.col(j)) * X.col(j);

    }

    X.col(i) /= sqrt(arma::accu(X.col(i) % X.col(i)));

  }

  return X;

}

double root_quad(double a, double b, double c) {

  // Solution to quadratic interpolation:

  double res = 0.5 * (- b + sqrt(b * b - 4 * a * c) ) / a;

  return res;

}

// Conjugate-gradient method to solve the Riemannian Newton equation:

void tcg(base_manifold *manifold, base_criterion *criterion, arguments_rotate x,
         arma::mat& dir, bool& att_bnd, double ng, arma::vec c, double rad) {

  /*
   * Truncated conjugate gradient sub-solver for the trust-region sub-problem
   * From Liu (Algorithm 4; 2020)
   */

  dir.zeros();
  arma::mat dir0;

  double alpha, rr0, tau, beta, dHd;
  x.dT = -x.rg; // Initial search direction
  arma::mat r = x.dT; // Initial residual
  double rr = ng * ng;
  double tol = ng * std::min(pow(ng, c[0]), c[1]);

  int iter = 0;

  do{

    // Differential of L and P
    manifold->dLP(x);

    // Differential of the gradient of L and P
    criterion->dgLP(x);

    // Differential of g
    manifold->dgrad(x);

    // Riemannian hessian
    manifold->hess(x);

    dHd = arma::accu(x.dT % x.dH);

    if(dHd <= 0) {

      tau = root_quad(arma::accu(x.dT % x.dT), 2 * arma::accu(dir % x.dT),
                      arma::accu(dir % dir) - rad * rad); // Solve equation 39
      dir = dir + tau * x.dT;
      att_bnd = true;

      break;

    }

    rr0 = rr;
    alpha = rr0 / dHd;
    dir0 = dir;
    dir = dir + alpha * x.dT; // update proposal

    if (sqrt(arma::accu(dir % dir)) >= rad) {

      tau = root_quad(arma::accu(x.dT % x.dT), 2 * arma::accu(dir0 % x.dT),
                      arma::accu(dir0 % dir0) - rad * rad); // Solve equation 39
      dir = dir0 + tau * x.dT;
      att_bnd = true;

      break;

    }

    r = r - alpha * x.dH; // update gradient
    rr = arma::accu(r % r);

    if (sqrt(rr) < tol) {

      att_bnd = false;
      break;

    }

    beta = rr / rr0;
    x.dT = r + beta * x.dT;
    iter = iter + 1;

  } while (iter < 5);

}

// Newton Trust-region algorithm:

typedef std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> TRN;

TRN NPF(arguments_rotate x, base_manifold *manifold, base_criterion *criterion,
        double eps, int max_iter) {

  /*
   * Riemannian trust-region algorithm
   * From Liu (Algorithm 2; 2020)
   */

  // Parameterization
  manifold->param(x); // update x.L, x.Phi and x.Inv_T

  // Objective
  criterion->F(x); // update x.f, x.L2, x.IgCL2N, x.term, x.f1 and x.f2

  // Rcpp::Rcout << "x.f = " << x.f << std::endl;

  // Gradient wrt L
  criterion->gLP(x); // update x.gL, x.gP, x.f1, x.f2 and x.LoL2

  // Rcpp::Rcout << "x.gL = " << x.gL << std::endl;

  // Gradient wtr T
  manifold->grad(x); // update x.g

  // Riemannian gradient
  manifold->proj(x); // update x.rg and x.A

  // Differential of the gradient of L and P
  // criterion->dgLP(x); // update dgL and dgP

  // Rcpp::Rcout << "x.dgL = " << x.dgL << std::endl;

  double ng = sqrt(arma::accu(x.rg % x.rg));

  double max_rad = 10;

  arma::vec fac_rad(2);
  fac_rad[0] = 0.25;
  fac_rad[1] = 2;

  arma::vec crit_goa(3);
  crit_goa[0] = 0.2;
  crit_goa[1] = 0.25;
  crit_goa[2] = 0.75;

  arma::vec c(2);
  c[0] = 1;
  c[1] = 0.01;

  double rad = 1;
  bool att_bnd = false;

  int iteration = 0;
  double goa, preddiff;

  arguments_rotate new_x;
  arma::mat dir(x.q, x.q);

  do {

    if (ng < eps) break;

    ++iteration;

    // subsolver
    tcg(manifold, criterion, x, dir, att_bnd, ng, c, rad);
    x.dT = dir;
    new_x = x;
    new_x.T += dir;

    // Projection onto the manifold
    manifold->retr(new_x); // update x.T

    // Differential of L and P
    manifold->dLP(x); // update x.dL, x.dP and Inv_T_dt

    // Differential of the gradient of L and P
    criterion->dgLP(x); // update dgL and dgP

    // Differential of g
    manifold->dgrad(x); // update dg

    // Riemannian hessian
    manifold->hess(x); // update dH

    preddiff = - arma::accu(x.dT % ( x.rg + 0.5 * x.dH) );

    // Parameterization
    manifold->param(new_x);

    // objective
    criterion->F(new_x);

    if ( std::abs(preddiff) <= arma::datum::eps ) {

      goa = 1;

    } else {

      goa = (x.f - new_x.f) / preddiff;

    }
    if (goa < crit_goa[1]) {

      rad = fac_rad[0] * rad;

    } else if (goa > crit_goa[2] && att_bnd) {

      rad = std::min(fac_rad[1] * rad, max_rad);

    }

    // accepted iteration
    if (goa > crit_goa[0]) {

      x = new_x;

      // update gradient
      criterion->gLP(x);
      manifold->grad(x);

      // Riemannian gradient
      manifold->proj(x);

      ng = sqrt(arma::accu(x.rg % x.rg));

    }

  } while (iteration <= max_iter);


  bool convergence = true;
  if(iteration > max_iter) {

    convergence = false;

  }

  TRN result = std::make_tuple(x.L, x.Phi, x.T, x.f, iteration, convergence);

  return result;

}

