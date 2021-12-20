#include "criteria.h"

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

// Solution to quadratic interpolation:

double root_quad(double a, double b, double c) {

  double res = 0.5 * (- b + sqrt(b * b - 4 * a * c) ) / a;

  return res;

}

// Conjugate-gradient method to solve the Riemannian Newton equation:

void tcg(base_manifold *manifold, base_criterion *criterion,
         arma::mat& dir, bool& att_bnd, arma::mat T, arma::mat Inv_T,
         arma::mat lambda, arma::mat L, arma::mat Phi, arma::mat g, arma::mat gr,
         arma::mat A, arma::uvec oblq_indexes, arma::mat L2,
         arma::mat LoL2, arma::vec term, int p, double p2,                     // for geomin
         arma::mat I_gamma_C,  arma::mat IgCL2N, arma::mat N, arma::mat M,  // for oblimin
         arma::mat PhiWeight2, arma::mat Weight2, arma::mat gP,          // for xtarget
         double epsilon, double ng, arma::vec c, double rad, double k,
         std::vector<arma::uvec> blocks) {

  dir.zeros();
  arma::mat dir0, dg, Hd, Inv_T_dt, dL, dP, dgL, dgP;

  double alpha, rr0, tau, beta, dHd;
  arma::mat delta = -gr;
  arma::mat r = delta;
  double rr = ng * ng;
  double tol = ng * std::min(pow(ng, c[0]), c[1]);

  int iter = 0;

  do{

    // Differential of L and P
    manifold->dLP(dL, dP, Inv_T_dt, T, lambda, L, Inv_T, delta);

    // Differential of the gradient of L and P
    criterion->dgLP(dgL, dgP, dL, dP, L, L2, LoL2, term, p2,
                   epsilon, I_gamma_C, IgCL2N, N, M,
                   Weight2, PhiWeight2, k, blocks);

    // Differential of g
    manifold->dgrad(dg, dgL, dgP, gP, lambda, delta, T, Inv_T, L, g, Inv_T_dt);

    // Riemannian hessian
    manifold->hess(Hd, delta, g, dg, T, Phi, A, oblq_indexes);

    dHd = arma::accu(delta % Hd);

    if(dHd <= 0) {

      tau = root_quad(arma::accu(delta % delta), 2 * arma::accu(dir % delta),
                      arma::accu(dir % dir) - rad * rad);
      dir = dir + tau * delta;
      att_bnd = true;

      break;

    }

    rr0 = rr;
    alpha = rr0 / dHd;
    dir0 = dir;
    dir = dir + alpha * delta;

    if (sqrt(arma::accu(dir % dir)) >= rad) {

      tau = root_quad(arma::accu(delta % delta), 2 * arma::accu(dir0 % delta),
                      arma::accu(dir0 % dir0) - rad * rad);
      dir = dir0 + tau * delta;
      att_bnd = true;

      break;

    }

    r = r - alpha * Hd;
    rr = arma::accu(r % r);

    if (sqrt(rr) < tol) {

      att_bnd = false;
      break;

    }

    beta = rr / rr0;
    delta = r + beta * delta;
    iter = iter + 1;

  } while (iter < 5);

}

// Trust-region Riemannian Newton algorithm:

typedef std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> TRN;

TRN NPF(base_manifold *manifold, base_criterion *criterion,
        arma::mat T, arma::mat lambda, arma::mat Target, arma::mat Weight,
        arma::mat Phi_Target, arma::mat Phi_Weight,
        std::vector<arma::uvec> blocks,
        std::vector<arma::uvec> list_oblq_indexes, arma::uvec oblq_indexes,
        double w, double gamma, double epsilon,
        double eps, int max_iter, arma::mat Weight2, arma::mat PhiWeight2,
        arma::mat I_gamma_C, arma::mat N, arma::mat M, double p2, double k) {

  arma::mat Inv_T, Inv_T_dt, L, dL, dgL, dP, f1, f2, g, gL, gr, dg, Hd, L2, LoL2, IgCL2N, A,
  new_T, new_Inv_T, new_L, new_f1, new_f2, new_g, new_gr, new_L2, gP, dgP,
  new_LoL2, new_IgCL2N, new_dg, new_Hd;

  arma::vec term, new_term;

  double f, ng, preddiff, new_f, new_ng;

  int n = lambda.n_rows;
  int p = T.n_rows;
  arma::mat Phi;
  arma::mat new_Phi;
  arma::mat PW = Phi_Weight;

  // Rcpp::Rcout << blocks[1] << std::endl;
  // blocks = list_oblq_indexes;

  // Parameterization
  manifold->param(L, lambda, Phi, Inv_T, T);

  // Objective
  criterion->F(Inv_T, L, Phi, f1, f2, f, lambda, T, Target, Weight,
               L2, term, p, epsilon, IgCL2N, I_gamma_C, N, M,
               Phi_Target, Phi_Weight, w, k, blocks);

  // Gradient wrt L
  criterion->gLP(gL, gP, f1, f2, L, LoL2, L2, term, p2, epsilon,
                 I_gamma_C, IgCL2N, N, M, Weight, Phi_Weight, k,
                 blocks);

  // Gradient wtr T
  manifold->grad(g, lambda, L, gL, Inv_T, T, gP, w);

  // Riemannian gradient
  manifold->proj(gr, A, g, T, Phi, oblq_indexes);

  ng = sqrt(arma::accu(gr % gr));

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
  arma::mat dir(p, p);

  int iteration = 0;
  double goa;

  do{

    if (ng < eps) break;

    iteration = iteration + 1;

    // subsolver
    tcg(manifold, criterion, dir, att_bnd, T, Inv_T,
        lambda, L, Phi, g, gr, A, oblq_indexes, L2,
        LoL2,term, p, p2,  I_gamma_C, IgCL2N, N, M,
        PhiWeight2, Weight2, gP, epsilon, ng, c, rad, k,
        blocks);

    new_T = T + dir;

    // Projection onto the manifold
    manifold->retr(new_T, list_oblq_indexes);

    // Differential of L and P
    manifold->dLP(dL, dP, Inv_T_dt, T, lambda, L, Inv_T, dir);

    // Differential of the gradient of L and P
    criterion->dgLP(dgL, dgP, dL, dP, L, L2, LoL2, term, p2,
                   epsilon, I_gamma_C, IgCL2N, N, M,
                   Weight2, PhiWeight2, k, blocks);

    // Differential of g
    manifold->dgrad(dg, dgL, dgP, gP, lambda, dir, T, Inv_T, L, g, Inv_T_dt);

    // Riemannian hessian
    manifold->hess(Hd, dir, g, dg, T, Phi, A, oblq_indexes);

    preddiff = - arma::accu(dir % ( gr + 0.5 * Hd) );

    // Parameterization
    manifold->param(new_L, lambda, new_Phi, new_Inv_T, new_T);

    // objective
    criterion->F(new_Inv_T, new_L, new_Phi, new_f1, new_f2, new_f, lambda, new_T, Target, Weight,
                 new_L2, new_term, p, epsilon, new_IgCL2N, I_gamma_C, N, M,
                 Phi_Target, Phi_Weight, w, k, blocks);

    if ( std::abs(preddiff) <= arma::datum::eps ) {

      goa = 1;

    } else {

      goa = (f - new_f) / preddiff;

    }
    if (goa < crit_goa[1]) {

      rad = fac_rad[0] * rad;

    } else if (goa > crit_goa[2] && att_bnd) {

      rad = std::min(fac_rad[1] * rad, max_rad);

    }

    // accepted iteration
    if (goa > crit_goa[0]) {

      T = new_T;
      Phi = new_Phi;
      Inv_T = new_Inv_T;
      L = new_L;
      f1 = new_f1;
      f = new_f;

      L2 = new_L2;
      LoL2 = new_LoL2;
      term = new_term;

      IgCL2N = new_IgCL2N;

      f2 = new_f2;

      // update gradient
      criterion->gLP(gL, gP, f1, f2, L, LoL2, L2, term, p2, epsilon,
                     I_gamma_C, IgCL2N, N, M, Weight, Phi_Weight, k, blocks);
      manifold->grad(g, lambda, L, gL, Inv_T, T, gP, w);

      // Riemannian gradient
      manifold->proj(gr, A, g, T, Phi, oblq_indexes);

      ng = sqrt(arma::accu(gr % gr));

    }

  } while (iteration <= max_iter);


  bool convergence = true;
  if(iteration > max_iter) {

    convergence = false;

  }

  TRN result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

