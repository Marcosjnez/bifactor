// Conjugate-gradient method to solve the Riemannian Newton equation in the Trust-Region algorithm:

void tcg(cor_rotate x, cor_manifold *manifold, cor_criterion *criterion,
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

typedef std::tuple<arma::mat, arma::mat, double, int, bool> cor_NTR;
cor_NTR ntr(cor_rotate x, cor_manifold *manifold, cor_criterion *criterion) {

  /*
   * Riemannian trust-region algorithm
   * From Liu (Algorithm 2; 2020)
   */

  // Parameterization
  manifold->param(x); // update

  // Objective
  criterion->F(x); // update

  // Gradient wrt cor
  criterion->gcor(x); // update

  // Gradient wtr T
  manifold->grad(x); // update

  // Riemannian gradient
  manifold->proj(x); // update

  x.ng = sqrt(arma::accu(x.rg % x.rg));

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

  x.iteration = 0;
  double goa, preddiff;

  arguments_cor new_x;
  arma::mat dir(x.q, x.q);

  x.convergence = false;

  do{

    // subsolver
    tcg(x, manifold, criterion, dir, att_bnd, x.ng, c, rad);
    x.dT = dir;
    new_x = x;
    new_x.T += dir;

    // Projection onto the manifold
    manifold->retr(new_x); // update x.T

    // Differential of cor
    manifold->dcor(x); // update x.dcor

    // Differential of the gradient of cor
    criterion->dgcor(x); // update x.dgcor

    // Differential of g
    manifold->dgrad(x); // update x.dg

    // Riemannian hessian
    manifold->hess(x); // update x.dH

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

      // update the gradient
      criterion->gcor(x);
      manifold->grad(x);

      // Riemannian gradient
      manifold->proj(x);

      x.ng = sqrt(arma::accu(x.rg % x.rg));

    }

    ++x.iteration;
    if (x.ng < x.eps) {
      x.convergence = true;
      break;
    }

  } while (x.iteration < x.maxit);

  cor_NTR result = std::make_tuple(x.cor, x.T, x.f, x.iteration, x.convergence);

  return result;

}
