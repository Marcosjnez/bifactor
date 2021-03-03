double root_quad(double a, double b, double c) {

  double res = 0.5 * (- b + sqrt(b * b - 4 * a * c) ) / a;

  return res;

}

// xtarget

void f_xtarget(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi, arma::mat& f1, arma::mat& f2, double& f,
               arma::mat A, arma::mat T, arma::mat Target, arma::mat Weight, arma::mat Phi_Target, arma::mat Phi_Weight, double w) {

  Phi = T.t() * T;
  Inv_T = inv(T);
  L = A * Inv_T.t();
  f1 = Weight % (L - Target);
  f2 = Phi_Weight % (Phi - Phi_Target);

  f = 0.5*arma::accu(pow(f1, 2)) + 0.25*w*arma::accu(pow(f2, 2));

}

void g_xtarget(arma::mat& g, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat Weight, arma::mat Phi_Weight, double w) {

  arma::mat df1_dL = Weight % f1;
  arma::mat df1_dt = - Inv_T.t() * df1_dL.t() * L;
  arma::mat df2_dt = T * (Phi_Weight % f2);

  g = df1_dt + w*df2_dt;

}

arma::mat dg_xtarget(arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, arma::mat g, double w) {

  arma::mat Inv_T_dt = Inv_T * dT;
  arma::mat dL = - L * Inv_T_dt.t();

  arma::mat dg1L = W2 % dL;
  arma::mat TtdT = T.t() * dT;
  arma::mat dg2 = dT * (PW % f2) + T * (PW2 % (TtdT.t() + TtdT));

  arma::mat dg1 = - g * Inv_T_dt.t() - (dT * Inv_T).t() * g - (dg1L * Inv_T).t() * L;

  arma::mat dg = dg1 + w*dg2;

  return dg;

}

void tcg_xtarget(arma::mat& dir, bool& att_bnd, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g, arma::mat gr, arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2,
        double w, double ng, arma::vec c, double rad) {

  dir.zeros();
  arma::mat dir0, dg, Hd;
  double alpha, rr0, tau, beta, dHd;

  arma::mat delta = -gr;
  arma::mat r = delta;
  double rr = ng * ng;
  double tol = ng * std::min(pow(ng, c[0]), c[1]);

  int iter = 0;

  do{

    dg = dg_xtarget(delta, T, Inv_T, L, W2, f2, PW, PW2, g, w);
    Hd = dg - T * diagmat( T.t() * dg ) - delta * diagmat( T.t() * g);

    dHd = arma::accu(delta % Hd);

    if(dHd <= 0) {

      tau = root_quad(arma::accu(delta % delta), 2 * arma::accu(dir % delta),
                             arma::accu(dir % dir) - rad * rad);
      dir = dir + tau * delta;
      att_bnd = true;

      // break;

    }

    rr0 = rr;
    alpha = rr0 / dHd;
    dir0 = dir;
    dir = dir + alpha * delta;

    if (sqrt(arma::accu(pow(dir, 2))) >= rad) {

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

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> NPF_xtarget(arma::mat T, arma::mat A, arma::mat Target, arma::mat Weight, arma::mat Phi_Target, arma::mat Phi_Weight,
                 double w, double eps, int max_iter) {

  arma::mat W2, PW2, Inv_T, L, Phi, f1, f2, g, gr, dg, Hd;
  double f, ng, preddiff;

  arma::mat new_T, new_Inv_T, new_L, new_Phi, new_f1, new_f2, new_g, new_gr;
  double new_f, new_ng;

  W2 = pow(Weight, 2);
  PW2 = pow(Phi_Weight, 2);

  // Objective
  f_xtarget(Inv_T, L, Phi, f1, f2, f,
            A, T, Target, Weight, Phi_Target, Phi_Weight, w);

  // Gradient
  g_xtarget(g, f1, f2, T, Inv_T, L, Weight, Phi_Weight, w);

  // Riemannian gradient
  gr = g - T * diagmat( T.t() * g );

  ng = sqrt(arma::accu(pow(gr, 2)));
  // ng = sqrt(trace(gr.t() * gr));

  double max_rad = 10;

  arma::vec fac_rad(2);
  fac_rad(0) = 0.25;
  fac_rad(1) = 2;

  arma::vec crit_goa(3);
  crit_goa(0) = 0.2;
  crit_goa(1) = 0.25;
  crit_goa(2) = 0.75;

  arma::vec c(2);
  c(0) = 1;
  c(1) = 0.01;

  double rad = 1;
  bool att_bnd = false;
  int p = T.n_rows;
  arma::mat dir(p, p);

  int iteration = 0;
  double goa;

  do{

    iteration = iteration + 1;

    if (ng < eps) break;

    // subsolver
    tcg_xtarget(dir, att_bnd, T, Inv_T, L, g, gr, W2, f2, Phi_Weight, PW2,  w, ng, c, rad);

    new_T = T + dir;

    // Projection onto the oblique manifold
    new_T *= diagmat(1 / sqrt(diagvec(new_T.t() * new_T)));

    // Differential of g
    dg = dg_xtarget(dir, T, Inv_T, L, W2, f2, Phi_Weight, PW2, g, w);

    // Riemannian hessian
    Hd = dg - T * diagmat( T.t() * dg ) - dir * diagmat( T.t() * g);

    preddiff = - arma::accu(dir % ( gr + 0.5 * Hd) );

    // objective
    f_xtarget(new_Inv_T, new_L, new_Phi, new_f1, new_f2, new_f,
              A, new_T, Target, Weight, Phi_Target, Phi_Weight, w);

    if ( std::abs(preddiff) <= arma::datum::eps ) {

      goa = 1;

    } else {

      goa = (f - new_f) / preddiff;

    }
      if (goa < crit_goa(1)) {

        rad = fac_rad(0) * rad;

      } else if (goa > crit_goa(2) && att_bnd) {

          rad = std::min(fac_rad(1) * rad, max_rad);

  }

    // accepted iteration
    if (goa > crit_goa(0)) {

      T = new_T;
      Phi = new_Phi;
      Inv_T = new_Inv_T;
      L = new_L;
      f1 = new_f1;
      f2 = new_f2;
      f = new_f;

      // update gradient
      g_xtarget(g, f1, f2, T, Inv_T, L, Weight, Phi_Weight, w);

      // Riemannian gradient
      gr = g - T * diagmat( T.t() * g );
      ng = sqrt(arma::accu(pow(gr, 2)));
      // ng = sqrt(trace(gr.t() * gr));

    }

  } while (iteration < max_iter);

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

// Oblimin

void f_oblimin(arma::mat& Inv_T, arma::mat& L, arma::mat& L2, arma::mat& Phi, arma::mat& IgCL2N, double& f,
               arma::mat A, arma::mat T, arma::mat I_gamma_C, arma::mat N) {

  Phi = T.t() * T;
  Inv_T = inv(T);
  L = A * Inv_T.t();
  L2 = pow(L, 2);
  IgCL2N = I_gamma_C * L2 * N;

  f = trace(L2.t() * IgCL2N) / 4;

}

void g_oblimin(arma::mat& g, arma::mat Inv_T, arma::mat L, arma::mat IgCL2N) {

  arma::mat Gq = L % IgCL2N;

  g = - Inv_T.t() * Gq.t() * L;

}

arma::mat dg_oblimin(arma::mat dT, arma::mat Inv_T, arma::mat L, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat g) {

  arma::mat Inv_T_dt = Inv_T * dT;
  arma::mat dL = - L * Inv_T_dt.t();
  arma::mat dGq = dL % IgCL2N + L % (I_gamma_C * (2*dL % L) * N);

  arma::mat dg = - g * (Inv_T * dT).t() - (dT * Inv_T).t() * g - (dGq * Inv_T).t() * L;

  return dg;

}

void tcg_oblimin(arma::mat& dir, bool& att_bnd, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g, arma::mat gr, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N,
         double ng, arma::vec c, double rad) {

  dir.zeros();
  arma::mat dir0, dg, Hd;
  double alpha, rr0, tau, beta, dHd;

  arma::mat delta = -gr;
  arma::mat r = delta;
  double rr = ng * ng;
  double tol = ng * std::min(pow(ng, c[0]), c[1]);

  int iter = 0;

  do{

    dg = dg_oblimin(delta, Inv_T, L, I_gamma_C, IgCL2N, N, g);

    Hd = dg - T * diagmat( T.t() * dg ) - delta * diagmat( T.t() * g);
    dHd = arma::accu(delta % Hd);

    if(dHd <= 0) {

      tau = root_quad(arma::accu(delta % delta), 2 * arma::accu(dir % delta),
                      arma::accu(dir % dir) - rad * rad);
      dir = dir + tau * delta;
      att_bnd = true;

      break;

      // if(iter == 0) {
      //
      //   dir = delta;
      //
      // }

    }

    rr0 = rr;
    alpha = rr0 / dHd;
    dir0 = dir;
    dir = dir + alpha * delta;

    if (sqrt(arma::accu(pow(dir, 2))) >= rad) {

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

  } while (iter < 10);

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> NPF_oblimin(arma::mat T, arma::mat A, double gamma = 0, double eps = 1e-05, int max_iter = 10000) {

  arma::mat Inv_T, L, L2, Phi, IgCL2N, g, gr, dg, Hd;
  double f, ng, preddiff;

  arma::mat new_T, new_Inv_T, new_L, new_L2, new_Phi, new_IgCL2N, new_g, new_gr, new_dg, new_Hd;
  double new_f, new_ng;

  int p = T.n_rows;
  int n = A.n_rows;

  arma::mat I(n, n, arma::fill::eye);
  arma::mat gamma_C(n, n, arma::fill::ones);
  double gamma_n = gamma/n;
  gamma_C *= gamma_n;
  arma::mat N(p, p, arma::fill::ones);
  N.diag(0).zeros();
  arma::mat I_gamma_C = (I - gamma_C);

  f_oblimin(Inv_T, L, L2, Phi, IgCL2N, f,
            A, T, I_gamma_C, N);

  g_oblimin(g, Inv_T, L, IgCL2N);

  // Riemannian gradient

  gr = g - T * diagmat( T.t() * g );

  ng = sqrt(arma::accu(pow(gr, 2)));

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

    iteration = iteration + 1;

    // if (verbose) {
    //
    //   Rcpp::Rcout << "\r Iteration " << iteration << " f = " << f << " norm(gr) = " << ng << "\r";
    //
    // }

    if (ng < eps) break;

    // subsolver
    tcg_oblimin(dir, att_bnd, T, Inv_T, L, g, gr, I_gamma_C, IgCL2N, N, ng, c, rad);
    new_T = T + dir;

    // Projection onto the oblique manifold
    new_T *= diagmat(1 / sqrt(diagvec(new_T.t() * new_T)));

    // Differential of g
    dg = dg_oblimin(dir, Inv_T, L, I_gamma_C, IgCL2N, N, g);

    // Riemannian hessian
    Hd = dg - T * diagmat( T.t() * dg ) - dir * diagmat( T.t() * g);

    preddiff = - arma::accu(dir % ( gr + 0.5 * Hd) );

    // objective
    f_oblimin(new_Inv_T, new_L, new_L2, new_Phi, new_IgCL2N, new_f,
              A, new_T, I_gamma_C, N);

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
      IgCL2N = new_IgCL2N;
      f = new_f;

      // update gradient
      g_oblimin(g, Inv_T, L, IgCL2N);

      // Riemannian gradient
      gr = g - T * diagmat( T.t() * g );
      ng = sqrt(arma::accu(pow(gr, 2)));

    }

    iteration = iteration + 1;

  } while (iteration < max_iter);

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

// GeominQ

void f_geominQ(arma::mat& Inv_T, arma::mat& L, arma::mat& L2, arma::vec& term, arma::mat& Phi, double& f,
               arma::mat A, arma::mat T, int p, double epsilon) {

  Phi = T.t() * T;
  Inv_T = inv(T);
  L = A * Inv_T.t();
  L2 = arma::pow(L, 2);
  L2 += epsilon;
  term = arma::exp(arma::sum(arma::log(L2), 1) / p);

  f = arma::accu(term);

}

void g_geominQ(arma::mat& g, arma::mat& LoL2, double p2, arma::mat Inv_T, arma::mat L, arma::mat L2,
               arma::vec term, int p) {

  LoL2 = L / L2;
  arma::mat Gq = LoL2 * p2;
  Gq.each_col() %= term;
  g = - Inv_T.t() * Gq.t() * L;

}

arma::mat dg_geominQ(arma::mat dT, arma::mat Inv_T, arma::mat L, arma::mat L2,
                     arma::mat LoL2, arma::vec term, double p2, arma::mat g, double epsilon) {

  arma::mat Inv_T_dt = Inv_T * dT;
  arma::mat dL = - L * Inv_T_dt.t();

  arma::mat c1 = (epsilon - L % L) / (L2 % L2) % dL;
  c1.each_col() %= term;
  arma::mat c2 = LoL2;
  arma::vec term2 = p2 * term % arma::sum(LoL2 % dL, 1);
  c2.each_col() %= term2;
  arma::mat dGq = p2 * (c1 + c2);

  arma::mat dg = - g * (Inv_T * dT).t() - (dT * Inv_T).t() * g - (dGq * Inv_T).t() * L;

  return dg;

}

void tcg_geominQ(arma::mat& dir, bool& att_bnd, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat L2, arma::mat LoL2, arma::vec term, arma::mat g, arma::mat gr,
         int p, double p2, double epsilon, double ng, arma::vec c, double rad) {

  dir.zeros();
  arma::mat dir0, dg, Hd;
  double alpha, rr0, tau, beta, dHd;

  arma::mat delta = -gr;
  arma::mat r = delta;
  double rr = ng * ng;
  double tol = ng * std::min(pow(ng, c[0]), c[1]);

  int iter = 0;

  do{

    dg = dg_geominQ(delta, Inv_T, L, L2, LoL2, term, p2, g, epsilon);

    Hd = dg - T * diagmat( T.t() * dg ) - delta * diagmat( T.t() * g);
    dHd = arma::accu(delta % Hd);

    if(dHd <= 0) {

      tau = root_quad(arma::accu(delta % delta), 2 * arma::accu(dir % delta),
                      arma::accu(dir % dir) - rad * rad);
      dir = dir + tau * delta;
      att_bnd = true;

      break;

      // if(iter == 0) {
      //
      //   dir = delta;
      //
      // }

    }

    rr0 = rr;
    alpha = rr0 / dHd;
    dir0 = dir;
    dir = dir + alpha * delta;

    if (sqrt(arma::accu(pow(dir, 2))) >= rad) {

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

  } while (iter < 10);

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> NPF_geominQ(arma::mat T, arma::mat A, double epsilon = 1e-02, double eps = 1e-05, int max_iter = 10000) {

  arma::mat Inv_T, L, L2, LoL2, Phi, g, gr, dg, Hd;
  arma::vec term, new_term;
  double f, ng, preddiff;

  arma::mat new_T, new_Inv_T, new_L, new_L2, new_LoL2, new_Phi, new_g, new_gr, new_dg, new_Hd;
  double new_f, new_ng;

  double step_size = 1;
  int p = T.n_rows;
  double p1 = p + 0.0;
  double p2 = 2/p1;

  f_geominQ(Inv_T, L, L2, term, Phi, f, A, T, p, epsilon);
  g_geominQ(g, LoL2, p2, Inv_T, L, L2, term, p);

  // Riemannian gradient

  gr = g - T * diagmat( T.t() * g );

  ng = sqrt(arma::accu(pow(gr, 2)));

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

    iteration = iteration + 1;

    // if (verbose) {
    //
    //   Rcpp::Rcout << "\r Iteration: " << iteration << " f = " << f << " norm(gr) = " << ng << "\r";
    //
    // }

    if (ng < eps) break;

    // subsolver
    tcg_geominQ(dir, att_bnd, T, Inv_T, L, L2, LoL2, term, g, gr, p, p2, epsilon, ng, c, rad);
    new_T = T + dir;

    // Projection onto the oblique manifold
    new_T *= diagmat(1 / sqrt(diagvec(new_T.t() * new_T)));

    // Differential of g
    dg = dg_geominQ(dir, Inv_T, L, L2, LoL2, term, p2, g, epsilon);

    // Riemannian hessian
    Hd = dg - T * diagmat( T.t() * dg ) - dir * diagmat( T.t() * g);

    preddiff = - arma::accu(dir % ( gr + 0.5 * Hd) );

    // objective

    f_geominQ(new_Inv_T, new_L, new_L2, new_term, new_Phi, new_f,
              A, new_T, p, epsilon);

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
      L2 = new_L2;
      LoL2 = new_LoL2;
      term = new_term;
      f = new_f;

      // update gradient

      g_geominQ(g, LoL2, p2, Inv_T, L, L2, term, p);

      // Riemannian gradient
      gr = g - T * diagmat( T.t() * g );
      ng = sqrt(arma::accu(pow(gr, 2)));

    }

    iteration = iteration + 1;

  } while (iteration < max_iter);

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

// targetQ

void f_targetQ(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi, arma::mat& f1, double& f,
               arma::mat A, arma::mat T, arma::mat Target, arma::mat Weight) {

  Phi = T.t() * T;
  Inv_T = inv(T);
  L = A * Inv_T.t();
  f1 = Weight % (L - Target);

  f = 0.5*arma::accu(pow(f1, 2));

}

void g_targetQ(arma::mat& g, arma::mat f1, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat Weight) {

  arma::mat df1_dL = Weight % f1;
  arma::mat df1_dt = - Inv_T.t() * df1_dL.t() * L;

  g = df1_dt;

}

arma::mat dg_targetQ(arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat W2, arma::mat g) {

  arma::mat Inv_T_dt = Inv_T * dT;
  arma::mat dL = - L * Inv_T_dt.t();

  arma::mat dg1L = W2 % dL;

  arma::mat dg1 = - g * (Inv_T * dT).t() - (dT * Inv_T).t() * g - (dg1L * Inv_T).t() * L;

  arma::mat dg = dg1;

  return dg;

}

void tcg_targetQ(arma::mat& dir, bool& att_bnd, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g, arma::mat gr, arma::mat W2, double ng, arma::vec c, double rad) {

  dir.zeros();
  arma::mat dir0, dg, Hd;
  double alpha, rr0, tau, beta, dHd;

  arma::mat delta = -gr;
  arma::mat r = delta;
  double rr = ng * ng;
  double tol = ng * std::min(pow(ng, c[0]), c[1]);

  int iter = 0;

  do{

    dg = dg_targetQ(delta, T, Inv_T, L, W2, g);
    Hd = dg - T * diagmat( T.t() * dg ) - delta * diagmat( T.t() * g);

    dHd = arma::accu(delta % Hd);

    if(dHd <= 0) {

      tau = root_quad(arma::accu(delta % delta), 2 * arma::accu(dir % delta),
                      arma::accu(dir % dir) - rad * rad);
      dir = dir + tau * delta;
      att_bnd = true;

      break;

      // if(iter == 0) {
      //
      //   dir = delta;
      //
      // }

    }

    rr0 = rr;
    alpha = rr0 / dHd;
    dir0 = dir;
    dir = dir + alpha * delta;

    if (sqrt(arma::accu(pow(dir, 2))) >= rad) {

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

  } while (iter < 10);

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> NPF_targetQ(arma::mat T, arma::mat A, arma::mat Target, arma::mat Weight, double eps = 1e-05, int max_iter = 1e4) {

  arma::mat W2, Inv_T, L, Phi, f1, g, gr, dg, Hd;
  double f, ng, preddiff;

  arma::mat new_T, new_Inv_T, new_L, new_Phi, new_f1, new_g, new_gr;
  double new_f, new_ng;

  W2 = pow(Weight, 2);

  // Objective
  f_targetQ(Inv_T, L, Phi, f1, f,
            A, T, Target, Weight);

  // Gradient
  g_targetQ(g, f1, T, Inv_T, L, Weight);

  // Riemannian gradient
  gr = g - T * diagmat( T.t() * g );

  ng = sqrt(arma::accu(pow(gr, 2)));

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
  int p = T.n_rows;
  arma::mat dir(p, p);

  int iteration = 0;
  double goa;

  do{

    iteration = iteration + 1;

    // if (verbose) {
    //
    //   Rcpp::Rcout << "\r Iteration " << iteration << " f = " << f << " norm(gr) = " << ng << "\r";
    //
    // }

    if (ng < eps) break;

    // subsolver
    tcg_targetQ(dir, att_bnd, T, Inv_T, L, g, gr, W2, ng, c, rad);

    new_T = T + dir;

    // Projection onto the oblique manifold
    new_T *= diagmat(1 / sqrt(diagvec(new_T.t() * new_T)));

    // Differential of g
    dg = dg_targetQ(dir, T, Inv_T, L, W2, g);

    // Riemannian hessian
    Hd = dg - T * diagmat( T.t() * dg ) - dir * diagmat( T.t() * g);

    preddiff = - arma::accu(dir % ( gr + 0.5 * Hd) );

    // objective
    f_targetQ(new_Inv_T, new_L, new_Phi, new_f1, new_f,
              A, new_T, Target, Weight);

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

      // update gradient
      g_targetQ(g, f1, T, Inv_T, L, Weight);

      // Riemannian gradient
      gr = g - T * diagmat( T.t() * g );
      ng = sqrt(arma::accu(pow(gr, 2)));

    }

    iteration = iteration + 1;

  } while (iteration < max_iter);


  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}


