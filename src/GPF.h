double objective_cf(double k, arma::mat L2, arma::mat L2N, arma::mat ML2) {

  double f1 = trace(L2.t() * L2N) / 4;
  double f2 = trace(L2.t() * ML2) / 4;

  return (1-k)*f1 + k*f2;
}

arma::mat gradient_cfT(double k, arma::mat A, arma::mat L, arma::mat L2, arma::mat L2N, arma::mat ML2) {

  arma::mat gradient_L = (1-k)*L % L2N + k*L % ML2;
  arma::mat gradient = A.t() * gradient_L;

  return gradient;
}

arma::mat gradient_cfQ(double k, arma::mat A, arma::mat L, arma::mat L2, arma::mat L2N, arma::mat ML2, arma::mat Inv_T) {

  arma::mat gradient_L = (1-k)*L % L2N + k*L % ML2;
  arma::mat gradient = -(L.t() * gradient_L * Inv_T).t();

  return gradient;
}

double objective_varimax(arma::mat L2, arma::mat B) {

  double f = trace(L2.t() * B) / 4;

  return f;
}

arma::mat gradient_varimax(arma::mat A, arma::mat L, arma::mat B) {

  arma::mat gradient_L = L % B;
  arma::mat gradient = A.t() * gradient_L;

  return gradient;
}

double objective_varimax_2(arma::mat QL) {

  arma::mat QLQL = QL.t() * QL;
  double f = -trace(QLQL) / 4;

  return(f);
}

arma::mat gradient_varimax_2(arma::mat A, arma::mat L, arma::mat QL) {

  arma::mat gradient_L = -L % QL;
  arma::mat gradient = A.t() * gradient_L;

  return(gradient);
}

double objective_target(arma::mat L, arma::mat Target, arma::mat Weight) {

  arma::mat f1 = Weight % (L - Target);
  f1 = pow(f1, 2);
  double f = 0.5 * arma::accu(f1);

  return f;
}

arma::mat gradient_target(arma::mat A, arma::mat L, arma::mat Target, arma::mat Weight) {

  arma::mat gradient_L = Weight % (L - Target);

  arma::mat gradient = A.t() * gradient_L;

  return gradient;
}

arma::mat gradient_targetQ(arma::mat A, arma::mat L, arma::mat Target, arma::mat Weight, arma::mat Inv_T) {

  arma::mat gradient_L = Weight % (L - Target);

  arma::mat gradient = -(L.t() * gradient_L * Inv_T).t();

  return gradient;
}

double objective_xtarget(arma::mat L, arma::mat Phi, arma::mat Target, arma::mat Weight, arma::mat Phi_Target, arma::mat Phi_Weight, double w) {

  // L es la parametrización de la matriz de pesos sin rotar (A):
  // L = A * (T⁻¹)'
  arma::mat f1 = Weight % (L - Target);
  // Parte que depende de los loadings:
  f1 = pow(f1, 2);
  arma::mat f2 = Phi_Weight % (Phi - Phi_Target);
  // Parte que depende de Phi:
  f2 = pow(f2, 2);
  double f = 0.5*(arma::accu(f1) + w*arma::accu(f2));
  // accu == sum

  return f;
}

arma::mat gradient_xtarget(arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat Phi, arma::mat Target, arma::mat Weight, arma::mat Phi_Target, arma::mat Phi_Weight, double w) {

  arma::mat df1_dL = pow(Weight, 2) % (L - Target);
  // Derivada de f1 respecto a t:
  arma::mat df1_dt = - Inv_T.t() * df1_dL.t() * L;

  // Derivada de f2 respecto a t:
  arma::mat df2_dt = 2 * T * ( pow(Phi_Weight, 2) % (Phi - Phi_Target) );

  arma::mat gradient = df1_dt + w*df2_dt;

  return gradient;
}

double objective_oblimin(arma::mat L2, arma::mat B) {

  double f = trace(L2.t() * B) / 4;

  return f;
}

arma::mat gradient_oblimin(arma::mat L, arma::mat B, arma::mat Inv_T) {

  arma::mat gradient_L = L % B;
  arma::mat gradient = -(L.t() * gradient_L * Inv_T).t();

  return gradient;
}

double objective_geomin(arma::vec term) {

  double f = arma::accu(term);

  return f;
}

arma::mat gradient_geominQ(arma::mat L, arma::mat L2, arma::vec term, double q, arma::mat Inv_T) {

  arma::mat gradient_L = (L/L2)*(2/q);
  gradient_L.each_col() %= term;
  arma::mat gradient = -(L.t() * gradient_L * Inv_T).t();

  return gradient;
}

arma::mat gradient_geominT(arma::mat unrotated, arma::mat A, arma::mat A2, arma::vec term, double q, arma::mat T) {

  arma::mat gradient_A = (A/A2)*(2/q);
  gradient_A.each_col() %= term;
  arma::mat gradient = unrotated.t() * gradient_A;

  return gradient;
}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_cfT(arma::mat T, arma::mat Unrotated, double k, double eps = 1e-05, int max_iter = 10000) {

  int n_factors = T.n_cols;
  int p = Unrotated.n_rows;
  double step_size = 1;

  arma::mat N(n_factors, n_factors, arma::fill::ones);
  arma::mat M(p, p, arma::fill::ones);
  N.diag(0).zeros();
  M.diag(0).zeros();

  arma::mat L = Unrotated * T;
  arma::mat L2 = pow(L, 2);

  arma::mat L2N = L2 * N;
  arma::mat ML2 = M * L2;
  double f = objective_cf(k, L2, L2N, ML2);
  arma::mat fg = gradient_cfT(k, Unrotated, L, L2, L2N, ML2);

  int iteration = 0;
  double s, Q, ft;
  arma::mat T_temp;

  do {

    iteration = iteration + 1;
    arma::mat MM = T.t() * fg;
    arma::mat S = 0.5 * (MM + MM.t());
    arma::mat Gp = fg - T * S;
    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;

      arma::mat U;
      arma::vec ss;
      arma::mat V;
      arma::svd(U, ss, V, X);

      T_temp = U * V.t();
      L = Unrotated * T_temp;
      L2 = pow(L, 2);
      L2N = L2 * N;
      ML2 = M * L2;

      ft = objective_cf(k, L2, L2N, ML2);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = gradient_cfT(k, Unrotated, L, L2, L2N, ML2);

  } while (s > eps && iteration < max_iter);

  arma::mat Phi(n_factors, n_factors, arma::fill::eye);

 bool convergence = true;
 if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_cfQ(arma::mat T, arma::mat Unrotated, double k, double eps = 1e-05, int max_iter = 10000) {

  int n_factors = T.n_cols;
  int p = Unrotated.n_rows;
  double step_size = 1;

  arma::mat N(n_factors, n_factors, arma::fill::ones);
  arma::mat M(p, p, arma::fill::ones);
  N.diag(0).zeros();
  M.diag(0).zeros();

  arma::mat Inv_T = inv(T);
  arma::mat L = Unrotated * Inv_T.t();
  arma::mat L2 = pow(L, 2);

  arma::mat L2N = L2 * N;
  arma::mat ML2 = M * L2;
  double f = objective_cf(k, L2, L2N, ML2);
  arma::mat fg = gradient_cfQ(k, Unrotated, L, L2, L2N, ML2, Inv_T);

  int iteration = 0;
  double s, Q, ft;
  arma::mat T_temp;

  do {

    iteration = iteration + 1;
    arma::mat Gp = fg - T * diagmat(T.t() * fg);

    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;
      arma::mat XX = X.t() * X;
      arma::vec v = 1 / sqrt(diagvec(XX));
      T_temp = X * diagmat(v);

      Inv_T = inv(T_temp);
      L = Unrotated * Inv_T.t();
      L2 = pow(L, 2);
      L2N = L2 * N;
      ML2 = M * L2;

      ft = objective_cf(k, L2, L2N, ML2);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = gradient_cfQ(k, Unrotated, L, L2, L2N, ML2, Inv_T);

  } while (s > eps && iteration < max_iter);

  arma::mat Phi = T.t() * T;

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_varimax(arma::mat T, arma::mat Unrotated, double eps = 1e-05, int max_iter = 10000) {

  int n_factors = T.n_cols;
  int p = Unrotated.n_rows;
  double step_size = 1;

  arma::mat I(p, p, arma::fill::eye);
  arma::mat gamma_C(p, p, arma::fill::ones);
  gamma_C *= 1/p;
  arma::mat N(n_factors, n_factors, arma::fill::ones);
  N.diag(0).zeros();
  arma::mat I_gamma_C = (I - gamma_C);

  arma::mat L = Unrotated * T;
  arma::mat L2 = pow(L, 2);
  arma::mat B = I_gamma_C * L2 * N;
  double f = objective_varimax(L2, B);
  arma::mat fg = gradient_varimax(Unrotated, L, B);

  int iteration = 0;
  double s, Q, ft;
  arma::mat T_temp;

  do {

    iteration = iteration + 1;
    arma::mat M = T.t() * fg;
    arma::mat S = 0.5 * (M + M.t());
    arma::mat Gp = fg - T * S;
    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;

      arma::mat U;
      arma::vec ss;
      arma::mat V;
      arma::svd(U, ss, V, X);

      T_temp = U * V.t();
      L = Unrotated * T_temp;
      L2 = pow(L, 2);
      B = I_gamma_C * L2 * N;

      ft = objective_varimax(L2, B);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = gradient_varimax(Unrotated, L, B);

  } while (s > eps && iteration < max_iter);

  arma::mat Phi(n_factors, n_factors, arma::fill::eye);

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_varimax_2(arma::mat T, arma::mat Unrotated, double eps = 1e-05, int max_iter = 10000) {

  int n_factors = T.n_cols;
  double step_size = 1;
  arma::mat L = Unrotated * T;
  arma::mat L2 = pow(L, 2);
  arma::mat means_L2 = mean(L2, 0);
  arma::mat QL = L2;
  QL.each_row() -= means_L2;
  double f = objective_varimax_2(QL);
  arma::mat fg = gradient_varimax_2(Unrotated, L, QL);

  int iteration = 0;
  double s, Q, ft;
  arma::mat T_temp;

  do {

    iteration = iteration + 1;
    arma::mat M = T.t() * fg;
    arma::mat S = 0.5 * (M + M.t());
    arma::mat Gp = fg - T * S;
    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;

      arma::mat U;
      arma::vec ss;
      arma::mat V;
      arma::svd(U, ss, V, X);

      T_temp = U * V.t();
      L = Unrotated * T_temp;
      L2 = pow(L, 2);
      means_L2 = mean(L2, 0);
      QL = L2;
      QL.each_row() -= means_L2;

      ft = objective_varimax_2(QL);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = gradient_varimax_2(Unrotated, L, QL);

  } while (s > eps && iteration < max_iter);

  arma::mat Phi(n_factors, n_factors, arma::fill::eye);

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_target(arma::mat T, arma::mat Unrotated, arma::mat Target, arma::mat Weight, double eps = 1e-05, int max_iter = 10000) {

  int n_factors = T.n_cols;
  double step_size = 1;

  arma::mat L = Unrotated * T;
  double f = objective_target(L, Target, Weight);
  arma::mat fg = gradient_target(Unrotated, L, Target, Weight);

  int iteration = 0;
  double s, Q, ft;
  arma::mat T_temp;

  do {

    iteration = iteration + 1;
    arma::mat M = T.t() * fg;
    arma::mat S = 0.5 * (M + M.t());
    arma::mat Gp = fg - T * S;
    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;

      arma::mat U;
      arma::vec ss;
      arma::mat V;
      arma::svd(U, ss, V, X);

      T_temp = U * V.t();
      L = Unrotated * T_temp;

      ft = objective_target(L, Target, Weight);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = gradient_target(Unrotated, L, Target, Weight);

  } while (s > eps && iteration < max_iter);

  arma::mat Phi(n_factors, n_factors, arma::fill::eye);

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_xtarget(arma::mat T, arma::mat Unrotated, arma::mat Target, arma::mat Weight, arma::mat Phi_Target, arma::mat Phi_Weight,
                       double w, double eps, int max_iter) {

  int n_factors = T.n_cols;
  double step_size = 1;
  arma::mat Inv_T = inv(T);
  arma::mat L = Unrotated * Inv_T.t();
  arma::mat Phi = T.t() * T;
  double f = objective_xtarget(L, Phi, Target, Weight, Phi_Target, Phi_Weight, w);
  arma::mat fg = gradient_xtarget(T, Inv_T, L, Phi, Target, Weight, Phi_Target, Phi_Weight, w);

  int iteration = 0;
  double s, Q, ft;
  arma::mat I(n_factors, n_factors, arma::fill::eye), T_temp;

  do {

    iteration = iteration + 1;
    // arma::mat Gp = fg - T * diagmat(sum(T % fg, 0));
    arma::mat Gp = fg - T * diagmat(T.t() * fg);
    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;
      arma::mat XX = X.t() * X;
      arma::vec v = 1 / sqrt(diagvec(XX));
      T_temp = X * diagmat(v);
      Inv_T = inv(T_temp);
      L = Unrotated * Inv_T.t();
      Phi = T_temp.t() * T_temp;
      ft = objective_xtarget(L, Phi, Target, Weight, Phi_Target, Phi_Weight, w);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = gradient_xtarget(T, Inv_T, L, Phi, Target, Weight, Phi_Target, Phi_Weight, w);

  } while (s > eps && iteration < max_iter);

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);


  return result;

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_targetQ(arma::mat T, arma::mat Unrotated, arma::mat Target, arma::mat Weight, double eps = 1e-05, int max_iter = 10000) {

  int n_factors = T.n_cols;
  double step_size = 1;
  arma::mat Inv_T = inv(T);
  arma::mat L = Unrotated * Inv_T.t();
  double f = objective_target(L, Target, Weight);
  arma::mat fg = gradient_targetQ(Unrotated, L, Target, Weight, Inv_T);

  int iteration = 0;
  double s, Q, ft;
  arma::mat I(n_factors, n_factors, arma::fill::eye), T_temp;

  do {

    iteration = iteration + 1;
    arma::mat Gp = fg - T * diagmat(T.t() * fg);
    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;
      arma::mat XX = X.t() * X;
      arma::vec v = 1 / sqrt(diagvec(XX));
      T_temp = X * diagmat(v);
      Inv_T = inv(T_temp);
      L = Unrotated * Inv_T.t();
      ft = objective_target(L, Target, Weight);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = fg = gradient_targetQ(Unrotated, L, Target, Weight, Inv_T);

  } while (s > eps && iteration < max_iter);

  arma::mat Phi = T.t() * T;

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_oblimin(arma::mat T, arma::mat Unrotated, double gamma = 0, double eps = 1e-05, int max_iter = 10000) {

  int n_factors = T.n_cols;
  int p = Unrotated.n_rows;

  arma::mat I(p, p, arma::fill::eye);
  arma::mat gamma_C(p, p, arma::fill::ones);
  gamma_C *= gamma/p;
  arma::mat N(n_factors, n_factors, arma::fill::ones);
  N.diag(0).zeros();
  arma::mat I_gamma_C = (I - gamma_C);

  double step_size = 1;
  arma::mat Inv_T = inv(T);
  arma::mat L = Unrotated * Inv_T.t();
  arma::mat L2 = pow(L, 2);
  arma::mat B = I_gamma_C * L2 * N;
  double f = objective_oblimin(L2, B);
  arma::mat fg = gradient_oblimin(L, B, Inv_T);

  int iteration = 0;
  double s, Q, ft;
  arma::mat T_temp;

  do {

    iteration = iteration + 1;
    arma::mat Gp = fg - T * diagmat(T.t() * fg);

    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;
      arma::mat XX = X.t() * X;
      arma::vec v = 1 / sqrt(diagvec(XX));
      T_temp = X * diagmat(v);
      Inv_T = inv(T_temp);
      L = Unrotated * Inv_T.t();
      L2 = pow(L, 2);
      B = I_gamma_C * L2 * N;
      ft = objective_oblimin(L2, B);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = gradient_oblimin(L, B, Inv_T);

  } while (s > eps && iteration < max_iter);

  arma::mat Phi = T.t() * T;

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_geominT(arma::mat T, arma::mat Unrotated, double epsilon = 1e-02, double eps = 1e-05, int max_iter = 10000) {

  int n_factors = T.n_cols;
  int p = Unrotated.n_rows;

  double step_size = 1;

  arma::mat L = Unrotated * T;
  arma::mat L2 = pow(L, 2);
  L2 += epsilon;
  arma::vec term(p);
  arma::colvec inside = sum(log(L2), 1);
  inside /= n_factors;
  for(int j=0; j < p; j++) term(j) = exp(inside(j, 0));
  double f = objective_geomin(term);
  arma::mat fg = gradient_geominT(Unrotated, L, L2, term, n_factors, T);

  int iteration = 0;
  double s, Q, ft;
  arma::mat T_temp;

  do {

    iteration = iteration + 1;
    arma::mat M = T.t() * fg;
    arma::mat S = 0.5 * (M + M.t());
    arma::mat Gp = fg - T * S;
    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;

      arma::mat U;
      arma::vec ss;
      arma::mat V;
      arma::svd(U, ss, V, X);

      T_temp = U * V.t();
      L = Unrotated * T_temp;
      L2 = pow(L, 2);
      L2 += epsilon;
      inside = sum(log(L2), 1);
      inside /= n_factors;
      for(int j=0; j < p; j++) term(j) = exp(inside(j, 0));

      ft = objective_geomin(term);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = gradient_geominT(Unrotated, L, L2, term, n_factors, T);

  } while (s > eps && iteration < max_iter);

  arma::mat Phi(n_factors, n_factors, arma::fill::eye);

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}

std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> GPF_geominQ(arma::mat T, arma::mat Unrotated, double epsilon = 1e-02, double eps = 1e-05, int max_iter = 10000) {

  int n_factors = T.n_cols;
  int p = Unrotated.n_rows;

  double step_size = 1;
  arma::mat Inv_T = inv(T);
  arma::mat L = Unrotated * Inv_T.t();
  arma::mat L2 = arma::pow(L, 2);
  L2 += epsilon;
  arma::mat logL2 = arma::log(L2);
  arma::vec sumlogL2 = arma::sum(logL2, 1) / n_factors;
  arma::vec term = arma::exp(sumlogL2);
  // arma::vec inside = sum(log(L2), 1) / n_factors;
  // for(int j=0; j < p; j++) term[j] = exp(inside[j]);
  // term = arma::exp(arma::sum(arma::log(L2), 1) / n_factors);
  double f = objective_geomin(term);
  arma::mat fg = gradient_geominQ(L, L2, term, n_factors, Inv_T);

  int iteration = 0;
  double s, Q, ft;
  arma::mat T_temp;

  do {

    iteration = iteration + 1;

    arma::mat Gp = fg - T * diagmat(T.t() * fg);

    s = sqrt(trace(Gp.t() * Gp));

    step_size = 2 * step_size;
    int i = 0;


    for(int i=0; i < 10; ++i) {
      arma::mat X = T - step_size * Gp;
      arma::mat XX = X.t() * X;
      arma::vec v = 1 / sqrt(diagvec(XX));
      T_temp = X * diagmat(v);
      Inv_T = inv(T_temp);
      L = Unrotated * Inv_T.t();
      L2 = arma::pow(L, 2);
      L2 += epsilon;
      logL2 = arma::log(L2);
      sumlogL2 = arma::sum(logL2, 1) / n_factors;
      term = arma::exp(sumlogL2);
      // arma::vec inside = sum(log(L2), 1) / n_factors;
      // for(int j=0; j < p; j++) term[j] = exp(inside[j]);
      // term = arma::exp(arma::sum(arma::log(L2), 1) / n_factors);

      ft = objective_geomin(term);
      Q = f - ft;
      if( Q > (0.5 * pow(s, 2) * step_size) ) break;
      step_size = 0.5 * step_size;
    }

    T = T_temp;
    f = ft;
    fg = gradient_geominQ(L, L2, term, n_factors, Inv_T);

  } while (s > eps && iteration < max_iter);

  arma::mat Phi = T.t() * T;

  bool convergence = true;
  if(iteration == max_iter) {

    convergence = false;

  }

  std::tuple<arma::mat, arma::mat, arma::mat, double, int, bool> result = std::make_tuple(L, Phi, T, f, iteration, convergence);

  return result;

}
