arma::mat dxt(arma::mat X) { // derivative wrt transpose

  int p = X.n_rows;
  int q = X.n_cols;
  int pq = p*q;

  arma::mat res(pq, pq);
  arma::mat temp(p, q);

  for(int i=0; i < pq; ++i) {
    temp.zeros();
    temp(i) = 1;
    res.col(i) = arma::vectorise(temp.t(), 0);
  }

  return res;

}

arma::mat kdiag(arma::mat X) {

  /*
   * Transform every column subset into a diagonal matrix and bind
   */

  int pq = X.n_rows;
  int q = X.n_cols;
  int p = pq/q;

  arma::mat res2(pq, 0);

  for(int j=0; j < q; ++j) {

    arma::mat res1(0, p);

    for(int i=0; i < q; ++i) {
      int index_1 = i*p;
      int index_2 = index_1 + (p-1);
      arma::mat temp = arma::diagmat(X(arma::span(index_1, index_2), j));
      res1 = arma::join_cols(res1, temp);
    }

    res2 = arma::join_rows(res2, res1);

  }

  return res2;

}

arma::mat cbind_diag(arma::mat X) {

  /*
   * Transform every column into a diagonal matrix and bind
   */

  int p = X.n_rows;
  int q = X.n_cols;
  arma::mat res(p, 0);

  for(int i=0; i < q; ++i) {
    res = arma::join_rows(res, arma::diagmat(X.col(i)));
  }

  return res;

}

// Criteria

class base_criterion {

public:

  virtual void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
                 arma::mat& f1, arma::mat& f2, double& f,
                 arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
                 arma::mat& L2, arma::vec& term, int p, double epsilon,
                 arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
                 arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k) = 0;

  virtual void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T,
                 arma::mat Inv_T, arma::mat L,
                 arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
                 arma::mat IgCL2N, arma::mat N, arma::mat M,
                 arma::mat Weight, arma::mat Phi_Weight, double w, double k) = 0;

  virtual void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T,
                  arma::mat L, arma::mat g, arma::mat L2, arma::mat LoL2, arma::vec term,
                  double p2, double epsilon,
                  arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
                  arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k) = 0;

  virtual void d_constraint(arma::mat& d_constraints, arma::mat lambda, arma::mat Phi,
                          arma::mat Target, arma::mat Weight,
                          arma::mat PhiTarget, arma::mat PhiWeight,
                          double gamma, double k, double epsilon, double w) = 0;

};

/*
 * Crawford-Ferguson family
 */

class cfT: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k){

    L = lambda * T;
    L2 = L % L;
    double ff1 = (1-k) * arma::accu(L2 % (L2 * N)) / 4;
    double ff2 = k * arma::accu(L2 % (M * L2)) / 4;

    f = ff1 + ff2;

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k){

    f1 = (1-k) * L % (L2 * N);
    f2 = k * L % (M * L2);
    g = lambda.t() * (f1 + f2);

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k){

    arma::mat dL = lambda * dT;
    arma::mat dL2 = 2 * dL % L;
    arma::mat df1 = (1-k) * dL % (L2 * N) + (1-k) * L % (dL2 * N);
    arma::mat df2 = k * dL % (M * L2) + k * L % (M * dL2);
    dg = lambda.t() * (df1 + df2);

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);;
    int p = L.n_rows;
    int q = L.n_cols;
    arma::mat I1(p, p, arma::fill::eye);
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat N(q, q, arma::fill::ones);
    N.diag(0).zeros();
    arma::mat M(p, p, arma::fill::ones);
    M.diag(0).zeros();

    arma::mat L2 = L % L;
    arma::mat f1 = (1-k) * L % (L2 * N);
    arma::mat f2 = k * L % (M * L2);
    arma::mat gL = f1 + f2;

    arma::mat diagL = diagmat(arma::vectorise(L));
    arma::mat diag2L = 2*diagL;
    arma::mat c2 = arma::kron(N, I1) * diag2L;
    arma::mat gf1 = (1-k)*(diagL * c2 + arma::diagmat(arma::vectorise(L2 * N)));
    arma::mat c3 = arma::kron(I2, M) * diag2L;
    arma::mat gf2 = k * (diagL * c3 + arma::diagmat(arma::vectorise(M * L2)));
    arma::mat hL = gf1 + gf2;

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);
    d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    d_constraints.insert_cols(p*q, p);

  }

};

class cfQ: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k){

    Phi = T.t() * T;
    Inv_T = inv(T);
    L = lambda * Inv_T.t();
    L2 = L % L;
    double ff1 = (1-k) * arma::accu(L2 % (L2 * N)) / 4;
    double ff2 = k * arma::accu(L2 % (M * L2)) / 4;

    f = ff1 + ff2;

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k){

    f1 = (1-k) * L % (L2 * N);
    f2 = k * L % (M * L2);
    arma::mat gL = f1 + f2;

    g = - Inv_T.t() * gL.t() * L;

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k){

    arma::mat Inv_T_dt = Inv_T * dT;
    arma::mat dL = - L * Inv_T_dt.t();

    arma::mat dL2 = 2 * dL % L;
    arma::mat df1 = (1-k) * dL % (L2 * N) + (1-k) * L % (dL2 * N);
    arma::mat df2 = k * dL % (M * L2) + k * L % (M * dL2);
    arma::mat dgL = df1 + df2;

    dg = - g * (Inv_T * dT).t() - (dT * Inv_T).t() * g - (dgL * Inv_T).t() * L;

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    int p = L.n_rows;
    int q = L.n_cols;
    int pq = p*q;
    int qq = q*q;
    int q_cor = q*(q-1)/2;
    arma::uvec indexes_1(q);
    for(int i=0; i < q; ++i) indexes_1[i] = ((i+1)*q) - (q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::mat I1(p, p, arma::fill::eye);
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat N(q, q, arma::fill::ones);
    N.diag(0).zeros();
    arma::mat M(p, p, arma::fill::ones);
    M.diag(0).zeros();

    arma::mat L2 = L % L;
    arma::mat f1 = (1-k) * L % (L2 * N);
    arma::mat f2 = k * L % (M * L2);
    arma::mat gL = f1 + f2;

    arma::mat diagL = diagmat(arma::vectorise(L));
    arma::mat diag2L = 2*diagL;
    arma::mat c2 = arma::kron(N, I1) * diag2L;
    arma::mat gf1 = (1-k)*(diagL * c2 + arma::diagmat(arma::vectorise(L2 * N)));
    arma::mat c3 = arma::kron(I2, M) * diag2L;
    arma::mat gf2 = k * (diagL * c3 + arma::diagmat(arma::vectorise(M * L2)));
    arma::mat hL = gf1 + gf2;

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);

    /*
     * Multiply each column (q x q form) by inv_Phi
     */

    arma::mat inv_Phi = arma::inv_sympd(Phi);
    arma::cube B(qq, pq, 1);
    B.slice(0) = d_constraints_temp;
    B.reshape(q, q, pq);
    B.each_slice() *= inv_Phi;
    B.reshape(q*q, pq, 1);
    d_constraints_temp = B.slice(0);

    arma::mat c1p = -arma::kron(inv_Phi.t(), (L.t() * gL * inv_Phi));
    arma::mat HP_temp = c1p + c1p * dxt(Phi);
    arma::mat HP = HP_temp.cols(indexes_2);

    d_constraints = arma::join_rows(d_constraints_temp, HP);
    d_constraints.shed_rows(indexes_1);
    d_constraints.insert_cols(p*q + q_cor, p);

  }

};

// Oblimin

class obliminT: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k){

    L = lambda * T;
    L2 = pow(L, 2);
    IgCL2N = I_gamma_C * L2 * N;

    f = trace(L2.t() * IgCL2N) / 4;

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k){

    arma::mat gL = L % IgCL2N;
    g = lambda.t() * gL;

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k){

    arma::mat dL = lambda * dT;
    arma::mat dL2 = 2 * dL % L;
    dg = lambda.t() * (dL % IgCL2N + L % (I_gamma_C * dL2 * N));

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);;
    int p = L.n_rows;
    int q = L.n_cols;
    arma::mat I1(p, p, arma::fill::eye);
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat N(q, q, arma::fill::ones);
    N.diag(0).zeros();

    arma::mat gC(p, p, arma::fill::ones);
    gC *= gamma/p;
    arma::mat IgC = I1 - gC;
    arma::mat L2 = L % L;
    arma::mat IgC_L2_N = IgC * L2 * N;
    arma::mat gL = L % IgC_L2_N;

    arma::mat c1 = arma::diagmat(arma::vectorise(IgC_L2_N));
    arma::mat diagL = arma::diagmat(arma::vectorise(L));
    arma::mat diag2L = 2*diagL;
    arma::mat c2 = diagL * arma::kron(N, IgC) * diag2L;
    arma::mat hL = c1 + c2;

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);
    d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    d_constraints.insert_cols(p*q, p);

  }

};

class obliminQ: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k) {

    Phi = T.t() * T;
    Inv_T = inv(T);
    L = lambda * Inv_T.t();
    L2 = pow(L, 2);
    IgCL2N = I_gamma_C * L2 * N;

    f = trace(L2.t() * IgCL2N) / 4;

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k) {

    arma::mat Gq = L % IgCL2N;

    g = - Inv_T.t() * Gq.t() * L;

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k) {

    arma::mat Inv_T_dt = Inv_T * dT;
    arma::mat dL = - L * Inv_T_dt.t();
    arma::mat dGq = dL % IgCL2N + L % (I_gamma_C * (2*dL % L) * N);

    dg = - g * (Inv_T * dT).t() - (dT * Inv_T).t() * g - (dGq * Inv_T).t() * L;

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    int p = L.n_rows;
    int q = L.n_cols;
    int pq = p*q;
    int qq = q*q;
    int q_cor = q*(q-1)/2;
    arma::uvec indexes_1(q);
    for(int i=0; i < q; ++i) indexes_1[i] = ((i+1)*q) - (q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::mat I1(p, p, arma::fill::eye);
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat N(q, q, arma::fill::ones);
    N.diag(0).zeros();

    arma::mat gC(p, p, arma::fill::ones);
    gC *= gamma/p;
    arma::mat IgC = I1 - gC;
    arma::mat L2 = L % L;
    arma::mat IgC_L2_N = IgC * L2 * N;
    arma::mat gL = L % IgC_L2_N;

    arma::mat c1 = arma::diagmat(arma::vectorise(IgC_L2_N));
    arma::mat diagL = arma::diagmat(arma::vectorise(L));
    arma::mat diag2L = 2*diagL;
    arma::mat c2 = diagL * arma::kron(N, IgC) * diag2L;
    arma::mat hL = c1 + c2;

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);

    /*
     * Multiply each column (q x q form) by inv_Phi
     */

    arma::mat inv_Phi = arma::inv_sympd(Phi);
    arma::cube B(qq, pq, 1);
    B.slice(0) = d_constraints_temp;
    B.reshape(q, q, pq);
    B.each_slice() *= inv_Phi;
    B.reshape(q*q, pq, 1);
    d_constraints_temp = B.slice(0);

    arma::mat c1p = -arma::kron(inv_Phi.t(), (L.t() * gL * inv_Phi));
    arma::mat HP_temp = c1p + c1p * dxt(Phi);
    arma::mat HP = HP_temp.cols(indexes_2);

    d_constraints = arma::join_rows(d_constraints_temp, HP);
    d_constraints.shed_rows(indexes_1);
    d_constraints.insert_cols(p*q + q_cor, p);

  }

};

// Geomin

class geominT: public base_criterion {

public:
  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k){

    L = lambda * T;
    L2 = L % L;
    L2 += epsilon;
    term = arma::exp(arma::sum(arma::log(L2), 1) / p);

    f = arma::accu(term);

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k){

    LoL2 = L / L2;
    arma::mat c1 = LoL2 * p2;
    c1.each_col() %= term;
    g = lambda.t() * c1;

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k){

    arma::mat dL = lambda * dT;
    arma::mat c1 = LoL2 * p2;
    arma::mat dL2 = 2 * dL % L;
    arma::mat dc1 = p2 * (L2 % dL - L % dL2) / (L2 % L2);
    arma::mat dlogL2 = dL2/L2;
    arma::mat dterm = term % sum(dlogL2, 1);
    dc1.each_col() %= term;
    c1.each_col() %= dterm;

    dg = lambda.t() * (dc1 + c1);

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);;
    int p = L.n_rows;
    int q = L.n_cols;
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat L2 = L % L + epsilon;
    arma::mat LoL2 = L / L2;
    double q2 = 2/(q + 0.0);
    arma::mat cx = q2 * LoL2;
    arma::mat I(q, 1, arma::fill::ones);
    arma::mat term = exp(log(L2) * I / q);
    arma::mat gL = cx;
    gL.each_col() %= term;

    arma::mat c1 = q2*(arma::vectorise(L2) - arma::vectorise(2*L % L)) / arma::vectorise(L2 % L2);
    arma::mat gcx = arma::diagmat(c1);
    arma::mat c2 = (1/L2) % (2*L) / q;
    c2.each_col() %= term;
    arma::mat gterm = cbind_diag(c2);
    arma::mat v = gterm.t() * cx;
    arma::mat hL = gcx;
    arma::mat term2 = term;
    for(int i=0; i < (q-1); ++i) term2 = arma::join_cols(term2, term);
    hL.each_col() %= term2;
    hL += kdiag(v);

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);
    d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    d_constraints.insert_cols(p*q, p);

  }

};

class geominQ: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k) {

    Phi = T.t() * T;
    Inv_T = inv(T);
    L = lambda * Inv_T.t();
    L2 = L % L;
    L2 += epsilon;
    term = arma::exp(arma::sum(arma::log(L2), 1) / p);

    f = arma::accu(term);

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k) {

    LoL2 = L / L2;
    arma::mat Gq = LoL2 * p2;
    Gq.each_col() %= term;
    g = - Inv_T.t() * Gq.t() * L;

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k) {

    arma::mat Inv_T_dt = Inv_T * dT;
    arma::mat dL = - L * Inv_T_dt.t();

    arma::mat c1 = (epsilon - L % L) / (L2 % L2) % dL;
    c1.each_col() %= term;
    arma::mat c2 = LoL2;
    arma::vec term2 = p2 * term % arma::sum(LoL2 % dL, 1);
    c2.each_col() %= term2;
    arma::mat dGq = p2 * (c1 + c2);

    dg = - g * (Inv_T * dT).t() - (dT * Inv_T).t() * g - (dGq * Inv_T).t() * L;

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    int p = L.n_rows;
    int q = L.n_cols;
    int pq = p*q;
    int qq = q*q;
    int q_cor = q*(q-1)/2;
    arma::uvec indexes_1(q);
    for(int i=0; i < q; ++i) indexes_1[i] = ((i+1)*q) - (q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat L2 = L % L + epsilon;
    arma::mat LoL2 = L / L2;
    double q2 = 2/(q + 0.0);
    arma::mat cx = q2 * LoL2;
    arma::mat I(q, 1, arma::fill::ones);
    arma::mat term = exp(log(L2) * I / q);
    arma::mat gL = cx;
    gL.each_col() %= term;

    arma::mat c1 = q2*(arma::vectorise(L2) - arma::vectorise(2*L % L)) / arma::vectorise(L2 % L2);
    arma::mat gcx = arma::diagmat(c1);
    arma::mat c2 = (1/L2) % (2*L) / q;
    c2.each_col() %= term;
    arma::mat gterm = cbind_diag(c2);
    arma::mat v = gterm.t() * cx;
    arma::mat hL = gcx;
    arma::mat term2 = term;
    for(int i=0; i < (q-1); ++i) term2 = arma::join_cols(term2, term);
    hL.each_col() %= term2;
    hL += kdiag(v);

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);

    /*
     * Multiply each column (q x q form) by inv_Phi
     */

    arma::mat inv_Phi = arma::inv_sympd(Phi);
    arma::cube B(qq, pq, 1);
    B.slice(0) = d_constraints_temp;
    B.reshape(q, q, pq);
    B.each_slice() *= inv_Phi;
    B.reshape(q*q, pq, 1);
    d_constraints_temp = B.slice(0);

    arma::mat c1p = -arma::kron(inv_Phi.t(), (L.t() * gL * inv_Phi));
    arma::mat HP_temp = c1p + c1p * dxt(Phi);
    arma::mat HP = HP_temp.cols(indexes_2);

    d_constraints = arma::join_rows(d_constraints_temp, HP);
    d_constraints.shed_rows(indexes_1);
    d_constraints.insert_cols(p*q + q_cor, p);

  }

};

// Target

class targetT: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k){

    L = lambda * T;
    f1 = Weight % (L - Target);

    f = 0.5*arma::accu(f1 % f1);

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k){

    g = lambda.t() * (Weight % f1);

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k){

    dg = lambda.t() * (W2 % (lambda * dT));

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);;
    int p = L.n_rows;
    int q = L.n_cols;
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat W2 = Weight % Weight;
    arma::mat gL = W2 % (L - Target);
    arma::mat hL = arma::diagmat(arma::vectorise(W2));

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);
    d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    d_constraints.insert_cols(p*q, p);

  }

};

class targetQ: public base_criterion {

public:
  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k) {

    Phi = T.t() * T;
    Inv_T = inv(T);
    L = lambda * Inv_T.t();
    f1 = Weight % (L - Target);

    f = 0.5*arma::accu(f1 % f1);

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k) {

    arma::mat df1_dL = Weight % f1;
    arma::mat df1_dt = - Inv_T.t() * df1_dL.t() * L;

    g = df1_dt;

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k) {

    arma::mat Inv_T_dt = Inv_T * dT;
    arma::mat dL = - L * Inv_T_dt.t();

    arma::mat dg1L = W2 % dL;

    dg = - g * Inv_T_dt.t() - (dT * Inv_T).t() * g - (dg1L * Inv_T).t() * L;

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    int p = L.n_rows;
    int q = L.n_cols;
    int pq = p*q;
    int qq = q*q;
    int q_cor = q*(q-1)/2;
    arma::uvec indexes_1(q);
    for(int i=0; i < q; ++i) indexes_1[i] = ((i+1)*q) - (q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat W2 = Weight % Weight;
    arma::mat gL = W2 % (L - Target);
    arma::mat hL = arma::diagmat(arma::vectorise(W2));

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);

    /*
     * Multiply each column (q x q form) by inv_Phi
     */

    arma::mat inv_Phi = arma::inv_sympd(Phi);
    arma::cube B(qq, pq, 1);
    B.slice(0) = d_constraints_temp;
    B.reshape(q, q, pq);
    B.each_slice() *= inv_Phi;
    B.reshape(q*q, pq, 1);
    d_constraints_temp = B.slice(0);

    arma::mat c1p = -arma::kron(inv_Phi.t(), (L.t() * gL * inv_Phi));
    arma::mat HP_temp = c1p + c1p * dxt(Phi);
    arma::mat HP = HP_temp.cols(indexes_2);

    d_constraints = arma::join_rows(d_constraints_temp, HP);
    d_constraints.shed_rows(indexes_1);
    d_constraints.insert_cols(p*q + q_cor, p);

  }

};

// xTarget

class xtargetT: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k){

    L = lambda * T;
    f1 = Weight % (L - Target);

    f = 0.5*arma::accu(f1 % f1);

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k){

    g = lambda.t() * (Weight % f1);

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k){

    dg = lambda.t() * (W2 % (lambda * dT));

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);;
    int p = L.n_rows;
    int q = L.n_cols;
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat W2 = Weight % Weight;
    arma::mat gL = W2 % (L - Target);
    arma::mat hL = arma::diagmat(arma::vectorise(W2));

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);
    d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    d_constraints.insert_cols(p*q, p);

  }

};

class xtargetQ: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k) {

    Phi = T.t() * T;
    Inv_T = inv(T);
    L = lambda * Inv_T.t();
    f1 = Weight % (L - Target);
    f2 = Phi_Weight % (Phi - Phi_Target);

    f = 0.5*arma::accu(f1 % f1) + 0.25*w*arma::accu(f2 % f2);

  }

  void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
         arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
         arma::mat IgCL2N, arma::mat N, arma::mat M,
         arma::mat Weight, arma::mat Phi_Weight, double w, double k) {

    arma::mat df1_dL = Weight % f1;
    arma::mat df1_dt = - Inv_T.t() * df1_dL.t() * L;
    arma::mat df2_dt = T * (Phi_Weight % f2);

    g = df1_dt + w*df2_dt;

  }

  void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k) {

    arma::mat Inv_T_dt = Inv_T * dT;
    arma::mat dL = - L * Inv_T_dt.t();

    arma::mat dg1L = W2 % dL;
    arma::mat TtdT = T.t() * dT;
    arma::mat dg2 = dT * (PW % f2) + T * (PW2 % (TtdT.t() + TtdT));

    arma::mat dg1 = - g * Inv_T_dt.t() - (dT * Inv_T).t() * g - (dg1L * Inv_T).t() * L;

    dg = dg1 + w*dg2;

  }

  void d_constraint(arma::mat& d_constraints, arma::mat L, arma::mat Phi,
                    arma::mat Target, arma::mat Weight,
                    arma::mat PhiTarget, arma::mat PhiWeight,
                    double gamma, double k, double epsilon, double w){

    int p = L.n_rows;
    int q = L.n_cols;
    int pq = p*q;
    int qq = q*q;
    int q_cor = q*(q-1)/2;
    arma::uvec indexes_1(q);
    for(int i=0; i < q; ++i) indexes_1[i] = ((i+1)*q) - (q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::mat I2(q, q, arma::fill::eye);

    arma::mat W2 = Weight % Weight;
    arma::mat gL = W2 % (L - Target);
    arma::mat hL = arma::diagmat(arma::vectorise(W2));

    arma::mat d_constraints_temp = arma::kron(I2, L.t()) * hL + arma::kron(gL.t(), I2) * dxt(L);

    /*
     * Multiply each column (q x q form) by inv_Phi
     */

    arma::mat inv_Phi = arma::inv_sympd(Phi);
    arma::cube B(qq, pq, 1);
    B.slice(0) = d_constraints_temp;
    B.reshape(q, q, pq);
    B.each_slice() *= inv_Phi;
    B.reshape(q*q, pq, 1);
    d_constraints_temp = B.slice(0);

    arma::mat c1p = -arma::kron(inv_Phi.t(), (L.t() * gL * inv_Phi));
    arma::mat Phi_t = dxt(Phi);
    arma::mat HP_temp = c1p + c1p * Phi_t;
    arma::mat diag_PW2 = arma::diagmat(arma::vectorise(PhiWeight % PhiWeight));
    HP_temp -= diag_PW2 + diag_PW2 * Phi_t;
    arma::mat HP = HP_temp.cols(indexes_2);

    d_constraints = arma::join_rows(d_constraints_temp, HP);
    d_constraints.shed_rows(indexes_1);
    d_constraints.insert_cols(p*q + q_cor, p);

  }

};

/*
 * Derivatives of the gradient wrt the transformed lambda
 */

