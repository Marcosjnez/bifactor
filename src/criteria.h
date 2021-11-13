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
                 arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k,
                 std::vector<arma::uvec> indexes_list) = 0;

  virtual void gLP(arma::mat& gL, arma::mat& gP, arma::mat f1, arma::mat f2,
                  arma::mat L, arma::mat& LoL2, arma::mat L2,
                  arma::vec term, double p2, double epsilon,
                  arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
                  arma::mat Weight, arma::mat Phi_Weight, double k,
                  std::vector<arma::uvec> indexes_list) = 0;

  virtual void dgLP(arma::mat& dgL, arma::mat& dgP, arma::mat dL, arma::mat dP,
                   arma::mat L, arma::mat L2, arma::mat LoL2, arma::vec term,
                   double p2, double epsilon, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
                   arma::mat W2, arma::mat PW2, double k,
                   std::vector<arma::uvec> indexes_list) = 0;

  virtual void d_constraint(arma::mat& d_constraints, arma::mat lambda, arma::mat Phi,
                            arma::mat Target, arma::mat Weight,
                            arma::mat PhiTarget, arma::mat PhiWeight,
                            double gamma, double k, double epsilon, double w) = 0;

};

/*
 * Crawford-Ferguson family
 */

class cf: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k,
         std::vector<arma::uvec> indexes_list){

    L2 = L % L;
    double ff1 = (1-k) * arma::accu(L2 % (L2 * N)) / 4;
    double ff2 = k * arma::accu(L2 % (M * L2)) / 4;

    f = ff1 + ff2;

  }

  void gLP(arma::mat& gL, arma::mat& gP, arma::mat f1, arma::mat f2,
          arma::mat L, arma::mat& LoL2, arma::mat L2,
          arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat Weight, arma::mat Phi_Weight, double k,
          std::vector<arma::uvec> indexes_list){

    f1 = (1-k) * L % (L2 * N);
    f2 = k * L % (M * L2);
    gL = f1 + f2;

  }

  void dgLP(arma::mat& dgL, arma::mat& dgP, arma::mat dL, arma::mat dP,
           arma::mat L, arma::mat L2, arma::mat LoL2, arma::vec term,
           double p2, double epsilon, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
           arma::mat W2, arma::mat PW2, double k,
           std::vector<arma::uvec> indexes_list) {

    arma::mat dL2 = 2 * dL % L;
    arma::mat df1 = (1-k) * dL % (L2 * N) + (1-k) * L % (dL2 * N);
    arma::mat df2 = k * dL % (M * L2) + k * L % (M * dL2);

    dgL = df1 + df2;

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

    // Orthogonal case
    // arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    // arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);
    // d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    // d_constraints.insert_cols(p*q, p);

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

class oblimin: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k,
         std::vector<arma::uvec> indexes_list) {

    L2 = L % L;
    IgCL2N = I_gamma_C * L2 * N;

    f = trace(L2.t() * IgCL2N) / 4;

  }

  void gLP(arma::mat& gL, arma::mat& gP, arma::mat f1, arma::mat f2,
          arma::mat L, arma::mat& LoL2, arma::mat L2,
          arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat Weight, arma::mat Phi_Weight, double k,
          std::vector<arma::uvec> indexes_list) {

    gL = L % IgCL2N;

  }

  void dgLP(arma::mat& dgL, arma::mat& dgP, arma::mat dL, arma::mat dP,
           arma::mat L, arma::mat L2, arma::mat LoL2, arma::vec term,
           double p2, double epsilon, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
           arma::mat W2, arma::mat PW2, double k,
           std::vector<arma::uvec> indexes_list) {

    dgL = dL % IgCL2N + L % (I_gamma_C * (2*dL % L) * N);

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

    // Orthogonal case
    // arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    // arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);
    // d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    // d_constraints.insert_cols(p*q, p);

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

class geomin: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k,
         std::vector<arma::uvec> indexes_list) {

    L2 = L % L;
    L2 += epsilon;
    term = arma::exp(arma::sum(arma::log(L2), 1) / p);

    f = arma::accu(term);

  }

  void gLP(arma::mat& gL, arma::mat& gP, arma::mat f1, arma::mat f2,
          arma::mat L, arma::mat& LoL2, arma::mat L2,
          arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat Weight, arma::mat Phi_Weight, double k,
          std::vector<arma::uvec> indexes_list) {

    LoL2 = L / L2;
    gL = LoL2 * p2;
    gL.each_col() %= term;

  }

  void dgLP(arma::mat& dgL, arma::mat& dgP, arma::mat dL, arma::mat dP,
           arma::mat L, arma::mat L2, arma::mat LoL2, arma::vec term,
           double p2, double epsilon, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
           arma::mat W2, arma::mat PW2, double k,
           std::vector<arma::uvec> indexes_list) {

    arma::mat c1 = (epsilon - L % L) / (L2 % L2) % dL;
    c1.each_col() %= term;
    arma::mat c2 = LoL2;
    arma::vec term2 = p2 * term % arma::sum(LoL2 % dL, 1);
    c2.each_col() %= term2;

    dgL = p2 * (c1 + c2);

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

    arma::mat c1 = q2*(arma::vectorise(L2) - arma::vectorise(2*L % L)) /
      arma::vectorise(L2 % L2);
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

    // Orthogonal case
    // arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    // arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);
    // d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    // d_constraints.insert_cols(p*q, p);

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

class target: public base_criterion {

public:
  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k,
         std::vector<arma::uvec> indexes_list) {

    f1 = Weight % (L - Target);

    f = 0.5*arma::accu(f1 % f1);

  }

  void gLP(arma::mat& gL, arma::mat& gP, arma::mat f1, arma::mat f2,
          arma::mat L, arma::mat& LoL2, arma::mat L2,
          arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat Weight, arma::mat Phi_Weight, double k,
          std::vector<arma::uvec> indexes_list) {

    gL = Weight % f1;

  }

  void dgLP(arma::mat& dgL, arma::mat& dgP, arma::mat dL, arma::mat dP,
           arma::mat L, arma::mat L2, arma::mat LoL2, arma::vec term,
           double p2, double epsilon, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
           arma::mat W2, arma::mat PW2, double k,
           std::vector<arma::uvec> indexes_list) {

    dgL = W2 % dL;

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

    // Orthogonal case
    // arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    // arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);
    // d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    // d_constraints.insert_cols(p*q, p);

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

class xtarget: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k,
         std::vector<arma::uvec> indexes_list) {

    f1 = Weight % (L - Target);
    f2 = Phi_Weight % (Phi - Phi_Target);

    f = 0.5*arma::accu(f1 % f1) + 0.25*w*arma::accu(f2 % f2);

  }

  void gLP(arma::mat& gL, arma::mat& gP, arma::mat f1, arma::mat f2,
          arma::mat L, arma::mat& LoL2, arma::mat L2,
          arma::vec term, double p2, double epsilon,
          arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
          arma::mat Weight, arma::mat Phi_Weight, double k,
          std::vector<arma::uvec> indexes_list) {

    gL = Weight % f1;
    gP = Phi_Weight % f2;

  }

  void dgLP(arma::mat& dgL, arma::mat& dgP, arma::mat dL, arma::mat dP,
           arma::mat L, arma::mat L2, arma::mat LoL2, arma::vec term,
           double p2, double epsilon, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
           arma::mat W2, arma::mat PW2, double k,
           std::vector<arma::uvec> indexes_list) {

    dgL = W2 % dL;
    dgP = PW2 % dP;

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

    // Orthogonal case
    // arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    // arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);
    // d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    // d_constraints.insert_cols(p*q, p);

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

// Rep CF

class rep_cf: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k,
         std::vector<arma::uvec> indexes_list){

    int n_blocks = indexes_list.size();
    f = 0;

    for(int i=0; i < n_blocks; ++i) {

      arma::uvec indexes = indexes_list[i];
      arma::mat Li = L.cols(indexes);
      arma::mat Li2 = Li % Li;
      arma::mat Ni = N(indexes, indexes);
      double ff1 = (1-k) * arma::accu(Li2 % (Li2 * Ni)) / 4;
      double ff2 = k * arma::accu(Li2 % (M * Li2)) / 4;

      f += ff1 + ff2;

    }

  }

  void gLP(arma::mat& gL, arma::mat& gP, arma::mat f1, arma::mat f2,
           arma::mat L, arma::mat& LoL2, arma::mat L2,
           arma::vec term, double p2, double epsilon,
           arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
           arma::mat Weight, arma::mat Phi_Weight, double k,
           std::vector<arma::uvec> indexes_list){

    int n_blocks = indexes_list.size();
    gL.set_size(arma::size(L));
    gL.zeros();

    for(int i=0; i < n_blocks; ++i) {

      arma::uvec indexes = indexes_list[i];
      arma::mat Li = L.cols(indexes);
      arma::mat Li2 = Li % Li;
      arma::mat Ni = N(indexes, indexes);
      f1 = (1-k) * Li % (Li2 * Ni);
      f2 = k * Li % (M * Li2);
      gL.cols(indexes) += f1 + f2;

    }

  }

  void dgLP(arma::mat& dgL, arma::mat& dgP, arma::mat dL, arma::mat dP,
            arma::mat L, arma::mat L2, arma::mat LoL2, arma::vec term,
            double p2, double epsilon, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
            arma::mat W2, arma::mat PW2, double k,
            std::vector<arma::uvec> indexes_list) {

    int n_blocks = indexes_list.size();
    dgL.set_size(arma::size(L));
    dgL.zeros();

    for(int i=0; i < n_blocks; ++i) {

      arma::uvec indexes = indexes_list[i];
      arma::mat Li = L.cols(indexes);
      arma::mat Ni = N(indexes, indexes);
      arma::mat Li2 = Li % Li;
      arma::mat dLi = dL.cols(indexes);
      arma::mat dLi2 = 2 * dLi % Li;
      arma::mat df1 = (1-k) * dLi % (Li2 * Ni) + (1-k) * Li % (dLi2 * Ni);
      arma::mat df2 = k * dLi % (M * Li2) + k * Li % (M * dLi2);

      dgL.cols(indexes) += df1 + df2;

    }

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

    // Orthogonal case
    // arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    // arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);
    // d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    // d_constraints.insert_cols(p*q, p);

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

// Rep Oblimin

class rep_oblimin: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k,
         std::vector<arma::uvec> indexes_list) {

    int n_blocks = indexes_list.size();
    f = 0;

    for(int i=0; i < n_blocks; ++i) {

      arma::uvec indexes = indexes_list[i];
      arma::mat Li = L.cols(indexes);
      arma::mat Ni = N(indexes, indexes);
      arma::mat Li2 = Li % Li;
      arma::mat IgCL2Ni = I_gamma_C * Li2 * Ni;

      f += trace(Li2.t() * IgCL2Ni) / 4;

    }

  }

  void gLP(arma::mat& gL, arma::mat& gP, arma::mat f1, arma::mat f2,
           arma::mat L, arma::mat& LoL2, arma::mat L2,
           arma::vec term, double p2, double epsilon,
           arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
           arma::mat Weight, arma::mat Phi_Weight, double k,
           std::vector<arma::uvec> indexes_list) {

    int n_blocks = indexes_list.size();
    gL.set_size(arma::size(L));
    gL.zeros();

    for(int i=0; i < n_blocks; ++i) {

      arma::uvec indexes = indexes_list[i];
      arma::mat Li = L.cols(indexes);
      arma::mat Ni = N(indexes, indexes);
      arma::mat Li2 = Li % Li;
      arma::mat IgCL2Ni = I_gamma_C * Li2 * Ni;
      gL.cols(indexes) += Li % IgCL2Ni;

    }

  }

  void dgLP(arma::mat& dgL, arma::mat& dgP, arma::mat dL, arma::mat dP,
            arma::mat L, arma::mat L2, arma::mat LoL2, arma::vec term,
            double p2, double epsilon, arma::mat I_gamma_C, arma::mat IgCL2N,
            arma::mat N, arma::mat M, arma::mat W2, arma::mat PW2, double k,
            std::vector<arma::uvec> indexes_list) {

    int n_blocks = indexes_list.size();
    dgL.set_size(arma::size(L));
    dgL.zeros();

    for(int i=0; i < n_blocks; ++i) {

      arma::uvec indexes = indexes_list[i];
      arma::mat Li = L.cols(indexes);
      arma::mat Ni = N(indexes, indexes);
      arma::mat Li2 = Li % Li;
      arma::mat IgCL2Ni = I_gamma_C * Li2 * Ni;
      arma::mat dLi = dL.cols(indexes);
      dgL.cols(indexes) += dLi % IgCL2Ni + Li % (I_gamma_C * (2*dLi % Li) * Ni);

    }

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

    // Orthogonal case
    // arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    // arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);
    // d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    // d_constraints.insert_cols(p*q, p);

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

// Rep Geomin

class rep_geomin: public base_criterion {

public:

  void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
         arma::mat& f1, arma::mat& f2, double& f,
         arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
         arma::mat& L2, arma::vec& term, int p, double epsilon,
         arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
         arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k,
         std::vector<arma::uvec> indexes_list) {

    int n_blocks = indexes_list.size();
    f = 0;

    for(int i=0; i < n_blocks; ++i) {

      arma::uvec indexes = indexes_list[i];
      arma::mat Li = L.cols(indexes);
      arma::mat Li2 = Li % Li;
      Li2 += epsilon;
      arma::mat termi = arma::exp(arma::sum(arma::log(Li2), 1) / p);

      f += arma::accu(termi);

    }

  }

  void gLP(arma::mat& gL, arma::mat& gP, arma::mat f1, arma::mat f2,
           arma::mat L, arma::mat& LoL2, arma::mat L2,
           arma::vec term, double p2, double epsilon,
           arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
           arma::mat Weight, arma::mat Phi_Weight, double k,
           std::vector<arma::uvec> indexes_list) {

    int p = L.n_cols;
    int n_blocks = indexes_list.size();
    gL.set_size(arma::size(L));
    gL.zeros();

    for(int i=0; i < n_blocks; ++i) {

      arma::uvec indexes = indexes_list[i];
      arma::mat Li = L.cols(indexes);
      arma::mat Li2 = Li % Li;
      Li2 += epsilon;
      arma::mat LoLi2 = Li / Li2;
      arma::mat termi = arma::exp(arma::sum(arma::log(Li2), 1) / p);
      arma::mat gLi = LoLi2 * p2;
      gLi.each_col() %= termi;
      gL.cols(indexes) += gLi;

    }

  }

  void dgLP(arma::mat& dgL, arma::mat& dgP, arma::mat dL, arma::mat dP,
            arma::mat L, arma::mat L2, arma::mat LoL2, arma::vec term,
            double p2, double epsilon, arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
            arma::mat W2, arma::mat PW2, double k,
            std::vector<arma::uvec> indexes_list) {

    int p = L.n_cols;
    int n_blocks = indexes_list.size();
    dgL.set_size(arma::size(L));
    dgL.zeros();

    for(int i=0; i < n_blocks; ++i) {

      arma::uvec indexes = indexes_list[i];
      arma::mat Li = L.cols(indexes);
      arma::mat Li2 = Li % Li;
      Li2 += epsilon;
      arma::mat dLi = dL.cols(indexes);
      arma::mat LoLi2 = Li / Li2;
      arma::mat termi = arma::exp(arma::sum(arma::log(Li2), 1) / p);
      arma::mat c1 = (epsilon - Li % Li) / (Li2 % Li2) % dLi;
      c1.each_col() %= termi;
      arma::mat c2 = LoLi2;
      arma::vec termi2 = p2 * termi % arma::sum(LoLi2 % dLi, 1);
      c2.each_col() %= termi2;

      dgL.cols(indexes) += p2 * (c1 + c2);

    }

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

    arma::mat c1 = q2*(arma::vectorise(L2) - arma::vectorise(2*L % L)) /
      arma::vectorise(L2 % L2);
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

    // Orthogonal case
    // arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    // arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);
    // d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    // d_constraints.insert_cols(p*q, p);

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

/*
 * Derivatives of the gradient wrt the transformed lambda
 */

