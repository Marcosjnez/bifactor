// Criteria

class base_criterion {

public:

  virtual void F(arma::mat& Inv_T, arma::mat& L, arma::mat& Phi,
                 arma::mat& f1, arma::mat& f2, double& f,
                 arma::mat lambda, arma::mat T, arma::mat Target, arma::mat Weight,
                 arma::mat& L2, arma::vec& term, int p, double epsilon,
                 arma::mat& IgCL2N, arma::mat I_gamma_C, arma::mat N, arma::mat M,
                 arma::mat Phi_Target, arma::mat Phi_Weight, double w, double k) = 0;

  virtual void G(arma::mat& g, arma::mat lambda, arma::mat f1, arma::mat f2, arma::mat T, arma::mat Inv_T, arma::mat L,
                 arma::mat& LoL2, arma::mat L2, arma::vec term, int p, double p2,
                 arma::mat IgCL2N, arma::mat N, arma::mat M,
                 arma::mat Weight, arma::mat Phi_Weight, double w, double k) = 0;

  virtual void dG(arma::mat& dg, arma::mat lambda, arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
                  arma::mat L2, arma::mat LoL2, arma::vec term, double p2, double epsilon,
                  arma::mat I_gamma_C, arma::mat IgCL2N, arma::mat N, arma::mat M,
                  arma::mat W2, arma::mat f2, arma::mat PW, arma::mat PW2, double w, double k) = 0;

};


// CF

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
    L2 = arma::pow(L, 2);
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
    L2 = arma::pow(L, 2);
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

    f = 0.5*arma::accu(pow(f1, 2));

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

    f = 0.5*arma::accu(pow(f1, 2));

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
};
