arma::mat dxt(arma::mat X) {

  /*
   * derivative of a matrix wrt its transpose
   */

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

arma::mat skew(arma::mat X) {

  return 0.5 * (X - X.t());

}

arma::mat symm(arma::mat X) {

  return 0.5 * (X + X.t());

}

// Solve the lyapunov equation YX + XY = Q with symmetric Q and X:

arma::mat lyap_sym(arma::mat Y, arma::mat Q) {

  int q = Y.n_cols;
  arma::vec I(q, arma::fill::ones);

  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, Y);

  arma::mat M = eigvec.t() * Q * eigvec;
  arma::mat W1 = I * eigval.t();
  arma::mat W = W1 + W1.t();
  arma::mat YY = M / W;
  arma::mat A = eigvec * YY * eigvec.t();

  return A;

}

arma::mat syl(arma::mat Y, arma::mat Q) {

  arma::mat X = arma::syl(Y, Y, -Q);

  return X;

}

arma::mat lyapunov(arma::mat Y, arma::mat Q, arma::uvec indexes) {

  int q = Y.n_rows;
  arma::mat I(q, q, arma::fill::eye);
  arma::mat Y1 = arma::kron(I, Y);
  arma::mat Y2 = arma::kron(Y, I);
  arma::mat X = Y1 + Y2;

  Q = Q.elem(indexes);
  X = X(indexes, indexes);
  arma::vec Lambda = arma::solve(X, Q, arma::solve_opts::fast);
  arma::mat A(q, q, arma::fill::zeros);
  A(indexes) = Lambda;
  A = symmatl(A);

  return A;

}

arma::mat lyapunov_2(arma::mat Y, arma::mat Q, arma::uvec indexes) {

  int q = Y.n_rows;
  arma::mat I(q, q, arma::fill::eye);
  arma::mat Y1 = arma::kron(I, Y);
  arma::mat Y2 = arma::kron(Y, I);
  arma::mat X = Y1 + Y2;

  Q = Q.elem(indexes);
  X = X(indexes, indexes);
  arma::vec Lambda = arma::solve(X, Q, arma::solve_opts::fast);
  arma::mat A(q, q, arma::fill::zeros);
  A(indexes) = Lambda;

  return A;

}

arma::uvec consecutive(int lower, int upper) {

  int size = upper - lower + 1;
  arma::uvec ivec(size);
  std::iota(ivec.begin(), ivec.end(), lower);

  return ivec;
}

std::vector<arma::uvec> vector_to_list(arma::uvec v){

  int n = v.size();
  std::vector<arma::uvec> lista(n);
  v.insert_rows(0, 1);

  for(int i=0; i < n; ++i) {

    lista[i] = v[i] + consecutive(1, v[i+1]);

  }

  return lista;

}

std::vector<arma::uvec> vector_to_list2(arma::uvec v){

  int n = v.size();
  int add = 0;
  std::vector<arma::uvec> lista(n);

  for(int i=0; i < n; ++i) {

    if(i != 0) {
      add = lista[i-1].back() + 1;
    }

    lista[i] = add + consecutive(1, v[i]) - 1;

  }

  return lista;

}

arma::vec orthogonalize(arma::mat orthogonals, arma::vec x, int k) {

  for(int i=0; i < k; ++i) {

    // x -= arma::accu(orthogonals.col(i) % x) / arma::accu(orthogonals.col(i) % orthogonals.col(i)) * orthogonals.col(i);
    x -= arma::accu(orthogonals.col(i) % x) * orthogonals.col(i);

  }

  x /= sqrt(arma::accu(x % x));

  return x;

}

arma::uvec list_to_vector(std::vector<arma::uvec> X) {

  arma::uvec single_vector = std::accumulate(X.begin(), X.end(),
                                             arma::uvec(), [](arma::uvec a, arma::uvec b) {
                                               a = arma::join_cols(a, b);
                                               return a;
                                             });

  return single_vector;

}

std::vector<arma::uvec> increment(arma::uvec oblq_indexes, int p) {

  arma::uvec oblq_indexes_total = oblq_indexes;
  int n_blocks = oblq_indexes.size();
  int total = arma::accu(oblq_indexes);
  if(p != total) {
    oblq_indexes_total.insert_rows(n_blocks, 1);
    oblq_indexes_total[n_blocks] = (p - total + 0.00);
  }
  std::vector<arma::uvec> indexes_list = vector_to_list2(oblq_indexes_total);

  return indexes_list;

}

typedef struct arguments{

  int p, q;
  double w = 1, gamma = 0, a = 10, f, q2;
  arma::vec k = {0}, epsilon = {0.01};

  arma::mat lambda, T, L, Phi, Inv_T, dL, dP, Inv_T_dt, dT, g,
  gL, gP, dg, dgL, dgP, hL, hP, d_constr, d_constr_temp, rg, A,
  dH, f1, f2, L2, LoL2, IgCL2N, I_gamma_C, N, M, H, HL2,
  Target, Weight, Phi_Target, Phi_Weight, Weight2, Phi_Weight2, S,
  Ls, Lg, L2s, L2g, exp_aL2g, g_exp_aL2g, gL2g, gL2s, C, logC, logCN,
  gC1, gC, glogC, glogCN, gexplogCN, exp_lCN, gL1, gL2, I1, I2, Ng, dxt_L2s;
  std::string penalization = "none";
  bool penalize = false;

  arma::vec term;
  arma::uvec oblq_indexes, blocks_vector; // REMOVE blocks_vector?
  std::vector<arma::uvec> list_oblq_indexes, blocks_list;
  arma::vec block_weights;
  int n_blocks = 1, n_rotations = 1, i, n_loads;

  std::vector<std::string> rotations = {"oblimin"};
  std::string projection = "oblq";
  std::vector<arma::mat> Li, Li2, Ni, HLi2, LoLi2, IgCL2Ni,
  f1i, Weighti, Targeti;
  std::vector<arma::vec> termi;

} args;

// Manifolds

class base_manifold {

public:

  virtual void param(arguments& x) = 0;

  virtual void dLP(arguments& x) = 0;

  virtual void grad(arguments& x) = 0;

  virtual void dgrad(arguments& x) = 0;

  virtual void g_constraints(arguments& x) = 0;

  virtual void proj(arguments& x) = 0;

  virtual void hess(arguments& x) = 0;

  virtual void retr(arguments& x) = 0;

};

// Orthogonal manifold:

class orth:public base_manifold {

public:

  void param(arguments& x) {

    x.L = x.lambda*x.T;

  }

  void dLP(arguments& x) {

    x.dL = x.lambda * x.dT;

  }

  void grad(arguments& x) {

    x.g = x.lambda.t() * x.gL;

  }

  void dgrad(arguments& x) {

    x.dg = x.lambda.t() * x.dgL;

  }

  void g_constraints(arguments& x) {

    arma::mat I1(x.p, x.p, arma::fill::eye);
    arma::mat I2(x.q, x.q, arma::fill::eye);
    int pq = x.p*x.q;

    x.d_constr_temp = arma::kron(I2, x.L.t()) * x.hL +
      arma::kron(x.gL.t(), I2) * dxt(x.L);

    arma::uvec indexes_1 = arma::trimatl_ind(arma::size(I2), -1);
    arma::uvec indexes_2 = arma::trimatu_ind(arma::size(I2), 1);
    arma::mat d_constr2 = dxt(I2) * x.d_constr_temp;
    x.d_constr_temp -= d_constr2;
    x.d_constr = x.d_constr_temp.rows(indexes_1);
    x.d_constr.insert_cols(pq, x.p);

  }

  void proj(arguments& x) {

    x.rg = x.T * skew(x.T.t() * x.g);

  }

  void hess(arguments& x) {

    arma::mat drg = x.dg - x.dT * symm(x.T.t() * x.g); // update this?
    // Rcpp::Rcout << drg << std::endl;
    // arma::mat drg = x.dT * skew(x.T.t() * x.g) + x.T * skew(x.dT.t() * x.g) + x.T*skew(x.T.t() * x.dg);
    // Rcpp::Rcout << drg << std::endl;
    x.dH = x.T * skew(x.T.t() * drg);

  }

  void retr(arguments& x) {

    arma::mat Q, R;
    arma::qr_econ(Q, R, x.T);

    x.T = Q;

  }

};

// Oblique manifold:

class oblq:public base_manifold {

public:

  void param(arguments& x) {

    x.Phi = x.T.t() * x.T;
    x.Inv_T = arma::inv(x.T);
    x.L = x.lambda * x.Inv_T.t();

  }

  void dLP(arguments& x) {

    x.Inv_T_dt = x.Inv_T * x.dT;
    x.dL = - x.L * x.Inv_T_dt.t();

    // if(!dP.is_empty()) {

      x.dP = x.T.t() * x.dT;
      x.dP += x.dP.t();

    // }

  }

  void grad(arguments& x) {

    arma::mat g1 = - x.Inv_T.t() * x.gL.t() * x.L;

    if(x.gP.is_empty()) {

      x.g = g1;

    } else {

      arma::mat g2 = x.T * x.gP;
      x.g = g1 + g2;

    }

  }

  void dgrad(arguments& x) {

    arma::mat dg1 = - x.g * x.Inv_T_dt.t() - (x.dT * x.Inv_T).t() * x.g - (x.dgL * x.Inv_T).t() * x.L;

    if(x.gP.is_empty()) {

      x.dg = dg1;

    } else {

      arma::mat dg2 = x.dT * x.gP + x.T * x.dgP;
      x.dg = dg1 + dg2;

    }

  }

  void g_constraints(arguments& x) {

    int pq = x.p*x.q;
    int qq = x.q*x.q;
    int q_cor = x.q*(x.q-1)/2;
    arma::uvec indexes_1(x.q);

    for(int i=0; i < x.q; ++i) indexes_1[i] = ((i+1)*x.q) - (x.q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(x.Phi), -1);
    arma::mat I2(x.q, x.q, arma::fill::eye);

    x.d_constr_temp = arma::kron(I2, x.L.t()) * x.hL +
      arma::kron(x.gL.t(), I2) * dxt(x.L);

    arma::mat inv_Phi = arma::inv_sympd(x.Phi);
    arma::cube B(qq, pq, 1);
    B.slice(0) = x.d_constr_temp;
    B.reshape(x.q, x.q, pq);
    B.each_slice() *= inv_Phi;
    B.reshape(x.q*x.q, pq, 1);
    x.d_constr_temp = B.slice(0);

    arma::mat c1p = -arma::kron(inv_Phi.t(), (x.L.t() * x.gL * inv_Phi));
    arma::mat Phi_t = dxt(x.Phi);
    arma::mat HP_temp = c1p + c1p * Phi_t;
    if(!x.hP.is_empty()) HP_temp -= x.hP;
    arma::mat HP = HP_temp.cols(indexes_2);

    x.d_constr = arma::join_rows(x.d_constr_temp, HP);
    x.d_constr.shed_rows(indexes_1);
    x.d_constr.insert_cols(pq + q_cor, x.p);

  };

  void proj(arguments& x) {

    x.rg = x.g - x.T * arma::diagmat( x.T.t() * x.g );

  }

  void hess(arguments& x) {

    x.dH = x.dg - x.dT * arma::diagmat( x.T.t() * x.g) - x.T * arma::diagmat( x.T.t() * x.dg );
    // arma::mat drg = x.dg - x.dT * arma::diagmat( x.T.t() * x.g) - x.T * arma::diagmat( x.dT.t() * x.g ) -
      // x.T * arma::diagmat( x.T.t() * x.dg );
    // x.dH = drg - x.T * arma::diagmat( x.T.t() * drg );

  }

  void retr(arguments& x) {

    x.T *= arma::diagmat( 1 / sqrt(arma::sum(x.T % x.T, 0)) );

  }

};

// Partially oblique manifold:

class poblq:public base_manifold {

public:

  void param(arguments& x) {

    x.Phi = x.T.t() * x.T;
    x.Inv_T = arma::inv(x.T);
    x.L = x.lambda * x.Inv_T.t();

  }

  void dLP(arguments& x) {

    x.Inv_T_dt = x.Inv_T * x.dT;
    x.dL = - x.L * x.Inv_T_dt.t();

    x.dP = x.T.t() * x.dT;
    x.dP += x.dP.t();

  }

  void grad(arguments& x) {

    arma::mat g1 = - x.Inv_T.t() * x.gL.t() * x.L;

    if(x.gP.is_empty()) {

      x.g = g1;

    } else {

      arma::mat g2 = x.T * x.gP;
      x.g = g1 + g2;

    }

  }

  void dgrad(arguments& x) {

    arma::mat dg1 = - x.g * x.Inv_T_dt.t() - (x.dT * x.Inv_T).t() * x.g - (x.dgL * x.Inv_T).t() * x.L;

    if(x.gP.is_empty()) {

      x.dg = dg1;

    } else {

      arma::mat dg2 = x.dT * x.gP + x.T * x.dgP;
      x.dg = dg1 + dg2;

    }

  }

  void g_constraints(arguments& x) {

    int pq = x.p*x.q;
    int qq = x.q*x.q;
    int q_cor = x.q*(x.q-1)/2;
    arma::uvec indexes_1(x.q);

    for(int i=0; i < x.q; ++i) indexes_1[i] = ((i+1)*x.q) - (x.q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(x.Phi), -1);
    arma::mat I2(x.q, x.q, arma::fill::eye);

    x.d_constr_temp = arma::kron(I2, x.L.t()) * x.hL +
      arma::kron(x.gL.t(), I2) * dxt(x.L);

    arma::mat inv_Phi = arma::inv_sympd(x.Phi);
    arma::cube B(qq, pq, 1);
    B.slice(0) = x.d_constr_temp;
    B.reshape(x.q, x.q, pq);
    B.each_slice() *= inv_Phi;
    B.reshape(x.q*x.q, pq, 1);
    x.d_constr_temp = B.slice(0);

    arma::mat c1p = -arma::kron(inv_Phi.t(), (x.L.t() * x.gL * inv_Phi));
    arma::mat Phi_t = dxt(x.Phi);
    arma::mat HP_temp = c1p + c1p * Phi_t;
    if(!x.hP.is_empty()) HP_temp -= x.hP;
    arma::mat HP = HP_temp.cols(indexes_2);

    x.d_constr = arma::join_rows(x.d_constr_temp, HP);
    x.d_constr.shed_rows(indexes_1);
    x.d_constr.insert_cols(pq + q_cor, x.p);

  };

  void proj(arguments& x) {

    arma::mat c1 = x.T.t() * x.g;
    arma::mat X0 = c1 + c1.t();
    x.A = lyap_sym(x.Phi, X0);
    x.A(x.oblq_indexes).zeros();

    // x.A = syl(x.Phi, X0);
    // X0(x.oblq_indexes).zeros(); x.A = lyapunov(x.Phi, X0, x.oblq_indexes); x.A(x.oblq_indexes).zeros();
    // x.A = lyapunov_2(x.Phi, X0, x.oblq_indexes);
    arma::mat N = x.T * x.A;
    x.rg = x.g - N;

  }

  void hess(arguments& x) {

    // Implicit differentiation of APhi + PhiA = X0
    arma::mat dc1 = x.dT.t() * x.g + x.T.t() * x.dg; // Differential of c1
    arma::mat dX0 = dc1 + dc1.t(); // Differential of X0
    arma::mat c2 = x.A * x.dP + x.dP * x.A; // Differential of APhi + PhiA wrt Phi
    arma::mat Q = dX0 - c2;
    // dAPhi + PhidA = Q
    arma::mat dA = lyap_sym(x.Phi, Q);
    dA(x.oblq_indexes).zeros(); // should be set to 0?

    // arma::mat dA = syl(Phi, Q);
    // Q(x.oblq_indexes).zeros(); arma::mat dA = lyapunov(x.Phi, Q, x.oblq_indexes); dA(x.oblq_indexes).zeros();
    // arma::mat dA = lyapunov_2(x.Phi, Q, x.oblq_indexes);
    arma::mat drg = x.dg - (x.dT * x.A + x.T * dA);

    // projection
    arma::mat c = x.T.t() * drg;
    arma::mat X0 = c + c.t();
    arma::mat A = lyap_sym(x.Phi, X0);
    A(x.oblq_indexes).zeros();

    // X0(x.oblq_indexes).zeros(); arma::mat A = lyapunov(x.Phi, X0, x.oblq_indexes); A(x.oblq_indexes).zeros();
    // arma::mat A = lyapunov_2(x.Phi, X0, x.oblq_indexes);
    arma::mat N = x.T * A;
    x.dH = drg - N;

  }

  void retr(arguments& x) {

    x.T *= arma::diagmat( 1 / sqrt(arma::diagvec( x.T.t() * x.T )) );

    arma::mat Q;
    arma::mat R;
    qr_econ(Q, R, x.T);

    arma::mat X2 = x.T;
    x.T = Q;
    x.T.cols(x.list_oblq_indexes[0]) = X2.cols(x.list_oblq_indexes[0]);

    int J = x.T.n_cols;
    int I = x.list_oblq_indexes.size()-1;

    // In the following loop, the Gram-Schmidt process is performed between blocks:

    for(int i=0; i < I; ++i) {

      // Select the cumulative indexes of blocks:
      std::vector<arma::uvec> indexes(&x.list_oblq_indexes[0], &x.list_oblq_indexes[i+1]);
      arma::uvec cum_indexes = list_to_vector(indexes);

      arma::mat orthogonals = Q.cols(cum_indexes);
      int n = orthogonals.n_cols;

      for(int j=0; j < x.list_oblq_indexes[i+1].size(); ++j) {

        // orthogonalize every column of the following block:
        int index = x.list_oblq_indexes[i+1][j];
        x.T.col(index) = orthogonalize(orthogonals, X2.col(index), n);

      }

    }

  }

};

// Retractions:

arma::mat retr_orth(arma::mat X) {

  arma::mat Q, R;
  arma::qr_econ(Q, R, X);

  return Q;

}

arma::mat retr_oblq(arma::mat X) {

  X *= arma::diagmat(1 / sqrt(arma::sum(X % X, 0)));

  return X;

}

arma::mat retr_poblq(arma::mat X, arma::uvec oblq_blocks) {

  int n_factors = arma::accu(oblq_blocks);
  if(n_factors > X.n_rows || n_factors > X.n_cols) Rcpp::stop("Too many factors declared in oblq_blocks");

  std::vector<arma::uvec> list_oblq_blocks = vector_to_list2(oblq_blocks);

  poblq RR;
  arguments x;
  x.T = X;
  x.list_oblq_indexes = list_oblq_blocks;
  RR.retr(x);

  return x.T;

}

arma::mat random_orth(int p, int q) {

  arma::mat X(p, q, arma::fill::randn);
  arma::mat Q;
  arma::mat R;
  qr_econ(Q, R, X);

  return Q;

}

arma::mat random_oblq(int p, int q) {

  arma::mat X(p, q, arma::fill::randn);
  X *= arma::diagmat(1 / sqrt(arma::sum(X % X, 0)));

  return X;

}

arma::mat random_poblq(int p, int q, arma::uvec oblq_blocks) {

  int n_factors = arma::accu(oblq_blocks);
  if(n_factors > p || n_factors > q) Rcpp::stop("Too many factors declared in oblq_blocks");

  std::vector<arma::uvec> list_oblq_blocks = vector_to_list2(oblq_blocks);;

  arma::mat X(p, q, arma::fill::randn);
  poblq RR;
  arguments x;
  x.T = X;
  x.list_oblq_indexes = list_oblq_blocks;
  RR.retr(x);

  return x.T;

}
