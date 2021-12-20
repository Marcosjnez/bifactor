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

arma::mat skew(arma::mat X) {

  return 0.5 * (X - X.t());

}

arma::mat symm(arma::mat X) {

  return 0.5 * (X + X.t());

}

// Solve the lyapunov equation YX + XY = Q for symmetric Q and X:

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

  return(indexes_list);

}

// Manifolds

class base_manifold {

public:

  virtual void param(arma::mat& L, arma::mat lambda, arma::mat& Phi,
                     arma::mat& Inv_T, arma::mat T) = 0;

  virtual void dLP(arma::mat& dL, arma::mat& dP, arma::mat& Inv_T_dt, arma::mat T,
                   arma::mat lambda, arma::mat L, arma::mat Inv_T, arma::mat dT) = 0;

  virtual void grad(arma::mat& g, arma::mat lambda, arma::mat L,
                    arma::mat gL, arma::mat Inv_T, arma::mat T, arma::mat gP, double w) = 0;

  virtual void dgrad(arma::mat& dg, arma::mat dgL, arma::mat dgP, arma::mat gP, arma::mat lambda,
                     arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
                     arma::mat Inv_T_dt) = 0;

  virtual void g_constraints(arma::mat& d_constraints, arma::mat d_constraints_temp,
                             arma::mat L, arma::mat Phi, arma::mat gL) = 0;

  virtual void proj(arma::mat& rg, arma::mat& A, arma::mat g, arma::mat T,
                    arma::mat Phi, arma::uvec indexes_1) = 0;

  virtual void hess(arma::mat& dH, arma::mat delta, arma::mat g, arma::mat dg, arma::mat T,
                    arma::mat Phi, arma::mat A, arma::uvec indexes_1) = 0;

  virtual void retr(arma::mat& T, std::vector<arma::uvec> oblq_indexes) = 0;

};

// Orthogonal manifold:

class orth:public base_manifold {

public:

  void param(arma::mat& L, arma::mat lambda, arma::mat& Phi,
                arma::mat& Inv_T, arma::mat T) {

    L = lambda*T;

  }

  void dLP(arma::mat& dL, arma::mat& dP, arma::mat& Inv_T_dt, arma::mat T,
           arma::mat lambda, arma::mat L, arma::mat Inv_T, arma::mat dT) {

    dL = lambda * dT;

  }

  void grad(arma::mat& g, arma::mat lambda, arma::mat L,
         arma::mat gL, arma::mat Inv_T, arma::mat T, arma::mat gP, double w) {

    g = lambda.t() * gL;

  }

  void dgrad(arma::mat& dg, arma::mat dgL, arma::mat dgP, arma::mat gP, arma::mat lambda,
          arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat Inv_T_dt){

    dg = lambda.t() * dgL;

  }

  void g_constraints(arma::mat& d_constraints, arma::mat d_constraints_temp,
                     arma::mat L, arma::mat Phi, arma::mat gL) {

    int p = L.n_rows;
    int q = L.n_cols;
    int pq = p*q;

    arma::uvec indexes_1 = arma::trimatl_ind(arma::size(Phi), -1);
    arma::uvec indexes_2 = arma::trimatu_ind(arma::size(Phi), 1);
    d_constraints = d_constraints_temp.rows(indexes_1) - d_constraints_temp.rows(indexes_2);
    d_constraints.insert_cols(pq, p);

  }

  void proj(arma::mat& rg, arma::mat& A, arma::mat g, arma::mat T,
            arma::mat Phi, arma::uvec indexes_1) {

    // A = skew(T.t() * g);
    // rg = T * A;
    rg = T * skew(T.t() * g);

  }

  void hess(arma::mat& dH, arma::mat delta, arma::mat g, arma::mat dg, arma::mat T,
            arma::mat Phi, arma::mat A, arma::uvec indexes_1) {

    // arma::mat hessian = delta * A + T * skew(delta.t() * g + T.t() * dg);
    // proj_orth(dH, A, hessian, T, Phi, indexes_1);
    dg -= delta * symm(T.t() * g);
    proj(dH, A, dg, T, Phi, indexes_1);

  }

  void retr(arma::mat& T, std::vector<arma::uvec> oblq_indexes) {

    arma::mat Q, R;
    arma::qr_econ(Q, R, T);

    T = Q;

  }

};

// Oblique manifold:

class oblq:public base_manifold {

public:

  void param(arma::mat& L, arma::mat lambda, arma::mat& Phi,
                arma::mat& Inv_T, arma::mat T) {

    Phi = T.t() * T;
    Inv_T = inv(T);
    L = lambda * Inv_T.t();

  }

  void dLP(arma::mat& dL, arma::mat& dP, arma::mat& Inv_T_dt, arma::mat T,
          arma::mat lambda, arma::mat L, arma::mat Inv_T, arma::mat dT) {

    Inv_T_dt = Inv_T * dT;
    dL = - L * Inv_T_dt.t();

    // if(!dP.is_empty()) {

      dP = T.t() * dT;
      dP += dP.t();

    // }

  }

  void grad(arma::mat& g, arma::mat lambda, arma::mat L,
         arma::mat gL, arma::mat Inv_T, arma::mat T, arma::mat gP, double w) {

    arma::mat g1 = - Inv_T.t() * gL.t() * L;

    if(gP.is_empty()) {

      g = g1;

    } else {

      arma::mat g2 = w*T*gP;
      g = g1 + g2;

    }

  }

  void dgrad(arma::mat& dg, arma::mat dgL, arma::mat dgP, arma::mat gP, arma::mat lambda,
          arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
          arma::mat Inv_T_dt) {

    arma::mat dg1 = - g * Inv_T_dt.t() - (dT * Inv_T).t() * g - (dgL * Inv_T).t() * L;

    if(gP.is_empty()) {

      dg = dg1;

    } else {

      arma::mat dg2 = dT * gP + T * dgP;
      dg = dg1 + dg2;

    }

  }

  void g_constraints(arma::mat& d_constraints, arma::mat d_constraints_temp,
                     arma::mat L, arma::mat Phi, arma::mat gL) {

    int p = L.n_rows;
    int q = L.n_cols;
    int pq = p*q;
    int qq = q*q;
    int q_cor = q*(q-1)/2;

    arma::uvec indexes_1(q);
    for(int i=0; i < q; ++i) indexes_1[i] = ((i+1)*q) - (q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(Phi), -1);

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

  };

  void proj(arma::mat& rg, arma::mat& A, arma::mat g, arma::mat T,
            arma::mat Phi, arma::uvec indexes_1) {

    rg = g - T * diagmat( T.t() * g );

  }

  void hess(arma::mat& dH, arma::mat delta, arma::mat g, arma::mat dg, arma::mat T,
            arma::mat Phi, arma::mat A, arma::uvec indexes_1) {

    dH = dg - T * diagmat( T.t() * dg ) - delta * diagmat( T.t() * g);

  }

  void retr(arma::mat& T, std::vector<arma::uvec> oblq_indexes) {

    T *= arma::diagmat(1 / sqrt(arma::sum(T % T, 0)));

  }

};

// Partially oblique manifold:

class poblq:public base_manifold {

public:

  void param(arma::mat& L, arma::mat lambda, arma::mat& Phi,
             arma::mat& Inv_T, arma::mat T) {

    Phi = T.t() * T;
    Inv_T = inv(T);
    L = lambda * Inv_T.t();

  }

  void dLP(arma::mat& dL, arma::mat& dP, arma::mat& Inv_T_dt, arma::mat T,
           arma::mat lambda, arma::mat L, arma::mat Inv_T, arma::mat dT) {

    Inv_T_dt = Inv_T * dT;
    dL = - L * Inv_T_dt.t();

    // if(!dP.is_empty()) {

    dP = T.t() * dT;
    dP += dP.t();

    // }

  }

  void grad(arma::mat& g, arma::mat lambda, arma::mat L,
            arma::mat gL, arma::mat Inv_T, arma::mat T, arma::mat gP, double w) {

    arma::mat g1 = - Inv_T.t() * gL.t() * L;

    if(gP.is_empty()) {

      g = g1;

    } else {

      arma::mat g2 = w*T*gP;
      g = g1 + g2;

    }

  }

  void dgrad(arma::mat& dg, arma::mat dgL, arma::mat dgP, arma::mat gP, arma::mat lambda,
             arma::mat dT, arma::mat T, arma::mat Inv_T, arma::mat L, arma::mat g,
             arma::mat Inv_T_dt) {

    arma::mat dg1 = - g * Inv_T_dt.t() - (dT * Inv_T).t() * g - (dgL * Inv_T).t() * L;

    if(gP.is_empty()) {

      dg = dg1;

    } else {

      arma::mat dg2 = dT * gP + T * dgP;
      dg = dg1 + dg2;

    }

  }

  void g_constraints(arma::mat& d_constraints, arma::mat d_constraints_temp,
                     arma::mat L, arma::mat Phi, arma::mat gL) {

    int p = L.n_rows;
    int q = L.n_cols;
    int pq = p*q;
    int qq = q*q;
    int q_cor = q*(q-1)/2;

    arma::uvec indexes_1(q);
    for(int i=0; i < q; ++i) indexes_1[i] = ((i+1)*q) - (q-i);
    arma::uvec indexes_2 = arma::trimatl_ind(arma::size(Phi), -1);

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

  };

  void proj(arma::mat& rg, arma::mat& A, arma::mat g, arma::mat X,
            arma::mat Phi, arma::uvec indexes_1) {

    arma::mat c1 = X.t() * g;
    arma::mat X0 = c1 + c1.t();
    A = lyap_sym(Phi, X0);
    A(indexes_1).zeros();
    // A = syl(Phi, X0);
    // X0(indexes_1).zeros(); A = lyapunov(Phi, X0, indexes_1); A(indexes_1).zeros();
    // A = lyapunov_2(Phi, X0, indexes_1);
    arma::mat N = X * A;
    rg = g - N;

  }

  void hess(arma::mat& dH, arma::mat Z, arma::mat g, arma::mat dg, arma::mat X,
            arma::mat Phi, arma::mat A, arma::uvec indexes_1) {

    int q = X.n_cols;
    arma::mat c1 = Z.t() * g + X.t() * dg;
    arma::mat c2 = Z.t() * X;
    arma::mat c3 = A * (c2 + c2.t());
    arma::mat Q = c1 + c1.t() - c3 - c3.t();
    arma::mat dA = lyap_sym(Phi, Q);
    dA(indexes_1).zeros();
    // arma::mat dA = syl(Phi, Q);
    // Q(indexes_1).zeros(); arma::mat dA = lyapunov(Phi, Q, indexes_1); dA(indexes_1).zeros();
    // arma::mat dA = lyapunov_2(Phi, Q, indexes_1);
    arma::mat hessian = dg - (Z * A + X * dA);

    proj(dH, A, hessian, X, Phi, indexes_1);

  }

  void retr(arma::mat& X, std::vector<arma::uvec> oblq_indexes) {

    X *= diagmat(1 / sqrt(diagvec(X.t() * X)));

    arma::mat Q;
    arma::mat R;
    qr_econ(Q, R, X);

    arma::mat X2 = X;
    X = Q;
    X.cols(oblq_indexes[0]) = X2.cols(oblq_indexes[0]);

    int J = X.n_cols;
    int I = oblq_indexes.size()-1;

    // In the following loop, the Gram-Schmidt process is performed between blocks:

    for(int i=0; i < I; ++i) {

      // Select the cumulative indexes of blocks:
      std::vector<arma::uvec> indexes(&oblq_indexes[0], &oblq_indexes[i+1]);
      arma::uvec cum_indexes = list_to_vector(indexes);

      arma::mat orthogonals = Q.cols(cum_indexes);
      int n = orthogonals.n_cols;

      for(int j=0; j < oblq_indexes[i+1].size(); ++j) {

        // orthogonalize every column of the following block:
        int index = oblq_indexes[i+1][j];
        X.col(index) = orthogonalize(orthogonals, X2.col(index), n);

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
  RR.retr(X, list_oblq_blocks);

  return X;

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
  RR.retr(X, list_oblq_blocks);

  return X;

}
