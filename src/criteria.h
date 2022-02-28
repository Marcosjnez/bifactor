#include "manifolds.h"

arma::mat kdiag(arma::mat X) {

  /*
   * Transform every column into a diagonal matrix and bind
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

arma::mat bc(int g) {

  int k = g*(g-1)/2;
  arma::mat Ng(g, k);

  int i = 0;

  for(int k=0; k < (g-1); ++k) {
    for(int j=k+1; j < g; ++j) {

      Ng(k, i) = 1;
      Ng(j, i) = 1;
      ++i;

    }
  }

  return Ng;

}

// Criteria

class base_criterion {

public:

  virtual void F(arguments& x) = 0;

  virtual void gLP(arguments& x) = 0;

  virtual void hLP(arguments& x) = 0;

  virtual void dgLP(arguments& x) = 0;

};

/*
 * Crawford-Ferguson family
 */

class cf: public base_criterion {

public:

  void F(arguments& x){

    x.L2 = x.L % x.L;
    double ff1 = (1-x.k) * arma::accu(x.L2 % (x.L2 * x.N)) / 4;
    double ff2 = x.k * arma::accu(x.L2 % (x.M * x.L2)) / 4;

    x.f = ff1 + ff2;

  }

  void gLP(arguments& x){

    x.f1 = (1-x.k) * x.L % (x.L2 * x.N);
    x.f2 = x.k * x.L % (x.M * x.L2);
    x.gL = x.f1 + x.f2;

  }

  void hLP(arguments& x) {

    arma::mat Ip(x.p, x.p, arma::fill::eye);
    arma::mat c1 = arma::kron(x.N.t(), Ip) * arma::diagmat(arma::vectorise(2*x.L));
    arma::mat gf1 = (1-x.k)*arma::diagmat(arma::vectorise(x.L)) * c1 +
      arma::diagmat(arma::vectorise(x.L2 * x.N));

    arma::mat Iq(x.q, x.q, arma::fill::eye);
    arma::mat c2 = arma::kron(Iq, x.M) * arma::diagmat(arma::vectorise(2*x.L));
    arma::mat gf2 = x.k*arma::diagmat(arma::vectorise(x.L)) * c2 +
      arma::diagmat(arma::vectorise(x.M * x.L2));
    x.hL = gf1 + gf2;

  }

  void dgLP(arguments& x) {

    arma::mat dL2 = 2 * x.dL % x.L;
    arma::mat df1 = (1-x.k) * x.dL % (x.L2 * x.N) + (1-x.k) * x.L % (dL2 * x.N);
    arma::mat df2 = x.k * x.dL % (x.M * x.L2) + x.k * x.L % (x.M * dL2);

    x.dgL = df1 + df2;

  }

};

/*
 * Varimax
 */

class varimax: public base_criterion {

public:

  void F(arguments& x) {

    x.L2 = x.L % x.L;
    x.HL2 = x.H * x.L2;

    x.f = -arma::trace(x.HL2.t() * x.HL2) / 4;

  }

  void gLP(arguments& x) {

    x.gL = -x.L % x.HL2;

  }

  void hLP(arguments& x) {

    arma::mat c1 = arma::diagmat(arma::vectorise(x.HL2));
    arma::mat diagL = arma::diagmat(arma::vectorise(x.H * x.L));
    arma::mat diag2L = 2*diagL;
    arma::mat c2 = diagL * diag2L;
    x.hL = -c1 - c2;

  }

  void dgLP(arguments& x) {

    arma::mat dL2 = 2 * x.dL % x.L;
    x.dgL = -x.dL % x.HL2 - x.L % (x.H * dL2);

  }

};

/*
 * Varimin
 */

class varimin: public base_criterion {

public:

  void F(arguments& x) {

    x.L2 = x.L % x.L;
    x.HL2 = x.H * x.L2;

    x.f = arma::trace(x.HL2.t() * x.HL2) / 4;

  }

  void gLP(arguments& x) {

    x.gL = x.L % x.HL2;

  }

  void hLP(arguments& x) {

    arma::mat c1 = arma::diagmat(arma::vectorise(x.HL2));
    arma::mat diagL = arma::diagmat(arma::vectorise(x.H * x.L));
    arma::mat diag2L = 2*diagL;
    arma::mat c2 = diagL * diag2L;
    x.hL = c1 + c2;

  }

  void dgLP(arguments& x) {

    arma::mat dL2 = 2 * x.dL % x.L;
    x.dgL = x.dL % x.HL2 + x.L % (x.H * dL2);

  }

};

/*
 * Oblimin
 */

class oblimin: public base_criterion {

public:

  void F(arguments& x) {

    x.L2 = x.L % x.L;
    x.IgCL2N = x.I_gamma_C * x.L2 * x.N;

    x.f = arma::trace(x.L2.t() * x.IgCL2N) / 4;

  }

  void gLP(arguments& x) {

    x.gL = x.L % x.IgCL2N;

  }

  void hLP(arguments& x) {

    arma::mat c1 = arma::diagmat(arma::vectorise(x.IgCL2N));
    arma::mat diagL = arma::diagmat(arma::vectorise(x.L));
    arma::mat diag2L = 2*diagL;
    arma::mat c2 = diagL * arma::kron(x.N, x.I_gamma_C) * diag2L;
    x.hL = c1 + c2;

  }

  void dgLP(arguments& x) {

    x.dgL = x.dL % x.IgCL2N + x.L % (x.I_gamma_C * (2*x.dL % x.L) * x.N);

  }

};

/*
 * Geomin
 */

class geomin: public base_criterion {

public:

  void F(arguments& x) {

    x.L2 = x.L % x.L;
    x.L2 += x.epsilon;
    x.term = arma::exp(arma::sum(arma::log(x.L2), 1) / x.q);

    x.f = arma::accu(x.term);

  }

  void gLP(arguments& x) {

    x.LoL2 = x.L / x.L2;
    x.gL = x.LoL2 * x.q2;
    x.gL.each_col() %= x.term;

  }

  void hLP(arguments& x) {

    arma::mat cx = x.q2 * x.LoL2;

    arma::mat c1 = x.q2*(arma::vectorise(x.L2) - arma::vectorise(2*x.L % x.L)) /
      arma::vectorise(x.L2 % x.L2);
    arma::mat gcx = arma::diagmat(c1);
    arma::mat c2 = (1/x.L2) % (2*x.L) / x.q;
    c2.each_col() %= x.term;
    arma::mat gterm = cbind_diag(c2);
    arma::mat v = gterm.t() * cx;
    x.hL = gcx;
    arma::mat term2 = x.term;
    for(int i=0; i < (x.q-1); ++i) term2 = arma::join_cols(term2, x.term);
    x.hL.each_col() %= term2;
    x.hL += kdiag(v);

  }

  void dgLP(arguments& x) {

    arma::mat c1 = (x.epsilon - x.L % x.L) / (x.L2 % x.L2) % x.dL;
    c1.each_col() %= x.term;
    arma::mat c2 = x.LoL2;
    arma::vec term2 = x.q2 * x.term % arma::sum(x.LoL2 % x.dL, 1);
    c2.each_col() %= term2;

    x.dgL = x.q2 * (c1 + c2);

  }

};

/*
 * Target
 */

class target: public base_criterion {

public:
  void F(arguments& x) {

    x.f1 = x.Weight % (x.L - x.Target);

    x.f = 0.5*arma::accu(x.f1 % x.f1);

  }

  void gLP(arguments& x) {

    x.gL = x.Weight % x.f1;

  }

  void hLP(arguments& x) {

    arma::mat W2 = x.Weight % x.Weight;
    x.hL = arma::diagmat(arma::vectorise(W2));

  }

  void dgLP(arguments& x) {

    x.dgL = x.Weight2 % x.dL;

  }

};

/*
 * xTarget
 */

class xtarget: public base_criterion {

public:

  void F(arguments& x) {

    x.f1 = x.Weight % (x.L - x.Target);
    x.f2 = x.Phi_Weight % (x.Phi - x.Phi_Target);

    x.f = 0.5*arma::accu(x.f1 % x.f1) + 0.25*x.w*arma::accu(x.f2 % x.f2);

  }

  void gLP(arguments& x) {

    x.gL = x.Weight % x.f1;
    x.gP = x.w * x.Phi_Weight % x.f2;

  }

  void hLP(arguments& x) {

    arma::mat W2 = x.Weight % x.Weight;
    x.hL = arma::diagmat(arma::vectorise(W2));

    arma::mat Phi_t = dxt(x.Phi);
    arma::mat diag_PW2 = arma::diagmat(arma::vectorise(x.Phi_Weight % x.Phi_Weight));
    x.hP = diag_PW2 + diag_PW2 * Phi_t; // w¿?

  }

  void dgLP(arguments& x) {

    x.dgL = x.Weight2 % x.dL;
    x.dgP = x.w * x.Phi_Weight2 % x.dP;

  }

};

/*
 * Repeated Crawford-Ferguson family
 */

class rep_cf: public base_criterion {

public:

  void F(arguments& x){

    arma::uvec indexes = x.blocks_list[x.i];
    x.Li[x.i] = x.L.cols(indexes);
    x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
    x.Ni[x.i] = x.N(indexes, indexes);
    double ff1 = (1-x.k) * arma::accu(x.Li2[x.i] % (x.Li2[x.i] * x.Ni[x.i])) / 4;
    double ff2 = x.k * arma::accu(x.Li2[x.i] % (x.M * x.Li2[x.i])) / 4;

    x.f += (ff1 + ff2) * x.block_weights[x.i];

  }

  void gLP(arguments& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    arma::mat f1 = (1-x.k) * x.Li[x.i] % (x.Li2[x.i] * x.Ni[x.i]);
    arma::mat f2 = x.k * x.Li[x.i] % (x.M * x.Li2[x.i]);
    x.gL.cols(indexes) += (f1 + f2) * x.block_weights[x.i];

  }

  void hLP(arguments& x) {

    arma::uvec indexes;
    arma::uvec block_indexes = x.blocks_list[x.i];
    int n_blocks = block_indexes.size();
    for(int j=0; j < n_blocks; ++j) {

      int end = block_indexes[j] * x.p + x.p - 1;
      int start = end - x.p + 1;
      arma::uvec add = consecutive(start, end+1);
      indexes = arma::join_cols(indexes, add);

    }

    arma::mat Ip(x.p, x.p, arma::fill::eye);
    arma::mat c1 = arma::kron(x.Ni[x.i].t(), Ip) * arma::diagmat(arma::vectorise(2*x.Li[x.i]));
    arma::mat gf1 = (1-x.k)*arma::diagmat(arma::vectorise(x.Li[x.i])) * c1 +
      arma::diagmat(arma::vectorise(x.L2[x.i] * x.Ni[x.i]));

    arma::mat Iq(x.q, x.q, arma::fill::eye);
    arma::mat c2 = arma::kron(Iq, x.M) * arma::diagmat(arma::vectorise(2*x.Li[x.i]));
    arma::mat gf2 = x.k*arma::diagmat(arma::vectorise(x.Li[x.i])) * c2 +
      arma::diagmat(arma::vectorise(x.M * x.L2[x.i]));
    x.hL(indexes, indexes) += (gf1 + gf2) * x.block_weights[x.i];

  }

  void dgLP(arguments& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    arma::mat dLi = x.dL.cols(indexes);
    arma::mat dLi2 = 2 * dLi % x.Li[x.i];
    arma::mat df1 = (1-x.k) * dLi % (x.Li2[x.i] * x.Ni[x.i]) +
      (1-x.k) * x.Li[x.i] % (dLi2 * x.Ni[x.i]);
    arma::mat df2 = x.k * dLi % (x.M * x.Li2[x.i]) + x.k * x.Li[x.i] % (x.M * dLi2);

    x.dgL.cols(indexes) += (df1 + df2) * x.block_weights[x.i];

  }

};

/*
 * Repeated Varimax
 */

class rep_varimax: public base_criterion {

public:

  void F(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.Li[x.i] = x.L.cols(indexes);
      x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
      x.HLi2[x.i] = x.H * x.Li2[x.i];

      x.f -= trace(x.HLi2[x.i].t() * x.HLi2[x.i]) / 4 * x.block_weights[x.i];

  }

  void gLP(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.gL.cols(indexes) -= (x.Li[x.i] % x.HLi2[x.i]) * x.block_weights[x.i];

  }

  void hLP(arguments& x) {

    arma::uvec indexes;
    arma::uvec block_indexes = x.blocks_list[x.i];
    int n_blocks = block_indexes.size();
    for(int j=0; j < n_blocks; ++j) {

      int end = block_indexes[j] * x.p + x.p - 1;
      int start = end - x.p + 1;
      arma::uvec add = consecutive(start, end+1);
      indexes = arma::join_cols(indexes, add);

    }

      arma::mat c1 = arma::diagmat(arma::vectorise(x.HLi2[x.i]));
      arma::mat diagL = arma::diagmat(arma::vectorise(x.H * x.Li[x.i]));
      arma::mat diag2L = 2*diagL;
      arma::mat c2 = diagL * diag2L;
      x.hL(indexes, indexes) -= (c1 + c2) * x.block_weights[x.i];

  }

  void dgLP(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      arma::mat dLi = x.dL.cols(indexes);
      arma::mat dLi2 = 2 * dLi % x.Li[x.i];
      x.dgL.cols(indexes) -= (dLi % x.HLi2[x.i] - x.Li[x.i] % (x.H * dLi2)) * x.block_weights[x.i];

  }

};

/*
 * Repeated Varimin
 */

class rep_varimin: public base_criterion {

public:

  void F(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.Li[x.i] = x.L.cols(indexes);
      x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
      x.HLi2[x.i] = x.H * x.Li2[x.i];

      x.f += trace(x.HLi2[x.i].t() * x.HLi2[x.i]) / 4 * x.block_weights[x.i];

  }

  void gLP(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.Li[x.i] = x.L.cols(indexes);
      x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
      x.HLi2[x.i] = x.H * x.Li2[x.i];
      x.gL.cols(indexes) += (x.Li[x.i] % x.HLi2[x.i]) * x.block_weights[x.i];

  }

  void hLP(arguments& x) {

    arma::uvec indexes;
    arma::uvec block_indexes = x.blocks_list[x.i];
    int n_blocks = block_indexes.size();
    for(int j=0; j < n_blocks; ++j) {

      int end = block_indexes[j] * x.p + x.p - 1;
      int start = end - x.p + 1;
      arma::uvec add = consecutive(start, end+1);
      indexes = arma::join_cols(indexes, add);

    }

      arma::mat c1 = arma::diagmat(arma::vectorise(x.HLi2[x.i]));
      arma::mat diagL = arma::diagmat(arma::vectorise(x.H * x.Li[x.i]));
      arma::mat diag2L = 2*diagL;
      arma::mat c2 = diagL * diag2L;
      x.hL(indexes, indexes) += (c1 + c2) * x.block_weights[x.i];

  }

  void dgLP(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      arma::mat dLi = x.dL.cols(indexes);
      arma::mat dLi2 = 2 * dLi % x.Li[x.i];
      x.dgL.cols(indexes) -= (dLi % x.HLi2[x.i] - x.Li[x.i] % (x.H * dLi2)) * x.block_weights[x.i];

  }

};

/*
 * Repeated Oblimin
 */

class rep_oblimin: public base_criterion {

public:

  void F(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.Li[x.i] = x.L.cols(indexes);
      x.Ni[x.i] = x.N(indexes, indexes);
      x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
      x.IgCL2Ni[x.i] = x.I_gamma_C * x.Li2[x.i] * x.Ni[x.i];

      x.f += trace(x.Li2[x.i].t() * x.IgCL2Ni[x.i]) / 4 * x.block_weights[x.i];

  }

  void gLP(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.gL.cols(indexes) += (x.Li[x.i] % x.IgCL2Ni[x.i]) * x.block_weights[x.i];

  }

  void hLP(arguments& x) {

    arma::uvec indexes;
    arma::uvec block_indexes = x.blocks_list[x.i];
    int n_blocks = block_indexes.size();
    for(int j=0; j < n_blocks; ++j) {

      int end = block_indexes[j] * x.p + x.p - 1;
      int start = end - x.p + 1;
      arma::uvec add = consecutive(start, end+1);
      indexes = arma::join_cols(indexes, add);

    }

      arma::mat c1 = arma::diagmat(arma::vectorise(x.IgCL2Ni[x.i]));
      arma::mat diagL = arma::diagmat(arma::vectorise(x.Li[x.i]));
      arma::mat diag2L = 2*diagL;
      arma::mat c2 = diagL * arma::kron(x.Ni[x.i], x.I_gamma_C) * diag2L;
      x.hL(indexes, indexes) += (c1 + c2) * x.block_weights[x.i];

  }

  void dgLP(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      arma::mat dLi = x.dL.cols(indexes);
      x.dgL.cols(indexes) += (dLi % x.IgCL2Ni[x.i] +
        x.Li[x.i] % (x.I_gamma_C * (2*dLi % x.Li[x.i]) * x.Ni[x.i])) * x.block_weights[x.i];

  }

};

/*
 * Repeated Geomin
 */

class rep_geomin: public base_criterion {

public:

  void F(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.Li[x.i] = x.L.cols(indexes);
      x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
      x.Li2[x.i] += x.epsilon;
      x.termi[x.i] = arma::exp(arma::sum(arma::log(x.Li2[x.i]), 1) / x.q);

      x.f += arma::accu(x.termi[x.i]) * x.block_weights[x.i];

  }

  void gLP(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.LoLi2[x.i] = x.Li[x.i] / x.Li2[x.i];
      x.termi[x.i] = arma::exp(arma::sum(arma::log(x.Li2[x.i]), 1) / x.q);
      arma::mat gLi = x.LoLi2[x.i] * x.q2;
      gLi.each_col() %= x.termi[x.i];
      x.gL.cols(indexes) += gLi * x.block_weights[x.i];

  }

  void hLP(arguments& x) {

    arma::uvec indexes;
    arma::uvec block_indexes = x.blocks_list[x.i];
    int n_blocks = block_indexes.size();
    for(int j=0; j < n_blocks; ++j) {

      int end = block_indexes[j] * x.p + x.p - 1;
      int start = end - x.p + 1;
      arma::uvec add = consecutive(start, end+1);
      indexes = arma::join_cols(indexes, add);

    }

      arma::mat cx = x.q2 * x.LoLi2[x.i];

      arma::mat c1 = x.q2*(arma::vectorise(x.Li2[x.i]) - arma::vectorise(2*x.Li[x.i] % x.Li[x.i])) /
        arma::vectorise(x.Li2[x.i] % x.Li2[x.i]);
      arma::mat gcx = arma::diagmat(c1);
      arma::mat c2 = (1/x.Li2[x.i]) % (2*x.Li[x.i]) / x.q;
      c2.each_col() %= x.termi[x.i];
      arma::mat gterm = cbind_diag(c2);
      arma::mat v = gterm.t() * cx;
      x.hL(indexes, indexes) = gcx;
      arma::mat term2 = x.termi[x.i];
      for(int i=0; i < (x.q-1); ++i) term2 = arma::join_cols(term2, x.termi[x.i]);
      arma::mat hL = x.hL(indexes, indexes);
      hL.each_col() %= term2;
      hL += kdiag(v);
      x.hL(indexes, indexes) += hL * x.block_weights[x.i];

  }

  void dgLP(arguments& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      arma::mat dLi = x.dL.cols(indexes);
      arma::mat c1 = (x.epsilon - x.Li[x.i] % x.Li[x.i]) / (x.Li2[x.i] % x.Li2[x.i]) % dLi;
      c1.each_col() %= x.termi[x.i];
      arma::mat c2 = x.LoLi2[x.i];
      arma::vec termi2 = x.q2 * x.termi[x.i] % arma::sum(x.LoLi2[x.i] % dLi, 1);
      c2.each_col() %= termi2;

      x.dgL.cols(indexes) += x.q2 * (c1 + c2) * x.block_weights[x.i];

  }

};

/*
 * Tian & Liu penalization
 */

class TL: public base_criterion {

public:

  void F(arguments& x) {

    // Penalization
    x.Lg = x.L.cols(x.blocks_list[x.i]);
    x.Ls = x.L.cols(x.blocks_list[x.i + 1]);
    x.L2 = x.L % x.L;
    x.L2g = x.L2.cols(x.blocks_list[x.i]);
    x.L2s = x.L2.cols(x.blocks_list[x.i + 1]);
    x.Ng = bc(x.Lg.n_cols);

    x.exp_aL2g = exp(-x.a*x.L2g);
    x.C = x.L2s.t() * x.exp_aL2g;
    x.logC = log(x.C);
    x.logCN = x.logC * x.Ng;
    x.exp_lCN = exp(x.logCN);
    x.f += arma::accu(x.exp_lCN);

  }

  void gLP(arguments& x) {

    arma::uvec indexes1 = x.blocks_list[x.i];
    arma::uvec indexes2 = x.blocks_list[x.i + 1];
    arma::uvec indexes = arma::join_cols(indexes1, indexes2);
    int q1 = indexes1.size();
    int q2 = indexes2.size();
    int q = indexes.size();
    arma::mat I1(q1, q1, arma::fill::eye);
    arma::mat I2(q2, q2, arma::fill::eye);
    x.I1 = I1;
    x.I2 = I2;

    // Penalization
    x.gL2g = 2*x.Lg;
    x.gL2s = 2*x.Ls;
    x.g_exp_aL2g = -x.a*x.exp_aL2g % x.gL2g;
    x.dxt_L2s = dxt(x.L2s);
    x.gC1 = arma::join_rows(arma::kron(x.I1, x.L2s.t()),
                            arma::kron(x.exp_aL2g.t(), x.I2) * x.dxt_L2s);
    x.gC = x.gC1.t();
    x.gC.each_col() %= arma::vectorise(arma::join_rows(x.g_exp_aL2g, x.gL2s));
    x.glogC = x.gC.t();
    x.glogC.each_col() %= arma::vectorise(1/x.C);
    arma::cube cube_gexplogCN(x.glogC.memptr(), x.C.n_rows, x.C.n_cols, x.p*q);
    cube_gexplogCN.each_slice() *= x.Ng;
    cube_gexplogCN.each_slice() %= x.exp_lCN;
    arma::mat mat_gexplogCN = arma::sum(cube_gexplogCN, 0);
    arma::rowvec vec_gexplogCN = arma::sum(mat_gexplogCN, 0);

    x.gL.cols(indexes) += arma::reshape(vec_gexplogCN, x.p, q);

  }

  void hLP(arguments& x) {

  }

  void dgLP(arguments& x) {

    arma::uvec indexes1 = x.blocks_list[x.i];
    arma::uvec indexes2 = x.blocks_list[x.i + 1];

    arma::uvec indexes = arma::join_cols(indexes1, indexes2);
    int q = indexes.size();

    // Penalization
    arma::mat dLg = x.dL.cols(x.blocks_list[x.i]);
    arma::mat dLs = x.dL.cols(x.blocks_list[x.i + 1]);
    arma::mat dL2 = 2*x.dL % x.L;
    arma::mat dL2g = dL2.cols(x.blocks_list[x.i]);
    arma::mat dL2s = dL2.cols(x.blocks_list[x.i + 1]);
    arma::mat dexp_aL2g = -x.a*x.exp_aL2g % dL2g;
    arma::mat dgL2g = 2*dLg;
    arma::mat dgL2s = 2*dLs;
    arma::mat dg_exp_aL2g = -x.a*dexp_aL2g % x.gL2g + -x.a*x.exp_aL2g % dgL2g;
    arma::mat dC = dL2s.t() * x.exp_aL2g + x.L2s.t() * dexp_aL2g;
    arma::mat dgC1 = arma::join_rows(arma::kron(x.I1, dL2s.t()),
                                     arma::kron(dexp_aL2g.t(), x.I2) * x.dxt_L2s);
    arma::mat dgC11 = dgC1.t();
    dgC11.each_col() %= arma::vectorise(arma::join_rows(x.g_exp_aL2g, x.gL2s));
    arma::mat dgC12 = x.gC1.t();
    dgC12.each_col() %= arma::vectorise(arma::join_rows(dg_exp_aL2g, dgL2s));
    arma::mat dgC = dgC11 + dgC12;

    arma::mat dglogC11 = dgC.t();
    dglogC11.each_col() %= arma::vectorise(1/x.C);
    arma::mat dglogC12 = x.gC.t();
    dglogC12.each_col() %= arma::vectorise(-dC/(x.C%x.C));
    arma::mat dglogC = dglogC11 + dglogC12;
    arma::cube cube_dglogCN11(dglogC.memptr(), x.C.n_rows, x.C.n_cols, x.p*q);
    cube_dglogCN11.each_slice() *= x.Ng;
    cube_dglogCN11.each_slice() %= x.exp_lCN;

    arma::mat dlogCN = (dC/x.C) * x.Ng;
    arma::mat dexp_lCN = dlogCN % x.exp_lCN;
    arma::cube cube_dglogCN12(x.glogC.memptr(), x.C.n_rows, x.C.n_cols, x.p*q);
    cube_dglogCN12.each_slice() *= x.Ng;
    cube_dglogCN12.each_slice() %= dexp_lCN;
    arma::cube cube_dgexplogCN = cube_dglogCN11 + cube_dglogCN12;

    arma::mat mat_dgexplogCN = arma::sum(cube_dgexplogCN, 0);
    arma::rowvec vec_dgexplogCN = arma::sum(mat_dgexplogCN, 0);

    x.dgL.cols(indexes) += arma::reshape(vec_dgexplogCN, x.p, q);

  }

};

/*
 * Tian & Liu modified penalization
 */

class TLM: public base_criterion {

public:

  void F(arguments& x) {

    // Penalization
    x.Lg = x.L.cols(x.blocks_list[x.i]);
    x.Ls = x.L.cols(x.blocks_list[x.i + 1]);
    x.L2 = x.L % x.L;
    x.L2g = x.L2.cols(x.blocks_list[x.i]);
    x.L2s = x.L2.cols(x.blocks_list[x.i + 1]);
    x.Ng = bc(x.Lg.n_cols);

    x.C = x.L2s.t() * x.L2g;
    x.logC = log(x.C);
    x.logCN = x.logC * x.Ng;
    x.exp_lCN = exp(x.logCN);
    x.f += arma::accu(x.exp_lCN);

  }

  void gLP(arguments& x) {

    arma::uvec indexes1 = x.blocks_list[x.i];
    arma::uvec indexes2 = x.blocks_list[x.i + 1];
    arma::uvec indexes = arma::join_cols(indexes1, indexes2);
    int q1 = indexes1.size();
    int q2 = indexes2.size();
    int q = indexes.size();
    arma::mat I1(q1, q1, arma::fill::eye);
    arma::mat I2(q2, q2, arma::fill::eye);
    x.I1 = I1;
    x.I2 = I2;

    // Penalization
    x.gL2g = 2*x.Lg;
    x.gL2s = 2*x.Ls;
    x.dxt_L2s = dxt(x.L2s);
    x.gC1 = arma::join_rows(arma::kron(x.I1, x.L2s.t()),
                            arma::kron(x.L2g.t(), x.I2) * x.dxt_L2s);
    x.gC = x.gC1.t();
    x.gC.each_col() %= arma::vectorise(arma::join_rows(x.gL2g, x.gL2s));
    x.glogC = x.gC.t();
    x.glogC.each_col() %= arma::vectorise(1/x.C);
    arma::cube cube_gexplogCN(x.glogC.memptr(), x.C.n_rows, x.C.n_cols, x.p*q);
    cube_gexplogCN.each_slice() *= x.Ng;
    cube_gexplogCN.each_slice() %= x.exp_lCN;
    arma::mat mat_gexplogCN = arma::sum(cube_gexplogCN, 0);
    arma::rowvec vec_gexplogCN = arma::sum(mat_gexplogCN, 0);

    x.gL.cols(indexes) += arma::reshape(vec_gexplogCN, x.p, q);

  }

  void hLP(arguments& x) {

  }

  void dgLP(arguments& x) {

    arma::uvec indexes1 = x.blocks_list[x.i];
    arma::uvec indexes2 = x.blocks_list[x.i + 1];

    arma::uvec indexes = arma::join_cols(indexes1, indexes2);
    int q = indexes.size();

    // Penalization
    arma::mat dLg = x.dL.cols(x.blocks_list[x.i]);
    arma::mat dLs = x.dL.cols(x.blocks_list[x.i + 1]);
    arma::mat dL2 = 2*x.dL % x.L;
    arma::mat dL2g = dL2.cols(x.blocks_list[x.i]);
    arma::mat dL2s = dL2.cols(x.blocks_list[x.i + 1]);
    arma::mat dgL2g = 2*dLg;
    arma::mat dgL2s = 2*dLs;
    arma::mat dC = dL2s.t() * x.L2g + x.L2s.t() * dL2g;
    arma::mat dgC1 = arma::join_rows(arma::kron(x.I1, dL2s.t()),
                                     arma::kron(dL2g.t(), x.I2) * x.dxt_L2s);
    arma::mat dgC11 = dgC1.t();
    dgC11.each_col() %= arma::vectorise(arma::join_rows(x.gL2g, x.gL2s));
    arma::mat dgC12 = x.gC1.t();
    dgC12.each_col() %= arma::vectorise(arma::join_rows(dgL2g, dgL2s));
    arma::mat dgC = dgC11 + dgC12;

    arma::mat dglogC11 = dgC.t();
    dglogC11.each_col() %= arma::vectorise(1/x.C);
    arma::mat dglogC12 = x.gC.t();
    dglogC12.each_col() %= arma::vectorise(-dC/(x.C%x.C));
    arma::mat dglogC = dglogC11 + dglogC12;
    arma::cube cube_dglogCN11(dglogC.memptr(), x.C.n_rows, x.C.n_cols, x.p*q);
    cube_dglogCN11.each_slice() *= x.Ng;
    cube_dglogCN11.each_slice() %= x.exp_lCN;

    arma::mat dlogCN = (dC/x.C) * x.Ng;
    arma::mat dexp_lCN = dlogCN % x.exp_lCN;
    arma::cube cube_dglogCN12(x.glogC.memptr(), x.C.n_rows, x.C.n_cols, x.p*q);
    cube_dglogCN12.each_slice() *= x.Ng;
    cube_dglogCN12.each_slice() %= dexp_lCN;
    arma::cube cube_dgexplogCN = cube_dglogCN11 + cube_dglogCN12;

    arma::mat mat_dgexplogCN = arma::sum(cube_dgexplogCN, 0);
    arma::rowvec vec_dgexplogCN = arma::sum(mat_dgexplogCN, 0);

    x.dgL.cols(indexes) += arma::reshape(vec_dgexplogCN, x.p, q);

  }

};

/*
 * Mixed criteria
 */

// Choose rotation criteria for mixed criteria:

base_criterion* choose_rep_criterion(std::string rotation, std::string projection) {

  base_criterion* criterion;

  if (rotation == "target") {

    criterion = new target();

  }  else if(rotation == "xtarget") {

    if(projection == "orth") {
      criterion = new target();
    } else {
      criterion = new xtarget();
    }

  } else if(rotation == "cf") {

    criterion = new rep_cf();

  } else if(rotation == "oblimin") {

    criterion = new rep_oblimin();

  } else if(rotation == "geomin") {

    criterion = new rep_geomin();

  } else if(rotation == "varimax") {

    criterion = new rep_varimax();

  } else if(rotation == "varimin") {

    criterion = new rep_varimin();

  } else if(rotation == "none") {

  } else {

    Rcpp::stop("Available rotations: \n cf, oblimin, geomin, varimax, varimin, target, xtarget");

  }

  return criterion;

}
base_criterion* choose_penalization(std::string rotation) {

  base_criterion* criterion;

  if(rotation == "TL") {

    criterion = new TL();

  } else if(rotation == "TLM") {

    criterion = new TLM();

  } else {

    Rcpp::stop("Available penalizations: TL and TLM");

  }

  return criterion;

}

class mixed: public base_criterion {

public:

  void F(arguments& x) {

    base_criterion* criterion;
    x.f = 0;

    for(int i=0; i < x.n_blocks; ++i) {

      x.i = i;
      criterion = choose_rep_criterion(x.rotations[i], x.projection);
      criterion->F(x);

    }

    if(x.penalize) {

      criterion = choose_penalization(x.penalization);

      for(int i=0; i < (x.n_blocks-1); ++i) {

        x.i = i;
        criterion->F(x);

      }

    }

  }

  void gLP(arguments& x) {

    base_criterion* criterion;
    x.gL.set_size(arma::size(x.L));
    x.gL.zeros();
    x.gP.set_size(x.q, x.q);
    x.gP.zeros();

    for(int i=0; i < x.n_blocks; ++i) {

      x.i = i;
      criterion = choose_rep_criterion(x.rotations[i], x.projection);
      criterion->gLP(x);

    }

    if(x.penalize) {

      criterion = choose_penalization(x.penalization);

      for(int i=0; i < (x.n_blocks-1); ++i) {

        x.i = i;
        criterion->gLP(x);

      }

    }

  }

  void hLP(arguments& x) {

    base_criterion* criterion;
    int pq = x.p*x.q;
    int qq = x.q*x.q;
    x.hL.set_size(pq, pq);
    x.hL.zeros();
    x.hP.set_size(qq, qq);
    x.hP.zeros();

    for(int i=0; i < x.n_blocks; ++i) {

      x.i = i;
      criterion = choose_rep_criterion(x.rotations[i], x.projection);
      criterion->hLP(x);

    }

    if(x.penalize) {

      criterion = choose_penalization(x.penalization);

      for(int i=0; i < (x.n_blocks-1); ++i) {

        x.i = i;
        criterion->hLP(x);

      }

    }

  }

  void dgLP(arguments& x) {

    base_criterion* criterion;
    x.dgL.set_size(arma::size(x.L));
    x.dgL.zeros();
    x.dgP.set_size(x.q, x.q);
    x.dgP.zeros();

    for(int i=0; i < x.n_blocks; ++i) {

      x.i = i;
      criterion = choose_rep_criterion(x.rotations[i], x.projection);
      criterion->dgLP(x);

    }

    if(x.penalize) {

      criterion = choose_penalization(x.penalization);

      for(int i=0; i < (x.n_blocks-1); ++i) {

        x.i = i;
        criterion->dgLP(x);

      }

    }

  }

};
