/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

// #include "auxiliary_criteria.h"

// Criteria

class base_criterion {

public:

  virtual void F(arguments_rotate& x) = 0;

  virtual void gLP(arguments_rotate& x) = 0;

  virtual void hLP(arguments_rotate& x) = 0;

  virtual void dgLP(arguments_rotate& x) = 0;

};

/*
 * none
 */

class none: public base_criterion {

public:

  void F(arguments_rotate& x) {}

  void gLP(arguments_rotate& x) {}

  void hLP(arguments_rotate& x) {}

  void dgLP(arguments_rotate& x) {}

};

/*
 * Crawford-Ferguson family
 */

class cf: public base_criterion {

public:

  void F(arguments_rotate& x){

    x.L2 = x.L % x.L;
    x.L2N = x.L2 * x.N;
    x.ML2 = x.M * x.L2;
    double ff1 = (1-x.k[0]) * arma::accu(x.L2 % x.L2N) / 4;
    double ff2 = x.k[0] * arma::accu(x.L2 % x.ML2) / 4;

    x.f = ff1 + ff2;

  }

  void gLP(arguments_rotate& x){

    x.f1 = (1-x.k[0]) * x.L % x.L2N;
    x.f2 = x.k[0] * x.L % x.ML2;
    x.gL = x.f1 + x.f2;

  }

  void hLP(arguments_rotate& x) {

    arma::colvec L_vector = arma::vectorise(x.L);

    arma::mat Ip(x.p, x.p, arma::fill::eye);
    arma::mat c1 = arma::kron(x.N.t(), Ip) * arma::diagmat(2*L_vector);
    c1.each_col() %= L_vector;
    arma::mat c2 = arma::diagmat(arma::vectorise(x.L2 * x.N));
    arma::mat gf1 = (1-x.k[0]) * (c1 + c2);

    arma::mat Iq(x.q, x.q, arma::fill::eye);
    arma::mat c3 = arma::kron(Iq, x.M) * arma::diagmat(2*L_vector);
    c3.each_col() %= L_vector;
    arma::mat c4 = arma::diagmat(arma::vectorise(x.M * x.L2));
    arma::mat gf2 = x.k[0] * (c3 + c4);
    x.hL = gf1 + gf2;

  }

  void dgLP(arguments_rotate& x) {

    arma::mat dL2 = 2 * x.dL % x.L;
    arma::mat df1 = (1-x.k[0]) * x.dL % x.L2N + (1-x.k[0]) * x.L % (dL2 * x.N);
    arma::mat df2 = x.k[0] * x.dL % x.ML2 + x.k[0] * x.L % (x.M * dL2);

    x.dgL = df1 + df2;

  }

};

/*
 * Varimax
 */

class varimax: public base_criterion {

public:

  void F(arguments_rotate& x) {

    x.L2 = x.L % x.L;
    x.HL2 = x.H * x.L2;

    x.f = -arma::trace(x.HL2.t() * x.HL2) / 4;

  }

  void gLP(arguments_rotate& x) {

    x.gL = -x.L % x.HL2;

  }

  void hLP(arguments_rotate& x) {

    arma::mat Iq(x.q, x.q, arma::fill::eye);
    arma::mat c1 = arma::diagmat(arma::vectorise(x.HL2));
    arma::colvec L_vector = arma::vectorise(x.L);
    arma::mat c2 = arma::kron(Iq, x.H) * arma::diagmat(2*L_vector);
    c2.each_col() %= L_vector;
    x.hL = -(c1 + c2);

  }

  void dgLP(arguments_rotate& x) {

    arma::mat dL2 = 2 * x.dL % x.L;
    x.dgL = -x.dL % x.HL2 - x.L % (x.H * dL2);

  }

};

/*
 * Varimin
 */

class varimin: public base_criterion {

public:

  void F(arguments_rotate& x) {

    x.L2 = x.L % x.L;
    x.HL2 = x.H * x.L2;

    x.f = arma::trace(x.HL2.t() * x.HL2) / 4;

  }

  void gLP(arguments_rotate& x) {

    x.gL = x.L % x.HL2;

  }

  void hLP(arguments_rotate& x) {

    arma::mat Iq(x.q, x.q, arma::fill::eye);
    arma::mat c1 = arma::diagmat(arma::vectorise(x.HL2));
    arma::colvec L_vector = arma::vectorise(x.L);
    arma::mat c2 = arma::kron(Iq, x.H) * arma::diagmat(2*L_vector);
    c2.each_col() %= L_vector;
    x.hL = c1 + c2;

  }

  void dgLP(arguments_rotate& x) {

    arma::mat dL2 = 2 * x.dL % x.L;
    x.dgL = x.dL % x.HL2 + x.L % (x.H * dL2);

  }

};

/*
 * Oblimin
 */

class oblimin: public base_criterion {

public:

  void F(arguments_rotate& x) {

    x.L2 = x.L % x.L;
    x.IgCL2N = x.I_gamma_C * x.L2 * x.N;

    x.f = arma::accu(x.L2 % x.IgCL2N) / 4;

  }

  void gLP(arguments_rotate& x) {

    x.gL = x.L % x.IgCL2N;

  }

  void hLP(arguments_rotate& x) {

    arma::mat c1 = arma::diagmat(arma::vectorise(x.IgCL2N));
    arma::mat diagL = arma::diagmat(arma::vectorise(x.L));
    arma::mat diag2L = 2*diagL;
    arma::mat c2 = diagL * arma::kron(x.N, x.I_gamma_C) * diag2L;
    x.hL = c1 + c2;

  }

  void dgLP(arguments_rotate& x) {

    x.dgL = x.dL % x.IgCL2N + x.L % (x.I_gamma_C * (2*x.dL % x.L) * x.N);

  }

};

/*
 * Geomin
 */

class geomin: public base_criterion {

public:

  void F(arguments_rotate& x) {

    x.q2 = 2/(x.q + 0.0);
    x.L2 = x.L % x.L;
    x.L2 += x.epsilon[0];
    x.term = arma::trunc_exp(arma::sum(arma::trunc_log(x.L2), 1) / x.q);

    x.f = arma::accu(x.term);

  }

  void gLP(arguments_rotate& x) {

    x.LoL2 = x.L / x.L2;
    x.gL = x.LoL2 * x.q2;
    x.gL.each_col() %= x.term;

  }

  void hLP(arguments_rotate& x) {

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

  void dgLP(arguments_rotate& x) {

    arma::mat c1 = (x.epsilon[0] - x.L % x.L) / (x.L2 % x.L2) % x.dL;
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
  void F(arguments_rotate& x) {

    x.f1 = x.Weight % (x.L - x.Target);

    x.f = 0.5*arma::accu(x.f1 % x.f1);

  }

  void gLP(arguments_rotate& x) {

    x.gL = x.Weight % x.f1;

  }

  void hLP(arguments_rotate& x) {

    arma::mat W2 = x.Weight % x.Weight;
    x.hL = arma::diagmat(arma::vectorise(W2));

  }

  void dgLP(arguments_rotate& x) {

    x.dgL = x.Weight2 % x.dL;

  }

};

/*
 * xTarget
 */

class xtarget: public base_criterion {

public:

  void F(arguments_rotate& x) {

    x.f1 = x.Weight % (x.L - x.Target);
    x.f2 = x.Phi_Weight % (x.Phi - x.Phi_Target);

    x.f = 0.5*arma::accu(x.f1 % x.f1) + 0.25*x.w*arma::accu(x.f2 % x.f2);

  }

  void gLP(arguments_rotate& x) {

    x.gL = x.Weight % x.f1;
    x.gP = x.w * x.Phi_Weight % x.f2;

  }

  void hLP(arguments_rotate& x) {

    arma::mat W2 = x.Weight % x.Weight;
    x.hL = arma::diagmat(arma::vectorise(W2));

    arma::mat wPW2 = x.w * x.Phi_Weight % x.Phi_Weight;
    arma::uvec lower_indices = arma::trimatl_ind(arma::size(x.Phi), -1);
    x.hP = arma::diagmat(wPW2(lower_indices));

  }

  void dgLP(arguments_rotate& x) {

    x.dgL = x.Weight2 % x.dL;
    x.dgP = x.w * x.Phi_Weight2 % x.dP;

  }

};

/*
 * equavar (orthogonal)
 */

class equavar: public base_criterion {

public:

  void F(arguments_rotate& x) {

    double q = x.q + 0.0;
    x.var = arma::diagvec(x.L.t() * x.L);
    x.varq = arma::pow(x.var, 1/q);
    x.prodvarq = arma::prod(x.varq);

    x.f = x.prodvarq * -1;

  }

  void gLP(arguments_rotate& x) {

    double q = x.q + 0.0;
    x.dvarqdL = 2*x.L;
    x.dvarqdL.each_row() %= arma::pow(x.var, 1/q-1)/q;
    x.gL = x.dvarqdL;
    x.gL.each_row() %= (1/x.varq);
    x.gL *= x.prodvarq * -1;

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for equavar");

  }

  void dgLP(arguments_rotate& x) {

    double q = x.q + 0.0;
    arma::mat dvardL = 2*x.L;
    arma::mat DdvardL = 2*x.dL;
    arma::rowvec Dvar = arma::diagvec(x.dL.t() * x.L + x.L.t() * x.dL);
    arma::rowvec Dvarq = arma::pow(x.var, 1/q-1)/q % Dvar;
    arma::mat dvarqdL = dvardL;
    dvarqdL.each_row() %= arma::pow(x.var, 1/q-1)/q;
    arma::mat DdvarqdL1 = DdvardL;
    DdvarqdL1.each_row() %= arma::pow(x.var, 1/q-1)/q;
    arma::mat DdvarqdL2 = dvardL;
    DdvarqdL2.each_row() %= arma::pow(x.var, 1/q-2)/q * (1/q-1) % Dvar;
    arma::mat DdvarqdL = DdvarqdL1 + DdvarqdL2;
    double Dprodvarq = arma::accu(x.prodvarq / x.varq % Dvarq);

    arma::mat dgL1 = Dprodvarq * dvarqdL + x.prodvarq * DdvarqdL;
    dgL1.each_row() %= x.varq;
    arma::mat dgL2 = x.prodvarq * dvarqdL;
    dgL2.each_row() %= Dvarq;
    x.dgL = dgL1 - dgL2;
    x.dgL.each_row() /= (x.varq % x.varq) * -1;

  }

};

/*
 * simplix orthogonal
 */

class simplix_orth: public base_criterion {

public:

  void F(arguments_rotate& x) {

    x.L2 = x.L % x.L;
    arma::rowvec c1 = arma::sum(x.L2, 0);
    x.mu = x.a*(c1.t()-x.b);
    x.expmmu = arma::exp(-x.mu);
    x.c2 = 1/(1 + x.expmmu);
    x.prodc2 = arma::prod(x.c2);

    x.f = 1 - x.prodc2;

  }

  void gLP(arguments_rotate& x) {

    arma::mat dc1dL = 2*x.L;
    x.dmudL = x.a * dc1dL;
    x.dc2dmu = x.expmmu / arma::pow(1 + x.expmmu, 2);
    x.dc2dL = x.dmudL; x.dc2dL.each_row() %= x.dc2dmu.t();
    arma::mat dprodc2dL = x.prodc2 * x.dc2dL; dprodc2dL.each_row() /= x.c2.t();

    x.gL = -dprodc2dL;

  }

  void hLP(arguments_rotate& x) {

    Rcpp::warning("Standard errors not implement yet for simplix");

  }

  void dgLP(arguments_rotate& x) {

    arma::colvec expmu = arma::exp(x.mu);
    arma::mat Ddc1dL = 2*x.dL;
    arma::colvec Dc1 = arma::diagvec(x.dL.t() * x.L + x.L.t() * x.dL);
    arma::colvec Dmu = x.a * Dc1;
    arma::colvec Dc2 = x.dc2dmu % Dmu;
    double Dprodc2 = arma::accu(x.prodc2 * Dc2 / x.c2);
    arma::mat DdmudL = x.a * Ddc1dL;
    arma::colvec ddc2dmudmu = -expmu % (expmu - 1) / arma::pow(1 + expmu, 3);
    arma::colvec Ddc2dmu = ddc2dmudmu % Dmu;
    arma::mat Ddc2dL1 = x.dmudL; Ddc2dL1.each_row() %= Ddc2dmu.t();
    arma::mat Ddc2dL2 = DdmudL; Ddc2dL2.each_row() %= x.dc2dmu.t();
    arma::mat Ddc2dL = Ddc2dL1 + Ddc2dL2;
    arma::mat dgL1 = Dprodc2 * x.dc2dL + x.prodc2 * Ddc2dL;
    dgL1.each_row() %= x.c2.t();
    arma::mat dgL2 = x.prodc2 * x.dc2dL; dgL2.each_row() %= Dc2.t();
    arma::mat dgL = dgL1 - dgL2; dgL.each_row() /= (x.c2 % x.c2).t();

    x.dgL = -dgL;

  }

};

/*
 * simplix
 */

class simplix: public base_criterion {

public:

  void F(arguments_rotate& x) {

    x.L2 = x.L.t() * x.L;
    arma::colvec c1 = arma::diagvec(x.Phi * x.L2);
    x.mu = x.a*(c1-x.b);
    x.expmmu = arma::exp(-x.mu);
    x.c2 = 1/(1 + x.expmmu);
    x.prodc2 = arma::prod(x.c2);

    x.f = 1 - x.prodc2;

  }

  void gLP(arguments_rotate& x) {

    arma::mat dc1dL = x.H * (arma::kron(x.L.t(), x.Phi) * x.dxtL +
      arma::kron(x.I, x.Phi * x.L.t()));
    x.dmudL = x.a * dc1dL;
    x.dc2dmu = x.expmmu / arma::pow(1 + x.expmmu, 2);
    x.dc2dL = x.dmudL; x.dc2dL.each_col() %= x.dc2dmu;
    arma::mat dprodc2dL = x.prodc2 * x.dc2dL; dprodc2dL.each_col() /= x.c2;
    arma::mat gL = -arma::sum(dprodc2dL, 0);

    x.gL = arma::reshape(gL, x.p, x.q);

    x.LtLxI = arma::kron(x.L2, x.I);
    arma::mat dc1dP = x.H * (x.LtLxI + x.LtLxI * x.dxtP);
    x.dmudP = x.a * dc1dP;
    x.dc2dP = x.dmudP; x.dc2dP.each_col() %= x.dc2dmu;
    arma::mat dprodc2dP = x.prodc2 * x.dc2dP; dprodc2dP.each_col() /= x.c2;
    arma::mat gP = -arma::sum(dprodc2dP, 0);

    x.gP = arma::reshape(gP, x.q, x.q);
    x.gP.diag().zeros();

  }

  void hLP(arguments_rotate& x) {

    Rcpp::warning("Standard errors not implement yet for simplix");

  }

  void dgLP(arguments_rotate& x) {

    arma::colvec expmu = arma::exp(x.mu);
    arma::colvec c22 = x.c2 % x.c2;
    arma::colvec Dc1 = arma::diagvec(x.Phi * x.dL.t() * x.L +
      x.Phi * x.L.t() * x.dL);
    arma::colvec Dmu = x.a * Dc1;
    arma::mat Ddc1dL = x.H * (arma::kron(x.dL.t(), x.Phi) * x.dxtL +
      arma::kron(x.I, x.Phi * x.dL.t()));
    arma::colvec temp0 = -expmu % (expmu - 1) / arma::pow(1 + expmu, 3);
    arma::colvec Ddc2dmu = temp0 % Dmu;
    arma::mat DdmudL = x.a * Ddc1dL;
    arma::mat temp1 = DdmudL; temp1.each_col() %= x.dc2dmu;
    arma::mat temp2 = x.dmudL; temp2.each_col() %= Ddc2dmu;
    arma::mat Ddc2dL = temp1 + temp2;
    arma::colvec Dc2 = x.dc2dmu; Dc2.each_col() %= Dmu;
    arma::colvec temp3 = x.prodc2 * Dc2 / x.c2;
    double Dprodc2 = arma::accu(temp3);
    arma::mat Ddprodc2dL = Dprodc2 * x.dc2dL + x.prodc2 * Ddc2dL;
    Ddprodc2dL.each_col() %= x.c2;
    arma::mat temp4 = x.prodc2 * x.dc2dL; temp4.each_col() %= Dc2;
    Ddprodc2dL -= temp4;
    Ddprodc2dL.each_col() /= c22;
    arma::mat dgL = -arma::sum(Ddprodc2dL, 0);

    x.dgL = arma::reshape(dgL, x.p, x.q);

    arma::colvec Dc12 = arma::diagvec(x.dP * x.L2);
    arma::colvec Dmu2 = x.a * Dc12;
    arma::colvec Ddc2dmu2 = temp0; Ddc2dmu2.each_col() %= Dmu2;
    arma::mat Ddc2dP = x.dmudP; Ddc2dP.each_col() %= Ddc2dmu2;
    arma::mat Dc22 = x.dc2dmu; Dc22.each_col() %= Dmu2;
    double Dprodc22 = arma::accu(x.prodc2 / x.c2 % Dc22);
    arma::mat Ddprodc2dP = Dprodc22 * x.dc2dP + x.prodc2 * Ddc2dP;
    Ddprodc2dP.each_col() %= x.c2;
    arma::mat temp5 = x.prodc2 * x.dc2dP; temp5.each_col() %= Dc22;
    Ddprodc2dP -= temp5;
    Ddprodc2dP.each_col() /= c22;
    arma::mat dgP = -arma::sum(Ddprodc2dP, 0);

    x.dgP = arma::reshape(dgP, x.q, x.q);
    x.dgP.diag().zeros();

  }

};

/*
 * Repeated Crawford-Ferguson family
 */

class rep_cf: public base_criterion {

public:

  void F(arguments_rotate& x){

    arma::uvec indexes = x.blocks_list[x.i];
    x.Li[x.i] = x.L.cols(indexes);
    x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
    x.Ni[x.i] = x.N(indexes, indexes);
    double ff1 = (1-x.k[x.i]) * arma::accu(x.Li2[x.i] % (x.Li2[x.i] * x.Ni[x.i])) / 4;
    double ff2 = x.k[x.i] * arma::accu(x.Li2[x.i] % (x.M * x.Li2[x.i])) / 4;

    x.f += (ff1 + ff2) * x.block_weights[x.i];

  }

  void gLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    arma::mat f1 = (1-x.k[x.i]) * x.Li[x.i] % (x.Li2[x.i] * x.Ni[x.i]);
    arma::mat f2 = x.k[x.i] * x.Li[x.i] % (x.M * x.Li2[x.i]);
    x.gL.cols(indexes) += (f1 + f2) * x.block_weights[x.i];

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

    arma::colvec L_vector = arma::vectorise(x.Li[x.i]);

    arma::mat Ip(x.p, x.p, arma::fill::eye);
    arma::mat c1 = arma::kron(x.Ni[x.i].t(), Ip) * arma::diagmat(2*L_vector);
    c1.each_col() %= L_vector;
    arma::mat c2 = arma::diagmat(arma::vectorise(x.Li2[x.i] * x.Ni[x.i]));
    arma::mat gf1 = (1-x.k[x.i]) * (c1 + c2);

    arma::mat Iq(x.qi[x.i], x.qi[x.i], arma::fill::eye);
    arma::mat c3 = arma::kron(Iq, x.M) * arma::diagmat(2*L_vector);
    c3.each_col() %= L_vector;
    arma::mat c4 = arma::diagmat(arma::vectorise(x.M * x.Li2[x.i]));
    arma::mat gf2 = x.k[x.i] * (c3 + c4);

    x.hL += (gf1 + gf2) * x.block_weights[x.i];

  }

  void dgLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    arma::mat dLi = x.dL.cols(indexes);
    arma::mat dLi2 = 2 * dLi % x.Li[x.i];
    arma::mat df1 = (1-x.k[x.i]) * dLi % (x.Li2[x.i] * x.Ni[x.i]) +
      (1-x.k[x.i]) * x.Li[x.i] % (dLi2 * x.Ni[x.i]);
    arma::mat df2 = x.k[x.i] * dLi % (x.M * x.Li2[x.i]) + x.k[x.i] * x.Li[x.i] % (x.M * dLi2);

    x.dgL.cols(indexes) += (df1 + df2) * x.block_weights[x.i];

  }

};

/*
 * Repeated Varimax
 */

class rep_varimax: public base_criterion {

public:

  void F(arguments_rotate& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.Li[x.i] = x.L.cols(indexes);
      x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
      x.HLi2[x.i] = x.H * x.Li2[x.i];

      x.f -= trace(x.HLi2[x.i].t() * x.HLi2[x.i]) / 4 * x.block_weights[x.i];

  }

  void gLP(arguments_rotate& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.gL.cols(indexes) -= (x.Li[x.i] % x.HLi2[x.i]) * x.block_weights[x.i];

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

    arma::mat Iq(x.qi[x.i], x.qi[x.i], arma::fill::eye);
    arma::mat c1 = arma::diagmat(arma::vectorise(x.HL2i[x.i]));
    arma::colvec L_vector = arma::vectorise(x.Li[x.i]);
    arma::mat c2 = arma::kron(Iq, x.Hi[x.i]) * arma::diagmat(2*L_vector);
    c2.each_col() %= L_vector;

    x.hL -= (c1 + c2) * x.block_weights[x.i];

  }

  void dgLP(arguments_rotate& x) {

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

  void F(arguments_rotate& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.Li[x.i] = x.L.cols(indexes);
      x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
      x.HLi2[x.i] = x.H * x.Li2[x.i];

      x.f += trace(x.HLi2[x.i].t() * x.HLi2[x.i]) / 4 * x.block_weights[x.i];

  }

  void gLP(arguments_rotate& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      x.Li[x.i] = x.L.cols(indexes);
      x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
      x.HLi2[x.i] = x.H * x.Li2[x.i];
      x.gL.cols(indexes) += (x.Li[x.i] % x.HLi2[x.i]) * x.block_weights[x.i];

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

    arma::mat Iq(x.qi[x.i], x.qi[x.i], arma::fill::eye);
    arma::mat c1 = arma::diagmat(arma::vectorise(x.HL2i[x.i]));
    arma::colvec L_vector = arma::vectorise(x.Li[x.i]);
    arma::mat c2 = arma::kron(Iq, x.Hi[x.i]) * arma::diagmat(2*L_vector);
    c2.each_col() %= L_vector;

    x.hL += (c1 + c2) * x.block_weights[x.i];

  }

  void dgLP(arguments_rotate& x) {

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

  void F(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];

    x.Li[x.i] = x.L.cols(indexes);
    x.Ni[x.i] = x.N(indexes, indexes);
    x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
    x.IgCL2Ni[x.i] = x.I_gamma_Ci[x.i] * x.Li2[x.i] * x.Ni[x.i];

    x.f += trace(x.Li2[x.i].t() * x.IgCL2Ni[x.i]) / 4 * x.block_weights[x.i];

  }

  void gLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    x.gL.cols(indexes) += (x.Li[x.i] % x.IgCL2Ni[x.i]) * x.block_weights[x.i];

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

    arma::mat c1 = arma::diagmat(arma::vectorise(x.IgCL2Ni[x.i]));
    arma::mat diagL = arma::diagmat(arma::vectorise(x.Li[x.i]));
    arma::mat diag2L = 2*diagL;
    arma::mat c2 = diagL * arma::kron(x.Ni[x.i], x.I_gamma_Ci[x.i]) * diag2L;
    x.hL += (c1 + c2) * x.block_weights[x.i];

  }

  void dgLP(arguments_rotate& x) {

      arma::uvec indexes = x.blocks_list[x.i];
      arma::mat dLi = x.dL.cols(indexes);
      x.dgL.cols(indexes) += (dLi % x.IgCL2Ni[x.i] +
        x.Li[x.i] % (x.I_gamma_Ci[x.i] * (2*dLi % x.Li[x.i]) * x.Ni[x.i])) * x.block_weights[x.i];

  }

};

/*
 * Repeated Geomin
 */

class rep_geomin: public base_criterion {

public:

  void F(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    int q = indexes.size();
    x.q2 = 2/(q + 0.0);

    x.Li[x.i] = x.L.cols(indexes);
    x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
    x.Li2[x.i] += x.epsilon[x.i];
    x.termi[x.i] = arma::exp(arma::sum(arma::log(x.Li2[x.i]), 1) / q);

    x.f += arma::accu(x.termi[x.i]) * x.block_weights[x.i];

  }

  void gLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    int q = indexes.size();
    x.q2 = 2/(q + 0.0);

    x.LoLi2[x.i] = x.Li[x.i] / x.Li2[x.i];
    x.termi[x.i] = arma::exp(arma::sum(arma::log(x.Li2[x.i]), 1) / q);
    arma::mat gLi = x.LoLi2[x.i] * x.q2;
    gLi.each_col() %= x.termi[x.i];
    x.gL.cols(indexes) += gLi * x.block_weights[x.i];

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

    arma::mat cx = x.q2 * x.LoLi2[x.i];
    arma::mat c1 = x.q2*(arma::vectorise(x.Li2[x.i]) -
      arma::vectorise(2*x.Li[x.i] % x.Li[x.i])) /
      arma::vectorise(x.Li2[x.i] % x.Li2[x.i]);
    arma::mat gcx = arma::diagmat(c1);
    arma::mat c2 = (1/x.Li2[x.i]) % (2*x.Li[x.i]) / x.qi[x.i];
    c2.each_col() %= x.termi[x.i];
    arma::mat gterm = cbind_diag(c2);
    arma::mat v = gterm.t() * cx;
    arma::mat hL = gcx;
    arma::mat term2 = x.termi[x.i];
    for(int i=0; i < (x.qi[x.i]-1); ++i) term2 = arma::join_cols(term2, x.termi[x.i]);
    hL.each_col() %= term2;
    hL += kdiag(v);

    x.hL += hL * x.block_weights[x.i];

  }

  void dgLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    int q = indexes.size();
    x.q2 = 2/(q + 0.0);

    arma::mat dLi = x.dL.cols(indexes);
    arma::mat c1 = (x.epsilon[x.i] - x.Li[x.i] % x.Li[x.i]) / (x.Li2[x.i] % x.Li2[x.i]) % dLi;
    c1.each_col() %= x.termi[x.i];
    arma::mat c2 = x.LoLi2[x.i];
    arma::vec termi2 = x.q2 * x.termi[x.i] % arma::sum(x.LoLi2[x.i] % dLi, 1);
    c2.each_col() %= termi2;

    x.dgL.cols(indexes) += x.q2 * (c1 + c2) * x.block_weights[x.i];

  }

};

/*
 * Repeated Target
 */

class rep_target: public base_criterion {

public:
  void F(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    x.Li[x.i] = x.L.cols(indexes);

    x.f1i[x.i] = x.Weight % (x.Li[x.i] - x.Target);
    x.f += 0.5*arma::accu(x.f1i[x.i] % x.f1i[x.i]);

  }

  void gLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];

    x.gL.cols(indexes) += x.Weight % x.f1i[x.i];

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

    arma::mat W2 = x.Weight % x.Weight;
    x.hL += arma::diagmat(arma::vectorise(W2));

  }

  void dgLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    arma::mat dLi = x.dL.cols(indexes);

    x.dgL.cols(indexes) += x.Weight % x.Weight % dLi;

  }

};

/*
 * Repeated xTarget
 */

class rep_xtarget: public base_criterion {

public:

  void F(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    x.Li[x.i] = x.L.cols(indexes);
    x.Phii[x.i] = x.Phi.cols(indexes);

    x.f1i[x.i] = x.Weight % (x.Li[x.i] - x.Target);
    x.f2i[x.i] = x.Phi_Weight % (x.Phii[x.i] - x.Phi_Target);

    x.f += 0.5*arma::accu(x.f1i[x.i] % x.f1i[x.i]) +
      0.25*x.w*arma::accu(x.f2i[x.i] % x.f2i[x.i]);

  }

  void gLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];

    x.gL.cols(indexes) += x.Weight % x.f1i[x.i];
    x.gP(indexes, indexes) += x.w * x.Phi_Weight % x.f2i[x.i];

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

    x.hL += arma::diagmat(arma::vectorise(x.Weight2));

    arma::mat Phi_t = dxt(x.qi[x.i], x.qi[x.i]);
    arma::mat diag_PW2 = arma::diagmat(arma::vectorise(x.Phi_Weight2));
    x.hP += diag_PW2 + diag_PW2 * Phi_t; // w¿?

  }

  void dgLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    arma::mat dPi = x.dP(indexes, indexes);

    x.dgL.cols(indexes) += x.Weight2 % x.dL;
    x.dgP(indexes, indexes) += x.w * x.Phi_Weight2 % dPi;

  }

};

/*
 * Repeated equavar (orthogonal)
 */

class rep_equavar: public base_criterion {

public:

  void F(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    x.Li[x.i] = x.L.cols(indexes);

    double q = indexes.size() + 0.0;
    x.vari[x.i] = arma::diagvec(x.Li[x.i].t() * x.Li[x.i]);
    x.varqi[x.i] = arma::pow(x.vari[x.i], 1/q);
    x.prodvarqi[x.i] = arma::prod(x.varqi[x.i]);

    x.f += x.prodvarqi[x.i] * -1 * x.block_weights[x.i];

  }

  void gLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];

    double q = indexes.size() + 0.0;
    x.dvarqdLi[x.i] = 2*x.Li[x.i];
    x.dvarqdLi[x.i].each_row() %= arma::pow(x.vari[x.i], 1/q-1)/q;
    arma::mat gL = x.dvarqdLi[x.i];
    gL.each_row() %= (1/x.varqi[x.i]);
    gL *= x.prodvarqi[x.i];
    x.gL.cols(indexes) += gL * -1 * x.block_weights[x.i];

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

  }

  void dgLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    arma::mat dLi = x.dL.cols(indexes);

    double q = indexes.size() + 0.0;
    arma::mat dvardL = 2*x.Li[x.i];
    arma::mat DdvardL = 2*dLi;
    arma::rowvec Dvar = arma::diagvec(dLi.t() * x.Li[x.i] + x.Li[x.i].t() * dLi);
    arma::rowvec Dvarq = arma::pow(x.vari[x.i], 1/q-1)/q % Dvar;
    arma::mat dvarqdL = dvardL;
    dvarqdL.each_row() %= arma::pow(x.vari[x.i], 1/q-1)/q;
    arma::mat DdvarqdL1 = DdvardL;
    DdvarqdL1.each_row() %= arma::pow(x.vari[x.i], 1/q-1)/q;
    arma::mat DdvarqdL2 = dvardL;
    DdvarqdL2.each_row() %= arma::pow(x.vari[x.i], 1/q-2)/q * (1/q-1) % Dvar;
    arma::mat DdvarqdL = DdvarqdL1 + DdvarqdL2;
    double Dprodvarq = arma::accu(x.prodvarqi[x.i] / x.varqi[x.i] % Dvarq);

    arma::mat dgL1 = Dprodvarq * dvarqdL + x.prodvarqi[x.i] * DdvarqdL;
    dgL1.each_row() %= x.varqi[x.i];
    arma::mat dgL2 = x.prodvarqi[x.i] * dvarqdL;
    dgL2.each_row() %= Dvarq;
    arma::mat dgL = dgL1 - dgL2;
    dgL /= (x.varqi[x.i] % x.varqi[x.i]);

    x.dgL.cols(indexes) += dgL * -1 * x.block_weights[x.i];

  }

};

/*
 * Repeated simplix orthogonal
 */

class rep_simplix_orth: public base_criterion {

public:

  void F(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    x.Li[x.i] = x.L.cols(indexes);

    x.Li2[x.i] = x.Li[x.i] % x.Li[x.i];
    arma::rowvec c1 = arma::sum(x.Li2[x.i], 0);
    x.mui[x.i] = x.a*(c1.t()-x.b);
    x.expmmui[x.i] = arma::exp(-x.mui[x.i]);
    x.c2i[x.i] = 1/(1 + x.expmmui[x.i]);
    x.prodc2i[x.i] = arma::prod(x.c2i[x.i]);

    x.f += (1 - x.prodc2i[x.i]) * x.block_weights[x.i];

  }

  void gLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];

    arma::mat dc1dL = 2*x.Li[x.i];
    x.dmudLi[x.i] = x.a * dc1dL;
    x.dc2dmui[x.i] = x.expmmui[x.i] / arma::pow(1 + x.expmmui[x.i], 2);
    x.dc2dLi[x.i] = x.dmudLi[x.i]; x.dc2dLi[x.i].each_row() %= x.dc2dmui[x.i].t();
    arma::mat dprodc2dL = x.prodc2i[x.i] * x.dc2dLi[x.i];
    dprodc2dL.each_row() /= x.c2i[x.i].t();

    x.gL.cols(indexes) += (-dprodc2dL) * x.block_weights[x.i];

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

  }

  void dgLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    arma::mat dLi = x.dL.cols(indexes);

    arma::colvec expmu = arma::exp(x.mui[x.i]);
    arma::mat Ddc1dL = 2*dLi;
    arma::colvec Dc1 = arma::diagvec(dLi.t() * x.Li[x.i] + x.Li[x.i].t() * dLi);
    arma::colvec Dmu = x.a * Dc1;
    arma::colvec Dc2 = x.dc2dmui[x.i] % Dmu;
    double Dprodc2 = arma::accu(x.prodc2i[x.i] * Dc2 / x.c2i[x.i]);
    arma::mat DdmudL = x.a * Ddc1dL;
    arma::colvec ddc2dmudmu = -expmu % (expmu - 1) / arma::pow(1 + expmu, 3);
    arma::colvec Ddc2dmu = ddc2dmudmu % Dmu;
    arma::mat Ddc2dL1 = x.dmudLi[x.i]; Ddc2dL1.each_row() %= Ddc2dmu.t();
    arma::mat Ddc2dL2 = DdmudL; Ddc2dL2.each_row() %= x.dc2dmui[x.i].t();
    arma::mat Ddc2dL = Ddc2dL1 + Ddc2dL2;
    arma::mat dgL1 = Dprodc2 * x.dc2dLi[x.i] + x.prodc2i[x.i] * Ddc2dL;
    dgL1.each_row() %= x.c2i[x.i].t();
    arma::mat dgL2 = x.prodc2i[x.i] * x.dc2dLi[x.i]; dgL2.each_row() %= Dc2.t();
    arma::mat dgL = dgL1 - dgL2; dgL.each_row() /= (x.c2i[x.i] % x.c2i[x.i]).t();

    x.dgL.cols(indexes) += (-dgL) * x.block_weights[x.i];

  }

};

/*
 * Repeated simplix
 */

class rep_simplix: public base_criterion {

public:

  void F(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    x.Li[x.i] = x.L.cols(indexes);
    x.Phii[x.i] = x.Phi(indexes, indexes);

    x.Li2[x.i] = x.Li[x.i].t() * x.Li[x.i];
    arma::colvec c1 = arma::diagvec(x.Phii[x.i] * x.Li2[x.i]);
    x.mui[x.i] = x.a*(c1-x.b);
    x.expmmui[x.i] = arma::exp(-x.mui[x.i]);
    x.c2i[x.i] = 1/(1 + x.expmmui[x.i]);
    x.prodc2i[x.i] = arma::prod(x.c2i[x.i]);

    x.f += (1 - x.prodc2i[x.i]) * x.block_weights[x.i];

  }

  void gLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];

    arma::mat dc1dL = x.Hi[x.i] * (arma::kron(x.Li[x.i].t(), x.Phii[x.i]) * x.dxtLi[x.i] +
      arma::kron(x.Ii[x.i], x.Phii[x.i] * x.Li[x.i].t()));
    x.dmudLi[x.i] = x.a * dc1dL;
    x.dc2dmui[x.i] = x.expmmui[x.i] / arma::pow(1 + x.expmmui[x.i], 2);
    x.dc2dLi[x.i] = x.dmudLi[x.i]; x.dc2dLi[x.i].each_col() %= x.dc2dmui[x.i];
    arma::mat dprodc2dL = x.prodc2i[x.i] * x.dc2dLi[x.i];
    dprodc2dL.each_col() /= x.c2i[x.i];
    arma::mat gL = -arma::sum(dprodc2dL, 0);

    int q = indexes.size();
    x.gL.cols(indexes) += arma::reshape(gL, x.p, q) * x.block_weights[x.i];

    x.LtLxIi[x.i] = arma::kron(x.Li2[x.i], x.Ii[x.i]);
    arma::mat dc1dP = x.Hi[x.i] * (x.LtLxIi[x.i] + x.LtLxIi[x.i] * x.dxtPi[x.i]);
    x.dmudPi[x.i] = x.a * dc1dP;
    x.dc2dPi[x.i] = x.dmudPi[x.i]; x.dc2dPi[x.i].each_col() %= x.dc2dmui[x.i];
    arma::mat dprodc2dP = x.prodc2i[x.i] * x.dc2dPi[x.i];
    dprodc2dP.each_col() /= x.c2i[x.i];
    arma::mat gP = -arma::sum(dprodc2dP, 0);

    x.gP.cols(indexes) += arma::reshape(gP, q, q) * x.block_weights[x.i];
    x.gP.diag().zeros();

  }

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for block criteria");

  }

  void dgLP(arguments_rotate& x) {

    arma::uvec indexes = x.blocks_list[x.i];
    arma::mat dLi = x.dL.cols(indexes);
    arma::mat dPi = x.dP(indexes, indexes);

    arma::colvec expmu = arma::exp(x.mui[x.i]);
    arma::colvec c22 = x.c2i[x.i] % x.c2i[x.i];
    arma::colvec Dc1 = arma::diagvec(x.Phii[x.i] * dLi.t() * x.Li[x.i] +
      x.Phii[x.i] * x.Li[x.i].t() * dLi);
    arma::colvec Dmu = x.a * Dc1;
    arma::mat Ddc1dL = x.Hi[x.i] * (arma::kron(dLi.t(), x.Phii[x.i]) * x.dxtLi[x.i] +
      arma::kron(x.Ii[x.i], x.Phii[x.i] * dLi.t()));
    arma::colvec temp0 = -expmu % (expmu - 1) / arma::pow(1 + expmu, 3);
    arma::colvec Ddc2dmu = temp0 % Dmu;
    arma::mat DdmudL = x.a * Ddc1dL;
    arma::mat temp1 = DdmudL; temp1.each_col() %= x.dc2dmui[x.i];
    arma::mat temp2 = x.dmudLi[x.i]; temp2.each_col() %= Ddc2dmu;
    arma::mat Ddc2dL = temp1 + temp2;
    arma::colvec Dc2 = x.dc2dmui[x.i]; Dc2.each_col() %= Dmu;
    arma::colvec temp3 = x.prodc2i[x.i] * Dc2 / x.c2i[x.i];
    double Dprodc2 = arma::accu(temp3);
    arma::mat Ddprodc2dL = Dprodc2 * x.dc2dLi[x.i] + x.prodc2i[x.i] * Ddc2dL;
    Ddprodc2dL.each_col() %= x.c2i[x.i];
    arma::mat temp4 = x.prodc2i[x.i] * x.dc2dLi[x.i]; temp4.each_col() %= Dc2;
    Ddprodc2dL -= temp4;
    Ddprodc2dL.each_col() /= c22;
    arma::mat dgL = -arma::sum(Ddprodc2dL, 0);

    int q = indexes.size();
    x.dgL.cols(indexes) += arma::reshape(dgL, x.p, q) * x.block_weights[x.i];

    arma::colvec Dc12 = arma::diagvec(dPi * x.Li2[x.i]);
    arma::colvec Dmu2 = x.a * Dc12;
    arma::colvec Ddc2dmu2 = temp0; Ddc2dmu2.each_col() %= Dmu2;
    arma::mat Ddc2dP = x.dmudPi[x.i]; Ddc2dP.each_col() %= Ddc2dmu2;
    arma::mat Dc22 = x.dc2dmui[x.i]; Dc22.each_col() %= Dmu2;
    double Dprodc22 = arma::accu(x.prodc2i[x.i] / x.c2i[x.i] % Dc22);
    arma::mat Ddprodc2dP = Dprodc22 * x.dc2dPi[x.i] + x.prodc2i[x.i] * Ddc2dP;
    Ddprodc2dP.each_col() %= x.c2i[x.i];
    arma::mat temp5 = x.prodc2i[x.i] * x.dc2dPi[x.i]; temp5.each_col() %= Dc22;
    Ddprodc2dP -= temp5;
    Ddprodc2dP.each_col() /= c22;
    arma::mat dgP = -arma::sum(Ddprodc2dP, 0);

    x.dgP.cols(indexes) += arma::reshape(dgP, q, q) * x.block_weights[x.i];
    x.dgP.diag().zeros();

  }

};

/*
 * Tian & Liu between_blocks
 */

class TL: public base_criterion {

public:

  void F(arguments_rotate& x) {

    // between_blocks
    x.Lg = x.L.cols(x.blocks_list[x.i]);
    x.Ls = x.L.cols(x.blocks_list[x.i + 1]);
    x.L2 = x.L % x.L;
    x.L2g = x.L2.cols(x.blocks_list[x.i]);
    x.L2s = x.L2.cols(x.blocks_list[x.i + 1]);
    x.Ng = bc(x.Lg.n_cols);

    x.exp_aL2g = exp(-x.alpha*x.L2g);
    x.C = x.L2s.t() * x.exp_aL2g;
    x.logC = log(x.C);
    x.logCN = x.logC * x.Ng;
    x.exp_lCN = exp(x.logCN);
    x.f += arma::accu(x.exp_lCN);

  }

  void gLP(arguments_rotate& x) {

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

    // between_blocks
    x.gL2g = 2*x.Lg;
    x.gL2s = 2*x.Ls;
    x.g_exp_aL2g = -x.alpha*x.exp_aL2g % x.gL2g;
    x.dxt_L2s = dxt(x.p, q2);
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

  void hLP(arguments_rotate& x) {

    Rcpp::stop("Standard errors not implement yet for between-block criteria");

  }

  void dgLP(arguments_rotate& x) {

    arma::uvec indexes1 = x.blocks_list[x.i];
    arma::uvec indexes2 = x.blocks_list[x.i + 1];

    arma::uvec indexes = arma::join_cols(indexes1, indexes2);
    int q = indexes.size();

    // between_blocks
    arma::mat dLg = x.dL.cols(x.blocks_list[x.i]);
    arma::mat dLs = x.dL.cols(x.blocks_list[x.i + 1]);
    arma::mat dL2 = 2*x.dL % x.L;
    arma::mat dL2g = dL2.cols(x.blocks_list[x.i]);
    arma::mat dL2s = dL2.cols(x.blocks_list[x.i + 1]);
    arma::mat dexp_aL2g = -x.alpha*x.exp_aL2g % dL2g;
    arma::mat dgL2g = 2*dLg;
    arma::mat dgL2s = 2*dLs;
    arma::mat dg_exp_aL2g = -x.alpha*dexp_aL2g % x.gL2g + -x.alpha*x.exp_aL2g % dgL2g;
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
    dglogC12.each_col() %= arma::vectorise(-dC/(x.C % x.C));
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

    criterion = new rep_target();

  }  else if(rotation == "xtarget") {

    if(projection == "orth") {
      criterion = new rep_target();
    } else {
      criterion = new rep_xtarget();
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

  } else if(rotation == "equavar") {

    criterion = new rep_equavar();

  } else if(rotation == "simplix") {

    if(projection == "orth") {
      criterion = new rep_simplix_orth();
    } else {
      criterion = new rep_simplix_orth();
    }

  } else if(rotation == "none") {

    criterion = new none();

  } else {

    Rcpp::stop("Available rotations: \n cf, oblimin, geomin, varimax, varimin, target, xtarget, equavar, simplix");

  }

  return criterion;

}
base_criterion* choose_between_blocks(std::string rotation) {

  base_criterion* criterion;

  if(rotation == "TL") {

    criterion = new TL();

  } else {

    Rcpp::stop("Available between_blockss: TL");

  }

  return criterion;

}

class mixed: public base_criterion {

public:

  void F(arguments_rotate& x) {

    base_criterion* criterion;
    x.f = 0;

    for(int i=0; i < x.n_blocks; ++i) {

      x.i = i;
      // int report = x.blocks_list[x.i].size();
      // Rcpp::Rcout << report << std::endl;

      criterion = choose_rep_criterion(x.rotations[i], x.projection);
      criterion->F(x);

    }

    if(x.between) {

      criterion = choose_between_blocks(x.between_blocks);

      for(int i=0; i < (x.n_blocks-1); ++i) {

        x.i = i;
        criterion->F(x);

      }

    }

  }

  void gLP(arguments_rotate& x) {

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

    if(x.between) {

      criterion = choose_between_blocks(x.between_blocks);

      for(int i=0; i < (x.n_blocks-1); ++i) {

        x.i = i;
        criterion->gLP(x);

      }

    }

  }

  void hLP(arguments_rotate& x) {

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

    if(x.between) {

      criterion = choose_between_blocks(x.between_blocks);

      for(int i=0; i < (x.n_blocks-1); ++i) {

        x.i = i;
        criterion->hLP(x);

      }

    }

  }

  void dgLP(arguments_rotate& x) {

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

    if(x.between) {

      criterion = choose_between_blocks(x.between_blocks);

      for(int i=0; i < (x.n_blocks-1); ++i) {

        x.i = i;
        criterion->dgLP(x);

      }

    }

  }

};

// Choose the rotation criteria:

base_criterion* choose_criterion(std::vector<std::string> rotations, std::string projection,
                                 std::vector<arma::uvec> blocks_list) {

  base_criterion *criterion;

  if(!blocks_list.empty()) {

    // Rcpp::stop("Mixed rotation criteria not supported yet");
    criterion = new mixed();

  } else if (rotations[0] == "target") {

    criterion = new target();

  }  else if(rotations[0] == "xtarget") {

    if(projection == "orth") {
      criterion = new target();
    } else {
      criterion = new xtarget();
    }

  } else if(rotations[0] == "cf") {

    criterion = new cf();

  } else if(rotations[0] == "oblimin") {

    criterion = new oblimin();

  } else if(rotations[0] == "geomin") {

    criterion = new geomin();

  } else if(rotations[0] == "varimax") {

    criterion = new varimax();

  } else if(rotations[0] == "varimin") {

    criterion = new varimin();

  } else if(rotations[0] == "equavar") {

    criterion = new equavar();

  } else if(rotations[0] == "simplix") {

    if(projection == "orth") {
      criterion = new simplix_orth();
    } else {
      criterion = new simplix_orth();
    }

  } else if(rotations[0] == "none") {

  } else {

    Rcpp::stop("Available rotations: \n cf, oblimin, geomin, varimax, varimin, target, xtarget, equavar, simplix");

  }

  return criterion;

}
