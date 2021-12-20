// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// random_orth
arma::mat random_orth(int p, int q);
RcppExport SEXP _bifactor_random_orth(SEXP pSEXP, SEXP qSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    rcpp_result_gen = Rcpp::wrap(random_orth(p, q));
    return rcpp_result_gen;
END_RCPP
}
// random_oblq
arma::mat random_oblq(int p, int q);
RcppExport SEXP _bifactor_random_oblq(SEXP pSEXP, SEXP qSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    rcpp_result_gen = Rcpp::wrap(random_oblq(p, q));
    return rcpp_result_gen;
END_RCPP
}
// random_poblq
arma::mat random_poblq(int p, int q, arma::uvec oblq_blocks);
RcppExport SEXP _bifactor_random_poblq(SEXP pSEXP, SEXP qSEXP, SEXP oblq_blocksSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type oblq_blocks(oblq_blocksSEXP);
    rcpp_result_gen = Rcpp::wrap(random_poblq(p, q, oblq_blocks));
    return rcpp_result_gen;
END_RCPP
}
// retr_orth
arma::mat retr_orth(arma::mat X);
RcppExport SEXP _bifactor_retr_orth(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(retr_orth(X));
    return rcpp_result_gen;
END_RCPP
}
// retr_oblq
arma::mat retr_oblq(arma::mat X);
RcppExport SEXP _bifactor_retr_oblq(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(retr_oblq(X));
    return rcpp_result_gen;
END_RCPP
}
// retr_poblq
arma::mat retr_poblq(arma::mat X, arma::uvec oblq_blocks);
RcppExport SEXP _bifactor_retr_poblq(SEXP XSEXP, SEXP oblq_blocksSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type oblq_blocks(oblq_blocksSEXP);
    rcpp_result_gen = Rcpp::wrap(retr_poblq(X, oblq_blocks));
    return rcpp_result_gen;
END_RCPP
}
// sl
Rcpp::List sl(arma::mat R, int n_generals, int n_groups, Rcpp::Nullable<Rcpp::List> first_efa, Rcpp::Nullable<Rcpp::List> second_efa);
RcppExport SEXP _bifactor_sl(SEXP RSEXP, SEXP n_generalsSEXP, SEXP n_groupsSEXP, SEXP first_efaSEXP, SEXP second_efaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type n_generals(n_generalsSEXP);
    Rcpp::traits::input_parameter< int >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type first_efa(first_efaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type second_efa(second_efaSEXP);
    rcpp_result_gen = Rcpp::wrap(sl(R, n_generals, n_groups, first_efa, second_efa));
    return rcpp_result_gen;
END_RCPP
}
// rotate
Rcpp::List rotate(arma::mat loadings, std::string rotation, std::string projection, Rcpp::Nullable<arma::mat> Target, Rcpp::Nullable<arma::mat> Weight, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, Rcpp::Nullable<arma::uvec> blocks, Rcpp::Nullable<arma::uvec> oblq_blocks, double gamma, double epsilon, double k, double w, int random_starts, int cores, Rcpp::Nullable<Rcpp::List> rot_control);
RcppExport SEXP _bifactor_rotate(SEXP loadingsSEXP, SEXP rotationSEXP, SEXP projectionSEXP, SEXP TargetSEXP, SEXP WeightSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP blocksSEXP, SEXP oblq_blocksSEXP, SEXP gammaSEXP, SEXP epsilonSEXP, SEXP kSEXP, SEXP wSEXP, SEXP random_startsSEXP, SEXP coresSEXP, SEXP rot_controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type loadings(loadingsSEXP);
    Rcpp::traits::input_parameter< std::string >::type rotation(rotationSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Target(TargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Weight(WeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiTarget(PhiTargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiWeight(PhiWeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type blocks(blocksSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type oblq_blocks(oblq_blocksSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< double >::type k(kSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type random_starts(random_startsSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type rot_control(rot_controlSEXP);
    rcpp_result_gen = Rcpp::wrap(rotate(loadings, rotation, projection, Target, Weight, PhiTarget, PhiWeight, blocks, oblq_blocks, gamma, epsilon, k, w, random_starts, cores, rot_control));
    return rcpp_result_gen;
END_RCPP
}
// efast
Rcpp::List efast(arma::mat R, int n_factors, std::string method, std::string rotation, std::string projection, Rcpp::Nullable<arma::mat> Target, Rcpp::Nullable<arma::mat> Weight, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, Rcpp::Nullable<arma::uvec> blocks, Rcpp::Nullable<arma::uvec> oblq_blocks, bool normalize, double gamma, double epsilon, double k, double w, int random_starts, int cores, Rcpp::Nullable<arma::vec> init, Rcpp::Nullable<Rcpp::List> efa_control, Rcpp::Nullable<Rcpp::List> rot_control);
RcppExport SEXP _bifactor_efast(SEXP RSEXP, SEXP n_factorsSEXP, SEXP methodSEXP, SEXP rotationSEXP, SEXP projectionSEXP, SEXP TargetSEXP, SEXP WeightSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP blocksSEXP, SEXP oblq_blocksSEXP, SEXP normalizeSEXP, SEXP gammaSEXP, SEXP epsilonSEXP, SEXP kSEXP, SEXP wSEXP, SEXP random_startsSEXP, SEXP coresSEXP, SEXP initSEXP, SEXP efa_controlSEXP, SEXP rot_controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type n_factors(n_factorsSEXP);
    Rcpp::traits::input_parameter< std::string >::type method(methodSEXP);
    Rcpp::traits::input_parameter< std::string >::type rotation(rotationSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Target(TargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Weight(WeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiTarget(PhiTargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiWeight(PhiWeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type blocks(blocksSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type oblq_blocks(oblq_blocksSEXP);
    Rcpp::traits::input_parameter< bool >::type normalize(normalizeSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< double >::type k(kSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type random_starts(random_startsSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type init(initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type efa_control(efa_controlSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type rot_control(rot_controlSEXP);
    rcpp_result_gen = Rcpp::wrap(efast(R, n_factors, method, rotation, projection, Target, Weight, PhiTarget, PhiWeight, blocks, oblq_blocks, normalize, gamma, epsilon, k, w, random_starts, cores, init, efa_control, rot_control));
    return rcpp_result_gen;
END_RCPP
}
// get_target
arma::mat get_target(arma::mat loadings, Rcpp::Nullable<arma::mat> Phi, double cutoff);
RcppExport SEXP _bifactor_get_target(SEXP loadingsSEXP, SEXP PhiSEXP, SEXP cutoffSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type loadings(loadingsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Phi(PhiSEXP);
    Rcpp::traits::input_parameter< double >::type cutoff(cutoffSEXP);
    rcpp_result_gen = Rcpp::wrap(get_target(loadings, Phi, cutoff));
    return rcpp_result_gen;
END_RCPP
}
// bifactor
Rcpp::List bifactor(arma::mat R, int n_generals, int n_groups, std::string twoTier_method, std::string projection, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, Rcpp::Nullable<arma::uvec> blocks, Rcpp::Nullable<arma::uvec> oblq_blocks, Rcpp::Nullable<arma::mat> init_Target, std::string method, int maxit, double cutoff, double w, int random_starts, int cores, Rcpp::Nullable<arma::vec> init, Rcpp::Nullable<Rcpp::List> efa_control, Rcpp::Nullable<Rcpp::List> rot_control, Rcpp::Nullable<Rcpp::List> SL_first_efa, Rcpp::Nullable<Rcpp::List> SL_second_efa, bool verbose);
RcppExport SEXP _bifactor_bifactor(SEXP RSEXP, SEXP n_generalsSEXP, SEXP n_groupsSEXP, SEXP twoTier_methodSEXP, SEXP projectionSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP blocksSEXP, SEXP oblq_blocksSEXP, SEXP init_TargetSEXP, SEXP methodSEXP, SEXP maxitSEXP, SEXP cutoffSEXP, SEXP wSEXP, SEXP random_startsSEXP, SEXP coresSEXP, SEXP initSEXP, SEXP efa_controlSEXP, SEXP rot_controlSEXP, SEXP SL_first_efaSEXP, SEXP SL_second_efaSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type n_generals(n_generalsSEXP);
    Rcpp::traits::input_parameter< int >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< std::string >::type twoTier_method(twoTier_methodSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiTarget(PhiTargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiWeight(PhiWeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type blocks(blocksSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type oblq_blocks(oblq_blocksSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type init_Target(init_TargetSEXP);
    Rcpp::traits::input_parameter< std::string >::type method(methodSEXP);
    Rcpp::traits::input_parameter< int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< double >::type cutoff(cutoffSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type random_starts(random_startsSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type init(initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type efa_control(efa_controlSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type rot_control(rot_controlSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type SL_first_efa(SL_first_efaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type SL_second_efa(SL_second_efaSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(bifactor(R, n_generals, n_groups, twoTier_method, projection, PhiTarget, PhiWeight, blocks, oblq_blocks, init_Target, method, maxit, cutoff, w, random_starts, cores, init, efa_control, rot_control, SL_first_efa, SL_second_efa, verbose));
    return rcpp_result_gen;
END_RCPP
}
// asymp_cov
arma::mat asymp_cov(arma::mat R, Rcpp::Nullable<arma::mat> X, double eta, std::string type);
RcppExport SEXP _bifactor_asymp_cov(SEXP RSEXP, SEXP XSEXP, SEXP etaSEXP, SEXP typeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    rcpp_result_gen = Rcpp::wrap(asymp_cov(R, X, eta, type));
    return rcpp_result_gen;
END_RCPP
}
// se
Rcpp::List se(int n, Rcpp::Nullable<Rcpp::List> fit, Rcpp::Nullable<arma::mat> R, Rcpp::Nullable<arma::mat> Lambda, Rcpp::Nullable<arma::mat> Phi, Rcpp::Nullable<arma::mat> X, std::string method, std::string projection, std::string rotation, Rcpp::Nullable<arma::mat> Target, Rcpp::Nullable<arma::mat> Weight, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, double gamma, double k, double epsilon, double w, std::string type, double eta);
RcppExport SEXP _bifactor_se(SEXP nSEXP, SEXP fitSEXP, SEXP RSEXP, SEXP LambdaSEXP, SEXP PhiSEXP, SEXP XSEXP, SEXP methodSEXP, SEXP projectionSEXP, SEXP rotationSEXP, SEXP TargetSEXP, SEXP WeightSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP gammaSEXP, SEXP kSEXP, SEXP epsilonSEXP, SEXP wSEXP, SEXP typeSEXP, SEXP etaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type fit(fitSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type R(RSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Lambda(LambdaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Phi(PhiSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type X(XSEXP);
    Rcpp::traits::input_parameter< std::string >::type method(methodSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
    Rcpp::traits::input_parameter< std::string >::type rotation(rotationSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Target(TargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Weight(WeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiTarget(PhiTargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiWeight(PhiWeightSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< double >::type k(kSEXP);
    Rcpp::traits::input_parameter< double >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    rcpp_result_gen = Rcpp::wrap(se(n, fit, R, Lambda, Phi, X, method, projection, rotation, Target, Weight, PhiTarget, PhiWeight, gamma, k, epsilon, w, type, eta));
    return rcpp_result_gen;
END_RCPP
}
// parallel
Rcpp::List parallel(arma::mat X, int n_boot, double quant, bool mean, bool replace, bool hierarchical, Rcpp::Nullable<Rcpp::List> efa, int cores);
RcppExport SEXP _bifactor_parallel(SEXP XSEXP, SEXP n_bootSEXP, SEXP quantSEXP, SEXP meanSEXP, SEXP replaceSEXP, SEXP hierarchicalSEXP, SEXP efaSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type n_boot(n_bootSEXP);
    Rcpp::traits::input_parameter< double >::type quant(quantSEXP);
    Rcpp::traits::input_parameter< bool >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< bool >::type hierarchical(hierarchicalSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type efa(efaSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    rcpp_result_gen = Rcpp::wrap(parallel(X, n_boot, quant, mean, replace, hierarchical, efa, cores));
    return rcpp_result_gen;
END_RCPP
}
// cv_eigen
Rcpp::List cv_eigen(arma::mat X, int N, bool hierarchical, Rcpp::Nullable<Rcpp::List> efa, int cores);
RcppExport SEXP _bifactor_cv_eigen(SEXP XSEXP, SEXP NSEXP, SEXP hierarchicalSEXP, SEXP efaSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type N(NSEXP);
    Rcpp::traits::input_parameter< bool >::type hierarchical(hierarchicalSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type efa(efaSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    rcpp_result_gen = Rcpp::wrap(cv_eigen(X, N, hierarchical, efa, cores));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bifactor_random_orth", (DL_FUNC) &_bifactor_random_orth, 2},
    {"_bifactor_random_oblq", (DL_FUNC) &_bifactor_random_oblq, 2},
    {"_bifactor_random_poblq", (DL_FUNC) &_bifactor_random_poblq, 3},
    {"_bifactor_retr_orth", (DL_FUNC) &_bifactor_retr_orth, 1},
    {"_bifactor_retr_oblq", (DL_FUNC) &_bifactor_retr_oblq, 1},
    {"_bifactor_retr_poblq", (DL_FUNC) &_bifactor_retr_poblq, 2},
    {"_bifactor_sl", (DL_FUNC) &_bifactor_sl, 5},
    {"_bifactor_rotate", (DL_FUNC) &_bifactor_rotate, 16},
    {"_bifactor_efast", (DL_FUNC) &_bifactor_efast, 21},
    {"_bifactor_get_target", (DL_FUNC) &_bifactor_get_target, 3},
    {"_bifactor_bifactor", (DL_FUNC) &_bifactor_bifactor, 22},
    {"_bifactor_asymp_cov", (DL_FUNC) &_bifactor_asymp_cov, 4},
    {"_bifactor_se", (DL_FUNC) &_bifactor_se, 19},
    {"_bifactor_parallel", (DL_FUNC) &_bifactor_parallel, 8},
    {"_bifactor_cv_eigen", (DL_FUNC) &_bifactor_cv_eigen, 5},
    {NULL, NULL, 0}
};

RcppExport void R_init_bifactor(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
