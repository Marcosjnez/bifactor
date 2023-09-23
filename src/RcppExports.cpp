// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// asymptotic_general
arma::mat asymptotic_general(arma::mat X);
RcppExport SEXP _bifactor_asymptotic_general(SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(asymptotic_general(X));
    return rcpp_result_gen;
END_RCPP
}
// asymptotic_normal
arma::mat asymptotic_normal(arma::mat P);
RcppExport SEXP _bifactor_asymptotic_normal(SEXP PSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type P(PSEXP);
    rcpp_result_gen = Rcpp::wrap(asymptotic_normal(P));
    return rcpp_result_gen;
END_RCPP
}
// asymptotic_elliptical
arma::mat asymptotic_elliptical(arma::mat P, double eta);
RcppExport SEXP _bifactor_asymptotic_elliptical(SEXP PSEXP, SEXP etaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type P(PSEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    rcpp_result_gen = Rcpp::wrap(asymptotic_elliptical(P, eta));
    return rcpp_result_gen;
END_RCPP
}
// asymptotic_poly
arma::mat asymptotic_poly(const arma::mat X, const arma::mat R, const int cores);
RcppExport SEXP _bifactor_asymptotic_poly(SEXP XSEXP, SEXP RSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< const arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< const int >::type cores(coresSEXP);
    rcpp_result_gen = Rcpp::wrap(asymptotic_poly(X, R, cores));
    return rcpp_result_gen;
END_RCPP
}
// smoothing
arma::mat smoothing(arma::mat X, double min_eigval);
RcppExport SEXP _bifactor_smoothing(SEXP XSEXP, SEXP min_eigvalSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type min_eigval(min_eigvalSEXP);
    rcpp_result_gen = Rcpp::wrap(smoothing(X, min_eigval));
    return rcpp_result_gen;
END_RCPP
}
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
arma::mat random_poblq(int p, int q, arma::uvec oblq_factors);
RcppExport SEXP _bifactor_random_poblq(SEXP pSEXP, SEXP qSEXP, SEXP oblq_factorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type oblq_factors(oblq_factorsSEXP);
    rcpp_result_gen = Rcpp::wrap(random_poblq(p, q, oblq_factors));
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
arma::mat retr_poblq(arma::mat X, arma::uvec oblq_factors);
RcppExport SEXP _bifactor_retr_poblq(SEXP XSEXP, SEXP oblq_factorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type oblq_factors(oblq_factorsSEXP);
    rcpp_result_gen = Rcpp::wrap(retr_poblq(X, oblq_factors));
    return rcpp_result_gen;
END_RCPP
}
// sl
Rcpp::List sl(arma::mat X, int n_generals, int n_groups, std::string cor, std::string estimator, std::string missing, Rcpp::Nullable<int> nobs, Rcpp::Nullable<Rcpp::List> first_efa, Rcpp::Nullable<Rcpp::List> second_efa, int cores);
RcppExport SEXP _bifactor_sl(SEXP XSEXP, SEXP n_generalsSEXP, SEXP n_groupsSEXP, SEXP corSEXP, SEXP estimatorSEXP, SEXP missingSEXP, SEXP nobsSEXP, SEXP first_efaSEXP, SEXP second_efaSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type n_generals(n_generalsSEXP);
    Rcpp::traits::input_parameter< int >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< std::string >::type cor(corSEXP);
    Rcpp::traits::input_parameter< std::string >::type estimator(estimatorSEXP);
    Rcpp::traits::input_parameter< std::string >::type missing(missingSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<int> >::type nobs(nobsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type first_efa(first_efaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type second_efa(second_efaSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    rcpp_result_gen = Rcpp::wrap(sl(X, n_generals, n_groups, cor, estimator, missing, nobs, first_efa, second_efa, cores));
    return rcpp_result_gen;
END_RCPP
}
// rotate
Rcpp::List rotate(arma::mat loadings, Rcpp::CharacterVector rotation, std::string projection, arma::vec gamma, arma::vec epsilon, arma::vec k, double w, Rcpp::Nullable<arma::mat> Target, Rcpp::Nullable<arma::mat> Weight, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> blocks, Rcpp::Nullable<arma::vec> block_weights, Rcpp::Nullable<arma::uvec> oblq_factors, std::string normalization, Rcpp::Nullable<Rcpp::List> rot_control, int random_starts, int cores);
RcppExport SEXP _bifactor_rotate(SEXP loadingsSEXP, SEXP rotationSEXP, SEXP projectionSEXP, SEXP gammaSEXP, SEXP epsilonSEXP, SEXP kSEXP, SEXP wSEXP, SEXP TargetSEXP, SEXP WeightSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP blocksSEXP, SEXP block_weightsSEXP, SEXP oblq_factorsSEXP, SEXP normalizationSEXP, SEXP rot_controlSEXP, SEXP random_startsSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type loadings(loadingsSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type rotation(rotationSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type k(kSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Target(TargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Weight(WeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiTarget(PhiTargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiWeight(PhiWeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> >::type blocks(blocksSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type block_weights(block_weightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type oblq_factors(oblq_factorsSEXP);
    Rcpp::traits::input_parameter< std::string >::type normalization(normalizationSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type rot_control(rot_controlSEXP);
    Rcpp::traits::input_parameter< int >::type random_starts(random_startsSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    rcpp_result_gen = Rcpp::wrap(rotate(loadings, rotation, projection, gamma, epsilon, k, w, Target, Weight, PhiTarget, PhiWeight, blocks, block_weights, oblq_factors, normalization, rot_control, random_starts, cores));
    return rcpp_result_gen;
END_RCPP
}
// efast
Rcpp::List efast(arma::mat X, int nfactors, std::string cor, std::string estimator, Rcpp::CharacterVector rotation, std::string projection, std::string missing, Rcpp::Nullable<int> nobs, Rcpp::Nullable<arma::mat> Target, Rcpp::Nullable<arma::mat> Weight, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> blocks, Rcpp::Nullable<arma::vec> block_weights, Rcpp::Nullable<arma::uvec> oblq_factors, arma::vec gamma, arma::vec epsilon, arma::vec k, double w, int random_starts, int cores, Rcpp::Nullable<arma::vec> init, Rcpp::Nullable<Rcpp::List> efa_control, Rcpp::Nullable<Rcpp::List> rot_control);
RcppExport SEXP _bifactor_efast(SEXP XSEXP, SEXP nfactorsSEXP, SEXP corSEXP, SEXP estimatorSEXP, SEXP rotationSEXP, SEXP projectionSEXP, SEXP missingSEXP, SEXP nobsSEXP, SEXP TargetSEXP, SEXP WeightSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP blocksSEXP, SEXP block_weightsSEXP, SEXP oblq_factorsSEXP, SEXP gammaSEXP, SEXP epsilonSEXP, SEXP kSEXP, SEXP wSEXP, SEXP random_startsSEXP, SEXP coresSEXP, SEXP initSEXP, SEXP efa_controlSEXP, SEXP rot_controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type nfactors(nfactorsSEXP);
    Rcpp::traits::input_parameter< std::string >::type cor(corSEXP);
    Rcpp::traits::input_parameter< std::string >::type estimator(estimatorSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type rotation(rotationSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
    Rcpp::traits::input_parameter< std::string >::type missing(missingSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<int> >::type nobs(nobsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Target(TargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Weight(WeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiTarget(PhiTargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiWeight(PhiWeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> >::type blocks(blocksSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type block_weights(block_weightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type oblq_factors(oblq_factorsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type k(kSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type random_starts(random_startsSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type init(initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type efa_control(efa_controlSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type rot_control(rot_controlSEXP);
    rcpp_result_gen = Rcpp::wrap(efast(X, nfactors, cor, estimator, rotation, projection, missing, nobs, Target, Weight, PhiTarget, PhiWeight, blocks, block_weights, oblq_factors, gamma, epsilon, k, w, random_starts, cores, init, efa_control, rot_control));
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
Rcpp::List bifactor(arma::mat X, int n_generals, int n_groups, std::string method, std::string cor, std::string estimator, std::string projection, std::string missing, Rcpp::Nullable<int> nobs, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> blocks, Rcpp::Nullable<arma::vec> block_weights, Rcpp::Nullable<arma::uvec> oblq_factors, Rcpp::Nullable<arma::mat> init_Target, int maxit, double cutoff, std::string normalization, double w, int random_starts, int cores, Rcpp::Nullable<arma::vec> init, Rcpp::Nullable<Rcpp::List> efa_control, Rcpp::Nullable<Rcpp::List> rot_control, Rcpp::Nullable<Rcpp::List> first_efa, Rcpp::Nullable<Rcpp::List> second_efa, bool verbose);
RcppExport SEXP _bifactor_bifactor(SEXP XSEXP, SEXP n_generalsSEXP, SEXP n_groupsSEXP, SEXP methodSEXP, SEXP corSEXP, SEXP estimatorSEXP, SEXP projectionSEXP, SEXP missingSEXP, SEXP nobsSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP blocksSEXP, SEXP block_weightsSEXP, SEXP oblq_factorsSEXP, SEXP init_TargetSEXP, SEXP maxitSEXP, SEXP cutoffSEXP, SEXP normalizationSEXP, SEXP wSEXP, SEXP random_startsSEXP, SEXP coresSEXP, SEXP initSEXP, SEXP efa_controlSEXP, SEXP rot_controlSEXP, SEXP first_efaSEXP, SEXP second_efaSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type n_generals(n_generalsSEXP);
    Rcpp::traits::input_parameter< int >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< std::string >::type method(methodSEXP);
    Rcpp::traits::input_parameter< std::string >::type cor(corSEXP);
    Rcpp::traits::input_parameter< std::string >::type estimator(estimatorSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
    Rcpp::traits::input_parameter< std::string >::type missing(missingSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<int> >::type nobs(nobsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiTarget(PhiTargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiWeight(PhiWeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> >::type blocks(blocksSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type block_weights(block_weightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type oblq_factors(oblq_factorsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type init_Target(init_TargetSEXP);
    Rcpp::traits::input_parameter< int >::type maxit(maxitSEXP);
    Rcpp::traits::input_parameter< double >::type cutoff(cutoffSEXP);
    Rcpp::traits::input_parameter< std::string >::type normalization(normalizationSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    Rcpp::traits::input_parameter< int >::type random_starts(random_startsSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type init(initSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type efa_control(efa_controlSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type rot_control(rot_controlSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type first_efa(first_efaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type second_efa(second_efaSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(bifactor(X, n_generals, n_groups, method, cor, estimator, projection, missing, nobs, PhiTarget, PhiWeight, blocks, block_weights, oblq_factors, init_Target, maxit, cutoff, normalization, w, random_starts, cores, init, efa_control, rot_control, first_efa, second_efa, verbose));
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
Rcpp::List se(Rcpp::List fit, Rcpp::Nullable<int> nobs, Rcpp::Nullable<arma::mat> X, std::string type, double eta);
RcppExport SEXP _bifactor_se(SEXP fitSEXP, SEXP nobsSEXP, SEXP XSEXP, SEXP typeSEXP, SEXP etaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::List >::type fit(fitSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<int> >::type nobs(nobsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type X(XSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    rcpp_result_gen = Rcpp::wrap(se(fit, nobs, X, type, eta));
    return rcpp_result_gen;
END_RCPP
}
// parallel
Rcpp::List parallel(arma::mat X, int nboot, std::string cor, std::string missing, Rcpp::Nullable<arma::vec> quant, bool mean, bool replace, Rcpp::Nullable<std::vector<std::string>> PA, bool hierarchical, Rcpp::Nullable<Rcpp::List> efa, int cores);
RcppExport SEXP _bifactor_parallel(SEXP XSEXP, SEXP nbootSEXP, SEXP corSEXP, SEXP missingSEXP, SEXP quantSEXP, SEXP meanSEXP, SEXP replaceSEXP, SEXP PASEXP, SEXP hierarchicalSEXP, SEXP efaSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type nboot(nbootSEXP);
    Rcpp::traits::input_parameter< std::string >::type cor(corSEXP);
    Rcpp::traits::input_parameter< std::string >::type missing(missingSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type quant(quantSEXP);
    Rcpp::traits::input_parameter< bool >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::vector<std::string>> >::type PA(PASEXP);
    Rcpp::traits::input_parameter< bool >::type hierarchical(hierarchicalSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type efa(efaSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    rcpp_result_gen = Rcpp::wrap(parallel(X, nboot, cor, missing, quant, mean, replace, PA, hierarchical, efa, cores));
    return rcpp_result_gen;
END_RCPP
}
// check_deriv
Rcpp::List check_deriv(arma::mat L, arma::mat Phi, arma::mat dL, arma::mat dP, Rcpp::CharacterVector rotation, std::string projection, Rcpp::Nullable<arma::mat> Target, Rcpp::Nullable<arma::mat> Weight, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> blocks, Rcpp::Nullable<arma::vec> block_weights, Rcpp::Nullable<arma::uvec> oblq_factors, arma::vec gamma, arma::vec epsilon, arma::vec k, double w);
RcppExport SEXP _bifactor_check_deriv(SEXP LSEXP, SEXP PhiSEXP, SEXP dLSEXP, SEXP dPSEXP, SEXP rotationSEXP, SEXP projectionSEXP, SEXP TargetSEXP, SEXP WeightSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP blocksSEXP, SEXP block_weightsSEXP, SEXP oblq_factorsSEXP, SEXP gammaSEXP, SEXP epsilonSEXP, SEXP kSEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type L(LSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type Phi(PhiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dL(dLSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dP(dPSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type rotation(rotationSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Target(TargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type Weight(WeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiTarget(PhiTargetSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::mat> >::type PhiWeight(PhiWeightSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> >::type blocks(blocksSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type block_weights(block_weightsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::uvec> >::type oblq_factors(oblq_factorsSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type epsilon(epsilonSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type k(kSEXP);
    Rcpp::traits::input_parameter< double >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(check_deriv(L, Phi, dL, dP, rotation, projection, Target, Weight, PhiTarget, PhiWeight, blocks, block_weights, oblq_factors, gamma, epsilon, k, w));
    return rcpp_result_gen;
END_RCPP
}
// polyfast
Rcpp::List polyfast(arma::mat X, std::string missing, const std::string acov, const std::string smooth, double min_eigval, const int nboot, const bool fit, const int cores);
RcppExport SEXP _bifactor_polyfast(SEXP XSEXP, SEXP missingSEXP, SEXP acovSEXP, SEXP smoothSEXP, SEXP min_eigvalSEXP, SEXP nbootSEXP, SEXP fitSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< std::string >::type missing(missingSEXP);
    Rcpp::traits::input_parameter< const std::string >::type acov(acovSEXP);
    Rcpp::traits::input_parameter< const std::string >::type smooth(smoothSEXP);
    Rcpp::traits::input_parameter< double >::type min_eigval(min_eigvalSEXP);
    Rcpp::traits::input_parameter< const int >::type nboot(nbootSEXP);
    Rcpp::traits::input_parameter< const bool >::type fit(fitSEXP);
    Rcpp::traits::input_parameter< const int >::type cores(coresSEXP);
    rcpp_result_gen = Rcpp::wrap(polyfast(X, missing, acov, smooth, min_eigval, nboot, fit, cores));
    return rcpp_result_gen;
END_RCPP
}
// cfa
Rcpp::List cfa(arma::vec parameters, std::vector<arma::mat> X, arma::ivec nfactors, arma::ivec nobs, std::vector<arma::mat> lambda, std::vector<arma::mat> phi, std::vector<arma::mat> psi, std::vector<arma::uvec> lambda_indexes, std::vector<arma::uvec> phi_indexes, std::vector<arma::uvec> psi_indexes, std::vector<arma::uvec> target_indexes, std::vector<arma::uvec> targetphi_indexes, std::vector<arma::uvec> targetpsi_indexes, Rcpp::CharacterVector cor, Rcpp::CharacterVector estimator, Rcpp::CharacterVector projection, Rcpp::CharacterVector missing, int random_starts, int cores, Rcpp::Nullable<Rcpp::List> control);
RcppExport SEXP _bifactor_cfa(SEXP parametersSEXP, SEXP XSEXP, SEXP nfactorsSEXP, SEXP nobsSEXP, SEXP lambdaSEXP, SEXP phiSEXP, SEXP psiSEXP, SEXP lambda_indexesSEXP, SEXP phi_indexesSEXP, SEXP psi_indexesSEXP, SEXP target_indexesSEXP, SEXP targetphi_indexesSEXP, SEXP targetpsi_indexesSEXP, SEXP corSEXP, SEXP estimatorSEXP, SEXP projectionSEXP, SEXP missingSEXP, SEXP random_startsSEXP, SEXP coresSEXP, SEXP controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type parameters(parametersSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::mat> >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::ivec >::type nfactors(nfactorsSEXP);
    Rcpp::traits::input_parameter< arma::ivec >::type nobs(nobsSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::mat> >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::mat> >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::mat> >::type psi(psiSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::uvec> >::type lambda_indexes(lambda_indexesSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::uvec> >::type phi_indexes(phi_indexesSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::uvec> >::type psi_indexes(psi_indexesSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::uvec> >::type target_indexes(target_indexesSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::uvec> >::type targetphi_indexes(targetphi_indexesSEXP);
    Rcpp::traits::input_parameter< std::vector<arma::uvec> >::type targetpsi_indexes(targetpsi_indexesSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type cor(corSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type estimator(estimatorSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type projection(projectionSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type missing(missingSEXP);
    Rcpp::traits::input_parameter< int >::type random_starts(random_startsSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type control(controlSEXP);
    rcpp_result_gen = Rcpp::wrap(cfa(parameters, X, nfactors, nobs, lambda, phi, psi, lambda_indexes, phi_indexes, psi_indexes, target_indexes, targetphi_indexes, targetpsi_indexes, cor, estimator, projection, missing, random_starts, cores, control));
    return rcpp_result_gen;
END_RCPP
}
// cfa_test
Rcpp::List cfa_test(arma::mat R, arma::mat lambda, arma::mat phi, arma::mat psi, arma::mat dlambda, arma::mat dphi, arma::mat dpsi, arma::mat W, std::string estimator, std::string projection);
RcppExport SEXP _bifactor_cfa_test(SEXP RSEXP, SEXP lambdaSEXP, SEXP phiSEXP, SEXP psiSEXP, SEXP dlambdaSEXP, SEXP dphiSEXP, SEXP dpsiSEXP, SEXP WSEXP, SEXP estimatorSEXP, SEXP projectionSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type phi(phiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type psi(psiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dlambda(dlambdaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dphi(dphiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type dpsi(dpsiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type W(WSEXP);
    Rcpp::traits::input_parameter< std::string >::type estimator(estimatorSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
    rcpp_result_gen = Rcpp::wrap(cfa_test(R, lambda, phi, psi, dlambda, dphi, dpsi, W, estimator, projection));
    return rcpp_result_gen;
END_RCPP
}
// count
std::vector<int> count(const std::vector<int>& X, const int n, const int max_X);
RcppExport SEXP _bifactor_count(SEXP XSEXP, SEXP nSEXP, SEXP max_XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const int >::type n(nSEXP);
    Rcpp::traits::input_parameter< const int >::type max_X(max_XSEXP);
    rcpp_result_gen = Rcpp::wrap(count(X, n, max_X));
    return rcpp_result_gen;
END_RCPP
}
// joint_frequency_table
std::vector<std::vector<int>> joint_frequency_table(const std::vector<int>& X, const int n, const int max_X, const std::vector<int>& Y, const int max_Y);
RcppExport SEXP _bifactor_joint_frequency_table(SEXP XSEXP, SEXP nSEXP, SEXP max_XSEXP, SEXP YSEXP, SEXP max_YSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<int>& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const int >::type n(nSEXP);
    Rcpp::traits::input_parameter< const int >::type max_X(max_XSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const int >::type max_Y(max_YSEXP);
    rcpp_result_gen = Rcpp::wrap(joint_frequency_table(X, n, max_X, Y, max_Y));
    return rcpp_result_gen;
END_RCPP
}
// dbinorm
double dbinorm(double p, double x, double y);
RcppExport SEXP _bifactor_dbinorm(SEXP pSEXP, SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type p(pSEXP);
    Rcpp::traits::input_parameter< double >::type x(xSEXP);
    Rcpp::traits::input_parameter< double >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(dbinorm(p, x, y));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_bifactor_asymptotic_general", (DL_FUNC) &_bifactor_asymptotic_general, 1},
    {"_bifactor_asymptotic_normal", (DL_FUNC) &_bifactor_asymptotic_normal, 1},
    {"_bifactor_asymptotic_elliptical", (DL_FUNC) &_bifactor_asymptotic_elliptical, 2},
    {"_bifactor_asymptotic_poly", (DL_FUNC) &_bifactor_asymptotic_poly, 3},
    {"_bifactor_smoothing", (DL_FUNC) &_bifactor_smoothing, 2},
    {"_bifactor_random_orth", (DL_FUNC) &_bifactor_random_orth, 2},
    {"_bifactor_random_oblq", (DL_FUNC) &_bifactor_random_oblq, 2},
    {"_bifactor_random_poblq", (DL_FUNC) &_bifactor_random_poblq, 3},
    {"_bifactor_retr_orth", (DL_FUNC) &_bifactor_retr_orth, 1},
    {"_bifactor_retr_oblq", (DL_FUNC) &_bifactor_retr_oblq, 1},
    {"_bifactor_retr_poblq", (DL_FUNC) &_bifactor_retr_poblq, 2},
    {"_bifactor_sl", (DL_FUNC) &_bifactor_sl, 10},
    {"_bifactor_rotate", (DL_FUNC) &_bifactor_rotate, 18},
    {"_bifactor_efast", (DL_FUNC) &_bifactor_efast, 24},
    {"_bifactor_get_target", (DL_FUNC) &_bifactor_get_target, 3},
    {"_bifactor_bifactor", (DL_FUNC) &_bifactor_bifactor, 27},
    {"_bifactor_asymp_cov", (DL_FUNC) &_bifactor_asymp_cov, 4},
    {"_bifactor_se", (DL_FUNC) &_bifactor_se, 5},
    {"_bifactor_parallel", (DL_FUNC) &_bifactor_parallel, 11},
    {"_bifactor_check_deriv", (DL_FUNC) &_bifactor_check_deriv, 17},
    {"_bifactor_polyfast", (DL_FUNC) &_bifactor_polyfast, 8},
    {"_bifactor_cfa", (DL_FUNC) &_bifactor_cfa, 20},
    {"_bifactor_cfa_test", (DL_FUNC) &_bifactor_cfa_test, 10},
    {"_bifactor_count", (DL_FUNC) &_bifactor_count, 3},
    {"_bifactor_joint_frequency_table", (DL_FUNC) &_bifactor_joint_frequency_table, 5},
    {"_bifactor_dbinorm", (DL_FUNC) &_bifactor_dbinorm, 3},
    {NULL, NULL, 0}
};

RcppExport void R_init_bifactor(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
