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
Rcpp::List sl(arma::mat R, int n_generals, int n_groups, Rcpp::Nullable<int> nobs, Rcpp::Nullable<Rcpp::List> first_efa, Rcpp::Nullable<Rcpp::List> second_efa);
RcppExport SEXP _bifactor_sl(SEXP RSEXP, SEXP n_generalsSEXP, SEXP n_groupsSEXP, SEXP nobsSEXP, SEXP first_efaSEXP, SEXP second_efaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type n_generals(n_generalsSEXP);
    Rcpp::traits::input_parameter< int >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<int> >::type nobs(nobsSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type first_efa(first_efaSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type second_efa(second_efaSEXP);
    rcpp_result_gen = Rcpp::wrap(sl(R, n_generals, n_groups, nobs, first_efa, second_efa));
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
Rcpp::List efast(arma::mat R, int nfactors, std::string method, Rcpp::CharacterVector rotation, std::string projection, Rcpp::Nullable<int> nobs, Rcpp::Nullable<arma::mat> Target, Rcpp::Nullable<arma::mat> Weight, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> blocks, Rcpp::Nullable<arma::vec> block_weights, Rcpp::Nullable<arma::uvec> oblq_factors, arma::vec gamma, arma::vec epsilon, arma::vec k, double w, int random_starts, int cores, Rcpp::Nullable<arma::vec> init, Rcpp::Nullable<Rcpp::List> efa_control, Rcpp::Nullable<Rcpp::List> rot_control);
RcppExport SEXP _bifactor_efast(SEXP RSEXP, SEXP nfactorsSEXP, SEXP methodSEXP, SEXP rotationSEXP, SEXP projectionSEXP, SEXP nobsSEXP, SEXP TargetSEXP, SEXP WeightSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP blocksSEXP, SEXP block_weightsSEXP, SEXP oblq_factorsSEXP, SEXP gammaSEXP, SEXP epsilonSEXP, SEXP kSEXP, SEXP wSEXP, SEXP random_startsSEXP, SEXP coresSEXP, SEXP initSEXP, SEXP efa_controlSEXP, SEXP rot_controlSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type nfactors(nfactorsSEXP);
    Rcpp::traits::input_parameter< std::string >::type method(methodSEXP);
    Rcpp::traits::input_parameter< Rcpp::CharacterVector >::type rotation(rotationSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
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
    rcpp_result_gen = Rcpp::wrap(efast(R, nfactors, method, rotation, projection, nobs, Target, Weight, PhiTarget, PhiWeight, blocks, block_weights, oblq_factors, gamma, epsilon, k, w, random_starts, cores, init, efa_control, rot_control));
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
Rcpp::List bifactor(arma::mat R, int n_generals, int n_groups, std::string bifactor_method, std::string method, std::string projection, Rcpp::Nullable<int> nobs, Rcpp::Nullable<arma::mat> PhiTarget, Rcpp::Nullable<arma::mat> PhiWeight, Rcpp::Nullable<std::vector<std::vector<arma::uvec>>> blocks, Rcpp::Nullable<arma::vec> block_weights, Rcpp::Nullable<arma::uvec> oblq_factors, Rcpp::Nullable<arma::mat> init_Target, int maxit, double cutoff, std::string normalization, double w, int random_starts, int cores, Rcpp::Nullable<arma::vec> init, Rcpp::Nullable<Rcpp::List> efa_control, Rcpp::Nullable<Rcpp::List> rot_control, Rcpp::Nullable<Rcpp::List> first_efa, Rcpp::Nullable<Rcpp::List> second_efa, bool verbose);
RcppExport SEXP _bifactor_bifactor(SEXP RSEXP, SEXP n_generalsSEXP, SEXP n_groupsSEXP, SEXP bifactor_methodSEXP, SEXP methodSEXP, SEXP projectionSEXP, SEXP nobsSEXP, SEXP PhiTargetSEXP, SEXP PhiWeightSEXP, SEXP blocksSEXP, SEXP block_weightsSEXP, SEXP oblq_factorsSEXP, SEXP init_TargetSEXP, SEXP maxitSEXP, SEXP cutoffSEXP, SEXP normalizationSEXP, SEXP wSEXP, SEXP random_startsSEXP, SEXP coresSEXP, SEXP initSEXP, SEXP efa_controlSEXP, SEXP rot_controlSEXP, SEXP first_efaSEXP, SEXP second_efaSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type n_generals(n_generalsSEXP);
    Rcpp::traits::input_parameter< int >::type n_groups(n_groupsSEXP);
    Rcpp::traits::input_parameter< std::string >::type bifactor_method(bifactor_methodSEXP);
    Rcpp::traits::input_parameter< std::string >::type method(methodSEXP);
    Rcpp::traits::input_parameter< std::string >::type projection(projectionSEXP);
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
    rcpp_result_gen = Rcpp::wrap(bifactor(R, n_generals, n_groups, bifactor_method, method, projection, nobs, PhiTarget, PhiWeight, blocks, block_weights, oblq_factors, init_Target, maxit, cutoff, normalization, w, random_starts, cores, init, efa_control, rot_control, first_efa, second_efa, verbose));
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
Rcpp::List parallel(arma::mat X, int n_boot, std::string type, Rcpp::Nullable<arma::vec> quant, bool mean, bool replace, Rcpp::Nullable<std::vector<std::string>> PA, bool hierarchical, Rcpp::Nullable<Rcpp::List> efa, int cores);
RcppExport SEXP _bifactor_parallel(SEXP XSEXP, SEXP n_bootSEXP, SEXP typeSEXP, SEXP quantSEXP, SEXP meanSEXP, SEXP replaceSEXP, SEXP PASEXP, SEXP hierarchicalSEXP, SEXP efaSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type n_boot(n_bootSEXP);
    Rcpp::traits::input_parameter< std::string >::type type(typeSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<arma::vec> >::type quant(quantSEXP);
    Rcpp::traits::input_parameter< bool >::type mean(meanSEXP);
    Rcpp::traits::input_parameter< bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<std::vector<std::string>> >::type PA(PASEXP);
    Rcpp::traits::input_parameter< bool >::type hierarchical(hierarchicalSEXP);
    Rcpp::traits::input_parameter< Rcpp::Nullable<Rcpp::List> >::type efa(efaSEXP);
    Rcpp::traits::input_parameter< int >::type cores(coresSEXP);
    rcpp_result_gen = Rcpp::wrap(parallel(X, n_boot, type, quant, mean, replace, PA, hierarchical, efa, cores));
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
// poly
Rcpp::List poly(const arma::mat& X, const int cores);
RcppExport SEXP _bifactor_poly(SEXP XSEXP, SEXP coresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< const int >::type cores(coresSEXP);
    rcpp_result_gen = Rcpp::wrap(poly(X, cores));
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
    {"_bifactor_sl", (DL_FUNC) &_bifactor_sl, 6},
    {"_bifactor_rotate", (DL_FUNC) &_bifactor_rotate, 18},
    {"_bifactor_efast", (DL_FUNC) &_bifactor_efast, 22},
    {"_bifactor_get_target", (DL_FUNC) &_bifactor_get_target, 3},
    {"_bifactor_bifactor", (DL_FUNC) &_bifactor_bifactor, 25},
    {"_bifactor_asymp_cov", (DL_FUNC) &_bifactor_asymp_cov, 4},
    {"_bifactor_se", (DL_FUNC) &_bifactor_se, 5},
    {"_bifactor_parallel", (DL_FUNC) &_bifactor_parallel, 10},
    {"_bifactor_cv_eigen", (DL_FUNC) &_bifactor_cv_eigen, 5},
    {"_bifactor_check_deriv", (DL_FUNC) &_bifactor_check_deriv, 17},
    {"_bifactor_poly", (DL_FUNC) &_bifactor_poly, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_bifactor(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
