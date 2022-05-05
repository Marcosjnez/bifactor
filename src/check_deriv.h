Rcpp::List check_deriv(arma::mat L, arma::mat Phi,
                 arma::mat dL, arma::mat dP,
                 Rcpp::CharacterVector char_rotation,
                 std::string projection,
                 Rcpp::Nullable<arma::mat> nullable_Target,
                 Rcpp::Nullable<arma::mat> nullable_Weight,
                 Rcpp::Nullable<arma::mat> nullable_PhiTarget,
                 Rcpp::Nullable<arma::mat> nullable_PhiWeight,
                 Rcpp::Nullable<arma::uvec> nullable_blocks,
                 Rcpp::Nullable<std::vector<arma::uvec>> nullable_blocks_list,
                 Rcpp::Nullable<arma::vec> nullable_block_weights,
                 Rcpp::Nullable<arma::uvec> nullable_oblq_blocks,
                 std::string between_blocks,
                 arma::vec gamma, arma::vec epsilon, arma::vec k,
                 double w, double alpha, double a, double b) {

  std::vector<std::string> rotation = Rcpp::as<std::vector<std::string>>(char_rotation);

  // Structure of rotation arguments:

  arguments_rotate x;
  x.p = L.n_rows, x.q = L.n_cols;
  x.lambda = L;
  x.L = L;
  x.Phi = Phi;
  x.gamma = gamma, x.epsilon = epsilon, x.k = k, x.w = w,
    x.alpha = alpha, x.a = a, x.b = b;
  x.between_blocks = between_blocks;
  x.rotations = rotation;
  x.projection = projection;
  x.nullable_Target = nullable_Target;
  x.nullable_Weight = nullable_Weight;
  x.nullable_PhiTarget = nullable_PhiTarget;
  x.nullable_PhiWeight = nullable_PhiWeight;
  x.nullable_blocks = nullable_blocks;
  x.nullable_oblq_blocks = nullable_oblq_blocks;
  x.nullable_block_weights = nullable_block_weights;
  x.nullable_blocks_list = nullable_blocks_list;
  x.nullable_rot_control = R_NilValue;

  // Check rotation inputs and compute constants for rotation criteria:

  check_rotate(x, 1L, 1L);
  x.dL = dL; x.dP = dP;
  if(x.n_blocks > 1) {
    x.gL.set_size(x.p, x.q);
    x.gL.zeros();
    x.gP.set_size(x.q, x.q);
    x.gP.zeros();
    int pq = x.p*x.q;
    int qcor = x.q*(x.q-1)*0.5;
    x.hL.set_size(pq, pq);
    x.hL.zeros();
    x.hP.set_size(qcor, qcor);
    x.hP.zeros();
    x.dgL.set_size(x.p, x.q);
    x.dgL.zeros();
    x.dgP.set_size(x.q, x.q);
    x.dgP.zeros();
  }

  // Select one manifold:
  base_manifold* manifold = choose_manifold(x.projection);
  // Select one specific criteria or mixed criteria:
  base_criterion* criterion = choose_criterion(x.rotations, x.projection, x.blocks_list);

  x.T = arma::eye(x.q, x.q);

  criterion->F(x);

  criterion->gLP(x);

  criterion->dgLP(x);

  criterion->hLP(x);

  Rcpp::List result;
  result["f"] = x.f;
  result["gL"] = x.gL;
  result["gP"] = x.gP;
  result["dgL"] = x.dgL;
  result["dgP"] = x.dgP;
  result["hL"] = x.hL;
  result["hP"] = x.hP;
  result["gamma"] = x.gamma;

  return result;

}
