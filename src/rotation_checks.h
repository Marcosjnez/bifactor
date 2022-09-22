/*
 * Author: Marcos Jimenez
 * email: marcosjnezhquez@gmail.com
 * Modification date: 18/03/2022
 *
 */

// #include "structures.h"
// #include "auxiliary_checks.h"
// #include "auxiliary_manifolds.h"

void check_rotate(arguments_rotate& x, int random_starts, int cores) {

  x.n_rotations = x.rotations.size();

  // Create a list of column indexes for each block of factors:

  if(x.nullable_blocks.isNotNull() || x.nullable_blocks_list.isNotNull()) {

    if(x.nullable_blocks_list.isNull()) {

      // If the main input was a vector with the number of factors for each block,
      // transform into a list of nonoverlapping indexes for each block:

      x.blocks_vector = Rcpp::as<arma::uvec>(x.nullable_blocks);
      if(arma::accu(x.blocks_vector) > x.q) Rcpp::stop("To many factors declared in blocks");

      x.blocks_list = vector_to_list2(x.blocks_vector);

    } else {

      // If the main input was a list of indexes for each block of factors,
      // adjust the indexes to start from 0:

      x.blocks_list = Rcpp::as<std::vector<arma::uvec>>(x.nullable_blocks_list);
      arma::uvec v_blocks = list_to_vector(x.blocks_list);

      if(v_blocks.max() > x.q || v_blocks.min() < 1) {
        Rcpp::stop("blocks_list must contain values between 1 and the number of factors");
      }

      for(int i=0; i < x.blocks_list.size(); ++i) x.blocks_list[i] -= 1;

    }

    x.n_blocks = x.blocks_list.size();

    // Weights for each block of factors:

    if(x.nullable_block_weights.isNull()) {

      x.block_weights.set_size(x.n_blocks);
      x.block_weights.ones();

    } else {

      x.block_weights = Rcpp::as<arma::vec>(x.nullable_block_weights);

      if(x.block_weights.size() != x.n_blocks) {
        Rcpp::stop("The number of weights must equal the number of blocks");
      }

    }

    x.qi.resize(x.n_blocks), x.Li.resize(x.n_blocks), x.Li2.resize(x.n_blocks),
    x.Ni.resize(x.n_blocks), x.HLi2.resize(x.n_blocks), x.LoLi2.resize(x.n_blocks),
    x.termi.resize(x.n_blocks), x.IgCL2Ni.resize(x.n_blocks), x.f1i.resize(x.n_blocks),
    x.f2i.resize(x.n_blocks), x.I_gamma_Ci.resize(x.n_blocks),
    x.V.resize(x.n_blocks), x.vari.resize(x.n_blocks), x.varqi.resize(x.n_blocks),
    x.prodvari.resize(x.n_blocks), x.prodvarqi.resize(x.n_blocks),
    x.dvarqdLi.resize(x.n_blocks), x.dmudLi.resize(x.n_blocks), x.dc2dLi.resize(x.n_blocks),
    x.mui.resize(x.n_blocks), x.c2i.resize(x.n_blocks), x.dc2dmui.resize(x.n_blocks),
    x.prodc2i.resize(x.n_blocks), x.Hi.resize(x.n_blocks), x.dxtLi.resize(x.n_blocks),
    x.Ii.resize(x.n_blocks), x.dc2dLi.resize(x.n_blocks), x.dmudPi.resize(x.n_blocks),
    x.dc2dPi.resize(x.n_blocks), x.LtLxIi.resize(x.n_blocks), x.expmmui.resize(x.n_blocks),
    x.Phii.resize(x.n_blocks), x.dxtPi.resize(x.n_blocks), x.HL2i.resize(x.n_blocks);

    // Number of factors in each block (x.qi):
    for(int i=0; i < x.n_blocks; ++i) x.qi[i] = x.blocks_list[i].size();

    if(x.n_rotations == 1) {

      // Resize the vector of rotations of length one to match the length of blocks:
      x.rotations.resize(x.n_blocks, x.rotations[0]);

    } else {

      if(x.n_rotations != x.n_blocks) Rcpp::stop("The number of rotation criteria and blocks does not match. \n Provide either one rotation criteria for all the blocks or one for each block");

    }

  } else {

    if(x.n_rotations > 1) Rcpp::stop("Multiple rotation criteria specified but no block of factors was specified \n Provide a block for each rotation criteria");

  }

  // Check inputs for partially oblique projection:

  if(x.projection == "poblq") {

    if(x.nullable_oblq_blocks.isNull()) {

      Rcpp::stop("Please, provide a vector with the number of factors in each oblique block in the oblq_blocks argument");

    } else {

      // Create indexes to fix to zero the entries corresponding to the nonconstrained correlations:

      x.oblq_indexes = Rcpp::as<arma::uvec>(x.nullable_oblq_blocks);
      if(arma::accu(x.oblq_indexes) > x.q) Rcpp::stop("To many factors declared in oblq_blocks");
      x.list_oblq_indexes = vector_to_list2(x.oblq_indexes);

      arma::mat X(x.q, x.q, arma::fill::ones);
      arma::mat Q = zeros(X, x.list_oblq_indexes);
      x.oblq_indexes = arma::find(Q == 0);

    }

  }

  // Check and build defaults for each rotation criteria:

  std::vector<std::string> all_rotations = {"cf", "oblimin", "geomin", "target",
                                            "xtarget", "varimax", "varimin",
                                            "equavar", "simplix", "none"};

  for (auto i: x.rotations) {
    if (std::find(all_rotations.begin(), all_rotations.end(), i) == all_rotations.end()) {
      Rcpp::stop("Available rotations: \n cf, oblimin, geomin, varimax, varimin, target, xtarget, equavar, simplix");
    }
  }

  if(std::find(x.rotations.begin(), x.rotations.end(), "target") != x.rotations.end()) {

    if (x.nullable_Target.isNotNull()) {
      x.Target = Rcpp::as<arma::mat>(x.nullable_Target);
    } else {
      Rcpp::stop("Provide a Target for target rotation");
    }
    if (x.nullable_Weight.isNotNull()) {
      x.Weight = Rcpp::as<arma::mat>(x.nullable_Weight);
    } else {
      x.Weight = 1 - x.Target;
    }

    arma::mat dimcheck(x.p, x.q);
    if(x.n_blocks > 1) {
      // Find the position of the block corresponding to the target criteria and
      // resize dimcheck according to the number of factors specified in the target:
      int pos = std::find(x.rotations.begin(), x.rotations.end(), "target") - x.rotations.begin();
      dimcheck.set_size(x.p, x.qi[pos]);
    }

    if(arma::size(x.Target) != arma::size(dimcheck) ||
       arma::size(x.Weight) != arma::size(dimcheck)) {

      Rcpp::stop("Incompatible Target and Weight dimensions");

    }

    x.Weight2 = x.Weight % x.Weight;

  }

  if(std::find(x.rotations.begin(), x.rotations.end(), "xtarget") != x.rotations.end()) {

    if(x.w < 0) Rcpp::stop("w must be nonnegative");

    if (x.nullable_Target.isNotNull()) {
      x.Target = Rcpp::as<arma::mat>(x.nullable_Target);
    } else {
      Rcpp::stop("Provide a Target for xtarget rotation");
    }
    if (x.nullable_PhiTarget.isNotNull()) {
      x.Phi_Target = Rcpp::as<arma::mat>(x.nullable_PhiTarget);
    } else {
      Rcpp::stop("Provide a PhiTarget for xtarget rotation");
    }
    if (x.nullable_Weight.isNotNull()) {
      x.Weight = Rcpp::as<arma::mat>(x.nullable_Weight);
    } else {
      x.Weight = 1-x.Target;
    }
    if (x.nullable_PhiWeight.isNotNull()) {
      x.Phi_Weight = Rcpp::as<arma::mat>(x.nullable_PhiWeight);
    } else {
      x.Phi_Weight = 1-x.Phi_Target;
    }

    arma::mat dimcheck1(x.p, x.q);
    arma::mat dimcheck2(x.q, x.q);
    if(x.n_blocks > 1) {
      // Find the position of the block corresponding to the target criteria and
      // resize dimcheck according to the number of factors specified in the target:
      int pos = std::find(x.rotations.begin(), x.rotations.end(), "target") - x.rotations.begin();
      dimcheck1.set_size(x.p, x.qi[pos]);
      dimcheck2.set_size(x.qi[pos], x.qi[pos]);
    }

    if(arma::size(x.Target) != arma::size(dimcheck1) ||
       arma::size(x.Weight) != arma::size(dimcheck1) ||
       arma::size(x.Phi_Target) != arma::size(dimcheck2) ||
       arma::size(x.Phi_Weight) != arma::size(dimcheck2)) {

      Rcpp::stop("Incompatible dimensions between Target, PhiTarget, Weight or PhiWeight");

    }

    x.Phi_Weight.diag().zeros();
    x.Weight2 = x.Weight % x.Weight;
    x.Phi_Weight2 = x.Phi_Weight % x.Phi_Weight;

  }

  if(std::find(x.rotations.begin(), x.rotations.end(), "geomin") != x.rotations.end()) {

    if(x.epsilon.min() < 0) {

      Rcpp::stop("epsilon must be nonnegative");

    }

    x.epsilon = new_k(x.rotations, "geomin", x.epsilon);

  }

  if(std::find(x.rotations.begin(), x.rotations.end(), "oblimin") != x.rotations.end()) {

    if(x.gamma.min() < 0) {

      Rcpp::stop("gamma must be nonnegative");

    }

    x.N.set_size(x.q, x.q); x.N.ones();
    x.N.diag(0).zeros();
    x.gamma = new_k(x.rotations, "oblimin", x.gamma);

    arma::mat I(x.p, x.p, arma::fill::eye), gamma_C(x.p, x.p, arma::fill::ones);

    if(!x.blocks_list.empty()) {
      for(int i=0; i < x.n_blocks; ++i) { // if(!x.gamma[i].empty())
        gamma_C *= (x.gamma[i]/x.p);
        x.I_gamma_Ci[i] = (I - gamma_C);
      }
    } else {
      gamma_C *= (x.gamma[0]/x.p);
      x.I_gamma_C = (I - gamma_C);
    }

  }
  if(std::find(x.rotations.begin(), x.rotations.end(), "cf") != x.rotations.end()) {

    x.N.set_size(x.q, x.q); x.N.ones();
    x.N.diag(0).zeros();
    x.M.set_size(x.p, x.p); x.M.ones();
    x.M.diag(0).zeros();

    x.k = new_k(x.rotations, "cf", x.k);

  }
  if(std::find(x.rotations.begin(), x.rotations.end(), "varimax") != x.rotations.end()) {

    if((x.projection == "oblq") | (x.projection == "poblq")) {
      Rcpp::warning("Usually, the varimax criterion does not converge with (partially) oblique projection. \n Consider using cf with k = 1/(number of items), which is equivalent to varimax for orthogonal rotation but also converges with (partially) oblique projection.");
    }

    // if(!x.blocks_list.empty()) {
    //   for(int i=0; i < x.n_blocks; ++i) {
    //     arma::vec v(x.p, arma::fill::ones);
    //     arma::mat I(x.p, x.p, arma::fill::eye);
    //     x.Hi[i] = I - v * v.t() / (x.p + 0.0); // Centering matrix
    //   }
    // } else {
      arma::vec v(x.p, arma::fill::ones);
      arma::mat I(x.p, x.p, arma::fill::eye);
      x.H = I - v * v.t() / (x.p + 0.0); // Centering matrix
    // }

  }
  if(std::find(x.rotations.begin(), x.rotations.end(), "varimin") != x.rotations.end()) {

    arma::vec v(x.p, arma::fill::ones);
    arma::mat I(x.p, x.p, arma::fill::eye);
    x.H = I - v * v.t() / (x.p + 0.0); // Centering matrix

  }

  if(x.rotations[0] == "simplix") {

    // For oblique rotation:

    if(!x.blocks_list.empty()) {
      for(int i=0; i < x.blocks_list.size(); ++i) {

        arma::uvec indexes = x.blocks_list[i];
        int q = indexes.size();
        x.Hi[i] = diagg(q);
        x.Ii[i] = arma::eye(q, q);
        x.dxtLi[i] = dxt(x.p, q);
        x.dxtPi[i] = dxt(q, q);

      }
    } else {

      x.H = diagg(x.q);
      x.I = arma::eye(x.q, x.q);
      x.dxtL = dxt(x.p, x.q);
      x.dxtP = dxt(x.q, x.q);

    }

    x.dP = arma::zeros(x.q, x.q);
    x.gP = arma::zeros(x.q, x.q);
    x.dgP = arma::zeros(x.q, x.q);

  }
  if(x.rotations[0] == "none") {

    // do nothing

  }

  if(x.between_blocks == "TL") {

    if(x.nullable_between_blocks_list.isNull()) {
      Rcpp::stop("Between-blocks criteria used but the between_blocks_list argument is empty");
    } else {
      x.between_blocks_list = Rcpp::as<std::vector<arma::uvec>>(x.nullable_between_blocks_list);
    }

    x.between = true;

    if(!x.blocks_list.empty()) {

      if(x.n_blocks < 2) {
        Rcpp::stop("between_blocks can only be used with more than one block of factors");
      }

    } else {
      Rcpp::stop("between_blocks can only be used with more than one block of factors");
    }

  } else if(x.between_blocks != "none") {

    Rcpp::stop("Unkown between_blocks. Available between-blocks criteria: 'TL'");

  }

  // Check rotation optimization parameters:

  Rcpp::List rot_control;


  if (x.nullable_rot_control.isNotNull()) {

    rot_control = Rcpp::as<Rcpp::List>(x.nullable_rot_control);

  }

  if(rot_control.containsElementNamed("maxit")) {

    x.maxit = rot_control["maxit"];

  }
  if(rot_control.containsElementNamed("eps")) {

    x.eps = rot_control["eps"];

  }
  if(rot_control.containsElementNamed("optim")) {

    std::string optim = rot_control["optim"];
    x.optim = optim;

  }

  // Check parallelization setup:

  if(random_starts < 1) Rcpp::stop("random_starts must be a positive integer");
  if(cores < 1) Rcpp::stop("The number of cores must be a positive integer");

}
