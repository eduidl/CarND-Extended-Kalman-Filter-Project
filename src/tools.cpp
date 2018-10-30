#include "tools.h"

#include <cmath>
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const std::vector<VectorXd> &estimations,
                              const std::vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if (estimations.size() != ground_truth.size() || estimations.empty()) {
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
    return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){
    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse / estimations.size();

  //calculate the squared root
  return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = std::pow(px, 2) + std::pow(py, 2);
  float c2 = std::pow(c1, 0.5);
  float c3 = std::pow(c1, 1.5);

  //check division by zero
  if (c1 < 0.0001){
    std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
    return Hj;
  }

  //compute the Jacobian matrix
  Hj << px / c2, py / c2, 0, 0,
        -py / c1, px / c1, 0, 0,
        py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3,
        px / c2, py / c2;

  return Hj;
}
