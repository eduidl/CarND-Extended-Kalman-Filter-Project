#include "tools.h"

#include <cmath>
#include <iostream>

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const std::vector<VectorXd> &estimations,
                              const std::vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() != ground_truth.size() || estimations.empty()) {
    std::cout << "Invalid estimation or ground_truth data" << std::endl;
    return rmse;
  }

  // accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];

    // coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  return rmse.array().sqrt();
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  MatrixXd Hj(3, 4);
  // recover state parameters
  const float px = x_state(0);
  const float py = x_state(1);
  const float vx = x_state(2);
  const float vy = x_state(3);

  // pre-compute a set of terms to avoid repeated calculation
  const float c1 = std::pow(px, 2) + std::pow(py, 2);
  const float c2 = std::sqrt(c1);
  const float c3 = c1 * c2;

  // check division by zero
  if (c1 < 0.0001) {
    std::cout << "CalculateJacobian () - Error - Division by Zero" << std::endl;
    return Hj;
  }

  // compute the Jacobian matrix
  Hj << px / c2, py / c2, 0, 0, -py / c1, px / c1, 0, 0,
      py * (vx * py - vy * px) / c3, px * (px * vy - py * vx) / c3, px / c2,
      py / c2;

  return Hj;
}
