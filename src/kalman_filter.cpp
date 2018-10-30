#include "kalman_filter.h"

#include <algorithm>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

#include <iostream>

void KalmanFilter::Predict() {
  std::cout << F_.rows() << F_.cols() << std::endl;
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::CommonUpdate(const VectorXd &y, const VectorXd &z) {
  MatrixXd PHt = P_ * H_.transpose();
  MatrixXd S = H_ * PHt + R_;
  MatrixXd K = PHt * S.inverse();

  x_ = x_ + K * y;
  auto size = x_.size();
  MatrixXd I = MatrixXd::Identity(size, size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  CommonUpdate(y, z);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = std::max(x_(0), 0.0001);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  VectorXd h(3);

  h(0) = std::sqrt(std::pow(px, 2) + std::pow(py, 2));
  h(1) = atan2(py, px);
  h(2) = (px * vx + py * vy) / h(0);

  VectorXd y = z - h;

  while (y(1) < -M_PI) {
    y(1) += M_PI * 2;
  }
  while (y(1) > M_PI) {
    y(1) -= M_PI * 2;
  }
  
  CommonUpdate(y, z);
}
