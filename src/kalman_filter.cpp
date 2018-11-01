#include "kalman_filter.h"

// Please note that the Eigen library does not initialize
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(const VectorXd &x_in, const MatrixXd &P_in,
                        const MatrixXd &F_in, const MatrixXd &H_in,
                        const MatrixXd &R_in, const MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  I_ = MatrixXd::Identity(x_.size(), x_.size());
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::CommonUpdate(const VectorXd &y, const VectorXd &z) {
  MatrixXd PHt = P_ * H_.transpose();
  MatrixXd S = H_ * PHt + R_;
  MatrixXd K = PHt * S.inverse();

  x_ = x_ + K * y;
  P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_ * x_;
  CommonUpdate(y, z);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  float px = x_(0);
  const float py = x_(1);
  const float vx = x_(2);
  const float vy = x_(3);

  const float eps = 0.0001;
  if (px < eps) {
    px += eps;
  } else if (-eps < px && px < 0) {
    px -= eps;
  }

  const float rho = std::sqrt(px * px + py * py);
  const float phi = std::atan2(py, px);
  const float rho_dot = (px * vx + py * vy) / rho;

  VectorXd h(3);
  h << rho, phi, rho_dot;

  VectorXd y = z - h;

  while (y(1) < -M_PI) {
    y(1) += M_PI * 2;
  }
  while (y(1) > M_PI) {
    y(1) -= M_PI * 2;
  }

  CommonUpdate(y, z);
}
