#include "FusionEKF.h"

#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Constructor.
 */
FusionEKF::FusionEKF()
  : is_initialized_(false), previous_timestamp_(0),
    R_laser_(2, 2), R_radar_(3, 3), H_laser_(2, 4) {
  // initializing matrices
  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  // Set the process and measurement noises
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
}

/**
 * Destructor.
 */
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
     * Initialize the state ekf_.x_ with the first measurement.
     * Create the covariance matrix.
     */
    // first measurement
    std::cout << "EKF: " << std::endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    float px(0.0), py(0.0), vx(0.0), vy(0.0);

    if (measurement_pack.IsRADAR()) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      const float rho = measurement_pack.raw_measurements_[0];
      const float phi = measurement_pack.raw_measurements_[1];
      const float rho_dot = measurement_pack.raw_measurements_[2];

      px = rho * std::cos(phi);
      py = rho * std::sin(phi);
      vx = rho_dot * std::cos(phi);
      vy = rho_dot * std::sin(phi);
    } else if (measurement_pack.IsLASER()) {
      // Initialize state.
      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
    }

    VectorXd x(4);
    x << px, py, vx, vy;

    MatrixXd P(4, 4);
    P << 1, 0, 0, 0,
         0, 1, 0, 0,
         0, 0, 1000, 0,
         0, 0, 0, 1000;

    MatrixXd F(4, 4);
    F << 1, 0, 1, 0,
         0, 1, 0, 1,
         0, 0, 1, 0,
         0, 0, 0, 1;

    MatrixXd Q(4, 4);
    Q << 1, 0, 1, 0,
         0, 1, 0, 1,
         1, 0, 1, 0,
         0, 1, 0, 1;

    ekf_.Init(x, P, F, H_laser_, R_laser_, Q);
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   * Update the state transition matrix F according to the new elapsed time.
     - Time is measured in seconds.
   * Update the process noise covariance matrix.
   * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // compute the time elapsed between the current and previous measurements
  const float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;  // dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  const auto dt_2 = dt * dt;
  const auto dt_3 = dt_2 * dt;
  const auto dt_4 = dt_3 * dt;

  // Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //set the acceleration noise components
  const auto noise_ax = 9;
  const auto noise_ay = 9;

  // set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
             0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
             dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
             0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   * Use the sensor type to perform the update step.
   * Update the state and covariance matrices.
   */

  if (measurement_pack.IsRADAR()) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  std::cout << "x_ = " << ekf_.x_ << std::endl;
  std::cout << "P_ = " << ekf_.P_ << std::endl;
}
