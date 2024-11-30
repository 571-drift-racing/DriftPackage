#include "kalman.hpp"

KalmanFilter::KalmanFilter(int state_dim, int meas_dim)
    : state_dim_(state_dim), meas_dim_(meas_dim), x_(Eigen::VectorXd::Zero(state_dim)),
      P_(Eigen::MatrixXd::Identity(state_dim, state_dim)),
      Q_(Eigen::MatrixXd::Identity(state_dim, state_dim)),
      R_(Eigen::MatrixXd::Identity(meas_dim, meas_dim)),
      A_(Eigen::MatrixXd::Identity(state_dim, state_dim)),
      B_(Eigen::MatrixXd::Zero(state_dim, state_dim)),
      H_(Eigen::MatrixXd::Zero(meas_dim, state_dim)) {}

void KalmanFilter::initialize(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0) {
    x_ = x0;
    P_ = P0;
}

void KalmanFilter::setProcessNoise(const Eigen::MatrixXd& Q) {
    Q_ = Q;
}

void KalmanFilter::setMeasurementNoise(const Eigen::MatrixXd& R) {
    R_ = R;
}

void KalmanFilter::setModel(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& H) {
    A_ = A;
    B_ = B;
    H_ = H;
}

void KalmanFilter::predict(const Eigen::VectorXd& u) {
    x_ = A_ * x_ + B_ * u;
    P_ = A_ * P_ * A_.transpose() + Q_;
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    Eigen::MatrixXd K = P_ * H_.transpose() * (H_ * P_ * H_.transpose() + R_).inverse();
    x_ = x_ + K * (z - H_ * x_);
    P_ = (Eigen::MatrixXd::Identity(state_dim_, state_dim_) - K * H_) * P_;
}

Eigen::VectorXd KalmanFilter::getState() const {
    return x_;
}
