// src/kalman_filter.hpp

#ifndef KALMAN_FILTER_HPP
#define KALMAN_FILTER_HPP

#include <Eigen/Dense>

class KalmanFilter {
public:
    KalmanFilter(int state_dim, int meas_dim);

    void initialize(const Eigen::VectorXd& x0, const Eigen::MatrixXd& P0);
    void setProcessNoise(const Eigen::MatrixXd& Q);
    void setMeasurementNoise(const Eigen::MatrixXd& R);
    void setModel(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& H);

    void predict(const Eigen::VectorXd& u);
    void update(const Eigen::VectorXd& z);

    Eigen::VectorXd getState() const;

private:
    int state_dim_, meas_dim_;
    Eigen::VectorXd x_;     // State estimate
    Eigen::MatrixXd P_;     // Estimate covariance
    Eigen::MatrixXd Q_;     // Process noise covariance
    Eigen::MatrixXd R_;     // Measurement noise covariance
    Eigen::MatrixXd A_, B_, H_; // Model matrices
};

#endif // KALMAN_FILTER_HPP
