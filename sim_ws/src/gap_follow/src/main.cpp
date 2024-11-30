#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "pid.h"
#include "kalman.hpp"
#include <math.h>
#include "helper.hpp"

using json = nlohmann::json;

constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

const double SPEED_LIMIT = 65.0;  // mph

int main()
{
  uWS::Hub h;

  // Define PID controllers
  PID position_pid;
  PID orientation_pid;
  PID velocity_pid;
  PID throttle_pid;

  position_pid.Init(0.1, 0.002, 0.2);
  orientation_pid.Init(0.1, 0.002, 0.2);
  velocity_pid.Init(0.1, 0.002, 0.2);
  throttle_pid.Init(0.12, 0.00001, 2.5);

  // Initialize Kalman Filter with dimensions for state and measurement
  int state_dim = 4;  // example: [x, y, v, phi]
  int meas_dim = 2;   // example: [measured_x, measured_y]
  KalmanFilter kf(state_dim, meas_dim);

  // Initial state and covariance
  Eigen::VectorXd x0(state_dim);
  x0 << 0, 0, 0, 0;  // Initial values for [x, y, v, phi]
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(state_dim, state_dim);
  kf.initialize(x0, P0);

  // Define the Kalman filter model matrices
  Eigen::MatrixXd A = Eigen::MatrixXd::Identity(state_dim, state_dim);
  Eigen::MatrixXd B = Eigen::MatrixXd::Zero(state_dim, 1);  // Adjust based on control inputs
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(meas_dim, state_dim);
  H(0, 0) = 1;  // Measurement of x
  H(1, 1) = 1;  // Measurement of y
  kf.setModel(A, B, H);

  // Process and measurement noise
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(state_dim, state_dim) * 0.1;
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(meas_dim, meas_dim) * 0.1;
  kf.setProcessNoise(Q);
  kf.setMeasurementNoise(R);

  h.onMessage([&position_pid, &orientation_pid, &velocity_pid, &throttle_pid, &kf](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data).substr(0, length));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // Get telemetry data
          double cte = std::stod(j[1]["cte"].get<std::string>());
          double speed = std::stod(j[1]["speed"].get<std::string>());
          double angle = std::stod(j[1]["steering_angle"].get<std::string>());
          double measured_x = /* Extract or calculate the measured x position */;
          double measured_y = /* Extract or calculate the measured y position */;

          // Prepare measurement vector
          Eigen::VectorXd z(meas_dim);
          z << measured_x, measured_y;

          // Kalman Filter Prediction and Update
          Eigen::VectorXd u(1); // Control input (e.g., throttle or steering angle)
          u << throttle_pid.TotalError();  // Example control input
          kf.predict(u);
          kf.update(z);
          Eigen::VectorXd estimated_state = kf.getState();

          // Define targets
          double target_x = /* target x */;
          double target_y = /* target y */;
          double target_speed = SPEED_LIMIT;
          double target_orientation = /* target orientation phi */;

          // Calculate errors
          double position_error = /* Compute position error */;
          double orientation_error = /* Compute orientation error */;

          // Position and orientation control
          position_pid.UpdateError(position_error);
          orientation_pid.UpdateError(orientation_error);
          double position_control = position_pid.TotalError();
          double orientation_control = orientation_pid.TotalError();
          double curvature_feedforward = /* Calculate curvature feedforward */;

          double steer_value = position_control + orientation_control + curvature_feedforward;
          steer_value = sigmoid(steer_value, 1.25, -1.25);

          // Speed control
          double speed_error = target_speed - speed;
          velocity_pid.UpdateError(speed_error);
          double throttle_value = velocity_pid.TotalError();
          throttle_value = sigmoid(throttle_value, 1.0, -1.0);

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = throttle_value;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
