#include <thread>
#include <deque>
#include <array>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <string>

#include <filesystem>
#include <fstream>

#include "ros/ros.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <unistd.h> // For read function
#include <fcntl.h>  // For fcntl function
#include <iostream> // For error output
#include "std_msgs/Float64MultiArray.h"

#include <iomanip>
#include <sstream>

#include <numeric>
#include "log4z.h"

using namespace zsummer::log4z;
const std::string model_path = "/home/xbot/policy_1.pt"; 

constexpr int CONTROL_FREQUENCY = 100;
constexpr int YAW_HIST_LENGTH = 50;
constexpr int N_POLICY_OBS = 47;
constexpr int N_HIST_LEN = 15;

constexpr std::array<double, 3> GRAV = {0.0, 0.0, -1.0};

constexpr int N_JOINTS = 30;
constexpr int N_HAND_JOINTS = 18; // left arm(7) + neck(2) + right arm(7) + waist(2)
constexpr int N_LEG_JOINTS = 12;  // left leg(6) + right leg(6)

constexpr int N_SINGLE_INS_HAND_JOINT = 6;

const float ins_hand_upper_from_thumb[12] = {2.878, 0.9878, 3.0718, 3.0718, 3.0718, 3.0718, 2.878, 0.9878, 3.0718, 3.0718, 3.0718, 3.0718}; // 上限，从大拇指开始，左手在前
const float ins_hand_lower_from_thumb[12] = {1.58825, -0.2094, 0.349, 0.349, 0.349, 0.349, 1.58825, -0.2094, 0.349, 0.349, 0.349, 0.349};   // 下限，从大拇指开始，左手在前
bool handless_ = false;
void handsNormalize(const std::vector<double> &hand_pos, std::vector<double> &hand_pos_norm)
{
  if ((hand_pos.size() != 12) || (hand_pos_norm.size() != 12))
  {
    std::cout << "hand_pos.size():" << hand_pos.size() << std::endl;
    std::cout << "hand_pos_norm.size() :" << hand_pos_norm.size() << std::endl;
    std::cout << "err:[handsNormalize] hand_pos.size() != 12) || (hand_pos_norm.size() ! = 12)" << std::endl;
    return;
  }
  for (int j = 0; j < 12; j++)
  {
    hand_pos_norm[j] = (hand_pos[j] - ins_hand_lower_from_thumb[j]) / (ins_hand_upper_from_thumb[j] - ins_hand_lower_from_thumb[j]);
  }
  return;
}

void handsDenormalize(const std::vector<double> &hand_pos_norm, std::vector<double> &hand_pos)
{
  if ((hand_pos_norm.size() != 12) || (hand_pos.size() != 12))
  {
    std::cout << "hand_pos_norm.size():" << hand_pos_norm.size() << std::endl;
    std::cout << "hand_pos.size() :" << hand_pos.size() << std::endl;
    std::cout << "err:[handsDenormalize] hand_pos_norm.size() != 12) || (hand_pos.size() ! = 12)" << std::endl;
    return;
  }

  for (int i = 0; i < 12; i++)
  {
    hand_pos[i] = hand_pos_norm[i] * (ins_hand_upper_from_thumb[i] - ins_hand_lower_from_thumb[i]) + ins_hand_lower_from_thumb[i];
  }
  return;
}


struct Command
{
  // 1 for moving, 0 for standing used for yaw while stand
  double x, y, yaw, cycle_time, move;
  Command(double _x = 0.0, double _y = 0.0, double _yaw = 0.0, double _cycle_time = 0.64, double _move = 0.) : x(_x), y(_y), yaw(_yaw), cycle_time(_cycle_time), move(_move) {}
} user_cmd;

std::mutex cmd_mutex;

struct Quaternion
{
  double w, x, y, z;
};

struct EulerAngle
{
  double roll, pitch, yaw;
};

int global_time;
bool damping_mode;

std::vector<torch::jit::IValue> tensor;
torch::Tensor out;
torch::Tensor critic_obs;
torch::Tensor tmp_obs;
torch::jit::script::Module policy;

std::deque<std::vector<double>> hist_obs;
std::deque<double> hist_yaw;

std::ofstream obs_buffer;
std::ofstream critic_obs_buffer;

std::vector<double> measured_q_;
std::vector<double> measured_v_;
std::vector<double> measured_tau_;
std::array<double, 4> quat_est;
std::array<double, 3> angular_vel_local;
std::array<double, 3> imu_accel;
std::vector<double> measure_q_ins_hand_;
std::vector<double> cmd_ins_hand_pos;

EulerAngle QuaternionToEuler(const Quaternion &q)
{
  EulerAngle angles;

  // Roll (x-axis rotation)
  double sinr_cosp = 2 * (q.w * q.x + q.y * q.z);
  double cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y);
  angles.roll = std::atan2(sinr_cosp, cosr_cosp);

  // Pitch (y-axis rotation)
  double sinp = 2 * (q.w * q.y - q.z * q.x);
  if (std::abs(sinp) >= 1)
    angles.pitch = std::copysign(M_PI / 2, sinp); // Use 90 degrees if out of range
  else
    angles.pitch = std::asin(sinp);

  // Yaw (z-axis rotation)
  double siny_cosp = 2 * (q.w * q.z + q.x * q.y);
  double cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z);
  angles.yaw = std::atan2(siny_cosp, cosy_cosp);

  return angles;
}

template <typename T>
T clip(T value, T min, T max)
{
  return std::max(min, std::min(value, max));
}

std::array<double, 3> quat_rotate_inverse(const std::array<double, 4> &quat,
                                          const std::array<double, 3> &vel)
{
  double w;
  double x, y, z;
  double vx, vy, vz;
  double a1, a2, a3, b1, b2, b3, c1, c2, c3, tmp;
  std::array<double, 3> proj_vel;

  w = quat[0];
  x = quat[1];
  y = quat[2];
  z = quat[3];

  vx = vel[0];
  vy = vel[1];
  vz = vel[2];

  tmp = 2.0 * w * w - 1.0;
  a1 = vx * tmp;
  a2 = vy * tmp;
  a3 = vz * tmp;

  b1 = (y * vz - z * vy) * w * 2;
  b2 = (z * vx - x * vz) * w * 2;
  b3 = (x * vy - y * vx) * w * 2;

  tmp = (x * vx + y * vy + z * vz) * 2;
  c1 = x * tmp;
  c2 = y * tmp;
  c3 = z * tmp;

  proj_vel[0] = a1 - b1 + c1;
  proj_vel[1] = a2 - b2 + c2;
  proj_vel[2] = a3 - b3 + c3;

  return proj_vel;
}

bool checkJointLimit(const std::vector<double> &q)
{
  //   return true;

  for (int i = 0; i < N_JOINTS; i++)
  {
    if (i < N_HAND_JOINTS)
    {
      if ((i == 1 || i == 10) && (q[6 + i] < -0.05 || q[6 + i] > 0.05))
      {
        return false;
      }
      continue;
    }

  }
  return true;
}

void init()
{
  global_time = 0;
  damping_mode = false;

  tmp_obs = torch::zeros({N_POLICY_OBS * N_HIST_LEN});
  tensor.push_back(torch::zeros({N_POLICY_OBS * N_HIST_LEN}));
  policy = torch::jit::load(model_path.c_str());

  for (int i = 0; i < 100; i++)
  {
    tensor[0] = torch::zeros({N_POLICY_OBS * N_HIST_LEN}).unsqueeze(0);
    // out, critic_obs = policy.forward(tensor);
    auto output = policy.forward(tensor).toTensor().index({0});
  }

  hist_obs = std::deque<std::vector<double>>(N_HIST_LEN, std::vector<double>(N_POLICY_OBS, 0.0));
  // hist_yaw = std::deque<double>(100, 0.);

  // Get current time
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);


  cmd_ins_hand_pos.resize(2 * N_SINGLE_INS_HAND_JOINT);

  for (int i = 0; i < 2 * N_SINGLE_INS_HAND_JOINT; i++)
  {
    cmd_ins_hand_pos[i] = 1.0;
  }
  measure_q_ins_hand_.resize(2 * N_SINGLE_INS_HAND_JOINT);
  for (int i = 0; i < 2 * N_SINGLE_INS_HAND_JOINT; i++)
  {
    measure_q_ins_hand_[i] = 0.0;
  }
}

ros::Publisher pub;
ros::Publisher ins_hand_cmd_pub;
ros::Subscriber sub_ins_hand;
std_msgs::Float64MultiArray resiverMsg;
void callback(const std_msgs::Float64MultiArray::ConstPtr &msg);
void ins_hand_callback(const std_msgs::Float64MultiArray::ConstPtr &msg);

void sendHandCommand(const std::vector<double> &pos_des_norm)
{
  // hand state sequence : 0~12
  // left thumb 0, left thumb 1,  left  index 1,  left middle 0, left ring 0, left pinky 0
  // right thumb 0 ,right thumb 1, right  index 1,right middle 0,right ring 0, right pinky 0
  if (pos_des_norm.size() != 2 * N_SINGLE_INS_HAND_JOINT)
  {
    std::cout << "Invalid hand command size:" << pos_des_norm.size() << "!= " << 2 * N_SINGLE_INS_HAND_JOINT << std::endl;
    return;
  }
  std::vector<double> pos_des;
  pos_des.resize(2 * N_SINGLE_INS_HAND_JOINT);

  handsDenormalize(pos_des_norm, pos_des); // denormalize the hand command

  std_msgs::Float64MultiArray msg;
  // left hand
  for (int i = 0; i < N_SINGLE_INS_HAND_JOINT; i++)
  {
    msg.data.push_back(pos_des[i]);
  }
  // right hand
  for (int i = 0; i < N_SINGLE_INS_HAND_JOINT; i++)
  {
    msg.data.push_back(pos_des[N_SINGLE_INS_HAND_JOINT + i]);
  }
  ins_hand_cmd_pub.publish(msg);
  return;
}
void sendJointCommand(std::vector<double> &pos_des,
                      std::vector<double> &vel_des,
                      std::vector<double> &kp,
                      std::vector<double> &kd,
                      std::vector<double> &torque)
{
  std_msgs::Float64MultiArray msg;

  for (int i = 0; i < N_JOINTS; i++)
  {
    msg.data.push_back(pos_des[i]);
  }
  for (int i = 0; i < N_JOINTS; i++)
  {
    msg.data.push_back(vel_des[i]);
  }
  for (int i = 0; i < N_JOINTS; i++)
  {
    msg.data.push_back(kp[i]);
  }
  for (int i = 0; i < N_JOINTS; i++)
  {
    msg.data.push_back(kd[i]);
  }
  for (int i = 0; i < N_JOINTS; i++)
  {
    msg.data.push_back(torque[i]);
  }

  pub.publish(msg);
}

void update()
{

  Command local_cmd;
  {
    std::lock_guard<std::mutex> lock(cmd_mutex);
    local_cmd = user_cmd; // Copy the shared command to a local variable
  }

  constexpr double PI = 3.14159265358979323846;

  constexpr int STARTUP_TIME = 100;
  constexpr int INITIAL_TIME = 500;
  constexpr int START_TO_WALK_TIME = 2000;

  static std::array<double, N_LEG_JOINTS> last_action;
  static std::vector<double> initial_pos;

  std::vector<double> pos_des, vel_des, kp, kd, torque;
  pos_des.resize(N_JOINTS);
  vel_des.resize(N_JOINTS);
  kp.resize(N_JOINTS);
  kd.resize(N_JOINTS);
  torque.resize(N_JOINTS);
  if(!handless_)
  {
    sendHandCommand(cmd_ins_hand_pos);
  }

  // static double base_yaw = 0.0;

  std::array<double, 3> proj_grav = quat_rotate_inverse(quat_est, GRAV);
  proj_grav[0] *= -1.;

  // if (global_time > INITIAL_TIME && (damping_mode || !checkJointLimit(measured_q_) || proj_grav[2] > -0.8))
  // {
  //   if (proj_grav[2] > -0.8)
  //   {
  //     std::cout << "grav " << proj_grav[2] << std::endl;
  //   }

  //   if(!damping_mode)
  //   {
  //     std::cout << "joint limit" << std::endl;
  //   }

  //   std::cout << "damping" << std::endl;
  //   damping_mode = true;

  //   // TODO(wyj): Check the scake of the Kd.
  //   for (int i = 0; i < N_JOINTS; i++)
  //   {
  //     kp[i] = 0.0;
  //     kd[i] = 2.0;
  //     pos_des[i] = 0.0;
  //     vel_des[i] = 0.0;
  //     torque[i] = 0.0;
  //   }

  //   sendJointCommand(pos_des, vel_des, kp, kd, torque);
  //   ros::shutdown();
  //   return;
  // }

  global_time++;

  // printf("GLOBAL TIME: %d\n", global_time);

  // Do nothing.
  if (global_time < STARTUP_TIME)
  {
    initial_pos = measured_q_;

    sendJointCommand(pos_des, vel_des, kp, kd, torque);
    return;
  }

  // Set to initial position.
  if (global_time < INITIAL_TIME)
  {
    for (int i = 0; i < N_HAND_JOINTS; i++)
    {
      kp[i] = 200.0;
      kd[i] = 10.0;
      pos_des[i] = initial_pos[6 + i] + double(global_time - STARTUP_TIME) / double(INITIAL_TIME - STARTUP_TIME) * (0.0 - initial_pos[6 + i]);
      vel_des[i] = 0.0;
      torque[i] = 0.0;
    }

    for (int i = 0; i < N_LEG_JOINTS; i++)
    {
      kp[N_HAND_JOINTS + i] = 300.0;
      kd[N_HAND_JOINTS + i] = 10.0;
      pos_des[N_HAND_JOINTS + i] = initial_pos[6 + N_HAND_JOINTS + i] + double(global_time - STARTUP_TIME) / double(INITIAL_TIME - STARTUP_TIME) * (0.0 - initial_pos[6 + N_HAND_JOINTS + i]);
      vel_des[N_HAND_JOINTS + i] = 0.0;
      torque[N_HAND_JOINTS + i] = 0.0;
    }

    sendJointCommand(pos_des, vel_des, kp, kd, torque);
    return;
  }

  if (global_time < START_TO_WALK_TIME)
  {

    EulerAngle euler = QuaternionToEuler(Quaternion{quat_est[0], quat_est[1], quat_est[2], quat_est[3]});
    // base_yaw += euler.yaw * -1.0;
    hist_yaw.push_back(euler.yaw * -1.0);

    if (hist_yaw.size() > YAW_HIST_LENGTH)
    {
      hist_yaw.pop_front();
    }

    for (int i = 0; i < N_HAND_JOINTS; i++)
    {
      kp[i] = 200.0;
      kd[i] = 10.0;
      pos_des[i] = 0.0;
      vel_des[i] = 0.0;
      torque[i] = 0.0;
    }

    for (int i = 0; i < N_LEG_JOINTS; i++)
    {
      kp[N_HAND_JOINTS + i] = 300.0;
      kd[N_HAND_JOINTS + i] = 10.;
      pos_des[N_HAND_JOINTS + i] = 0.0;
      vel_des[N_HAND_JOINTS + i] = 0.0;
      torque[N_HAND_JOINTS + i] = 0.0;
    }

    sendJointCommand(pos_des, vel_des, kp, kd, torque);
    return;
  }

  std::vector<double> curr_obs;
  curr_obs.reserve(N_POLICY_OBS);

  EulerAngle euler = QuaternionToEuler(Quaternion{quat_est[0], quat_est[1], quat_est[2], quat_est[3]});

  double cycle_time = local_cmd.cycle_time;
  curr_obs.push_back(sin(2 * PI * (global_time - START_TO_WALK_TIME) * (1.0 / CONTROL_FREQUENCY) / cycle_time));
  LOGFMTD("Received Obs[0] sin_p: %f", curr_obs[0]);
  curr_obs.push_back(cos(2 * PI * (global_time - START_TO_WALK_TIME) * (1.0 / CONTROL_FREQUENCY) / cycle_time));
  LOGFMTD("Received Obs[1] cos_p: %f", curr_obs[1]);
  if (local_cmd.move == 0)
  {
    // base_yaw = euler.yaw * -1.0;
    hist_yaw.push_back(euler.yaw * -1.0);
  }

  // double mean = sum / hist_yaw.size();
  if (hist_yaw.size() > YAW_HIST_LENGTH)
  {
    hist_yaw.pop_front();
  }
  double base_yaw = std::accumulate(hist_yaw.begin(), hist_yaw.end(), 0.0) / hist_yaw.size();

  double cmd_x = 2 * local_cmd.x;
  double cmd_y = 2 * local_cmd.y;
  double target_yaw_angular = -0.5 * ((euler.yaw * -1.0 - base_yaw) - local_cmd.yaw) * local_cmd.move; // heading command
  curr_obs.push_back(cmd_x);
  LOGFMTD("Received Obs[2] 2 * cmd_x: %f", curr_obs[2]);
  curr_obs.push_back(cmd_y);
  LOGFMTD("Received Obs[3] 2 * cmd_y: %f", curr_obs[3]);

  curr_obs.push_back(clip(target_yaw_angular, -0.4, 0.4));
  LOGFMTD("Received Obs[4] cliped target_yaw_angular: %f", curr_obs[4]);

  for (int i = 0; i < N_LEG_JOINTS; i++)
  {
    LOGFMTD("Received Obs[%d] measured_joint_pos[%d]: %f", 5 + i, i, curr_obs[5 + i]);
    curr_obs.push_back(1.0 * measured_q_[6 + N_HAND_JOINTS + i]);
  }

  for (int i = 0; i < N_LEG_JOINTS; i++)
  {
    LOGFMTD("Received Obs[%d] measured_joint_vel[%d]: %f", 5 + i + N_LEG_JOINTS, i, curr_obs[5 + i + N_LEG_JOINTS]);
    curr_obs.push_back(0.05 * clip(measured_v_[6 + N_HAND_JOINTS + i], -14., 14.));
  }

  for (int i = 0; i < N_LEG_JOINTS; i++)
  {
    LOGFMTD("Received Obs[%d] measured_joint_last_action[%d]: %f", 5 + i + N_LEG_JOINTS * 2, i, curr_obs[5 + i + N_LEG_JOINTS * 2]);
    curr_obs.push_back(last_action[i]);
  }

  for (int i = 0; i < 3; i++)
  {
    LOGFMTD("Received Obs[%d] angular_vel_local[%d]: %f", 5 + i + N_LEG_JOINTS * 3, i, curr_obs[5 + i + N_LEG_JOINTS * 3]);
    curr_obs.push_back(angular_vel_local[i] * (i > 0 ? -1 : 1));
  }
  curr_obs.push_back(euler.roll);
  LOGFMTD("Received Obs[%d] euler.roll: %f", 5 + 3 + N_LEG_JOINTS * 3, euler.roll);
  curr_obs.push_back(euler.pitch * -1.0);
  LOGFMTD("Received Obs[%d] euler.pitch: %f", 5 + 4 + N_LEG_JOINTS * 3, euler.pitch * -1.0);
  curr_obs.push_back(euler.yaw * -1.0 - base_yaw);
  LOGFMTD("Received Obs[%d] euler.yaw: %f", 5 + 5 + N_LEG_JOINTS * 3, euler.yaw * -1.0 - base_yaw);


  hist_obs.push_back(curr_obs);
  hist_obs.pop_front();

  for (int i = 0; i < (int)hist_obs.size(); i++)
  {
    for (int j = 0; j < (int)hist_obs[0].size(); j++)
    {
      obs_buffer << "," << hist_obs[i][j];
      tmp_obs.index_put_({i * (int)hist_obs[0].size() + j}, hist_obs[i][j]);
    }
  }
  obs_buffer << '\n';

  constexpr double MIN_CLIP = -18.0;
  constexpr double MAX_CLIP = 18.0;
  tensor[0] = tmp_obs.unsqueeze(0).clamp(MIN_CLIP, MAX_CLIP);
  // out, critic_obs = policy.forward(tensor);
  auto output = policy.forward(tensor);
  out = output.toTensor().index({0});
  for (int i = 0; i < N_LEG_JOINTS; i++)
  {
    double unclipped_action = out.index({i}).item().toFloat();
    double clipped_action = std::min(std::max(MIN_CLIP, unclipped_action), MAX_CLIP);
    last_action[i] = clipped_action;
  }


  for (int i = 0; i < N_HAND_JOINTS; i++)
  {
    kp[i] = 200.0;
    kd[i] = 10.0;
    pos_des[i] = 0.0;
    vel_des[i] = 0.0;
    torque[i] = 0.0;
    if (i == 16 || i == 17)
    {
      kp[i] = 400.0;
      kd[i] = 10.0;
    }
  }

  LOGD("Output action! +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
  for (int i = 0; i < N_LEG_JOINTS; i++)
  {
    kp[N_HAND_JOINTS + i] = 200.0;
    kd[N_HAND_JOINTS + i] = 10.0;
    pos_des[N_HAND_JOINTS + i] = 0.25 * last_action[i];
    LOGFMTD("leg joint[%d] pos_des:%f", N_HAND_JOINTS + i, pos_des[N_HAND_JOINTS + i]);
    vel_des[N_HAND_JOINTS + i] = 0.0;
    torque[N_HAND_JOINTS + i] = 0.0;
  }

  for (int i : {4, 5, 10, 11})
  {
    kp[N_HAND_JOINTS + i] = 15.0;
    kd[N_HAND_JOINTS + i] = 10.0;
  }

  for (int i : {2, 3, 8, 9})
  {
    kp[N_HAND_JOINTS + i] = 350.0;
    kd[N_HAND_JOINTS + i] = 10.0;
  }
  sendJointCommand(pos_des, vel_des, kp, kd, torque);
  return;
}

void processKeyboardInput(char ch)
{
  std::lock_guard<std::mutex> lock(cmd_mutex);
  if (ch == 'w')
  {
    user_cmd.x = std::min(0.6, user_cmd.x + 0.1);
    user_cmd.move = 1.;
  }
  if (ch == 's')
  {
    user_cmd.x = std::max(-0.4, user_cmd.x - 0.1);
    user_cmd.move = 1.;
  }
  if (ch == 'a')
  {
    user_cmd.y = std::min(0.5, user_cmd.y + 0.1);
    user_cmd.move = 1.;
  }
  if (ch == 'd')
  {
    user_cmd.y = std::max(-0.5, user_cmd.y - 0.1);
    user_cmd.move = 1.;
  }

  if (ch == 'r')
  {
    user_cmd.x = 0.;
    user_cmd.y = 0.;
    user_cmd.yaw = 0.;
    user_cmd.move = 0.;
    user_cmd.cycle_time = 0.64;
  }
  if (ch == 'n')
  {
    user_cmd.x += 0.01;
    user_cmd.y = 0.;
    user_cmd.yaw = 0.;
    user_cmd.move = 0.;
    user_cmd.cycle_time = 0.64;
  }
  if (ch == 'm')
  {
    user_cmd.x -= 0.01;
    user_cmd.y = 0.;
    user_cmd.yaw = 0.;
    user_cmd.move = 0.;
    user_cmd.cycle_time = 0.64;
  }
  if (ch == 'f')
  {
    user_cmd.x = 0.4;
    user_cmd.move = 1.;
  }
  if (ch == 't')
  {
    user_cmd.x = 0.45;
    user_cmd.move = 1.;
    user_cmd.cycle_time = 0.64;
  } // walk up stair
  if (ch == 'b')
  {
    user_cmd.x = -0.3;
    user_cmd.move = 1.;
  }

  if (ch == 'i')
  {
    user_cmd.cycle_time = std::min(1., user_cmd.cycle_time + 0.04);
    user_cmd.move = 1.;
  }
  if (ch == 'o')
  {
    user_cmd.cycle_time = std::max(0.6, user_cmd.cycle_time - 0.04);
    user_cmd.move = 1.;
  }

  if (ch == 'k')
  {
    user_cmd.yaw = std::min(3., user_cmd.yaw + 0.3);
    user_cmd.move = 1.;
  }
  if (ch == 'l')
  {
    user_cmd.yaw = std::max(-3., user_cmd.yaw - 0.3);
    user_cmd.move = 1.;
  }
}

void handleInput()
{

  system("stty raw -echo");
  fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);

  char ch;
  while (ros::ok())
  {
    if (read(STDIN_FILENO, &ch, 1) > 0)
    {
      processKeyboardInput(ch);
      if (ch == 'q')
        break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  ros::shutdown();
  system("stty cooked echo");
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "xbot_rl_guest_node");
  ROS_INFO("policy start loading");

  ILog4zManager::getRef().start();
  ILog4zManager::getRef().setLoggerLevel(LOG4Z_MAIN_LOGGER_ID, LOG_LEVEL_DEBUG);
  
  init();

  std_msgs::Float64MultiArray sendMsg;

  ros::NodeHandle nh;
  nh.getParam("/handless",handless_);
  pub = nh.advertise<std_msgs::Float64MultiArray>("/policy_input", 1);
  ros::Subscriber sub = nh.subscribe("/controllers/xbot_controller/policy_output", 1, callback, ros::TransportHints().tcpNoDelay());

  if(!handless_)
  {
    ins_hand_cmd_pub = nh.advertise<std_msgs::Float64MultiArray>("/ins_hand_cmd", 1);
    sub_ins_hand = nh.subscribe("/controllers/xbot_controller/ins_hand_state", 1, ins_hand_callback, ros::TransportHints().tcpNoDelay());
  }

  std::thread inputThread(handleInput);
  while (ros::ok())
  {
    auto start_time = std::chrono::steady_clock::now();
    update();

    auto duration = std::chrono::steady_clock::now() - start_time;
    auto micro_sec = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    int sleep_time = 1000000 / CONTROL_FREQUENCY - micro_sec;

    std::this_thread::sleep_for(std::chrono::microseconds(sleep_time));

    ros::spinOnce();
  }

  if (inputThread.joinable())
    inputThread.join();

  return 0;
}

void ins_hand_callback(const std_msgs::Float64MultiArray::ConstPtr &msg)
{

  if (msg == nullptr)
    return;

  if (msg->data.size() != 12)
  {
    std::cout << "err ins_hand_callback msg->data.size():" << msg->data.size() << "!= 12" << std::endl;
    return;
  }

  // hand state sequence : 0~12
  // left thumb 0, left thumb 1,  left  index 1,  left middle 0, left ring 0, left pinky 0
  // right thumb 0 ,right thumb 1, right  index 1,right middle 0,right ring 0, right pinky 0

  for (int i = 0; i < 12; i++)
  {
    measure_q_ins_hand_[i] = msg->data[i];
  }

  return;
}
void callback(const std_msgs::Float64MultiArray::ConstPtr &msg)
{


  measured_q_.clear();
  measured_v_.clear();
  measured_tau_.clear();
  LOGFMTD("Callback function called!");
  LOGFMTD("Received data size: %lu", msg->data.size());

  for (int i = 0; i < 6 + N_JOINTS; i++) // float base(6)  + jnt space
  {
    measured_q_.push_back(msg->data[i]);
  }

  LOGFMTD("Assigned measured_q_'s size: %ld", measured_q_.size());
  for (int i = 0; i < 6 + N_JOINTS; i++) // float base(6)  + jnt space
  {
    measured_v_.push_back(msg->data[6 + N_JOINTS + i]);
  }

  LOGFMTD("Assigned measured_v_'s size: %ld", measured_v_.size());
  for (int i = 0; i < 6 + N_JOINTS; i++) // float base(6)  + jnt space
  {
    measured_tau_.push_back(msg->data[(6 + N_JOINTS) * 2 + i]);
  }

  LOGFMTD("Assigned measured_tau_'s size: %ld", measured_q_.size());
  for (int i = 0; i < 4; i++) // quat_est
  {
    quat_est[i] = msg->data[(6 + N_JOINTS) * 3 + i];
  }

  LOGFMTD("Received quaternion: (%f, %f, %f, %f)", quat_est[0], quat_est[1], quat_est[2], quat_est[3]);

  for (int i = 0; i < 3; i++) // angular_vel_local
  {
    angular_vel_local[i] = msg->data[(6 + N_JOINTS) * 3 + 4 + i];
  }

  LOGFMTD("Received angular velocity: (%f, %f, %f)", angular_vel_local[0], angular_vel_local[1], angular_vel_local[2]);

  for (int i = 0; i < 3; i++) // imu_accel
  {
    imu_accel[i] = msg->data[(6 + N_JOINTS) * 3 + 4 + 3 + i];
  }
  LOGFMTD("Received imu_accel: (%f, %f, %f)", imu_accel[0], imu_accel[1], imu_accel[2]);
  
  return;
};
