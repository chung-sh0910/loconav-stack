#!/usr/bin/env python3
"""
ROS2 /cmd_ctr → LocomotionNode → NNPolicyController → MuJoCo 시각화

외부 motion_controller(with_ros2.py)가 /cmd_ctr 를 publish하면
LocomotionNode 가 이를 수신해 ONNX policy를 호출하고 MuJoCo 로 제어한다.

실행 순서:
    # 터미널 1 — 이 스크립트
    python3 tools/run_ros2_mujoco_demo.py

    # 터미널 2 — 외부 command publisher (motion_controller 또는 직접)
    ros2 topic pub /cmd_ctr geometry_msgs/TwistStamped \
        "{twist: {linear: {x: 0.3}}}" --rate 20

    # 또는 motion_controller 실행
    cd /home/csh/smll_project/tools/motion_controller
    python3 with_ros2.py --profile_path default_profile.yaml

주의:
    /cmd_ctr 가 0.5 초 안에 도착하지 않으면 LocomotionNode 의 watchdog 이
    step() 호출을 막으므로 ONNX 추론이 실행되지 않는다.
"""
import sys
import threading
import time

# ── DDS mock ─────────────────────────────────────────────────────────────
from unittest.mock import MagicMock

def _mock(name, **attrs):
    sys.modules.setdefault(name, MagicMock(**attrs))

for _n in [
    'unitree_sdk2py', 'unitree_sdk2py.utils',
    'unitree_sdk2py.core', 'unitree_sdk2py.core.channel',
    'unitree_sdk2py.idl', 'unitree_sdk2py.idl.unitree_go',
    'unitree_sdk2py.idl.unitree_go.msg', 'unitree_sdk2py.idl.unitree_go.msg.dds_',
    'unitree_sdk2py.idl.default',
    'unitree_sdk2py.go2', 'unitree_sdk2py.go2.sport',
    'unitree_sdk2py.go2.sport.sport_client',
    'unitree_sdk2py.comm', 'unitree_sdk2py.comm.motion_switcher',
    'unitree_sdk2py.comm.motion_switcher.motion_switcher_client',
]:
    _mock(_n)
_mock('unitree_sdk2py.utils.crc', CRC=MagicMock(return_value=MagicMock()))

# ── package path ─────────────────────────────────────────────────────────
sys.path.insert(0, '/home/csh/smll_project/locomotion/go2_locomotion')

import argparse
import numpy as np
import mujoco
import mujoco.viewer
import rclpy
from rclpy.executors import MultiThreadedExecutor

from go2_locomotion.controllers.nn_policy_controller import NNPolicyController
from go2_locomotion.utils.go2_constants import DEFAULT_JOINT_POS, NUM_JOINTS

# ── 상수 ─────────────────────────────────────────────────────────────────
SCENE_XML   = '/home/csh/unitree_mujoco/unitree_robots/go2/scene.xml'
POLICY_ONNX = '/home/csh/go2_logs/2026-06-26_02-37-40/policy.onnx'
KP_SIM      = [25.0] * NUM_JOINTS
KD_SIM      = [0.5]  * NUM_JOINTS
SIM_DT      = 0.005
# SDK/sensor 순서(FR,FL,RR,RL) → qpos joint 순서(FL,FR,RL,RR) 매핑
SDK_TO_QPOS = [10, 11, 12, 7, 8, 9, 16, 17, 18, 13, 14, 15]


# ── MuJoCo 시뮬레이터 ──────────────────────────────────────────────────────
class MujocoSim:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT
        self.lock  = threading.Lock()
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2]   = 0.35
        self.data.qpos[3]   = 1.0
        self.data.qpos[4:7] = 0.0
        self.data.ctrl[:]   = 0.0
        for i in range(NUM_JOINTS):
            self.data.qpos[SDK_TO_QPOS[i]] = DEFAULT_JOINT_POS[i]
        mujoco.mj_forward(self.model, self.data)

    def make_lowstate(self):
        sd = self.data.sensordata
        ls = MagicMock()
        mm = [MagicMock() for _ in range(NUM_JOINTS)]
        for i in range(NUM_JOINTS):
            mm[i].q  = float(sd[i])
            mm[i].dq = float(sd[i + NUM_JOINTS])
        ls.motor_state.__getitem__.side_effect = lambda i: mm[i]
        imu = NUM_JOINTS * 3   # 36: quat(4), 40: gyro(3)
        ls.imu_state.quaternion = [float(sd[imu + j])     for j in range(4)]
        ls.imu_state.gyroscope  = [float(sd[imu + 4 + j]) for j in range(3)]
        return ls

    def apply_cmd(self, target_q):
        sd  = self.data.sensordata
        q   = np.array(sd[0:NUM_JOINTS])
        dq  = np.array(sd[NUM_JOINTS:NUM_JOINTS * 2])
        tau = np.array(KP_SIM) * (target_q - q) + np.array(KD_SIM) * (0.0 - dq)
        self.data.ctrl[:] = tau

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)


# ── NNPolicyController DDS 초기화 no-op 패치 ─────────────────────────────
def _fake_init_dds(self):
    self._cmd_pub   = MagicMock()
    self._state_sub = MagicMock()

NNPolicyController._init_dds            = _fake_init_dds
NNPolicyController._release_sport_mode  = lambda self: None
NNPolicyController._hold_current_pos    = lambda self: None


# ── 인자 파싱 ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--policy', default=POLICY_ONNX)
    p.add_argument('--scene',  default=SCENE_XML)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print(f'Policy : {args.policy}')
    print(f'Scene  : {args.scene}')

    sim      = MujocoSim(args.scene)
    shutdown = threading.Event()

    # ── ROS2 초기화 + LocomotionNode 생성 ────────────────────────────────
    rclpy.init()
    from go2_locomotion.locomotion_node import LocomotionNode
    loco_node = LocomotionNode()

    nn_ctrl = loco_node._controllers['nn_policy']

    # policy 주입 (파라미터로 model_path 미전달 → None)
    if nn_ctrl._policy is None:
        from go2_locomotion.policy.onnx_policy import OnnxPolicy
        nn_ctrl._policy = OnnxPolicy(args.policy)
        print(f'ONNX policy 로드됨: {args.policy}')

    nn_ctrl._obs_dim = 45

    # _send_low_cmd → MuJoCo 리다이렉트
    # ONNX 호출 횟수 카운터 (디버그용)
    onnx_counter = [0]

    def mujoco_send_low_cmd(target_q):
        onnx_counter[0] += 1
        with sim.lock:
            sim.apply_cmd(target_q)

    nn_ctrl._send_low_cmd = mujoco_send_low_cmd

    # nn_policy 모드로 전환
    loco_node._activate_controller('nn_policy')

    # ── executor ─────────────────────────────────────────────────────────
    executor = MultiThreadedExecutor()
    executor.add_node(loco_node)

    # ── 물리 스레드: sensordata → LowState 주입 + physics step ────────────
    def physics_loop():
        while not shutdown.is_set():
            # sim.lock 밖에서 lowstate 생성 후 state_lock만 잠금
            # (sim.lock ↔ state_lock 순서 역전 방지)
            with sim.lock:
                lowstate = sim.make_lowstate()
                sim.step(1)
            nn_ctrl._on_low_state(lowstate)
            time.sleep(SIM_DT)

    # ── ROS2 executor 스레드 ──────────────────────────────────────────────
    def ros_loop():
        executor.spin()

    # ── 상태 출력 스레드 (2초마다) ────────────────────────────────────────
    def status_loop():
        prev = 0
        while not shutdown.is_set():
            time.sleep(2.0)
            calls = onnx_counter[0]
            hz    = (calls - prev) / 2.0
            prev  = calls
            wdog  = loco_node._watchdog_triggered
            with sim.lock:
                h = sim.data.qpos[2]
            print(f'[status] ONNX {hz:.1f} Hz | watchdog={wdog} '
                  f'| base_height={h:.3f} m')
            if wdog:
                print('         ↑ /cmd_ctr 미수신 — motion_controller 실행 필요')

    for target, name in [
        (physics_loop, 'physics'),
        (ros_loop,     'ros2'),
        (status_loop,  'status'),
    ]:
        t = threading.Thread(target=target, name=name, daemon=True)
        t.start()

    # ── 조작 안내 ─────────────────────────────────────────────────────────
    print()
    print('외부에서 /cmd_ctr 발행 필요 (0.5초 내에):')
    print('  ros2 topic pub /cmd_ctr geometry_msgs/TwistStamped \\')
    print('    "{twist: {linear: {x: 0.3}}}" --rate 20')
    print()
    print('또는 motion_controller 실행:')
    print('  cd /home/csh/smll_project/tools/motion_controller')
    print('  python3 with_ros2.py')
    print()
    print('뷰어 창을 닫으면 종료')

    # ── 뷰어 (메인 스레드) ────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            with sim.lock:
                viewer.sync()
            time.sleep(0.02)

    # ── 종료 ──────────────────────────────────────────────────────────────
    shutdown.set()
    executor.shutdown(timeout_sec=1.0)
    loco_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
