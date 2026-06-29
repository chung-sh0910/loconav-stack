#!/usr/bin/env python3
"""
MuJoCo 시각화 + 조이스틱 데모.

실제 NNPolicyController + ONNX policy를 MuJoCo에서 구동한다.
조이스틱(Xbox/Switch)으로 속도 명령을 보내고 뷰어로 실시간 확인.

Usage:
    python3 tools/run_mujoco_demo.py [--policy PATH] [--js xbox|switch]

    # 조이스틱 없이 키보드만 사용:
    python3 tools/run_mujoco_demo.py --no-joystick

Controls (조이스틱):
    왼쪽 스틱 Y축   → vx (전진/후진)
    왼쪽 스틱 X축   → vyaw (좌/우 회전)
    오른쪽 스틱 X축 → vy (좌/우 이동)
    B 버튼          → 비상정지 / 해제 토글

Controls (키보드, MuJoCo 뷰어 포커스 필요):
    W/S → 전진/후진   A/D → 좌/우 회전   Q/E → 횡이동   Space → 정지
"""

import argparse
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
import pygame

# go2_locomotion 패키지 경로
sys.path.insert(0, '/home/csh/smll_project/locomotion/go2_locomotion')

# DDS mock — 데모에서는 unitree_sdk2py를 사용하지 않으므로 .so 로드를 막는다
from unittest.mock import MagicMock, patch
import sys as _sys

def _mock(name, **attrs):
    m = MagicMock(**attrs)
    _sys.modules.setdefault(name, m)

_mock('unitree_sdk2py')
_mock('unitree_sdk2py.utils')
_mock('unitree_sdk2py.utils.crc',   CRC=MagicMock(return_value=MagicMock()))
_mock('unitree_sdk2py.core')
_mock('unitree_sdk2py.core.channel')
_mock('unitree_sdk2py.idl')
_mock('unitree_sdk2py.idl.unitree_go')
_mock('unitree_sdk2py.idl.unitree_go.msg')
_mock('unitree_sdk2py.idl.unitree_go.msg.dds_')
_mock('unitree_sdk2py.idl.default')

from go2_locomotion.controllers.nn_policy_controller import NNPolicyController
from go2_locomotion.policy.onnx_policy import OnnxPolicy
from go2_locomotion.utils.go2_constants import DEFAULT_JOINT_POS, NUM_JOINTS

# ---------------------------------------------------------------------------
# 기본 경로
# ---------------------------------------------------------------------------

SCENE_XML   = "/home/csh/unitree_mujoco/unitree_robots/go2/scene.xml"
POLICY_ONNX = "/home/csh/go2_logs/2026-06-26_02-37-40/policy.onnx"

KP_SIM = [25.0] * NUM_JOINTS
KD_SIM = [0.5]  * NUM_JOINTS

SIM_DT         = 0.005
CTRL_HZ        = 50
CTRL_DT        = 1.0 / CTRL_HZ
STEPS_PER_CTRL = max(1, round(CTRL_DT / SIM_DT))


# ---------------------------------------------------------------------------
# 조이스틱 리더
# ---------------------------------------------------------------------------

AXIS_MAPS = {
    "xbox":   {"LX": 0, "LY": 1, "RX": 3, "RY": 4},
    "switch": {"LX": 0, "LY": 1, "RX": 2, "RY": 3},
}
BTN_MAPS = {
    "xbox":   {"B": 1, "A": 0, "X": 2, "Y": 3},
    "switch": {"B": 1, "A": 0, "X": 3, "Y": 4},
}

class JoystickReader:
    def __init__(self, js_type="xbox"):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("조이스틱을 찾을 수 없습니다.")
        self._js   = pygame.joystick.Joystick(0)
        self._js.init()
        self._axis = AXIS_MAPS[js_type]
        self._btn  = BTN_MAPS[js_type]
        print(f"조이스틱 연결됨: {self._js.get_name()}")

    def read(self):
        pygame.event.pump()
        vx   = -self._js.get_axis(self._axis["LY"])  # 앞이 양수
        vy   =  self._js.get_axis(self._axis["RX"])
        vyaw = -self._js.get_axis(self._axis["LX"])  # 왼쪽 회전이 양수
        b_btn = self._js.get_button(self._btn["B"])
        return vx, vy, vyaw, b_btn

    def close(self):
        pygame.quit()


# ---------------------------------------------------------------------------
# MuJoCo 래퍼 (테스트 코드의 HeadlessSim과 동일 구조)
# ---------------------------------------------------------------------------

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
        SDK_TO_QPOS = [10, 11, 12, 7, 8, 9, 16, 17, 18, 13, 14, 15]
        for i in range(NUM_JOINTS):
            self.data.qpos[SDK_TO_QPOS[i]] = DEFAULT_JOINT_POS[i]
        mujoco.mj_forward(self.model, self.data)

    def make_lowstate(self):
        sd = self.data.sensordata
        ls = MagicMock()
        motor_mocks = [MagicMock() for _ in range(NUM_JOINTS)]
        for i in range(NUM_JOINTS):
            motor_mocks[i].q  = float(sd[i])
            motor_mocks[i].dq = float(sd[i + NUM_JOINTS])
        ls.motor_state.__getitem__.side_effect = lambda i: motor_mocks[i]
        imu = NUM_JOINTS * 3
        ls.imu_state.quaternion = [float(sd[imu+j]) for j in range(4)]
        ls.imu_state.gyroscope  = [float(sd[imu+4+j]) for j in range(3)]
        return ls

    def apply_cmd(self, target_q, kp, kd):
        sd  = self.data.sensordata
        q   = sd[0:NUM_JOINTS]
        dq  = sd[NUM_JOINTS:NUM_JOINTS*2]
        tau = np.array(kp) * (target_q - q) + np.array(kd) * (0.0 - dq)
        self.data.ctrl[:] = tau

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    @property
    def base_height(self):
        return float(self.data.qpos[2])


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy",       default=POLICY_ONNX)
    p.add_argument("--scene",        default=SCENE_XML)
    p.add_argument("--js",           default="xbox", choices=["xbox", "switch"])
    p.add_argument("--no-joystick",  action="store_true")
    p.add_argument("--action-scale", type=float, default=0.25)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Policy  : {args.policy}")
    print(f"Scene   : {args.scene}")

    policy = OnnxPolicy(args.policy)
    sim    = MujocoSim(args.scene)
    ctrl   = NNPolicyController(
        policy=policy,
        obs_dim=45,
        action_scale=args.action_scale,
        kp=KP_SIM,
        kd=KD_SIM,
    )

    # 조이스틱 초기화
    joystick = None
    if not args.no_joystick:
        try:
            joystick = JoystickReader(args.js)
        except RuntimeError as e:
            print(f"[경고] {e} — 키보드 모드로 전환")

    # 상태 공유
    cmd = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}
    flags = {"estop": False, "running": True, "b_prev": False}

    # --------------- 뷰어 키콜백 (키보드 제어) ---------------
    SPEED = 0.3
    def key_callback(keycode):
        k = chr(keycode) if keycode < 128 else ""
        if k == "W":  cmd["vx"]   =  SPEED
        elif k == "S":  cmd["vx"]   = -SPEED / 2
        elif k == "A":  cmd["vyaw"] =  SPEED
        elif k == "D":  cmd["vyaw"] = -SPEED
        elif k == "Q":  cmd["vy"]   =  SPEED
        elif k == "E":  cmd["vy"]   = -SPEED
        elif k == " ":
            cmd["vx"] = cmd["vy"] = cmd["vyaw"] = 0.0
        elif k == "R":
            flags["estop"] = False
            sim.reset()
            ctrl._prev_actions[:] = 0.0
            print("리셋")

    # --------------- 제어 스레드 ---------------
    def control_loop():
        prev_b = False
        while flags["running"]:
            t0 = time.perf_counter()

            # 조이스틱 읽기
            if joystick:
                vx, vy, vyaw, b_btn = joystick.read()
                cmd["vx"]   = vx   * 1.0
                cmd["vy"]   = vy   * 0.4
                cmd["vyaw"] = vyaw * 1.0
                # B 버튼 — 에지 감지로 estop 토글
                if b_btn and not prev_b:
                    flags["estop"] = not flags["estop"]
                    if flags["estop"]:
                        print("비상정지 ON")
                    else:
                        sim.reset()
                        ctrl._prev_actions[:] = 0.0
                        print("비상정지 해제 + 리셋")
                prev_b = b_btn

            if flags["estop"]:
                with sim.lock:
                    sim.data.ctrl[:] = 0.0
                    sim.step(STEPS_PER_CTRL)
                time.sleep(CTRL_DT)
                continue

            ctrl.update_cmd_vel(cmd["vx"], cmd["vy"], cmd["vyaw"])

            with sim.lock:
                ctrl._lowstate = sim.make_lowstate()

            # _send_low_cmd 가로채기
            captured = {}
            def intercept(q, _c=captured):
                _c["q"] = q.copy()

            with patch.object(ctrl, '_send_low_cmd', side_effect=intercept):
                ctrl.step()

            with sim.lock:
                if "q" in captured:
                    sim.apply_cmd(captured["q"], KP_SIM, KD_SIM)
                sim.step(STEPS_PER_CTRL)

            elapsed = time.perf_counter() - t0
            sleep_t = CTRL_DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()

    # --------------- 뷰어 (메인 스레드) ---------------
    print("\n[조작법]")
    if joystick:
        print("  왼쪽 스틱 Y: 전진/후진  |  왼쪽 스틱 X: 회전  |  오른쪽 스틱 X: 횡이동")
        print("  B 버튼: 비상정지 토글 + 리셋")
    else:
        print("  W/S: 전진/후진  |  A/D: 회전  |  Q/E: 횡이동  |  Space: 정지  |  R: 리셋")
    print("  뷰어 창을 닫으면 종료\n")

    with mujoco.viewer.launch_passive(
        sim.model, sim.data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            with sim.lock:
                viewer.sync()
            time.sleep(0.02)   # ~50fps 뷰어

    flags["running"] = False
    if joystick:
        joystick.close()


if __name__ == "__main__":
    main()
