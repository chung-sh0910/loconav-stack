#!/usr/bin/env python3
"""
MuJoCo 시각화 + 조이스틱 데모 (Discrete-MoE 정책 버전).

run_mujoco_demo.py 와 동일한 하네스(NNPolicyController + MuJoCo)를 쓰되, ONNX 대신 rsl_rl
ActorCriticMoE 체크포인트(.pt)를 직접 로드하고 **현재 어떤 Expert(gait mode)로 제어 중인지**를
실시간으로 콘솔에 표시한다.

Usage:
    python3 tools/run_mujoco_demo_moe.py --checkpoint /path/to/model_XXXX.pt \
        [--scene scene.xml] [--js xbox|switch] [--no-joystick] [--rsl-rl-path /home/chung/workspace/rsl_rl]

의존성: 이 데모는 torch + rsl_rl 이 필요하다(ONNX 데모와 달리). mujoco/pygame 이 깔린 학습 env에서
실행하거나 해당 패키지를 설치할 것.

Controls (조이스틱):
    왼쪽 스틱 Y → vx | 왼쪽 스틱 X → vyaw | 오른쪽 스틱 X → vy | B → 비상정지 토글
Controls (키보드, 뷰어 포커스):
    W/S 전진/후진 | A/D 회전 | Q/E 횡이동 | Space 정지 | R 리셋
"""

import argparse
import os
import select
import sys
import termios
import threading
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np

# --- go2_locomotion 패키지 경로 (이 파일 기준으로 계산) ---
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

# DDS mock — 데모에서는 unitree_sdk2py .so 로드를 막는다 (NNPolicyController가 CRC를 import)
from unittest.mock import MagicMock, patch
import sys as _sys


def _mock(name, **attrs):
    _sys.modules.setdefault(name, MagicMock(**attrs))


for _m, _a in [
    ("unitree_sdk2py", {}),
    ("unitree_sdk2py.utils", {}),
    ("unitree_sdk2py.utils.crc", {"CRC": MagicMock(return_value=MagicMock())}),
    ("unitree_sdk2py.core", {}),
    ("unitree_sdk2py.core.channel", {}),
    ("unitree_sdk2py.idl", {}),
    ("unitree_sdk2py.idl.unitree_go", {}),
    ("unitree_sdk2py.idl.unitree_go.msg", {}),
    ("unitree_sdk2py.idl.unitree_go.msg.dds_", {}),
    ("unitree_sdk2py.idl.default", {}),
]:
    _mock(_m, **_a)

from go2_locomotion.controllers.nn_policy_controller import NNPolicyController
from go2_locomotion.policy.moe_policy import MoEPolicy
from go2_locomotion.policy.mlp_policy import MlpPolicy
from go2_locomotion.utils.go2_constants import DEFAULT_JOINT_POS, NUM_JOINTS

# ---------------------------------------------------------------------------
SCENE_XML_DEFAULT = "/home/chung/workspace/unitree_mujoco/unitree_robots/go2/scene.xml"

KP_SIM = [25.0] * NUM_JOINTS
KD_SIM = [0.5] * NUM_JOINTS

SIM_DT = 0.005
CTRL_HZ = 50
CTRL_DT = 1.0 / CTRL_HZ
STEPS_PER_CTRL = max(1, round(CTRL_DT / SIM_DT))


# ---------------------------------------------------------------------------
AXIS_MAPS = {"xbox": {"LX": 0, "LY": 1, "RX": 3, "RY": 4}, "switch": {"LX": 0, "LY": 1, "RX": 2, "RY": 3}}
BTN_MAPS = {"xbox": {"B": 1}, "switch": {"B": 1}}


class JoystickReader:
    def __init__(self, js_type="xbox"):
        import pygame

        self._pygame = pygame
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError("조이스틱을 찾을 수 없습니다.")
        self._js = pygame.joystick.Joystick(0)
        self._js.init()
        self._axis = AXIS_MAPS[js_type]
        self._btn = BTN_MAPS[js_type]
        print(f"조이스틱 연결됨: {self._js.get_name()}")

    def read(self):
        self._pygame.event.pump()
        vx = -self._js.get_axis(self._axis["LY"])
        vy = self._js.get_axis(self._axis["RX"])
        vyaw = -self._js.get_axis(self._axis["LX"])
        b_btn = self._js.get_button(self._btn["B"])
        return vx, vy, vyaw, b_btn

    def close(self):
        self._pygame.quit()


class TerminalKeyReader:
    """Read single keypresses from the TERMINAL (stdin), not the MuJoCo window.

    Typing into the MuJoCo viewer triggers its built-in shortcuts (camera / render toggles), so we read the
    terminal instead -- keep the TERMINAL focused. cbreak mode delivers each key immediately (no Enter); the
    terminal is restored on ``close()``. If stdin is not a tty (piped/redirected), ``poll()`` yields nothing.
    """

    def __init__(self):
        self._enabled = sys.stdin.isatty()
        self._fd = None
        self._old = None
        if self._enabled:
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            new = termios.tcgetattr(self._fd)
            new[3] &= ~(termios.ICANON | termios.ECHO)  # immediate keys, no echo; keep ISIG so Ctrl-C still works
            termios.tcsetattr(self._fd, termios.TCSANOW, new)

    def poll(self):
        if not self._enabled:
            return None
        r, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if r else None

    def close(self):
        if self._enabled and self._old is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


# incremental command steps and clamps (match the training limit_ranges)
_VX_RANGE, _VY_RANGE, _VYAW_RANGE = (-0.5, 1.0), (-0.4, 0.4), (-1.0, 1.0)


def _clamp(v, lo_hi):
    return max(lo_hi[0], min(lo_hi[1], v))


def _apply_key(ch, cmd, flags, sim, ctrl, policy):
    """Map one terminal keypress to an INCREMENTAL velocity command (so you can sweep speed and watch experts)."""
    ch = ch.lower()
    if ch == "w":
        cmd["vx"] = _clamp(cmd["vx"] + 0.1, _VX_RANGE)
    elif ch == "s":
        cmd["vx"] = _clamp(cmd["vx"] - 0.1, _VX_RANGE)
    elif ch == "a":
        cmd["vyaw"] = _clamp(cmd["vyaw"] + 0.2, _VYAW_RANGE)
    elif ch == "d":
        cmd["vyaw"] = _clamp(cmd["vyaw"] - 0.2, _VYAW_RANGE)
    elif ch == "q":
        cmd["vy"] = _clamp(cmd["vy"] + 0.1, _VY_RANGE)
    elif ch == "e":
        cmd["vy"] = _clamp(cmd["vy"] - 0.1, _VY_RANGE)
    elif ch == " ":
        cmd["vx"] = cmd["vy"] = cmd["vyaw"] = 0.0
    elif ch == "r":
        flags["estop"] = False
        with sim.lock:
            sim.reset()
        ctrl._prev_actions[:] = 0.0
        policy.reset()
        cmd["vx"] = cmd["vy"] = cmd["vyaw"] = 0.0
        print("\n리셋")


class MujocoSim:
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.35
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        self.data.ctrl[:] = 0.0
        SDK_TO_QPOS = [10, 11, 12, 7, 8, 9, 16, 17, 18, 13, 14, 15]
        for i in range(NUM_JOINTS):
            self.data.qpos[SDK_TO_QPOS[i]] = DEFAULT_JOINT_POS[i]
        mujoco.mj_forward(self.model, self.data)

    def make_lowstate(self):
        sd = self.data.sensordata
        ls = MagicMock()
        motor_mocks = [MagicMock() for _ in range(NUM_JOINTS)]
        for i in range(NUM_JOINTS):
            motor_mocks[i].q = float(sd[i])
            motor_mocks[i].dq = float(sd[i + NUM_JOINTS])
        ls.motor_state.__getitem__.side_effect = lambda i: motor_mocks[i]
        imu = NUM_JOINTS * 3
        ls.imu_state.quaternion = [float(sd[imu + j]) for j in range(4)]
        ls.imu_state.gyroscope = [float(sd[imu + 4 + j]) for j in range(3)]
        return ls

    def apply_cmd(self, target_q, kp, kd):
        sd = self.data.sensordata
        q = sd[0:NUM_JOINTS]
        dq = sd[NUM_JOINTS : NUM_JOINTS * 2]
        tau = np.array(kp) * (target_q - q) + np.array(kd) * (0.0 - dq)
        self.data.ctrl[:] = tau

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="rsl_rl ActorCriticMoE .pt checkpoint")
    p.add_argument("--rsl-rl-path", default="/home/chung/workspace/rsl_rl")
    p.add_argument("--scene", default=SCENE_XML_DEFAULT)
    p.add_argument("--js", default="xbox", choices=["xbox", "switch"])
    p.add_argument("--no-joystick", action="store_true")
    p.add_argument("--action-scale", type=float, default=0.25)
    p.add_argument("--stochastic-routing", action="store_true",
                   help="expert 를 게이트에서 샘플링 (기본: deterministic argmax 라우팅)")
    p.add_argument("--hysteresis", type=float, default=1.0,
                   help="라우팅 히스테리시스 logit 마진 (0=off). 도전 expert 의 logit 이 현재 mode 보다 이만큼 커야 전환")
    p.add_argument("--min-dwell", type=int, default=10,
                   help="mode 최소 유지 스텝 수 (50Hz 기준 10=0.2s, 0=off)")
    p.add_argument("--force-expert", type=int, default=None,
                   help="매 스텝 이 expert(0..K-1)로 강제 라우팅 (게이트 무시). 특정 expert gait 단독 관찰용")
    p.add_argument("--mlp", action="store_true",
                   help="MoE 대신 Hist-MLP(ActorCriticVel) 체크포인트를 로드 (라우팅 없는 비교 baseline)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Scene      : {args.scene}")

    if args.mlp:
        policy = MlpPolicy(args.checkpoint, rsl_rl_path=args.rsl_rl_path)
        print("Policy     : Hist-MLP baseline (no routing)")
    else:
        policy = MoEPolicy(
            args.checkpoint,
            rsl_rl_path=args.rsl_rl_path,
            deterministic=not args.stochastic_routing,
            route_hysteresis=args.hysteresis,
            route_min_dwell=args.min_dwell,
            force_expert=args.force_expert,
        )
        if args.force_expert is not None:
            print(f"Routing    : FORCED -> expert {args.force_expert} (gate bypassed)")
        else:
            print(f"Routing    : {'stochastic (gate sampling)' if args.stochastic_routing else 'DETERMINISTIC (argmax)'}"
                  f" | hysteresis={args.hysteresis} min_dwell={args.min_dwell}")
    sim = MujocoSim(args.scene)
    ctrl = NNPolicyController(policy=policy, obs_dim=45, action_scale=args.action_scale, kp=KP_SIM, kd=KD_SIM)

    joystick = None
    if not args.no_joystick:
        try:
            joystick = JoystickReader(args.js)
        except RuntimeError as e:
            print(f"[경고] {e} — 키보드 모드로 전환")

    cmd = {"vx": 0.0, "vy": 0.0, "vyaw": 0.0}
    flags = {"estop": False, "running": True}
    WINDOW = 100  # ~2s @ 50Hz -- live distribution reflects the CURRENT command, not the whole session
    disp = {"last_expert": -1, "last_print": 0.0, "t0": time.perf_counter(),
            "recent": deque(maxlen=WINDOW), "held_since": time.perf_counter()}

    # terminal keyboard reader (used when there's no joystick) -- avoids MuJoCo eating the keys
    term_keys = TerminalKeyReader() if not joystick else None

    def report_expert():
        e = policy.current_expert
        if e < 0:  # MLP baseline: no routing/experts -- just show the command line
            now = time.perf_counter() - disp["t0"]
            if now - disp["last_print"] > 0.5:
                print(f"\rMLP (no routing) | cmd=({cmd['vx']:+.2f},{cmd['vy']:+.2f},{cmd['vyaw']:+.2f})   ",
                      end="", flush=True)
                disp["last_print"] = now
            return
        disp["recent"].append(e)  # sliding window over EVERY step (reflects the current condition)
        now = time.perf_counter() - disp["t0"]
        if e != disp["last_expert"]:
            # 모드가 바뀔 때마다 즉시 한 줄 (직전 모드를 얼마나 유지했는지 함께)
            held = time.perf_counter() - disp["held_since"]
            print(f"\n[t={now:6.2f}s] expert {disp['last_expert']} -> {e}  (held {held:.2f}s)", flush=True)
            disp["last_expert"] = e
            disp["held_since"] = time.perf_counter()
        elif now - disp["last_print"] > 0.5:
            # 유지 중 0.5s마다: 분포는 세션 누적이 아니라 **최근 ~2s 윈도우** (지금 조건을 반영)
            rec = disp["recent"]
            w = np.bincount(np.asarray(rec), minlength=8) / max(len(rec), 1)
            bar = " ".join(f"{i}:{w[i]*100:4.1f}%" for i in range(len(w)))
            print(f"\rexpert={e} | cmd=({cmd['vx']:+.2f},{cmd['vy']:+.2f},{cmd['vyaw']:+.2f}) | 최근2s {bar}   ",
                  end="", flush=True)
            disp["last_print"] = now

    def control_loop():
        prev_b = False
        while flags["running"]:
            t0 = time.perf_counter()
            if joystick:
                vx, vy, vyaw, b_btn = joystick.read()
                cmd["vx"], cmd["vy"], cmd["vyaw"] = vx * 1.0, vy * 0.4, vyaw * 1.0
                if b_btn and not prev_b:
                    flags["estop"] = not flags["estop"]
                    if flags["estop"]:
                        print("\n비상정지 ON")
                    else:
                        sim.reset()
                        ctrl._prev_actions[:] = 0.0
                        policy.reset()
                        print("\n비상정지 해제 + 리셋")
                prev_b = b_btn
            elif term_keys is not None:
                # drain all keys typed in the terminal since the last tick
                ch = term_keys.poll()
                while ch is not None:
                    _apply_key(ch, cmd, flags, sim, ctrl, policy)
                    ch = term_keys.poll()

            if flags["estop"]:
                with sim.lock:
                    sim.data.ctrl[:] = 0.0
                    sim.step(STEPS_PER_CTRL)
                time.sleep(CTRL_DT)
                continue

            ctrl.update_cmd_vel(cmd["vx"], cmd["vy"], cmd["vyaw"])
            with sim.lock:
                ctrl._lowstate = sim.make_lowstate()

            captured = {}

            def intercept(q, _c=captured, **_kw):  # step() calls _send_low_cmd(q, kp=..., kd=...); absorb kwargs
                _c["q"] = q.copy()

            with patch.object(ctrl, "_send_low_cmd", side_effect=intercept):
                ctrl.step()

            report_expert()  # <-- 현재 제어 중인 expert 표시

            with sim.lock:
                if "q" in captured:
                    sim.apply_cmd(captured["q"], KP_SIM, KD_SIM)
                sim.step(STEPS_PER_CTRL)

            sleep_t = CTRL_DT - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    ctrl_thread = threading.Thread(target=control_loop, daemon=True)
    ctrl_thread.start()

    print("\n[조작법]")
    if joystick:
        print("  왼쪽 스틱 Y: 전진/후진 | 왼쪽 스틱 X: 회전 | 오른쪽 스틱 X: 횡이동 | B: 비상정지")
    else:
        print("  ** 터미널 창을 포커스한 채로 ** 키 입력 (뷰어 창에 입력하면 MuJoCo 설정이 바뀜):")
        print("    W/S: 전진속도 ±0.1 | A/D: 회전 ±0.2 | Q/E: 횡이동 ±0.1 | Space: 정지 | R: 리셋")
        print("    (증분 제어 — 속도를 올려가며 어떤 속도에서 expert 가 바뀌는지 관찰). Ctrl-C 또는 뷰어 닫기로 종료.")
    print("  아래에 현재 제어 중인 expert 가 실시간 표시됨.\n")

    try:
        with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
            while viewer.is_running():
                with sim.lock:
                    viewer.sync()
                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n종료")
    finally:
        flags["running"] = False
        time.sleep(0.1)
        if term_keys is not None:
            term_keys.close()  # restore the terminal (critical -- else it stays in no-echo mode)
        if joystick:
            joystick.close()

    if getattr(policy, "current_expert", 0) < 0 and args.mlp:
        print("\n\n=== Hist-MLP baseline (no experts) ===")
    else:
        counts = policy.expert_counts
        print("\n\n=== expert 사용 히스토그램 (control steps) ===")
        for i, c in enumerate(counts):
            print(f"  expert {i}: {c:8d}  ({c / max(counts.sum(),1) * 100:5.1f}%)")


if __name__ == "__main__":
    main()
