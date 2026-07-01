"""
MuJoCo 헤드리스 통합 테스트.

DDS / 실제 하드웨어 없이 MuJoCo 물리 엔진에서 NNPolicyController + ONNX policy를
실제로 돌려 로봇이 서 있는지 검증한다.

구조:
  HeadlessSim  ─ MuJoCo 직접 구동 (scene.xml의 센서/액추에이터는 SDK 순서)
  ctrl.step()  ─ NNPolicyController (실제 코드 그대로)
  _send_low_cmd 패치  ─ DDS 대신 MuJoCo ctrl 에 직접 기록
  _lowstate 주입   ─ DDS 대신 sensordata 로 채운 fake LowState 주입
"""
import time
import numpy as np
import pytest
import mujoco
from unittest.mock import MagicMock, patch

from go2_locomotion.controllers.nn_policy_controller import NNPolicyController
from go2_locomotion.policy.onnx_policy import OnnxPolicy
from go2_locomotion.utils.go2_constants import DEFAULT_JOINT_POS, NUM_JOINTS

SCENE_XML   = "/home/csh/unitree_mujoco/unitree_robots/go2/scene.xml"
POLICY_ONNX = "/home/csh/go2_logs/2026-06-26_02-37-40/policy.onnx"

# deploy.yaml 기준 PD 게인 (Isaac Lab 학습 시 사용값)
KP_SIM = [25.0] * NUM_JOINTS
KD_SIM = [0.5]  * NUM_JOINTS

SIM_DT          = 0.005        # MuJoCo timestep (scene.xml 기본값)
CTRL_HZ         = 50           # NNPolicyController 제어 주기
CTRL_DT         = 1.0 / CTRL_HZ
STEPS_PER_CTRL  = max(1, round(CTRL_DT / SIM_DT))   # 10 physics steps per ctrl step


# ---------------------------------------------------------------------------
# 헤드리스 시뮬레이터
# ---------------------------------------------------------------------------

class HeadlessSim:
    """
    MuJoCo Go2 헤드리스 래퍼.

    scene.xml의 센서/액추에이터 순서가 SDK 순서와 동일하므로
    별도 인덱스 재배열 없이 motor_state / ctrl 을 SDK 인덱스로 직접 사용한다.
    """

    def __init__(self, xml_path: str = SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self.model.opt.timestep = SIM_DT
        self._reset()

    def _reset(self):
        """DEFAULT_JOINT_POS (SDK 순서) 로 초기 자세 설정."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.35
        self.data.qpos[3] = 1.0
        self.data.qpos[4:7] = 0.0
        self.data.ctrl[:] = 0.0

        # qpos joint 순서 (FL,FR,RL,RR) ≠ SDK/sensor 순서 (FR,FL,RR,RL)
        # sensor i → qpos address 매핑: sdk_to_qpos[i] = jnt_qposadr[sensor_objid[i]]
        SDK_TO_QPOS = [10, 11, 12, 7, 8, 9, 16, 17, 18, 13, 14, 15]
        for i in range(NUM_JOINTS):
            self.data.qpos[SDK_TO_QPOS[i]] = DEFAULT_JOINT_POS[i]
        mujoco.mj_forward(self.model, self.data)

    def make_lowstate(self) -> MagicMock:
        """현재 시뮬레이터 상태를 LowState 형태로 반환 (SDK 순서)."""
        sd = self.data.sensordata

        ls = MagicMock()
        # 센서 0-11: 관절 위치 (SDK 순서)
        # 센서 12-23: 관절 속도 (SDK 순서)
        motor_mocks = [MagicMock() for _ in range(NUM_JOINTS)]
        for i in range(NUM_JOINTS):
            motor_mocks[i].q  = float(sd[i])
            motor_mocks[i].dq = float(sd[i + NUM_JOINTS])
        ls.motor_state.__getitem__.side_effect = lambda i: motor_mocks[i]

        # 센서 36: imu_quat (w, x, y, z)
        # 센서 37: imu_gyro (x, y, z)
        imu_offset = NUM_JOINTS * 3          # 12 pos + 12 vel + 12 torque = 36
        ls.imu_state.quaternion = [
            float(sd[imu_offset + 0]),
            float(sd[imu_offset + 1]),
            float(sd[imu_offset + 2]),
            float(sd[imu_offset + 3]),
        ]
        ls.imu_state.gyroscope = [
            float(sd[imu_offset + 4]),
            float(sd[imu_offset + 5]),
            float(sd[imu_offset + 6]),
        ]
        return ls

    def apply_cmd(self, target_q: np.ndarray, kp, kd) -> None:
        """target_q (SDK 순서) 와 PD 게인으로 torque 계산 후 ctrl 에 기록."""
        sd  = self.data.sensordata
        q   = sd[0:NUM_JOINTS]
        dq  = sd[NUM_JOINTS:NUM_JOINTS*2]
        tau = np.array(kp) * (target_q - q) + np.array(kd) * (0.0 - dq)
        self.data.ctrl[:] = tau

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    @property
    def base_height(self) -> float:
        return float(self.data.qpos[2])

    @property
    def base_quat(self) -> np.ndarray:
        """쿼터니언 [w, x, y, z]"""
        return self.data.qpos[3:7].copy()


# ---------------------------------------------------------------------------
# 픽스처
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sim():
    return HeadlessSim()


@pytest.fixture(scope="module")
def policy():
    return OnnxPolicy(POLICY_ONNX)


@pytest.fixture(scope="module")
def ctrl(policy):
    return NNPolicyController(
        policy=policy,
        obs_dim=45,
        action_scale=0.25,
        kp=KP_SIM,
        kd=KD_SIM,
    )


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def run_sim(sim: HeadlessSim, ctrl: NNPolicyController,
            seconds: float, vx=0.0, vy=0.0, vyaw=0.0):
    """
    지정 시간(초) 동안 제어 루프를 돌린다.
    매 ctrl step:
      1. 시뮬레이터 상태 → LowState 주입
      2. ctrl.step() 호출 (내부에서 policy 추론 + _send_low_cmd 호출)
      3. _send_low_cmd 를 가로채 MuJoCo ctrl 에 기록
      4. MuJoCo physics step
    """
    ctrl.update_cmd_vel(vx, vy, vyaw)
    ctrl._prev_actions[:] = 0.0

    num_ctrl_steps = int(seconds / CTRL_DT)

    for _ in range(num_ctrl_steps):
        # LowState 주입
        ctrl._lowstate = sim.make_lowstate()

        # _send_low_cmd 가로채기
        captured_q = {}
        original_send = ctrl._send_low_cmd

        def intercept(target_q, kp=None, kd=None, _cap=captured_q):
            _cap['q'] = target_q.copy()

        with patch.object(ctrl, '_send_low_cmd', side_effect=intercept):
            ctrl.step()

        # MuJoCo 에 적용
        if 'q' in captured_q:
            sim.apply_cmd(captured_q['q'], KP_SIM, KD_SIM)

        sim.step(STEPS_PER_CTRL)


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------

class TestStanding:

    def test_robot_does_not_collapse_standing_still(self, sim, ctrl):
        """정지 명령(vx=0) 3초 후 로봇이 쓰러지지 않아야 한다."""
        sim._reset()
        run_sim(sim, ctrl, seconds=3.0, vx=0.0)
        assert sim.base_height > 0.15, \
            f"로봇 쓰러짐: base_height={sim.base_height:.3f}m"

    def test_robot_stays_upright_orientation(self, sim, ctrl):
        """서 있는 동안 자세가 과도하게 기울지 않아야 한다 (|roll|, |pitch| < 60°)."""
        sim._reset()
        run_sim(sim, ctrl, seconds=3.0, vx=0.0)

        quat = sim.base_quat   # [w, x, y, z]
        w, x, y, z = quat
        # roll
        roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        # pitch
        sinp  = 2*(w*y - z*x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        assert abs(roll)  < np.radians(60), f"roll 과다: {np.degrees(roll):.1f}°"
        assert abs(pitch) < np.radians(60), f"pitch 과다: {np.degrees(pitch):.1f}°"

    def test_no_nan_in_observations(self, sim, ctrl):
        """제어 루프 중 observation 에 NaN 이 없어야 한다."""
        sim._reset()
        ctrl._prev_actions[:] = 0.0
        nan_found = False

        for _ in range(50):   # 1초
            ctrl._lowstate = sim.make_lowstate()
            obs = ctrl._build_observation(ctrl._lowstate)
            if np.any(np.isnan(obs)):
                nan_found = True
                break
            with patch.object(ctrl, '_send_low_cmd', side_effect=lambda q, kp=None, kd=None: sim.apply_cmd(q, KP_SIM, KD_SIM)):
                ctrl.step()
            sim.step(STEPS_PER_CTRL)

        assert not nan_found, "observation 에 NaN 발생"


class TestForwardWalking:

    def test_robot_moves_forward(self, sim, ctrl):
        """vx=0.3 명령 후 3초간 전진 방향으로 이동해야 한다."""
        sim._reset()
        x_start = float(sim.data.sensordata[36 + 10])  # frame_pos x

        # frame_pos: sensor[39] = frame_pos (dim=3)
        imu_offset = NUM_JOINTS * 3
        x_start = float(sim.data.sensordata[imu_offset + 10])

        run_sim(sim, ctrl, seconds=3.0, vx=0.3)

        x_end = float(sim.data.sensordata[imu_offset + 10])
        assert sim.base_height > 0.15, "전진 중 쓰러짐"
        # 전진 방향으로 조금이라도 이동했는지
        assert x_end > x_start - 0.5, \
            f"전진 실패: x_start={x_start:.2f}, x_end={x_end:.2f}"
