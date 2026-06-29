import time
import threading
import numpy as np

from .base_controller import BaseController
from go2_locomotion.utils.go2_constants import (
    NUM_JOINTS, DEFAULT_JOINT_POS, KP_DEFAULT, KD_DEFAULT, KD_PASSIVE,
    KP_ESTOP_DESCENT, KD_ESTOP_DESCENT, PRONE_JOINT_POS,
    MAX_VX, MIN_VX, MAX_VY, MAX_VYAW,
    TOPIC_LOW_STATE, TOPIC_LOW_CMD,
    POS_STOP_F, VEL_STOP_F, LOWLEVEL,
    JOINT_IDS_MAP,
)
from go2_locomotion.utils.imu_utils import quat_to_projected_gravity


class NNPolicyController(BaseController):
    """
    Mode 2: Low-level NN policy control via LowCmd motor position commands.

    start() 순서:
      1. Publisher/Subscriber 먼저 초기화 (제어 공백 최소화)
      2. StandDown() → 로봇 엎드림 (안전한 낮은 자세)
      3. ReleaseMode() → sport mode 해제
      4. 현재 관절 위치를 즉시 hold → 제어 공백 없음
      5. step()이 50Hz로 NN 제어 시작

    recover() 순서 (비상정지 후 재개):
      1. RecoveryStand() → sport mode로 일어서기
      2. start()로 low-level 재진입
    """

    name = "nn_policy"

    def __init__(
        self,
        policy,
        obs_dim: int = 45,
        action_scale: float = 0.25,
        kp: list = None,
        kd: list = None,
    ):
        self._policy = policy
        self._obs_dim = obs_dim
        self._action_scale = action_scale
        self._kp = kp if kp is not None else KP_DEFAULT
        self._kd = kd if kd is not None else KD_DEFAULT

        self._cmd_lock = threading.Lock()
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0

        self._state_lock = threading.Lock()
        self._lowstate = None

        self._prev_actions = np.zeros(NUM_JOINTS, dtype=np.float32)
        self._default_pos = DEFAULT_JOINT_POS.copy()   # SDK 순서
        self._joint_ids_map = JOINT_IDS_MAP            # policy[i] → sdk[map[i]]

        self._state_sub = None
        self._cmd_pub = None

        from unitree_sdk2py.utils.crc import CRC
        self._crc = CRC()

    def start(self) -> None:
        """
        초기 진입: 서있는 상태에서 바로 ReleaseMode → 즉시 hold → NN 제어 시작
        StandDown()을 하지 않는 이유: _hold_current_pos()가 ReleaseMode() 직후
        현재 관절 위치를 잡아주므로 제어 공백 없이 서있는 자세 유지 가능
        """
        self._init_dds()
        self._release_sport_mode()
        self._hold_current_pos()
        self._reset_policy()

    def recover(self) -> None:
        """
        비상정지 후 복구.
        robot이 바닥에 있음 → RecoveryStand() → 서있는 상태에서 바로 ReleaseMode → hold
        StandDown()을 다시 하지 않음 (서있다 앉았다 반복 방지)
        """
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        sc = SportClient()
        sc.SetTimeout(5.0)
        sc.Init()

        sc.RecoveryStand()
        time.sleep(3.0)   # 일어서기 완료 대기

        # 이미 서있으므로 StandDown 없이 바로 ReleaseMode
        self._init_dds()
        self._release_sport_mode()
        self._hold_current_pos()
        self._reset_policy()

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _init_dds(self) -> None:
        """Publisher/Subscriber 초기화 (이미 있으면 재사용)."""
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_

        if self._state_sub is None:
            self._state_sub = ChannelSubscriber(TOPIC_LOW_STATE, LowState_)
            self._state_sub.Init(self._on_low_state, 10)

        if self._cmd_pub is None:
            self._cmd_pub = ChannelPublisher(TOPIC_LOW_CMD, LowCmd_)
            self._cmd_pub.Init()

        # 첫 LowState 수신 대기 (최대 2초)
        for _ in range(20):
            with self._state_lock:
                has_state = self._lowstate is not None
            if has_state:
                break
            time.sleep(0.1)

    def _release_sport_mode(self) -> None:
        """sport mode 해제. 자세 변경 없이 바로 해제 — _hold_current_pos()가 공백을 메움."""
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()

        status, result = msc.CheckMode()
        if result.get('name'):
            msc.ReleaseMode()
            time.sleep(0.3)

    def _hold_current_pos(self) -> None:
        """ReleaseMode() 직후 현재 관절 위치를 즉시 hold — 제어 공백 방지."""
        with self._state_lock:
            lowstate = self._lowstate
        if lowstate is not None:
            hold_pos = np.array(
                [lowstate.motor_state[i].q for i in range(NUM_JOINTS)],
                dtype=np.float32
            )
            self._send_low_cmd(hold_pos)

    def _reset_policy(self) -> None:
        self._prev_actions[:] = 0.0
        if self._policy is not None:
            self._policy.reset()

    # ------------------------------------------------------------------

    def _on_low_state(self, msg) -> None:
        with self._state_lock:
            self._lowstate = msg

    def update_cmd_vel(self, vx: float, vy: float, vyaw: float) -> None:
        with self._cmd_lock:
            self._vx   = max(MIN_VX,    min(MAX_VX,    vx))
            self._vy   = max(-MAX_VY,   min(MAX_VY,    vy))
            self._vyaw = max(-MAX_VYAW, min(MAX_VYAW, vyaw))

    def step(self) -> None:
        if self._policy is None:
            return

        with self._state_lock:
            lowstate = self._lowstate
        if lowstate is None:
            return

        obs = self._build_observation(lowstate)
        raw_action = self._policy(obs)

        action = np.clip(raw_action, -1.0, 1.0) * self._action_scale
        self._prev_actions = raw_action.copy()

        # policy 순서 action을 SDK 순서 target_q로 재배열
        target_q = self._default_pos.copy()
        target_q[self._joint_ids_map] = self._default_pos[self._joint_ids_map] + action
        self._send_low_cmd(target_q)

    def _build_observation(self, lowstate) -> np.ndarray:
        obs = np.zeros(self._obs_dim, dtype=np.float32)

        # [0:3] base_ang_vel (scale=0.2, matches training)
        obs[0] = lowstate.imu_state.gyroscope[0] * 0.2
        obs[1] = lowstate.imu_state.gyroscope[1] * 0.2
        obs[2] = lowstate.imu_state.gyroscope[2] * 0.2

        # [3:6] projected_gravity
        quat = [
            lowstate.imu_state.quaternion[0],
            lowstate.imu_state.quaternion[1],
            lowstate.imu_state.quaternion[2],
            lowstate.imu_state.quaternion[3],
        ]
        obs[3:6] = quat_to_projected_gravity(quat)

        # [6:9] velocity_commands
        with self._cmd_lock:
            obs[6] = self._vx
            obs[7] = self._vy
            obs[8] = self._vyaw

        # [9:21] joint_pos_rel, [21:33] joint_vel_rel (scale=0.05)
        # policy 순서로 읽기: motor_state[sdk_i] → obs[policy_i]
        for policy_i, sdk_i in enumerate(self._joint_ids_map):
            obs[9 + policy_i]  = lowstate.motor_state[sdk_i].q - self._default_pos[sdk_i]
            obs[21 + policy_i] = lowstate.motor_state[sdk_i].dq * 0.05

        # [33:45] last_action
        obs[33:45] = self._prev_actions
        return obs

    def _send_low_cmd(self, target_q: np.ndarray, kp=None, kd=None) -> None:
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

        kp = self._kp if kp is None else kp
        kd = self._kd if kd is None else kd

        msg = unitree_go_msg_dds__LowCmd_()
        msg.head[0] = 0xFE
        msg.head[1] = 0xEF
        msg.level_flag = LOWLEVEL
        msg.gpio = 0

        for i in range(20):
            msg.motor_cmd[i].mode = 0x01
            msg.motor_cmd[i].q   = POS_STOP_F
            msg.motor_cmd[i].kp  = 0
            msg.motor_cmd[i].dq  = VEL_STOP_F
            msg.motor_cmd[i].kd  = 0
            msg.motor_cmd[i].tau = 0

        for i in range(NUM_JOINTS):
            msg.motor_cmd[i].q   = float(target_q[i])
            msg.motor_cmd[i].dq  = 0.0
            msg.motor_cmd[i].kp  = float(kp[i])
            msg.motor_cmd[i].kd  = float(kd[i])
            msg.motor_cmd[i].tau = 0.0

        msg.crc = self._crc.Crc(msg)
        self._cmd_pub.Write(msg)

    def stop(self) -> None:
        if self._cmd_pub is not None:
            self._send_low_cmd(self._default_pos)

    def emergency_stop(self) -> None:
        """
        안전 정지(low-level): sport 모드로 넘어가지 않고 현재 채널에서 처리한다.
        현재 자세 → 엎드린 자세(prone)로 LowCmd를 보간 publish해 천천히 내려앉힌 뒤,
        low-level damp(kp=0, kd=passive)로 힘을 뺀다.

        sport takeover에 의존하지 않아 즉시·결정론적 — 비상정지에 적합하다.
        sdk_velocity 모드는 sport가 항상 켜져 있어 StandDown()을 쓰지만,
        nn_policy는 ReleaseMode 상태라 모드 전환 의존을 피하려 low-level로 처리한다.

        cb_realtime 스레드에서 단발로 호출되며(step()과 인터리브 안 됨) 블로킹으로 수행한다.
        """
        if self._cmd_pub is None:
            return

        DT = 0.02            # 50 Hz publish 주기
        DESCENT_S = 1.5      # 내려앉는 데 걸리는 시간
        SETTLE_S = 0.3       # 엎드린 자세 정착 유지 시간

        # 1) 현재 관절각 읽기 (SDK 순서). 상태 없으면 보간 생략하고 바로 damp.
        with self._state_lock:
            lowstate = self._lowstate
        if lowstate is not None:
            start_q = np.array(
                [lowstate.motor_state[i].q for i in range(NUM_JOINTS)],
                dtype=np.float32,
            )

            # 2) 현재 → prone 선형 보간 (compliant 게인으로 부드럽게)
            steps = max(1, int(DESCENT_S / DT))
            for k in range(1, steps + 1):
                alpha = k / steps
                q = (1.0 - alpha) * start_q + alpha * PRONE_JOINT_POS
                self._send_low_cmd(q, kp=KP_ESTOP_DESCENT, kd=KD_ESTOP_DESCENT)
                time.sleep(DT)

            # 3) 엎드린 자세 잠깐 유지 (정착)
            for _ in range(max(1, int(SETTLE_S / DT))):
                self._send_low_cmd(
                    PRONE_JOINT_POS, kp=KP_ESTOP_DESCENT, kd=KD_ESTOP_DESCENT
                )
                time.sleep(DT)

        # 4) damp 전환 — 토크 차단, 바닥에 닿은 상태에서 힘 빠짐
        self._send_damp()

    def _send_damp(self) -> None:
        """모든 모터를 damp 모드(kp=0, 약한 kd)로 전환 — 수동 상태."""
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_

        msg = unitree_go_msg_dds__LowCmd_()
        msg.head[0] = 0xFE
        msg.head[1] = 0xEF
        msg.level_flag = LOWLEVEL
        msg.gpio = 0

        for i in range(20):
            msg.motor_cmd[i].mode = 0x00        # Damp (servo 아님)
            msg.motor_cmd[i].q   = 0.0
            msg.motor_cmd[i].kp  = 0.0
            msg.motor_cmd[i].dq  = 0.0
            msg.motor_cmd[i].kd  = KD_PASSIVE[i % 12]
            msg.motor_cmd[i].tau = 0.0

        msg.crc = self._crc.Crc(msg)
        self._cmd_pub.Write(msg)
