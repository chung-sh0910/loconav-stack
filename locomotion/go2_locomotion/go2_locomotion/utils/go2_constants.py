import numpy as np

JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

NUM_JOINTS = 12

# Standing pose from unitree_rl_lab go2 config (FixStand)
DEFAULT_JOINT_POS = np.array([
    -0.1, 0.8, -1.5,   # FR
     0.1, 0.8, -1.5,   # FL
    -0.1, 1.0, -1.5,   # RR
     0.1, 1.0, -1.5,   # RL
], dtype=np.float32)

# PD gains — FixStand from unitree_rl_lab/deploy/robots/go2/config/config.yaml
# (hip=60/5, thigh=80/4, calf=80/4 — thigh/calf stiffer than SDK example's uniform 60/5)
# 일어서기/hold(start·recover·stop)에서만 사용. RL 보행 중에는 KP_POLICY/KD_POLICY 사용.
KP_DEFAULT = [60.0, 80.0, 80.0, 60.0, 80.0, 80.0, 60.0, 80.0, 80.0, 60.0, 80.0, 80.0]
KD_DEFAULT = [5.0,  4.0,  4.0,  5.0,  4.0,  4.0,  5.0,  4.0,  4.0,  5.0,  4.0,  4.0]

# RL 보행(Velocity) 게인 — 학습 actuator = 정책 export params/deploy.yaml = stiffness 25 / damping 0.5.
# 공식 deploy(State_RLBase.h)가 RL 정책 구간에 joint_stiffness/joint_damping로 넣는 값과 동일.
# FixStand의 60~80/4~5를 정책에 쓰면 과다 stiffness + 과다 damping으로 동적 보행이 안 나온다.
KP_POLICY = [25.0] * 12
KD_POLICY = [0.5] * 12

# Passive (emergency stop) gains — Passive state in unitree_rl_lab config: kp=0, kd=3
KP_PASSIVE = [0.0] * 12
KD_PASSIVE = [3.0] * 12

# Emergency-stop 내려앉기 게인 — 서있는 게인(60/80)보다 낮춰 compliant하게
# (덜 뻣뻣하게 엎드림). damp 직전 보간 구간에서만 사용.
KP_ESTOP_DESCENT = [40.0] * 12
KD_ESTOP_DESCENT = [4.0] * 12

# Prone / lie-down pose (SDK order) — Unitree go2 low-level example _targetPos_3.
# Emergency stop이 damp 전에 이 자세로 천천히 내려앉힌다 (다리 완전히 접음).
PRONE_JOINT_POS = np.array([
    -0.35, 1.36, -2.65,   # FR
     0.35, 1.36, -2.65,   # FL
    -0.50, 1.36, -2.65,   # RR
     0.50, 1.36, -2.65,   # RL
], dtype=np.float32)

# Joint index mapping: policy order (Isaac Lab) → SDK motor index
# deploy.yaml: joint_ids_map: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
# policy[i] commands SDK joint JOINT_IDS_MAP[i]
JOINT_IDS_MAP = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8], dtype=np.intp)

# Velocity command limits
MAX_VX   =  1.0   # m/s forward
MIN_VX   = -0.5   # m/s backward
MAX_VY   =  0.4   # m/s lateral
MAX_VYAW =  1.0   # rad/s yaw rate

# Safety stop values for motors not under active control
POS_STOP_F = 2.146e9   # position stop flag (from unitree_legged_const)
VEL_STOP_F = 16000.0   # velocity stop flag

# Low-level mode flag
LOWLEVEL = 0xFF

# DDS topic names (Go2 uses unitree_go namespace)
TOPIC_LOW_STATE = "rt/lowstate"
TOPIC_LOW_CMD   = "rt/lowcmd"
