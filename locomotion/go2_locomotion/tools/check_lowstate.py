#!/usr/bin/env python3
"""
실제 Unitree Go2 로봇에서 rt/lowstate 상태가 제대로 수신되는지 확인하는 진단 스크립트.

사용법:
  python3 tools/check_lowstate.py [--interface eth0] [--count 10] [--interval 0.5]

  --interface : 네트워크 인터페이스 (기본값: eth0)
  --count     : 출력할 상태 횟수 (0이면 무한, 기본값: 0)
  --interval  : 출력 간격(초) (기본값: 0.5)
"""

import argparse
import signal
import sys
import time
import threading

JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]


def parse_args():
    p = argparse.ArgumentParser(description="Go2 LowState 수신 진단")
    p.add_argument("--interface", default="eth0", help="네트워크 인터페이스 (기본값: eth0)")
    p.add_argument("--count",     type=int, default=0, help="출력 횟수 (0=무한, 기본값: 0)")
    p.add_argument("--interval",  type=float, default=0.5, help="출력 간격 (초, 기본값: 0.5)")
    return p.parse_args()


def main():
    args = parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    ChannelFactoryInitialize(0, args.interface)

    state_lock = threading.Lock()
    latest_state = [None]
    recv_count = [0]

    def on_low_state(msg):
        with state_lock:
            latest_state[0] = msg
            recv_count[0] += 1

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_low_state, 10)

    print(f"[check_lowstate] 인터페이스={args.interface}, rt/lowstate 수신 대기 중...")

    # 첫 메시지 대기 (최대 5초)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with state_lock:
            if latest_state[0] is not None:
                break
        time.sleep(0.05)
    else:
        print("[ERROR] 5초 동안 LowState 수신 없음 — 로봇 연결 또는 인터페이스 확인 필요")
        sys.exit(1)

    print("[OK] LowState 수신 시작\n")

    stop = threading.Event()

    def on_sigint(sig, frame):
        stop.set()

    signal.signal(signal.SIGINT, on_sigint)

    iteration = 0
    while not stop.is_set():
        time.sleep(args.interval)

        with state_lock:
            msg = latest_state[0]
            total = recv_count[0]

        if msg is None:
            continue

        iteration += 1
        _print_state(msg, total, iteration)

        if args.count > 0 and iteration >= args.count:
            break

    print("\n[check_lowstate] 종료")


def _print_state(msg, total_recv: int, iteration: int):
    imu = msg.imu_state
    qw, qx, qy, qz = (imu.quaternion[i] for i in range(4))
    gx, gy, gz = (imu.gyroscope[i] for i in range(3))
    ax, ay, az = (imu.accelerometer[i] for i in range(3))

    print(f"─── 수신 #{iteration}  (누적 메시지: {total_recv}) ──────────────────────────")

    print(f"  IMU 쿼터니언   w={qw:+.4f}  x={qx:+.4f}  y={qy:+.4f}  z={qz:+.4f}")
    print(f"  자이로(rad/s)  x={gx:+.4f}  y={gy:+.4f}  z={gz:+.4f}")
    print(f"  가속도(m/s²)   x={ax:+.4f}  y={ay:+.4f}  z={az:+.4f}")

    print(f"  {'관절':<12}  {'q(rad)':>9}  {'dq(rad/s)':>10}  {'tau(Nm)':>8}  {'온도(℃)':>7}")
    for i, name in enumerate(JOINT_NAMES):
        ms = msg.motor_state[i]
        print(f"  {name:<12}  {ms.q:>+9.4f}  {ms.dq:>+10.4f}  {ms.tau_est:>+8.3f}  {ms.temperature:>7.1f}")

    print(f"  foot_force     FL={msg.foot_force[0]:>5}  FR={msg.foot_force[1]:>5}  RL={msg.foot_force[2]:>5}  RR={msg.foot_force[3]:>5}")
    print()


if __name__ == "__main__":
    main()
