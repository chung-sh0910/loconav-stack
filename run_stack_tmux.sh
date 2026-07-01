#!/usr/bin/env bash
# joystick / motion_controller / locomotion 를 tmux 한 창에 3분할로 띄운다.
# joystick, motion_controller는 바로 실행하고, locomotion은 대기 상태(cd만, 실행 X)로 둔다.
#
# Usage: ./run_stack_tmux.sh
# 이미 같은 이름 세션이 떠 있으면 새로 만들지 않고 그냥 붙는다(attach).

set -e

SESSION="go2_stack"
SMLL_ROOT="/home/csh/smll_project"
JOYSTICK_DIR="$SMLL_ROOT/tools/joystick"
MOTION_DIR="$SMLL_ROOT/tools/motion_controller"
LOCOMOTION_DIR="$SMLL_ROOT/locomotion/go2_locomotion/go2_locomotion"

ROS_SETUP="source /opt/ros/foxy/setup.bash && source $SMLL_ROOT/install/setup.bash"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "세션 '$SESSION' 이미 존재 — 붙습니다."
    tmux attach -t "$SESSION"
    exit 0
fi

# pane 0: joystick — 바로 실행
tmux new-session -d -s "$SESSION" -n main -c "$JOYSTICK_DIR"
tmux send-keys -t "${SESSION}:main.0" "$ROS_SETUP && python3 with_ros2.py" C-m

# pane 1: motion_controller — 바로 실행
tmux split-window -h -t "${SESSION}:main" -c "$MOTION_DIR"
tmux send-keys -t "${SESSION}:main.1" "$ROS_SETUP && python3 with_ros2.py" C-m

# pane 2: locomotion — ROS2 환경만 세팅하고 대기 (직접 실행할 때까지 명령 안 침)
tmux split-window -h -t "${SESSION}:main" -c "$LOCOMOTION_DIR"
tmux send-keys -t "${SESSION}:main.2" "$ROS_SETUP" C-m

# 3분할을 균등한 너비로, pane 제목이 보이도록
tmux select-layout -t "${SESSION}:main" even-horizontal
tmux set-option -t "$SESSION" -g pane-border-status top
tmux set-option -t "$SESSION" -g pane-border-format "#{pane_index}: #{pane_title}"
tmux select-pane -t "${SESSION}:main.0" -T "joystick"
tmux select-pane -t "${SESSION}:main.1" -T "motion_controller"
tmux select-pane -t "${SESSION}:main.2" -T "locomotion (standby)"

tmux attach -t "$SESSION"
