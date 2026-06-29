# Vast.ai + Isaac Lab + unitree_rl_lab 환경 세팅 가이드
> 실제 세팅하면서 겪은 이슈와 해결책을 전부 정리. 다음에 새 인스턴스 만들 때 이 문서대로만 하면 막힘 없이 진행 가능.

---

## 0. 시작 전 체크리스트

- [ ] Vast.ai 계정 + **크레딧 충전** (Credit $0이면 RENT 불가)
- [ ] 로컬에 SSH 키 존재 확인: `ls ~/.ssh/` → `id_ed25519`, `id_ed25519.pub`
- [ ] Vast.ai에 **공개키 등록**: Account → SSH Keys → Add SSH Key (`cat ~/.ssh/id_ed25519.pub` 내용 붙여넣기)
- [ ] VS Code + Remote-SSH 익스텐션 설치 (로컬)

---

## 1. 인스턴스 선택 기준 (실패 줄이는 필터)

| 항목 | 기준 | 이유 |
|---|---|---|
| Reliability | **99% 이상** | 학습 중 중단 방지 |
| 타입 | **On-Demand** (Interruptible 금지) | 장시간 학습 회수 방지 |
| 위치 | **Asia / Korea** | rsync·SSH 레이턴시 |
| VRAM | 16GB+ (5080) / 32GB(5090) | depth cam 쓰면 5090 |
| Disk | **200GB로 설정** | Isaac Lab 이미지가 큼 |

### GPU별 판단
- **RTX 5080 (16GB)**: blind/heightmap 4096 env OK, depth cam은 2048 env 이하
- **RTX 5090 (32GB)**: 전부 여유. depth cam 쓸 거면 권장

---

## 2. ★ 가장 중요한 이슈: Launch Type을 SSH로

### 겪은 문제
- 기본 템플릿이 **Jupyter 모드**로 켜짐
- `Open Jupyter terminal` → `218.55.235.21:30383` 연결 시도 → **Unable to connect** (포트 안 열림)
- Jupyter 포트가 제대로 노출 안 되는 케이스 빈번

### 해결
- 인스턴스 생성 시 **Launch Type = SSH** 선택 (Jupyter 아님)
- **SSH 키는 인스턴스 생성 전에 등록**해야 함
  - "Adding a key only applies to **new** instances" → 기존 인스턴스엔 자동 적용 안 됨
  - 키 등록 안 하고 만들었으면 → 인스턴스 삭제 후 재생성이 깔끔

---

## 3. Docker 이미지 설정

### 템플릿에 Isaac Lab이 없음 (37개 템플릿 중 없음)
→ 직접 입력해야 함

### 방법 (PyTorch Vast 템플릿 기반 수정 권장)
1. Template에서 **PyTorch (Vast)** 선택 (NGC 공식 템플릿은 수정 시 찜찜하므로 회피)
2. 연필(✏️) 아이콘 → Edit
3. **Image Path:Tag** 필드만 교체:
   ```
   nvcr.io/nvidia/isaac-lab:2.3.2
   ```
4. 나머지(Version Tag, Docker Options, Ports, Env Vars)는 **건드리지 않음**
5. Container Size: **200GB**
6. 저장

### 주의
- 이미지가 30~50GB라 첫 pull에 시간 걸림 (켜놓고 대기)

---

## 4. SSH / 포트 이슈

### 겪은 문제
- `ssh vastai` → `Connection timed out`: **포트번호를 잘못 넣음** (8080 가정 → 실제 다름)
- `Permission denied (publickey)`: **키 파일 미지정**

### 해결
- 정확한 포트는 인스턴스 **`>_` 아이콘** 클릭 → SSH 커맨드의 `-p` 뒤 숫자
- 항상 **키 명시**:
  ```bash
  ssh -p <포트> root@<IP> -i ~/.ssh/id_ed25519
  ```
- `~/.ssh/config`에 등록해두면 편함:
  ```
  Host vastai
      HostName <IP>
      Port <포트>
      User root
      IdentityFile ~/.ssh/id_ed25519
  ```
  → 이후 `ssh vastai`, VS Code도 이 host로 연결

### 포트포워딩 (TensorBoard/Streamlit 등)
```bash
# 로컬에서
ssh -p <포트> root@<IP> -L 6006:localhost:6006 -N -i ~/.ssh/id_ed25519
# -i 키 빼먹으면 publickey 에러남
```

---

## 5. 레포 클론 & 경로 이슈 ★ 자주 막힘

### 겪은 문제 1: 클론 위치
- 작업 디렉토리가 `/root/workspace/` 였음 (`/root/`가 아님)
- 가이드의 `~/unitree_rl_lab` 경로가 실제와 안 맞아서 `sed: can't read ...` 에러

### 해결: 경로 먼저 확인하는 습관
```bash
find ~ -name "unitree.py" 2>/dev/null
find / -name "isaaclab.sh" 2>/dev/null
```
→ 실제 경로 확인 후 명령어의 경로를 맞춤

### 클론 순서 (실제 동작한 버전)
```bash
cd ~/workspace   # 실제 작업 디렉토리 확인 후 이동
git clone https://github.com/unitreerobotics/unitree_rl_lab.git
git clone https://github.com/unitreerobotics/unitree_ros.git
```

### 겪은 문제 2: UNITREE_ROS_DIR 경로 치환
- sed 대상 경로가 실제 클론 위치와 달라 실패
- 실제 경로에 맞춰 수정:
  ```bash
  sed -i 's|UNITREE_ROS_DIR = "path/to/unitree_ros"|UNITREE_ROS_DIR = "/root/workspace/unitree_ros/unitree_ros"|' \
    ~/workspace/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
  ```
- ★ 치환 후 반드시 확인:
  ```bash
  grep "UNITREE_ROS_DIR" .../assets/robots/unitree.py
  ```

### 패키지 설치
```bash
pip install -e ~/workspace/unitree_rl_lab/source/unitree_rl_lab/
```

---

## 6. ★ USD 파일 없음 이슈 (unitree_rl_lab의 큰 함정)

### 겪은 문제
```
FileNotFoundError: USD file not found at path at: 'path/to/unitree_model/Go2/usd/go2.usd'
```
- unitree_ros에는 **URDF만 있고 USD가 없음**
- `find ~/workspace/unitree_ros -name "*.usd"` → 결과 없음
- Isaac Lab은 URDF 직접 못 씀, **USD 필요**

### 해결책 A (가장 빠름): Isaac Lab 공식 Go2 task 사용
unitree_rl_lab 대신 Isaac Lab 내장 task가 USD 경로까지 다 세팅돼 있음:
```bash
cd /workspace/isaaclab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless
```
등록된 task 확인:
```bash
cat /workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/__init__.py
```
사용 가능: `Isaac-Velocity-Flat-Unitree-Go2-v0`, `...-Rough-...`, `...-Play-v0`

### 해결책 B: URDF → USD 변환 (unitree_rl_lab 쓸 때)
```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /root/workspace/unitree_ros/robots/go2_description/urdf/go2_description.urdf \
  /root/workspace/unitree_ros/robots/go2_description/usd/go2.usd \
  --headless
```

---

## 7. ★ conda / 실행 스크립트 이슈

### 겪은 문제
- `./unitree_rl_lab.sh` 실행 → `No conda environment activated` + `Permission denied`
- `conda: command not found` (이미지에 conda 없음)

### 원인
- Isaac Lab 도커 이미지는 conda 안 씀. 시스템 python3 + `isaaclab.sh` 래퍼 사용

### 해결: isaaclab.sh를 통해 실행
```bash
# python 위치 확인
which python3            # /usr/bin/python3
find / -name "isaaclab.sh" 2>/dev/null   # /workspace/isaaclab/isaaclab.sh

# 항상 isaaclab.sh -p 로 실행
cd /workspace/isaaclab && ./isaaclab.sh -p <스크립트> --task <task> --headless
```

---

## 8. 학습 실행 (실제 동작 버전)

```bash
cd /workspace/isaaclab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --headless
```

### tmux로 백그라운드 (SSH 끊겨도 유지)
```bash
tmux new -s train          # 새 세션
# (학습 실행)
# Ctrl+B, D 로 detach
tmux attach -t train       # 다시 붙기
# 세션 안에서 새 창: Ctrl+B, C  /  창 이동: Ctrl+B, 0~9
```

### 인자 전달 주의
- 백슬래시(`\`) 줄바꿈이 깨지면 `--task`가 None으로 들어가
  `AttributeError: 'NoneType' object has no attribute 'split'` 발생
- 안 되면 **한 줄로 붙여서** 실행

---

## 9. 결과 확인

### 로그/체크포인트 위치
```bash
find /workspace/isaaclab/logs -type d
# /workspace/isaaclab/logs/rsl_rl/unitree_go2_flat/<날짜시각>/model_*.pt
```

### TensorBoard
```bash
# 서버
tensorboard --logdir=/workspace/isaaclab/logs/rsl_rl/unitree_go2_flat --port=6006
# 로컬 (포트포워딩, 키 명시 필수)
ssh -p <포트> root@<IP> -L 6006:localhost:6006 -N -i ~/.ssh/id_ed25519
# 브라우저: http://localhost:6006
```

### 영상 저장 (play 모드, checkpoint 절대경로 권장)
```bash
cd /workspace/isaaclab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Unitree-Go2-Play-v0 --num_envs 16 --video --video_length 200 --headless --load_run <날짜시각> --checkpoint /workspace/isaaclab/logs/rsl_rl/unitree_go2_flat/<날짜시각>/model_299.pt
```
- ★ `--checkpoint model_299.pt`만 주면 `Unable to find the file` 날 수 있음 → **절대경로**로

### 영상 로컬로 가져오기 (scp, 대문자 -P)
```bash
scp -P <포트> -r root@<IP>:/workspace/isaaclab/logs/rsl_rl/unitree_go2_flat/<날짜시각>/videos/play ~/Downloads/
```

---

## 10. Streamlit / 추가 패키지 설치 시 (도커 환경 보호)

### 겪은 고민
- venv 만들었더니 bash에서 안 읽힘 / 도커 환경 깰까 걱정

### 해결
```bash
# 시스템 건드리지 말고 --user 로
pip install streamlit --user
python -m streamlit run app.py
# 외부 접속은 포트포워딩으로 (External URL 직접 안 열림)
ssh -p <포트> root@<IP> -L 8501:localhost:8501 -N -i ~/.ssh/id_ed25519
```
- 가능하면 추가 설치 최소화. 모니터링은 이미 있는 **TensorBoard** 우선

---

## 11. ★ 비용 관리 (까먹으면 돈 샌다)

| 동작 | GPU 과금 | 스토리지 과금 |
|---|---|---|
| 인스턴스 ON | O | O |
| **Stop** | X | **O (계속)** |
| **Delete** | X | X |

- **Stop만으론 스토리지 과금 안 멈춤. 완전히 끝나면 Delete.**
- 크레딧 0 → 자동 Stop(삭제 아님), 카드 등록돼 있으면 자동 청구됨
- 루틴: **학습 완료 → checkpoint scp로 백업 → Delete**
- 코드는 GitHub에 push 해두면 재생성 시 git clone 한 줄로 복구
- Docker 이미지는 캐싱되어 매번 50GB 다시 안 받음

### 비용 감각 (RTX 5080 $0.272/hr 기준)
- 상시 24h: 약 $195/월
- 하루 8h: 약 $65/월
- 1회 학습 72h: 약 $20

---

## 12. 다음 세팅 시 빠른 실행 순서 (요약)

```bash
# (1) 인스턴스: SSH 모드 + isaac-lab:2.3.2 + 200GB + Reliability 99%+ On-Demand
# (2) 키 등록은 생성 전에. 연결:
ssh -p <포트> root@<IP> -i ~/.ssh/id_ed25519

# (3) 경로 확인
find / -name "isaaclab.sh" 2>/dev/null

# (4) 바로 공식 Go2 task로 학습 (USD 문제 회피)
cd /workspace/isaaclab && ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Unitree-Go2-v0 --headless

# (5) tmux 권장, TensorBoard 포트포워딩으로 모니터링
# (6) 끝나면 scp 백업 → Delete
```

---

## 핵심 교훈 5가지
1. **SSH 모드로 생성** (Jupyter 모드 포트 안 열림), 키는 **생성 전** 등록
2. 명령어 경로는 **항상 find로 실제 위치 확인** 후 사용
3. unitree_rl_lab은 **USD 없음** → Isaac Lab 공식 Go2 task가 가장 빠름
4. 도커엔 conda 없음 → **isaaclab.sh -p** 로 실행
5. **Stop ≠ 과금중단**. 끝나면 반드시 **Delete**, 그 전에 scp 백업