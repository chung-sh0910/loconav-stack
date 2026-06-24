import os
from pathlib import Path

import streamlit as st

ROOT_DEFAULT = "/workspace/isaaclab/logs/rsl_rl"


@st.cache_data(ttl=30)
def scan_videos(root: str) -> dict:
    """Scan root → task → timestamp → videos/mode → [*.mp4]"""
    result = {}
    root_path = Path(root)
    if not root_path.is_dir():
        return result

    for task_entry in sorted(root_path.iterdir()):
        if not task_entry.is_dir():
            continue
        timestamps = {}
        for ts_entry in sorted(task_entry.iterdir(), reverse=True):
            if not ts_entry.is_dir():
                continue
            videos_dir = ts_entry / "videos"
            if not videos_dir.is_dir():
                continue
            modes = {}
            for mode_entry in sorted(videos_dir.iterdir()):
                if not mode_entry.is_dir():
                    continue
                mp4_files = sorted(mode_entry.glob("*.mp4"))
                if mp4_files:
                    modes[mode_entry.name] = mp4_files
            if modes:
                timestamps[ts_entry.name] = modes
        if timestamps:
            result[task_entry.name] = timestamps

    return result


def main():
    st.set_page_config(page_title="IsaacLab Video Finder", layout="wide")

    root = os.environ.get("ISAACLAB_LOG_DIR", ROOT_DEFAULT)

    with st.sidebar:
        st.title("IsaacLab Videos")
        st.caption(f"Root: `{root}`")

        if st.button("새로고침"):
            scan_videos.clear()
            st.rerun()

        data = scan_videos(root)

        if not data:
            st.error(f"비디오를 찾을 수 없습니다.\n\n경로를 확인하세요:\n`{root}`")
            st.stop()

        task = st.selectbox("Task", list(data.keys()))
        timestamps = data[task]

        timestamp = st.selectbox("Timestamp", list(timestamps.keys()))
        modes = timestamps[timestamp]

        mode = st.selectbox("Mode", list(modes.keys()))
        mp4_files = modes[mode]

    st.header(f"{task} / {timestamp} / {mode}")

    if not mp4_files:
        st.info("비디오 파일이 없습니다.")
        return

    for path in mp4_files:
        st.subheader(path.name)
        st.video(str(path))


if __name__ == "__main__":
    main()
