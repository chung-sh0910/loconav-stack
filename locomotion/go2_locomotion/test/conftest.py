"""
하드웨어 SDK mock — unitree_sdk2py의 .so 의존 모듈을 테스트 수집 전에 가짜로 교체한다.
pytest가 이 파일을 자동으로 로드하므로 각 테스트 파일에서 별도 처리 불필요.
"""
import sys
from unittest.mock import MagicMock

def _mock_if_absent(module_name, **attrs):
    if module_name not in sys.modules:
        m = MagicMock()
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[module_name] = m

# CRC — crc_amd64.so 로드를 막기 위해 통째로 교체
mock_crc_instance = MagicMock()
mock_crc_instance.Crc.return_value = 0xDEADBEEF
MockCRC = MagicMock(return_value=mock_crc_instance)
_mock_if_absent("unitree_sdk2py.utils.crc", CRC=MockCRC)

# DDS channel — 실제 네트워크 없이 import만 되도록
_mock_if_absent("unitree_sdk2py.core.channel")
_mock_if_absent("unitree_sdk2py.idl.unitree_go.msg.dds_")
_mock_if_absent("unitree_sdk2py.idl.default")
