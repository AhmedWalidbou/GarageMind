"""
Unit tests for GarageMind M1 - UDS layer and DTC handling.
Run with: pytest
"""

import pytest

from src.uds.iso_tp import encode, Reassembler, make_flow_control
from src.uds.uds_ecu_sim import (
    UdsEcu, StoredDTC, build_demo_ecu,
    dtc_to_bytes, bytes_to_dtc,
    SID_READ_DTC, SID_READ_DATA_BY_ID, SID_CLEAR_DTC,
    STATUS_CONFIRMED, STATUS_TEST_FAILED,
)
from src.uds.uds_client import UdsClient, decode_status
from src.scan_engine.dtc_reader import interpret_dtc, scan_dtcs, validate_dtc


# ---------- DTC textual encoding/decoding (ISO 14229) ----------

class TestDtcEncoding:
    @pytest.mark.parametrize("code", ["P0301", "P0401", "P2002", "U0100", "C0035", "B1234"])
    def test_roundtrip(self, code):
        """Encoding a DTC to 3 bytes then decoding must return the same code."""
        encoded = dtc_to_bytes(code)
        assert len(encoded) == 3
        decoded = bytes_to_dtc(encoded)
        assert decoded == code

    def test_system_letters(self):
        assert bytes_to_dtc(dtc_to_bytes("P0000"))[0] == "P"
        assert bytes_to_dtc(dtc_to_bytes("C0000"))[0] == "C"
        assert bytes_to_dtc(dtc_to_bytes("B0000"))[0] == "B"
        assert bytes_to_dtc(dtc_to_bytes("U0000"))[0] == "U"


# ---------- DTC interpretation (the reader) ----------

class TestDtcReader:
    def test_valid_format(self):
        assert validate_dtc("P0301") is True
        assert validate_dtc("BADCODE") is False
        assert validate_dtc("P030") is False

    def test_known_code_enriched(self):
        result = interpret_dtc("P0301", lang="fr")
        assert result["valid"] is True
        assert result["in_database"] is True
        assert "cylindre 1" in result["description_fr"].lower()
        assert result["severity_key"] == "high"

    def test_unknown_code_structural(self):
        """An unknown code should still be structurally decoded, not rejected."""
        result = interpret_dtc("P1234", lang="fr")
        assert result["valid"] is True
        assert result["in_database"] is False
        assert result["system_fr"] is not None

    def test_severity_sorting(self):
        report = scan_dtcs(["P0128", "P0301", "P0420"], lang="fr")
        severities = [r["severity_key"] for r in report["results"]]
        # high must come before medium/low
        assert severities[0] == "high"


# ---------- ISO-TP transport ----------

class TestIsoTp:
    def test_single_frame(self):
        msg = bytes([0x22, 0xF1, 0x90])
        frames = encode(msg)
        assert len(frames) == 1
        assert len(frames[0]) == 8

    def test_single_frame_roundtrip(self):
        msg = bytes([0x22, 0xF1, 0x90])
        reasm = Reassembler()
        result = None
        for f in encode(msg):
            result = reasm.push(f)
        assert result == msg

    def test_multiframe_roundtrip(self):
        msg = bytes(range(1, 26))  # 25 bytes -> multi-frame
        frames = encode(msg)
        assert len(frames) > 1
        reasm = Reassembler()
        result = None
        for f in frames:
            out = reasm.push(f)
            if out is not None:
                result = out
        assert result == msg

    def test_large_message(self):
        msg = bytes([i % 256 for i in range(100)])
        reasm = Reassembler()
        result = None
        for f in encode(msg):
            out = reasm.push(f)
            if out is not None:
                result = out
        assert result == msg
        assert len(result) == 100

    def test_sequence_error_detected(self):
        msg = bytes(range(1, 26))
        reasm = Reassembler()
        reasm.push(encode(msg)[0])  # first frame
        bad_cf = bytes([0x23, 0, 0, 0, 0, 0, 0, 0])  # wrong sequence number
        with pytest.raises(ValueError):
            reasm.push(bad_cf)

    def test_flow_control_frame(self):
        fc = make_flow_control(flow_status=0, block_size=0, st_min=10)
        assert len(fc) == 8
        assert (fc[0] >> 4) == 0x3  # flow control PCI


# ---------- ECU simulator ----------

class TestEcuSimulator:
    def test_read_vin(self):
        ecu = build_demo_ecu()
        resp = ecu.request(bytes([SID_READ_DATA_BY_ID, 0xF1, 0x90]))
        assert resp[0] == SID_READ_DATA_BY_ID + 0x40
        vin = resp[3:].decode("ascii")
        assert len(vin) == 17

    def test_read_dtcs_count(self):
        ecu = build_demo_ecu()
        resp = ecu.request(bytes([SID_READ_DTC, 0x02, 0xFF]))
        body = resp[3:]
        assert len(body) % 4 == 0
        assert len(body) // 4 == 3  # demo ECU has 3 DTCs

    def test_clear_dtcs(self):
        ecu = build_demo_ecu()
        ecu.request(bytes([SID_CLEAR_DTC, 0xFF, 0xFF, 0xFF]))
        resp = ecu.request(bytes([SID_READ_DTC, 0x02, 0xFF]))
        assert len(resp[3:]) == 0  # no DTCs left

    def test_unsupported_service(self):
        ecu = build_demo_ecu()
        resp = ecu.request(bytes([0x99]))
        assert resp[0] == 0x7F  # negative response

    def test_freeze_frame_subfunction(self):
        ecu = build_demo_ecu()
        dtc_bytes = dtc_to_bytes("P0301")[:3]
        resp = ecu.request(bytes([SID_READ_DTC, 0x04]) + dtc_bytes)
        assert resp[0] == SID_READ_DTC + 0x40
        assert len(resp) > 7  # contains freeze-frame data


# ---------- Full client integration ----------

class TestUdsClientIntegration:
    def test_full_scan(self):
        client = UdsClient(build_demo_ecu())
        report = client.full_scan(lang="fr")
        assert report["vin"] is not None
        assert report["dtc_count"] == 3
        assert report["interpretation"]["highest_severity"] == "high"

    def test_freeze_frames_attached(self):
        client = UdsClient(build_demo_ecu())
        report = client.full_scan(lang="fr")
        for result in report["interpretation"]["results"]:
            assert "freeze_frame" in result
            if result["code"] == "P0301":
                assert result["freeze_frame"]["engine_rpm"]["value"] == 617

    def test_transport_used(self):
        """Ensure the scan actually went through ISO-TP framing."""
        client = UdsClient(build_demo_ecu())
        client.full_scan()
        assert client.frames_sent > 0
        assert client.frames_received > 0

    def test_status_decoding(self):
        bits = decode_status(STATUS_CONFIRMED | STATUS_TEST_FAILED)
        assert "confirmedDTC" in bits
        assert "testFailed" in bits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])