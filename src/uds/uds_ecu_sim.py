"""
UDS ECU Simulator - GarageMind UDS layer
A virtual Electronic Control Unit (ECU) that answers UDS diagnostic requests
over ISO-TP, exactly as a real vehicle controller would.

Implements the core UDS services used by every diagnostic tool:
  0x10 DiagnosticSessionControl
  0x22 ReadDataByIdentifier      (live parameters via DID)
  0x19 ReadDTCInformation        (stored fault codes + status)
  0x14 ClearDiagnosticInformation

DTCs are stored as a 3-byte code + 1 status byte, per ISO 14229.
"""

from dataclasses import dataclass, field


# --- UDS service IDs ---
SID_DIAGNOSTIC_SESSION = 0x10
SID_READ_DATA_BY_ID = 0x22
SID_READ_DTC = 0x19
SID_CLEAR_DTC = 0x14

POSITIVE_RESPONSE_OFFSET = 0x40  # response SID = request SID + 0x40
NEGATIVE_RESPONSE = 0x7F

# --- Negative Response Codes (NRC) ---
NRC_SERVICE_NOT_SUPPORTED = 0x11
NRC_SUBFUNCTION_NOT_SUPPORTED = 0x12
NRC_REQUEST_OUT_OF_RANGE = 0x31

# --- DTC status bits (ISO 14229) ---
STATUS_TEST_FAILED = 0x01
STATUS_CONFIRMED = 0x08
STATUS_TEST_FAILED_THIS_CYCLE = 0x04


def dtc_to_bytes(code: str) -> bytes:
    """
    Convert a textual DTC like 'P0301' to its 3-byte ISO 14229 encoding.
    First 2 bits = system letter, remaining bits = the hex digits.
    """
    system_map = {"P": 0b00, "C": 0b01, "B": 0b10, "U": 0b11}
    system = system_map[code[0].upper()]
    first_digit = int(code[1], 16)
    high_byte = (system << 6) | (first_digit << 4) | int(code[2], 16)
    mid_byte = int(code[3], 16) << 4 | int(code[4], 16)
    return bytes([high_byte, mid_byte, 0x00])


def bytes_to_dtc(data: bytes) -> str:
    """Convert a 3-byte ISO 14229 DTC back into textual form like 'P0301'."""
    system_map = {0b00: "P", 0b01: "C", 0b10: "B", 0b11: "U"}
    high = data[0]
    system = system_map[(high >> 6) & 0b11]
    first_digit = (high >> 4) & 0b11
    second_digit = high & 0x0F
    mid = data[1]
    third_digit = (mid >> 4) & 0x0F
    fourth_digit = mid & 0x0F
    return f"{system}{first_digit}{second_digit:X}{third_digit:X}{fourth_digit:X}"


@dataclass
class StoredDTC:
    """A fault code as stored in the ECU memory, with freeze-frame data."""
    code: str
    status: int = STATUS_CONFIRMED | STATUS_TEST_FAILED
    freeze_frame: dict = field(default_factory=dict)
    occurrence_count: int = 1


@dataclass
class UdsEcu:
    """
    A simulated ECU. Holds an identity, a set of readable data identifiers
    (DIDs), and a list of stored DTCs. Responds to UDS requests.
    """
    name: str = "Engine ECU"
    dids: dict = field(default_factory=dict)
    dtcs: list = field(default_factory=list)
    session: int = 0x01  # default session

    def request(self, payload: bytes) -> bytes:
        """Process one UDS request payload and return the response payload."""
        if not payload:
            return self._negative(0x00, NRC_REQUEST_OUT_OF_RANGE)

        sid = payload[0]

        if sid == SID_DIAGNOSTIC_SESSION:
            return self._handle_session(payload)
        if sid == SID_READ_DATA_BY_ID:
            return self._handle_read_did(payload)
        if sid == SID_READ_DTC:
            return self._handle_read_dtc(payload)
        if sid == SID_CLEAR_DTC:
            return self._handle_clear_dtc(payload)

        return self._negative(sid, NRC_SERVICE_NOT_SUPPORTED)

    def _handle_session(self, payload: bytes) -> bytes:
        if len(payload) < 2:
            return self._negative(SID_DIAGNOSTIC_SESSION, NRC_SUBFUNCTION_NOT_SUPPORTED)
        session_type = payload[1]
        self.session = session_type
        # positive response: echo session + dummy timing params
        return bytes([SID_DIAGNOSTIC_SESSION + POSITIVE_RESPONSE_OFFSET,
                      session_type, 0x00, 0x32, 0x01, 0xF4])

    def _handle_read_did(self, payload: bytes) -> bytes:
        if len(payload) < 3:
            return self._negative(SID_READ_DATA_BY_ID, NRC_REQUEST_OUT_OF_RANGE)
        did = (payload[1] << 8) | payload[2]
        if did not in self.dids:
            return self._negative(SID_READ_DATA_BY_ID, NRC_REQUEST_OUT_OF_RANGE)
        value = self.dids[did]
        return bytes([SID_READ_DATA_BY_ID + POSITIVE_RESPONSE_OFFSET,
                      payload[1], payload[2]]) + value

    def _handle_read_dtc(self, payload: bytes) -> bytes:
        if len(payload) < 2:
            return self._negative(SID_READ_DTC, NRC_SUBFUNCTION_NOT_SUPPORTED)
        sub = payload[1]
        resp_sid = SID_READ_DTC + POSITIVE_RESPONSE_OFFSET
        if sub == 0x01:
            count = len(self.dtcs)
            return bytes([resp_sid, sub, 0xFF, 0x01]) + count.to_bytes(2, "big")
        if sub == 0x02:
            response = bytes([resp_sid, sub, 0xFF])
            for dtc in self.dtcs:
                response += dtc_to_bytes(dtc.code)[:3] + bytes([dtc.status])
            return response
        if sub == 0x04:
            if len(payload) < 5:
                return self._negative(SID_READ_DTC, NRC_REQUEST_OUT_OF_RANGE)
            requested = payload[2:5]
            target_code = bytes_to_dtc(requested)
            for dtc in self.dtcs:
                if dtc.code == target_code:
                    response = bytes([resp_sid, sub]) + requested + bytes([dtc.status, 0x01])
                    for did, value in dtc.freeze_frame.items():
                        response += did.to_bytes(2, "big") + int(value).to_bytes(2, "big", signed=True)
                    return response
            return self._negative(SID_READ_DTC, NRC_REQUEST_OUT_OF_RANGE)
        return self._negative(SID_READ_DTC, NRC_SUBFUNCTION_NOT_SUPPORTED)

    def _handle_clear_dtc(self, payload: bytes) -> bytes:
        cleared = len(self.dtcs)
        self.dtcs = []
        return bytes([SID_CLEAR_DTC + POSITIVE_RESPONSE_OFFSET]) + \
            cleared.to_bytes(1, "big")

    def _negative(self, sid: int, nrc: int) -> bytes:
        return bytes([NEGATIVE_RESPONSE, sid, nrc])


def build_demo_ecu() -> UdsEcu:
    """A demo engine ECU with realistic identity, live data and stored faults."""
    return UdsEcu(
        name="Engine ECU (simulated)",
        dids={
            0xF190: b"WAUZZZ8V1JA123456",       # VIN (DID F190)
            0xF18C: b"ECU-GM-2019-0472",         # ECU serial number
            0xF195: bytes([0x01, 0x04]),         # software version 1.4
        },
        dtcs=[
            StoredDTC("P0301",
                      STATUS_CONFIRMED | STATUS_TEST_FAILED,
                      freeze_frame={0xF40C: 617, 0xF405: 92, 0xF40D: 0},
                      occurrence_count=4),
            StoredDTC("P0401",
                      STATUS_CONFIRMED,
                      freeze_frame={0xF40C: 1850, 0xF405: 88, 0xF40D: 45},
                      occurrence_count=2),
            StoredDTC("P2002",
                      STATUS_CONFIRMED | STATUS_TEST_FAILED_THIS_CYCLE,
                      freeze_frame={0xF40C: 2400, 0xF405: 95, 0xF40D: 70},
                      occurrence_count=1),
        ],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("GarageMind - UDS ECU simulator demo")
    print("=" * 60)

    ecu = build_demo_ecu()
    print(f"\nECU: {ecu.name}")

    # 1. Start an extended diagnostic session (0x10 0x03)
    print("\n[0x10] DiagnosticSessionControl (extended):")
    resp = ecu.request(bytes([0x10, 0x03]))
    print(f"  request : 1003")
    print(f"  response: {resp.hex()}")

    # 2. Read the VIN (0x22 0xF190)
    print("\n[0x22] ReadDataByIdentifier (VIN, DID F190):")
    resp = ecu.request(bytes([0x22, 0xF1, 0x90]))
    vin = resp[3:].decode("ascii", errors="replace")
    print(f"  request : 22f190")
    print(f"  response: {resp.hex()}")
    print(f"  decoded VIN: {vin}")

    # 3. Read stored DTCs (0x19 0x02)
    print("\n[0x19] ReadDTCInformation (by status mask):")
    resp = ecu.request(bytes([0x19, 0x02, 0xFF]))
    print(f"  request : 1902ff")
    print(f"  response: {resp.hex()}")
    print("  stored DTCs:")
    body = resp[3:]
    for i in range(0, len(body), 4):
        code = bytes_to_dtc(body[i:i + 3])
        status = body[i + 3]
        print(f"    {code}  status=0x{status:02X}")

    # 4. Clear DTCs (0x14)
    print("\n[0x14] ClearDiagnosticInformation:")
    resp = ecu.request(bytes([0x14, 0xFF, 0xFF, 0xFF]))
    print(f"  response: {resp.hex()} (cleared {resp[1]} codes)")

    # 5. Read DTCs again (should be empty now)
    resp = ecu.request(bytes([0x19, 0x02, 0xFF]))
    remaining = (len(resp) - 3) // 4
    print(f"  DTCs remaining after clear: {remaining}")

    # 6. Unsupported service -> negative response
    print("\n[0x99] Unsupported service:")
    resp = ecu.request(bytes([0x99]))
    print(f"  response: {resp.hex()} (7F = negative, NRC=0x{resp[2]:02X})")