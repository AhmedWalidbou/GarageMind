"""
UDS Client - GarageMind UDS layer
The diagnostic tester (the "scan tool" side). Sends UDS requests to an ECU
THROUGH the ISO-TP transport layer, reassembles the responses, and interprets
stored DTCs by connecting to the DTC reader.

This closes the M1 chain:
    ECU (faults) -> UDS -> ISO-TP -> raw DTC bytes -> DTC interpretation
"""

from src.uds.iso_tp import encode, Reassembler
from src.uds.uds_ecu_sim import (
    UdsEcu, build_demo_ecu, bytes_to_dtc,
    SID_DIAGNOSTIC_SESSION, SID_READ_DATA_BY_ID, SID_READ_DTC, SID_CLEAR_DTC,
    NEGATIVE_RESPONSE, POSITIVE_RESPONSE_OFFSET,
)
from src.scan_engine.dtc_reader import scan_dtcs


# DTC status bit meanings (ISO 14229), for human-readable reporting
STATUS_BITS = {
    0x01: "testFailed",
    0x02: "testFailedThisOperationCycle",
    0x04: "pendingDTC",
    0x08: "confirmedDTC",
    0x10: "testNotCompletedSinceLastClear",
    0x20: "testFailedSinceLastClear",
    0x40: "testNotCompletedThisOperationCycle",
    0x80: "warningIndicatorRequested",
}


def decode_status(status: int) -> list[str]:
    """Expand a DTC status byte into the list of active status bits."""
    return [name for bit, name in STATUS_BITS.items() if status & bit]


class UdsClient:
    """
    A diagnostic client that talks to an ECU over ISO-TP.
    Every request is segmented into CAN frames and every response reassembled,
    faithfully simulating the real transport path.
    """

    def __init__(self, ecu: UdsEcu):
        self.ecu = ecu
        self.frames_sent = 0
        self.frames_received = 0

    def _transact(self, request_payload: bytes) -> bytes:
        """
        Send a UDS request through ISO-TP and get the reassembled response.
        Simulates: client segments -> ECU reassembles -> ECU responds ->
        response segmented -> client reassembles.
        """
        # 1. Client segments the request into CAN frames
        request_frames = encode(request_payload)
        self.frames_sent += len(request_frames)

        # 2. ECU side reassembles the request
        ecu_reasm = Reassembler()
        received_request = None
        for frame in request_frames:
            out = ecu_reasm.push(frame)
            if out is not None:
                received_request = out

        # 3. ECU processes and produces a response
        response_payload = self.ecu.request(received_request)

        # 4. ECU segments the response into CAN frames
        response_frames = encode(response_payload)
        self.frames_received += len(response_frames)

        # 5. Client reassembles the response
        client_reasm = Reassembler()
        reassembled = None
        for frame in response_frames:
            out = client_reasm.push(frame)
            if out is not None:
                reassembled = out
        return reassembled

    def _is_positive(self, response: bytes, request_sid: int) -> bool:
        return (len(response) > 0
                and response[0] == request_sid + POSITIVE_RESPONSE_OFFSET)

    def start_session(self, session_type: int = 0x03) -> bool:
        """Open a diagnostic session (0x03 = extended)."""
        resp = self._transact(bytes([SID_DIAGNOSTIC_SESSION, session_type]))
        return self._is_positive(resp, SID_DIAGNOSTIC_SESSION)

    def read_vin(self) -> str | None:
        """Read the VIN via DID 0xF190."""
        resp = self._transact(bytes([SID_READ_DATA_BY_ID, 0xF1, 0x90]))
        if not self._is_positive(resp, SID_READ_DATA_BY_ID):
            return None
        return resp[3:].decode("ascii", errors="replace")

    def read_dtcs(self) -> list[dict]:
        """Read stored DTCs and return them with decoded status."""
        resp = self._transact(bytes([SID_READ_DTC, 0x02, 0xFF]))
        if not self._is_positive(resp, SID_READ_DTC):
            return []
        body = resp[3:]  # skip response SID, subfunction, status availability mask
        dtcs = []
        for i in range(0, len(body), 4):
            chunk = body[i:i + 4]
            if len(chunk) < 4:
                break
            code = bytes_to_dtc(chunk[:3])
            status = chunk[3]
            dtcs.append({
                "code": code,
                "status_byte": status,
                "status_bits": decode_status(status),
            })
        return dtcs

    def clear_dtcs(self) -> int:
        """Clear stored DTCs, return how many were cleared."""
        resp = self._transact(bytes([SID_CLEAR_DTC, 0xFF, 0xFF, 0xFF]))
        if self._is_positive(resp, SID_CLEAR_DTC) and len(resp) > 1:
            return resp[1]
        return 0

    def full_scan(self, lang: str = "fr") -> dict:
        """
        Complete diagnostic scan: session -> VIN -> DTCs -> interpretation.
        Connects UDS-read DTCs to the DTC reader for full enrichment.
        """
        self.start_session(0x03)
        vin = self.read_vin()
        raw_dtcs = self.read_dtcs()

        # Bridge to the DTC reader: interpret the codes read over UDS
        codes = [d["code"] for d in raw_dtcs]
        interpretation = scan_dtcs(codes, lang=lang)

        # Merge UDS status info into the interpreted results
        status_by_code = {d["code"]: d for d in raw_dtcs}
        for result in interpretation["results"]:
            uds_info = status_by_code.get(result["code"], {})
            result["uds_status_byte"] = uds_info.get("status_byte")
            result["uds_status_bits"] = uds_info.get("status_bits", [])

        return {
            "vin": vin,
            "dtc_count": len(raw_dtcs),
            "interpretation": interpretation,
            "transport_stats": {
                "frames_sent": self.frames_sent,
                "frames_received": self.frames_received,
            },
        }


if __name__ == "__main__":
    print("=" * 64)
    print("GarageMind - UDS Client: full diagnostic scan over ISO-TP")
    print("=" * 64)

    ecu = build_demo_ecu()
    client = UdsClient(ecu)

    report = client.full_scan(lang="fr")

    print(f"\nVIN lu           : {report['vin']}")
    print(f"DTC trouves      : {report['dtc_count']}")
    print(f"Gravite max      : {report['interpretation']['highest_severity']}")
    print(f"Trames CAN       : {report['transport_stats']['frames_sent']} envoyees, "
          f"{report['transport_stats']['frames_received']} recues")

    print("\n--- Diagnostic complet (via UDS + interpretation) ---")
    for d in report["interpretation"]["results"]:
        print(f"\n[{d['code']}] {d['description_fr']}")
        print(f"  Gravite     : {d['severity_fr']}")
        print(f"  Statut UDS  : 0x{d['uds_status_byte']:02X} -> "
              f"{', '.join(d['uds_status_bits'])}")
        print(f"  Causes probables :")
        for cause in d["likely_causes_fr"][:3]:
            print(f"    - {cause}")