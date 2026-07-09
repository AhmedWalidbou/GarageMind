"""
ISO-TP (ISO 15765-2) - GarageMind UDS layer
Transport protocol that carries UDS messages over CAN.

A CAN frame holds at most 8 data bytes. Diagnostic messages are often longer,
so ISO-TP splits them across multiple frames and reassembles them:

  - Single Frame (SF)   : PCI 0x0_, payload <= 7 bytes, fits in one frame
  - First Frame (FF)    : PCI 0x1_, starts a multi-frame message (12-bit length)
  - Consecutive Frame(CF): PCI 0x2_, continues it (4-bit rolling sequence number)
  - Flow Control (FC)   : PCI 0x3_, receiver tells sender to continue

This module implements segmentation (encode) and reassembly (decode).
"""

from dataclasses import dataclass, field


# PCI (Protocol Control Information) frame types, encoded in the high nibble
PCI_SINGLE = 0x0
PCI_FIRST = 0x1
PCI_CONSECUTIVE = 0x2
PCI_FLOW_CONTROL = 0x3

CAN_FRAME_LEN = 8
PADDING_BYTE = 0xAA  # conventional padding for unused bytes


def encode(payload: bytes) -> list[bytes]:
    """
    Segment a UDS payload into one or more ISO-TP CAN frames.
    Returns a list of 8-byte frames (padded).
    """
    if len(payload) <= 7:
        # Single Frame: first byte = 0x0<len>, then data
        frame = bytes([PCI_SINGLE << 4 | len(payload)]) + payload
        return [_pad(frame)]

    frames = []
    total_len = len(payload)

    # First Frame: 0x1<len_high>, len_low, then first 6 data bytes
    ff = bytes([
        (PCI_FIRST << 4) | ((total_len >> 8) & 0x0F),
        total_len & 0xFF,
    ]) + payload[:6]
    frames.append(_pad(ff))

    # Consecutive Frames: 0x2<seq>, then up to 7 data bytes each
    remaining = payload[6:]
    seq = 1
    while remaining:
        chunk = remaining[:7]
        cf = bytes([(PCI_CONSECUTIVE << 4) | (seq & 0x0F)]) + chunk
        frames.append(_pad(cf))
        remaining = remaining[7:]
        seq = (seq + 1) & 0x0F
    return frames


def make_flow_control(flow_status: int = 0, block_size: int = 0,
                      st_min: int = 0) -> bytes:
    """
    Build a Flow Control frame.
    flow_status: 0=Continue, 1=Wait, 2=Overflow
    block_size : number of CFs before next FC (0 = send all)
    st_min     : minimum separation time between CFs
    """
    fc = bytes([
        (PCI_FLOW_CONTROL << 4) | (flow_status & 0x0F),
        block_size & 0xFF,
        st_min & 0xFF,
    ])
    return _pad(fc)


@dataclass
class Reassembler:
    """Reassembles a multi-frame ISO-TP message from incoming CAN frames."""

    expected_length: int = 0
    buffer: bytearray = field(default_factory=bytearray)
    in_progress: bool = False
    next_seq: int = 1

    def push(self, frame: bytes) -> bytes | None:
        """
        Feed one CAN frame. Returns the full payload when complete, else None.
        Raises ValueError on protocol violations (bad sequence).
        """
        pci_type = (frame[0] >> 4) & 0x0F

        if pci_type == PCI_SINGLE:
            length = frame[0] & 0x0F
            return bytes(frame[1:1 + length])

        if pci_type == PCI_FIRST:
            self.expected_length = ((frame[0] & 0x0F) << 8) | frame[1]
            self.buffer = bytearray(frame[2:8])
            self.in_progress = True
            self.next_seq = 1
            return None

        if pci_type == PCI_CONSECUTIVE:
            if not self.in_progress:
                raise ValueError("Consecutive frame without a first frame")
            seq = frame[0] & 0x0F
            if seq != self.next_seq:
                raise ValueError(
                    f"Bad sequence: expected {self.next_seq}, got {seq}")
            self.buffer.extend(frame[1:8])
            self.next_seq = (self.next_seq + 1) & 0x0F
            if len(self.buffer) >= self.expected_length:
                result = bytes(self.buffer[:self.expected_length])
                self._reset()
                return result
            return None

        if pci_type == PCI_FLOW_CONTROL:
            return None  # flow control is handled by the sender, ignore here

        raise ValueError(f"Unknown PCI type: {pci_type}")

    def _reset(self):
        self.expected_length = 0
        self.buffer = bytearray()
        self.in_progress = False
        self.next_seq = 1


def _pad(frame: bytes) -> bytes:
    """Pad a frame to 8 bytes with the conventional padding byte."""
    if len(frame) >= CAN_FRAME_LEN:
        return frame[:CAN_FRAME_LEN]
    return frame + bytes([PADDING_BYTE] * (CAN_FRAME_LEN - len(frame)))


if __name__ == "__main__":
    print("=" * 60)
    print("GarageMind - ISO-TP transport layer demo")
    print("=" * 60)

    # Test 1: short message (single frame)
    short_msg = bytes([0x22, 0xF1, 0x90])  # ReadDataByIdentifier example
    frames = encode(short_msg)
    print(f"\nShort message {short_msg.hex()} -> {len(frames)} frame(s):")
    for f in frames:
        print(f"  {f.hex()}")

    reasm = Reassembler()
    result = None
    for f in frames:
        result = reasm.push(f)
    print(f"  Reassembled: {result.hex()} -> {'OK' if result == short_msg else 'FAIL'}")

    # Test 2: long message (multi-frame, needs segmentation)
    long_msg = bytes(range(1, 26))  # 25 bytes
    frames = encode(long_msg)
    print(f"\nLong message ({len(long_msg)} bytes) -> {len(frames)} frame(s):")
    for f in frames:
        print(f"  {f.hex()}")

    reasm = Reassembler()
    result = None
    for f in frames:
        out = reasm.push(f)
        if out is not None:
            result = out
    ok = result == long_msg
    print(f"  Reassembled {len(result)} bytes -> {'OK' if ok else 'FAIL'}")

    # Test 3: flow control frame
    fc = make_flow_control(flow_status=0, block_size=0, st_min=10)
    print(f"\nFlow Control frame: {fc.hex()}")

    # Test 4: sequence error detection
    print("\nSequence error detection:")
    reasm = Reassembler()
    reasm.push(encode(long_msg)[0])  # first frame
    try:
        bad_cf = bytes([0x23, 0, 0, 0, 0, 0, 0, 0])  # wrong seq (3 instead of 1)
        reasm.push(bad_cf)
        print("  FAIL - error not detected")
    except ValueError as e:
        print(f"  OK - caught: {e}")