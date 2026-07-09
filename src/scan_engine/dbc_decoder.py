"""
DBC Decoder - GarageMind M1 Scan Engine
Translates raw CAN frames into named physical signals using a DBC file.
This is the multi-brand core: swap the DBC, decode any manufacturer.

Robust by design: malformed DBC files (common in the wild) are loaded in
permissive mode instead of crashing the whole scan.
"""

import cantools
from cantools.database.errors import UnsupportedDatabaseFormatError


class DbcDecoder:
    """Loads a DBC database and decodes CAN frames into physical signals."""

    def __init__(self, dbc_path: str):
        self.dbc_path = dbc_path
        self.loaded = False
        self.load_mode = None
        self.error = None
        self.db = None
        self.known_ids = set()

        try:
            self.db = cantools.database.load_file(dbc_path)
            self.load_mode = "strict"
            self.loaded = True
        except UnsupportedDatabaseFormatError:
            try:
                self.db = cantools.database.load_file(dbc_path, strict=False)
                self.load_mode = "permissive"
                self.loaded = True
            except Exception as exc:
                self.error = str(exc)
        except Exception as exc:
            self.error = str(exc)

        if self.loaded:
            self.known_ids = {msg.frame_id for msg in self.db.messages}

    def list_messages(self) -> list[dict]:
        """Return all message definitions in the DBC."""
        if not self.loaded:
            return []
        result = []
        for msg in self.db.messages:
            result.append({
                "name": msg.name,
                "can_id_hex": f"{msg.frame_id:X}",
                "signals": [sig.name for sig in msg.signals],
            })
        return result

    def find_signal(self, keyword: str) -> list[dict]:
        """
        Search all messages for signals whose name contains a keyword.
        Useful to locate 'speed', 'rpm', 'temp' across a DBC.
        """
        if not self.loaded:
            return []
        keyword = keyword.lower()
        hits = []
        for msg in self.db.messages:
            for sig in msg.signals:
                if keyword in sig.name.lower():
                    hits.append({
                        "message": msg.name,
                        "can_id_hex": f"{msg.frame_id:X}",
                        "signal": sig.name,
                        "unit": sig.unit or "",
                        "minimum": sig.minimum,
                        "maximum": sig.maximum,
                    })
        return hits

    def decode_frame(self, can_id_hex: str, payload: list[int]) -> dict | None:
        """
        Decode one frame into named signals.
        can_id_hex: CAN ID as hex string (e.g. '0316').
        payload: list of integer bytes.
        Returns None if the CAN ID is not in the DBC or cannot be decoded.
        """
        if not self.loaded:
            return None
        frame_id = int(can_id_hex, 16)
        if frame_id not in self.known_ids:
            return None
        try:
            message = self.db.get_message_by_frame_id(frame_id)
            data = bytes(payload)
            if len(data) < message.length:
                data = data + bytes(message.length - len(data))
            decoded = self.db.decode_message(frame_id, data)
            return {
                "message_name": message.name,
                "can_id_hex": can_id_hex.upper(),
                "signals": decoded,
            }
        except (KeyError, ValueError):
            return None

    def coverage(self, can_ids: list[str]) -> dict:
        """
        Given a list of observed CAN IDs, report how many are known to the DBC.
        Measures how well a DBC matches a given vehicle's traffic.
        """
        if not self.loaded:
            return {"observed_ids": 0, "known_ids_in_dbc": 0,
                    "matched_ids": 0, "coverage_percent": 0}
        observed = {int(cid, 16) for cid in can_ids}
        matched = observed & self.known_ids
        return {
            "observed_ids": len(observed),
            "known_ids_in_dbc": len(self.known_ids),
            "matched_ids": len(matched),
            "coverage_percent": round(100 * len(matched) / len(observed), 1) if observed else 0,
        }

    def info(self) -> dict:
        """Return a short status report for this DBC."""
        return {
            "path": self.dbc_path.split("/")[-1],
            "loaded": self.loaded,
            "load_mode": self.load_mode,
            "message_count": len(self.known_ids),
            "error": self.error,
        }


def _demo():
    """Multi-brand demo: same code, different DBC, proving brand-agnostic design."""

    base = "data/dbc/opendbc-master/opendbc/dbc/"
    brands = {
        "Hyundai": "hyundai_2015_ccan.dbc",
        "Toyota": "toyota_prius_2010_pt.dbc",
        "Volkswagen": "vw_mqb.dbc",
        "PSA (Peugeot/Citroen)": "psa_aee2010_r3.dbc",
        "Chrysler (Stellantis)": "chrysler_cusw.dbc",
        "Ford": "ford_fusion_2018_pt.dbc",
        "Tesla": "tesla_can.dbc",
    }

    print("=" * 60)
    print("GarageMind - Multi-brand DBC decoder demo")
    print("=" * 60)

    decoders = {}
    for brand, filename in brands.items():
        decoder = DbcDecoder(base + filename)
        decoders[brand] = decoder
        status = decoder.info()
        flag = "OK" if status["loaded"] else "FAIL"
        mode = f"({status['load_mode']})" if status["loaded"] else f"({status['error'][:40]})"
        print(f"[{flag:4}] {brand:24} {status['message_count']:4} messages {mode}")

    print("\n" + "=" * 60)
    print("Searching for key powertrain signals across brands")
    print("=" * 60)

    for keyword in ["speed", "rpm", "temp"]:
        print(f"\nSignals containing '{keyword}':")
        for brand, decoder in decoders.items():
            hits = decoder.find_signal(keyword)
            if hits:
                example = hits[0]
                print(f"  {brand:24} -> {example['message']}.{example['signal']} "
                      f"[{example['unit']}] ({len(hits)} total)")

    print("\n" + "=" * 60)
    print("Decoding a real frame example (Hyundai)")
    print("=" * 60)
    hyundai = decoders["Hyundai"]
    sample = hyundai.list_messages()[3]
    result = hyundai.decode_frame(sample["can_id_hex"], [0, 0, 0, 0, 0, 0, 0, 0])
    if result:
        print(f"  Message: {result['message_name']} (ID {result['can_id_hex']})")
        for sig_name, sig_val in list(result["signals"].items())[:6]:
            print(f"    {sig_name}: {sig_val}")


if __name__ == "__main__":
    _demo()