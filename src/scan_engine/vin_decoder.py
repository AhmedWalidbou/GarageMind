"""
VIN Decoder - GarageMind M1 Scan Engine
Decodes a 17-character Vehicle Identification Number into vehicle info.
Uses local WMI tables first, NHTSA public API as fallback.
"""

import re
import requests

WMI_TABLE = {
    "VF1": {"manufacturer": "Renault", "country": "France"},
    "VF3": {"manufacturer": "Peugeot", "country": "France"},
    "VF7": {"manufacturer": "Citroen", "country": "France"},
    "VR1": {"manufacturer": "DS Automobiles", "country": "France"},
    "W0L": {"manufacturer": "Opel", "country": "Germany"},
    "WVW": {"manufacturer": "Volkswagen", "country": "Germany"},
    "WAU": {"manufacturer": "Audi", "country": "Germany"},
    "WBA": {"manufacturer": "BMW", "country": "Germany"},
    "WDB": {"manufacturer": "Mercedes-Benz", "country": "Germany"},
    "TMB": {"manufacturer": "Skoda", "country": "Czech Republic"},
    "JT": {"manufacturer": "Toyota", "country": "Japan"},
    "JH": {"manufacturer": "Honda", "country": "Japan"},
    "KMH": {"manufacturer": "Hyundai", "country": "South Korea"},
    "KNA": {"manufacturer": "Kia", "country": "South Korea"},
    "ZFA": {"manufacturer": "Fiat", "country": "Italy"},
    "1FA": {"manufacturer": "Ford", "country": "USA"},
    "1G": {"manufacturer": "General Motors", "country": "USA"},
    "5YJ": {"manufacturer": "Tesla", "country": "USA"},
}

YEAR_CODES = {
    "A": 2010, "B": 2011, "C": 2012, "D": 2013, "E": 2014,
    "F": 2015, "G": 2016, "H": 2017, "J": 2018, "K": 2019,
    "L": 2020, "M": 2021, "N": 2022, "P": 2023, "R": 2024,
    "S": 2025, "T": 2026,
}

NHTSA_API = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"


def validate_vin(vin: str) -> bool:
    """Check VIN format: 17 chars, no I, O, Q."""
    if not isinstance(vin, str):
        return False
    vin = vin.strip().upper()
    return bool(re.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", vin))


def decode_wmi(vin: str) -> dict:
    """Decode manufacturer from World Manufacturer Identifier (positions 1-3)."""
    vin = vin.strip().upper()
    for prefix_len in (3, 2):
        prefix = vin[:prefix_len]
        if prefix in WMI_TABLE:
            return WMI_TABLE[prefix]
    return {"manufacturer": "Unknown", "country": "Unknown"}


def decode_year(vin: str) -> int | None:
    """Decode model year from position 10."""
    return YEAR_CODES.get(vin[9].upper())


def decode_vin_nhtsa(vin: str, timeout: int = 5) -> dict | None:
    """Fallback: query the free NHTSA vPIC API for detailed info."""
    try:
        response = requests.get(NHTSA_API.format(vin=vin), timeout=timeout)
        response.raise_for_status()
        results = response.json()["Results"][0]
        return {
            "manufacturer": results.get("Make") or "Unknown",
            "model": results.get("Model") or "Unknown",
            "year": results.get("ModelYear") or None,
            "fuel_type": results.get("FuelTypePrimary") or "Unknown",
            "source": "nhtsa",
        }
    except (requests.RequestException, KeyError, IndexError):
        return None


def decode_vin(vin: str, use_api: bool = True) -> dict:
    """
    Main entry point. Decodes a VIN into structured vehicle info.
    Local tables first, NHTSA API enrichment if available.
    """
    if not validate_vin(vin):
        return {"valid": False, "error": "Invalid VIN format"}

    vin = vin.strip().upper()
    info = {
        "valid": True,
        "vin": vin,
        "year": decode_year(vin),
        "source": "local",
        **decode_wmi(vin),
    }

    if use_api:
        api_info = decode_vin_nhtsa(vin)
        if api_info:
            info.update({k: v for k, v in api_info.items() if v not in (None, "Unknown")})

    return info


if __name__ == "__main__":
    test_vins = [
        "TMBER6NJXKZ123456",
        "VF1RFB00X61234567",
        "KMHD35LE4EU123456",
        "INVALID_VIN",
    ]
    for v in test_vins:
        print(decode_vin(v, use_api=False))