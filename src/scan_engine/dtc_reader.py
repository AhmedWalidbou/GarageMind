"""
DTC Reader - GarageMind M1 Scan Engine
Parses and interprets OBD-II Diagnostic Trouble Codes (DTCs).

A DTC is a 5-character code like P0301:
  - Position 1 (letter): system
      P = Powertrain, C = Chassis, B = Body, U = Network/UDS
  - Position 2 (digit): 0 = standardized (SAE), 1 = manufacturer-specific
  - Position 3 (hex): subsystem
  - Positions 4-5 (hex): specific fault index

This module decodes that structure and enriches known codes with
bilingual (FR/EN) descriptions, likely causes and severity.
"""

SYSTEM_MAP = {
    "P": {"en": "Powertrain", "fr": "Groupe motopropulseur"},
    "C": {"en": "Chassis", "fr": "Chassis"},
    "B": {"en": "Body", "fr": "Carrosserie"},
    "U": {"en": "Network / UDS", "fr": "Reseau / UDS"},
}

CODE_TYPE_MAP = {
    "0": {"en": "Generic (SAE standard)", "fr": "Generique (standard SAE)"},
    "1": {"en": "Manufacturer-specific", "fr": "Specifique constructeur"},
    "2": {"en": "Generic (SAE standard)", "fr": "Generique (standard SAE)"},
    "3": {"en": "Manufacturer-specific", "fr": "Specifique constructeur"},
}

POWERTRAIN_SUBSYSTEM = {
    "0": {"en": "Fuel and air metering / auxiliary emission",
          "fr": "Dosage air/carburant et emissions"},
    "1": {"en": "Fuel and air metering", "fr": "Dosage air/carburant"},
    "2": {"en": "Fuel and air metering (injector circuit)",
          "fr": "Dosage air/carburant (circuit injecteur)"},
    "3": {"en": "Ignition system or misfire", "fr": "Allumage ou rate de combustion"},
    "4": {"en": "Auxiliary emission controls", "fr": "Controle des emissions"},
    "5": {"en": "Vehicle speed, idle control, aux inputs",
          "fr": "Vitesse vehicule, ralenti, entrees auxiliaires"},
    "6": {"en": "Computer output circuit", "fr": "Circuit de sortie calculateur"},
    "7": {"en": "Transmission", "fr": "Transmission / boite de vitesses"},
    "8": {"en": "Transmission", "fr": "Transmission / boite de vitesses"},
    "9": {"en": "Transmission", "fr": "Transmission / boite de vitesses"},
}

SEVERITY = {
    "low": {"en": "Low - monitor", "fr": "Faible - a surveiller"},
    "medium": {"en": "Medium - repair soon", "fr": "Moyen - reparer prochainement"},
    "high": {"en": "High - stop driving", "fr": "Eleve - arreter le vehicule"},
}

KNOWN_DTC = {
    "P0300": {
        "en": "Random / multiple cylinder misfire detected",
        "fr": "Rates de combustion aleatoires sur plusieurs cylindres",
        "causes_fr": ["Bougies usees", "Bobines d'allumage", "Injecteurs encrasses",
                      "Fuite d'admission", "Faible compression"],
        "severity": "high",
    },
    "P0301": {
        "en": "Cylinder 1 misfire detected",
        "fr": "Rate de combustion cylindre 1",
        "causes_fr": ["Bougie cylindre 1", "Bobine cylindre 1", "Injecteur cylindre 1"],
        "severity": "high",
    },
    "P0302": {
        "en": "Cylinder 2 misfire detected",
        "fr": "Rate de combustion cylindre 2",
        "causes_fr": ["Bougie cylindre 2", "Bobine cylindre 2", "Injecteur cylindre 2"],
        "severity": "high",
    },
    "P0171": {
        "en": "System too lean (bank 1)",
        "fr": "Melange trop pauvre (rangee 1)",
        "causes_fr": ["Prise d'air", "Debitmetre d'air defectueux",
                      "Pression carburant faible", "Injecteurs encrasses"],
        "severity": "medium",
    },
    "P0401": {
        "en": "Exhaust gas recirculation (EGR) flow insufficient",
        "fr": "Debit vanne EGR insuffisant",
        "causes_fr": ["Vanne EGR encrassee", "Vanne EGR bloquee",
                      "Conduits EGR obstrues", "Capteur de pression differentielle"],
        "severity": "medium",
    },
    "P0402": {
        "en": "EGR flow excessive",
        "fr": "Debit vanne EGR excessif",
        "causes_fr": ["Vanne EGR bloquee ouverte", "Capteur DPF defectueux"],
        "severity": "medium",
    },
    "P2002": {
        "en": "Diesel particulate filter (DPF) efficiency below threshold",
        "fr": "Efficacite du filtre a particules (FAP) sous le seuil",
        "causes_fr": ["FAP colmate", "Additif FAP epuise",
                      "Capteur de pression differentielle FAP", "Regenerations incompletes"],
        "severity": "medium",
    },
    "P0420": {
        "en": "Catalyst system efficiency below threshold (bank 1)",
        "fr": "Efficacite du catalyseur sous le seuil (rangee 1)",
        "causes_fr": ["Catalyseur use", "Sonde lambda defectueuse", "Fuite d'echappement"],
        "severity": "medium",
    },
    "P0128": {
        "en": "Coolant thermostat below regulating temperature",
        "fr": "Thermostat sous la temperature de regulation",
        "causes_fr": ["Thermostat bloque ouvert", "Capteur temperature liquide"],
        "severity": "low",
    },
    "P0016": {
        "en": "Crankshaft / camshaft position correlation (bank 1)",
        "fr": "Correlation vilebrequin / arbre a cames (rangee 1)",
        "causes_fr": ["Distribution decalee", "Chaine ou courroie distendue",
                      "Capteur arbre a cames", "Dephaseur (VVT)"],
        "severity": "high",
    },
    "P0113": {
        "en": "Intake air temperature sensor circuit high input",
        "fr": "Signal capteur temperature d'air d'admission trop haut",
        "causes_fr": ["Capteur IAT defectueux", "Cablage coupe", "Connecteur corrode"],
        "severity": "low",
    },
    "U0100": {
        "en": "Lost communication with ECM/PCM",
        "fr": "Perte de communication avec le calculateur moteur",
        "causes_fr": ["Bus CAN coupe", "Calculateur hors tension", "Cablage reseau"],
        "severity": "high",
    },
}


def validate_dtc(code: str) -> bool:
    """Check the DTC has the canonical 5-character shape: letter + 4 hex digits."""
    if not isinstance(code, str) or len(code) != 5:
        return False
    code = code.upper()
    if code[0] not in SYSTEM_MAP:
        return False
    return all(c in "0123456789ABCDEF" for c in code[1:])


def decode_structure(code: str) -> dict:
    """Decode the meaning encoded in the DTC characters themselves."""
    code = code.upper()
    system = SYSTEM_MAP[code[0]]
    code_type = CODE_TYPE_MAP.get(code[1], {"en": "Unknown", "fr": "Inconnu"})
    result = {
        "code": code,
        "system_en": system["en"],
        "system_fr": system["fr"],
        "type_en": code_type["en"],
        "type_fr": code_type["fr"],
        "subsystem_en": None,
        "subsystem_fr": None,
    }
    if code[0] == "P":
        sub = POWERTRAIN_SUBSYSTEM.get(code[2])
        if sub:
            result["subsystem_en"] = sub["en"]
            result["subsystem_fr"] = sub["fr"]
    return result


def interpret_dtc(code: str, lang: str = "fr") -> dict:
    """
    Full interpretation of a DTC: structural decoding plus enrichment
    from the known-codes database when available.
    """
    if not validate_dtc(code):
        return {"code": code, "valid": False, "error": "Invalid DTC format"}

    code = code.upper()
    structure = decode_structure(code)
    known = KNOWN_DTC.get(code)

    report = {
        "code": code,
        "valid": True,
        "in_database": known is not None,
        **structure,
    }

    if known:
        sev = SEVERITY[known["severity"]]
        report.update({
            "description_en": known["en"],
            "description_fr": known["fr"],
            "likely_causes_fr": known["causes_fr"],
            "severity_key": known["severity"],
            "severity_en": sev["en"],
            "severity_fr": sev["fr"],
        })
    else:
        report.update({
            "description_en": f"{structure['subsystem_en'] or structure['system_en']} fault",
            "description_fr": f"Defaut {structure['subsystem_fr'] or structure['system_fr']}",
            "likely_causes_fr": ["Non repertorie - consulter la documentation technique"],
            "severity_key": "unknown",
            "severity_en": "Unknown",
            "severity_fr": "Inconnu",
        })

    return report


def scan_dtcs(codes: list[str], lang: str = "fr") -> dict:
    """
    Interpret a list of DTCs and return a structured scan report,
    sorted by severity (high first), like a real diagnostic tool.
    """
    order = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
    interpreted = [interpret_dtc(c, lang) for c in codes]
    valid = [d for d in interpreted if d.get("valid")]
    invalid = [d for d in interpreted if not d.get("valid")]
    valid.sort(key=lambda d: order.get(d["severity_key"], 3))

    return {
        "total_codes": len(codes),
        "valid_codes": len(valid),
        "invalid_codes": len(invalid),
        "in_database": sum(1 for d in valid if d["in_database"]),
        "highest_severity": valid[0]["severity_key"] if valid else None,
        "results": valid,
        "rejected": [d["code"] for d in invalid],
    }


if __name__ == "__main__":
    print("=" * 60)
    print("GarageMind - DTC Reader demo")
    print("=" * 60)

    test_codes = ["P0301", "P2002", "P0401", "P0420", "U0100",
                  "P1234", "C0035", "BADCODE"]

    report = scan_dtcs(test_codes, lang="fr")

    print(f"\nCodes scannes : {report['total_codes']}")
    print(f"Valides       : {report['valid_codes']}")
    print(f"Repertories   : {report['in_database']}")
    print(f"Rejetes       : {report['rejected']}")
    print(f"Gravite max   : {report['highest_severity']}")

    print("\n" + "=" * 60)
    print("Detail des defauts (tries par gravite)")
    print("=" * 60)

    for d in report["results"]:
        print(f"\n[{d['code']}] {d['description_fr']}")
        print(f"  Systeme   : {d['system_fr']} ({d['type_fr']})")
        if d["subsystem_fr"]:
            print(f"  Sous-syst.: {d['subsystem_fr']}")
        print(f"  Gravite   : {d['severity_fr']}")
        print(f"  Causes probables :")
        for cause in d["likely_causes_fr"]:
            print(f"    - {cause}")