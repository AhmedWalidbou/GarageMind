"""
Unit tests for GarageMind M1 - CLI.
Covers argument parsing, exit codes, VIN validation, brand resolution,
and error handling for missing files.
Run with: pytest
"""

import pytest

from src.cli import (
    main, build_parser, cmd_decode_vin, cmd_version,
    _resolve_brand_dbcs, BRAND_DBC_GLOBS,
)


class _Args:
    """Lightweight namespace to call command handlers directly."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------- Parser ----------

class TestParser:
    def test_parser_requires_command(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_scan_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["scan"])
        assert args.lang == "fr"
        assert args.no_save is False

    def test_decode_vin_parsing(self):
        parser = build_parser()
        args = parser.parse_args(["decode-vin", "WAUZZZ8V1JA123456", "--no-api"])
        assert args.vin == "WAUZZZ8V1JA123456"
        assert args.no_api is True

    def test_analyze_can_brand(self):
        parser = build_parser()
        args = parser.parse_args(["analyze-can", "log.csv", "--brand", "toyota"])
        assert args.brand == "toyota"


# ---------- VIN command ----------

class TestVinCommand:
    def test_valid_vin_returns_zero(self):
        code = cmd_decode_vin(_Args(vin="WAUZZZ8V1JA123456", no_api=True))
        assert code == 0

    def test_invalid_vin_returns_two(self):
        code = cmd_decode_vin(_Args(vin="NOTAVIN", no_api=True))
        assert code == 2


# ---------- Brand resolution ----------

class TestBrandResolution:
    def test_known_brand_has_glob(self):
        assert "hyundai" in BRAND_DBC_GLOBS
        assert "toyota" in BRAND_DBC_GLOBS

    def test_unknown_brand_returns_empty(self):
        assert _resolve_brand_dbcs("ferrari") == []

    def test_alias_brands(self):
        # peugeot and citroen both map to PSA
        assert BRAND_DBC_GLOBS["peugeot"] == BRAND_DBC_GLOBS["psa"]
        assert BRAND_DBC_GLOBS["citroen"] == BRAND_DBC_GLOBS["psa"]


# ---------- Version & error handling ----------

class TestMisc:
    def test_version_returns_zero(self, capsys):
        code = cmd_version(_Args())
        captured = capsys.readouterr()
        assert code == 0
        assert "GarageMind" in captured.out

    def test_analyze_can_missing_file(self):
        code = main(["analyze-can", "does_not_exist.csv"])
        assert code == 2

    def test_decode_vin_via_main(self):
        code = main(["decode-vin", "WAUZZZ8V1JA123456", "--no-api"])
        assert code == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])