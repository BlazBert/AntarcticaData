"""Static schema invariants — guards against silent column-order drift."""

from __future__ import annotations

from ubx.schemas import ALL_SCHEMAS


def test_all_schemas_have_t_ns():
    for name, schema in ALL_SCHEMAS.items():
        assert "t_ns" in schema.names, f"{name} missing t_ns"


def test_no_duplicate_columns():
    for name, schema in ALL_SCHEMAS.items():
        names = list(schema.names)
        assert len(names) == len(set(names)), f"{name} has duplicate columns: {names}"


def test_long_form_ids_present():
    """Long-form tables must carry the keys we group by downstream."""
    expected_keys = {
        "nav_sat": {"gnssId", "svId"},
        "rxm_rawx": {"gnssId", "svId", "sigId"},
        "rxm_sfrbx": {"gnssId", "svId", "sigId"},
        "rxm_measx": {"gnssId", "svId"},
    }
    for name, keys in expected_keys.items():
        cols = set(ALL_SCHEMAS[name].names)
        missing = keys - cols
        assert not missing, f"{name} missing keys {missing}"
