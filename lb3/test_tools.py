"""Тести інструментів: без мовної моделі, без мережі, без API-ключів."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab_main  # noqa: E402


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """Усі файли пишемо в тимчасову теку, а не в справжню artifacts/."""
    out = tmp_path / "artifacts"
    monkeypatch.setattr(lab_main, "OUTPUT_DIR", out)
    return out


# ── Розрахунки ─────────────────────────────────────────────────────────
def test_feet_to_meters():
    assert lab_main.convert_value(2500, "ft", "m") == pytest.approx(762.0)


def test_aliases_and_back_conversion():
    assert lab_main.convert_value(1, "pounds", "kg") == pytest.approx(0.45359237)
    assert lab_main.convert_value(762, "meters", "feet") == pytest.approx(2500)


def test_pressure():
    assert lab_main.convert_value(1, "MPa", "kPa") == pytest.approx(1000)


def test_different_quantities_rejected():
    with pytest.raises(ValueError):
        lab_main.convert_value(1, "m", "kg")


def test_unknown_unit_rejected():
    with pytest.raises(ValueError):
        lab_main.convert_value(1, "парсек", "m")


def test_convert_tool_returns_all_results_at_once():
    out = lab_main.convert_units.invoke({"items": [
        {"value": 2500, "from_unit": "ft", "to_unit": "m"},
        {"value": 10, "from_unit": "lb", "to_unit": "kg"},
        {"value": 1, "from_unit": "m", "to_unit": "kg"},
    ]})
    data = json.loads(out)["conversions"]
    assert len(data) == 3
    assert data[0]["result"] == "762 m"
    assert data[1]["result"] == "4.53592 kg"
    assert "error" in data[2]            # одна помилка не валить решту


def test_order_weight_full_result():
    out = json.loads(lab_main.order_weight.invoke(
        {"length": 2500, "length_unit": "ft", "kg_per_meter": 0.888}))
    assert out["length_m"] == "762.00"
    assert out["total_weight_kg"] == "676.66"
    assert out["total_weight_tonnes"] == "0.677"


def test_order_weight_rejects_bad_input():
    assert lab_main.order_weight.invoke(
        {"length": -5, "length_unit": "m", "kg_per_meter": 0.888}).startswith("ERROR")
    assert lab_main.order_weight.invoke(
        {"length": 5, "length_unit": "foo", "kg_per_meter": 0.888}).startswith("ERROR")


# ── Безпека імені файлу ────────────────────────────────────────────────
def test_path_traversal_stays_inside(sandbox, tmp_path):
    res = lab_main.write_report.invoke({"filename": "../../pwned.txt", "content": "x"})
    assert res.startswith("Saved")
    assert (sandbox / "pwned.txt").exists()
    assert not (tmp_path.parent / "pwned.txt").exists()
    assert not (tmp_path / "pwned.txt").exists()


@pytest.mark.parametrize("bad", ["C:/Windows/system32/x.dll", "..\\..\\evil.txt", "/etc/passwd"])
def test_other_dangerous_names_stay_inside(sandbox, bad):
    lab_main.write_report.invoke({"filename": bad, "content": "x"})
    written = [p for p in sandbox.iterdir()]
    assert len(written) == 1 and written[0].resolve().parent == sandbox.resolve()


@pytest.mark.parametrize("bad", ["", "   ", "..", "con", "NUL.txt"])
def test_refused_names(sandbox, bad):
    res = lab_main.write_report.invoke({"filename": bad, "content": "x"})
    assert res.startswith("Refused")


def test_normal_file_is_written(sandbox):
    lab_main.write_report.invoke({"filename": "conversion_report.txt", "content": "привіт"})
    assert (sandbox / "conversion_report.txt").read_text(encoding="utf-8") == "привіт"
