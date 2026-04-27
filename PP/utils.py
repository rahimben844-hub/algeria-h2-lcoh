"""
utils.py — Financial helpers for the Algeria LCOH calculator.
No CSV dependency. Pure math only.
Data comes from atlite/ERA5 via atlite_grid.py.
"""
from __future__ import annotations
from typing import NamedTuple

LHV_H2 = 33.33  # kWh per kg H2 (lower heating value)


class TechEconomicParams(NamedTuple):
    # Financial
    discount_rate:          float   # WACC, e.g. 0.08
    project_lifetime:       int     # years, e.g. 25
    grid_price_dzd:         float   # DZD/kWh
    dzd_to_usd:             float   # exchange rate

    # Solar PV
    solar_capex:            float   # USD/kW
    solar_opex_pct:         float   # fraction of CAPEX/yr
    solar_lifetime:         int

    # Wind
    wind_capex:             float
    wind_opex_pct:          float
    wind_lifetime:          int

    # Electrolyzer (PEM)
    electrolyzer_capex:     float   # USD/kW_el
    electrolyzer_opex_pct:  float
    electrolyzer_efficiency:float   # kWh_el / kg_H2, e.g. 55
    electrolyzer_lifetime:  int

    # H2 Storage
    h2_storage_capex:       float   # USD/kg
    h2_storage_opex_pct:    float
    h2_storage_lifetime:    int

    # Battery
    battery_capex:          float   # USD/kWh
    battery_opex_pct:       float
    battery_lifetime:       int
    battery_efficiency:     float   # round-trip, e.g. 0.90


DEFAULT_PARAMS = TechEconomicParams(
    discount_rate=0.08,
    project_lifetime=25,
    grid_price_dzd=9.0,
    dzd_to_usd=134.5,
    solar_capex=400.0,
    solar_opex_pct=0.015,
    solar_lifetime=25,
    wind_capex=1400.0,
    wind_opex_pct=0.025,
    wind_lifetime=25,
    electrolyzer_capex=700.0,
    electrolyzer_opex_pct=0.03,
    electrolyzer_efficiency=55.0,
    electrolyzer_lifetime=15,
    h2_storage_capex=12.0,
    h2_storage_opex_pct=0.01,
    h2_storage_lifetime=25,
    battery_capex=250.0,
    battery_opex_pct=0.01,
    battery_lifetime=15,
    battery_efficiency=0.90,
)


def annuity_factor(discount_rate: float, lifetime: int) -> float:
    """Capital Recovery Factor: r(1+r)^n / [(1+r)^n - 1]"""
    r, n = discount_rate, lifetime
    if r == 0:
        return 1.0 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


if __name__ == "__main__":
    print("annuity_factor(0.08, 25) =", annuity_factor(0.08, 25))
    print("DEFAULT_PARAMS.solar_capex =", DEFAULT_PARAMS.solar_capex)
    print("LHV_H2 =", LHV_H2)
    print("✓ utils.py OK")
