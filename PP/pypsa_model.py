"""
pypsa_model.py — PyPSA network builder and LCOH optimizer.
Accepts CF arrays directly. No file I/O here.
CF data comes from atlite_grid.py (real ERA5).


Network:
  [Solar PV] ──┐
  [Wind]     ──┤──→ [AC Bus] ──→ [Electrolyzer Link] ──→ [H2 Bus]
  [Grid]     ──┘         │                                    │
                    [Battery]                          [H2 Tank Store]
                                                             │
                                                       [H2 Load]
"""
from __future__ import annotations


import logging
import warnings
from dataclasses import dataclass


import numpy as np
import pandas as pd
import pypsa


from utils import TechEconomicParams, DEFAULT_PARAMS, annuity_factor, LHV_H2


warnings.filterwarnings("ignore")
logging.getLogger("pypsa").setLevel(logging.ERROR)
logging.getLogger("linopy").setLevel(logging.ERROR)



@dataclass
class LCOHResult:
    status:                     str
    lcoh_usd_per_kg:            float
    lcoh_dzd_per_kg:            float
    solar_capacity_kw:          float
    wind_capacity_kw:           float
    electrolyzer_capacity_kw:   float
    battery_capacity_kwh:       float
    h2_storage_capacity_kg:     float
    solar_generation_kwh:       float
    wind_generation_kwh:        float
    grid_import_kwh:            float
    h2_produced_kg:             float
    electrolyzer_utilization_pct: float
    cost_solar:                 float
    cost_wind:                  float
    cost_electrolyzer:          float
    cost_h2_storage:            float
    cost_battery:               float
    cost_grid:                  float
    total_cost_usd:             float
    solar_full_load_hours:      float
    wind_cf_mean:               float
    pct_renewable:              float


    def to_dict(self) -> dict:
        return {
            "status":                       self.status,
            "lcoh_usd_per_kg":              round(self.lcoh_usd_per_kg, 4),
            "lcoh_dzd_per_kg":              round(self.lcoh_dzd_per_kg, 2),
            "solar_capacity_kw":            round(self.solar_capacity_kw, 2),
            "wind_capacity_kw":             round(self.wind_capacity_kw, 2),
            "electrolyzer_capacity_kw":     round(self.electrolyzer_capacity_kw, 2),
            "battery_capacity_kwh":         round(self.battery_capacity_kwh, 2),
            "h2_storage_capacity_kg":       round(self.h2_storage_capacity_kg, 2),
            "solar_generation_kwh":         round(self.solar_generation_kwh, 2),
            "wind_generation_kwh":          round(self.wind_generation_kwh, 2),
            "grid_import_kwh":              round(self.grid_import_kwh, 2),
            "h2_produced_kg":               round(self.h2_produced_kg, 2),
            "electrolyzer_utilization_pct": round(self.electrolyzer_utilization_pct, 2),
            "cost_solar":                   round(self.cost_solar, 2),
            "cost_wind":                    round(self.cost_wind, 2),
            "cost_electrolyzer":            round(self.cost_electrolyzer, 2),
            "cost_h2_storage":              round(self.cost_h2_storage, 2),
            "cost_battery":                 round(self.cost_battery, 2),
            "cost_grid":                    round(self.cost_grid, 2),
            "total_cost_usd":               round(self.total_cost_usd, 2),
            "solar_full_load_hours":        round(self.solar_full_load_hours, 1),
            "wind_cf_mean":                 round(self.wind_cf_mean, 4),
            "pct_renewable":                round(self.pct_renewable, 2),
        }



def build_network(
    solar_cf:               np.ndarray,
    wind_cf:                np.ndarray,
    params:                 TechEconomicParams = DEFAULT_PARAMS,
    h2_demand_kg_per_year:  float = 1000.0,
    allow_wind:             bool  = True,
    allow_grid:             bool  = True,
    representative_weeks:   int | None = 4,
) -> pypsa.Network:
    """
    Build a PyPSA network from CF arrays.

    Parameters
    ----------
    solar_cf              : Hourly solar PV capacity factors, shape (8760,)
    wind_cf               : Hourly wind capacity factors, shape (8760,)
    params                : TechEconomicParams
    h2_demand_kg_per_year : Annual H2 demand in kg
    allow_wind            : Include wind generator
    allow_grid            : Allow grid electricity backup
    representative_weeks  : Sub-sample for speed (4=fast, None=full year)
    """
    n_hours = len(solar_cf)

    # ── Sub-sample to representative weeks ────────────────────────────────────
    if representative_weeks:
        n_rep     = min(representative_weeks * 168, n_hours)
        step      = max(n_hours // n_rep, 1)
        idx       = np.arange(0, n_hours, step)[:n_rep]
        weight    = 8760.0 / n_rep
        solar_use = solar_cf[idx]
        wind_use  = wind_cf[idx]
        T         = n_rep
    else:
        solar_use, wind_use = solar_cf, wind_cf
        T, weight = n_hours, 1.0

    timestamps = pd.date_range("2023-01-01", periods=T, freq="h")
    r          = params.discount_rate

    def auc(capex, opex_pct, lifetime):
        """Annualised unit cost (USD/kW/yr or USD/kWh/yr)."""
        return capex * (annuity_factor(r, lifetime) + opex_pct)

    # Annualised costs per unit capacity
    solar_auc = auc(params.solar_capex,        params.solar_opex_pct,        params.solar_lifetime)
    wind_auc  = auc(params.wind_capex,         params.wind_opex_pct,         params.wind_lifetime)
    elec_auc  = auc(params.electrolyzer_capex, params.electrolyzer_opex_pct, params.electrolyzer_lifetime)
    h2t_auc   = auc(params.h2_storage_capex,   params.h2_storage_opex_pct,   params.h2_storage_lifetime)
    bat_auc   = auc(params.battery_capex,      params.battery_opex_pct,      params.battery_lifetime)

    grid_usd_kwh = params.grid_price_dzd / params.dzd_to_usd

    # Electrolyzer: kWh_H2_out / kWh_el_in = LHV_H2 / efficiency_kWh_per_kg
    # (electrolyzer_efficiency is in kWh_el/kg_H2, e.g. 55 kWh/kg)
    elec_eff = LHV_H2 / params.electrolyzer_efficiency

    # Fixed H2 withdrawal: kWh_H2 per hour
    h2_load_kwh_h = h2_demand_kg_per_year * LHV_H2 / 8760.0

    # ── Build network ─────────────────────────────────────────────────────────
    net = pypsa.Network()
    net.set_snapshots(timestamps)
    for col in net.snapshot_weightings.columns:
        net.snapshot_weightings[col] = weight

    net.add("Bus", "AC", carrier="AC")
    net.add("Bus", "H2", carrier="H2")

    net.add("Generator", "Solar PV",
            bus="AC", carrier="solar",
            p_nom_extendable=True, p_nom_min=0.0,
            capital_cost=solar_auc, marginal_cost=0.0,
            p_max_pu=pd.Series(solar_use, index=timestamps))

    if allow_wind:
        net.add("Generator", "Wind",
                bus="AC", carrier="wind",
                p_nom_extendable=True, p_nom_min=0.0,
                capital_cost=wind_auc, marginal_cost=0.0,
                p_max_pu=pd.Series(wind_use, index=timestamps))

    if allow_grid:
        net.add("Generator", "Grid",
                bus="AC", carrier="grid",
                p_nom=1e6, p_nom_extendable=False,
                capital_cost=0.0, marginal_cost=grid_usd_kwh)

    net.add("StorageUnit", "Battery",
            bus="AC", carrier="battery",
            p_nom_extendable=True, p_nom_min=0.0,
            capital_cost=bat_auc, marginal_cost=0.0,
            efficiency_store=params.battery_efficiency ** 0.5,
            efficiency_dispatch=params.battery_efficiency ** 0.5,
            max_hours=4.0, cyclic_state_of_charge=True)

    net.add("Link", "Electrolyzer",
            bus0="AC", bus1="H2",
            p_nom_extendable=True, p_nom_min=0.0,
            capital_cost=elec_auc, marginal_cost=0.0,
            efficiency=elec_eff, p_min_pu=0.0)

    # FIX 6: H2 Tank capital_cost units.
    # h2_storage_capex is in USD/kWh_H2 (energy-based storage cost).
    # PyPSA Store tracks energy in kWh, so capital_cost = USD/kWh/yr = h2t_auc.
    # Old code was:  capital_cost=h2t_auc * LHV_H2
    # That inflated cost by LHV_H2 (~33×) because it treated kWh cost as kg cost
    # and then multiplied by kWh/kg again. Correct form has no LHV_H2 here.
    #
    # NOTE: If your h2_storage_capex in utils.py is in USD/kg_H2 (not USD/kWh),
    # change this line to:  capital_cost=h2t_auc / LHV_H2
    net.add("Store", "H2 Tank",
            bus="H2",
            e_nom_extendable=True, e_nom_min=0.0,
            capital_cost=h2t_auc,          # FIX 6: was h2t_auc * LHV_H2
            marginal_cost=0.0, e_cyclic=True)

    net.add("Load", "H2 Demand",
            bus="H2", carrier="H2",
            p_set=pd.Series(h2_load_kwh_h, index=timestamps))

    return net



def solve_network(net: pypsa.Network, solver_name: str = "highs") -> str:
    try:
        net.optimize(solver_name=solver_name,
                     solver_options={"time_limit": 120})
        return "optimal"
    except Exception as e:
        logging.error(f"Solver error: {e}")
        return "error"


def extract_results(
    net:        pypsa.Network,
    params:     TechEconomicParams,
    wind_cf:    np.ndarray,
) -> LCOHResult:
    r = params.discount_rate

    def cap(df, name):
        return float(df.loc[name, "p_nom_opt"]) if name in df.index else 0.0

    solar_cap   = cap(net.generators,    "Solar PV")
    wind_cap    = cap(net.generators,    "Wind")
    elec_cap    = cap(net.links,         "Electrolyzer")
    bat_cap_kw  = cap(net.storage_units, "Battery")
    bat_cap_kwh = bat_cap_kw * 4.0
    h2t_kwh     = float(net.stores.loc["H2 Tank", "e_nom_opt"]) if "H2 Tank" in net.stores.index else 0.0
    h2t_kg      = h2t_kwh / LHV_H2

    w = net.snapshot_weightings["generators"]

    def gen_kwh(name):
        s = net.generators_t.p.get(name, pd.Series(0.0, index=net.snapshots))
        return float((s * w).sum())

    solar_kwh = gen_kwh("Solar PV")
    wind_kwh  = gen_kwh("Wind")
    grid_kwh  = gen_kwh("Grid")

    elec_p0        = net.links_t.p0.get("Electrolyzer", pd.Series(0.0, index=net.snapshots))
    elec_input_kwh = float((elec_p0 * w).sum())

    # electrolyzer_efficiency is in kWh_el/kg_H2
    # → kg_H2 = kWh_el_in / (kWh_el/kg_H2)
    h2_kg = elec_input_kwh / params.electrolyzer_efficiency

    def auc(capex, opex, life):
        return capex * (annuity_factor(r, life) + opex)

    c_solar = solar_cap  * auc(params.solar_capex,        params.solar_opex_pct,        params.solar_lifetime)
    c_wind  = wind_cap   * auc(params.wind_capex,         params.wind_opex_pct,         params.wind_lifetime)
    c_elec  = elec_cap   * auc(params.electrolyzer_capex, params.electrolyzer_opex_pct, params.electrolyzer_lifetime)
    c_bat   = bat_cap_kw * auc(params.battery_capex,      params.battery_opex_pct,      params.battery_lifetime)
    c_grid  = grid_kwh   * params.grid_price_dzd / params.dzd_to_usd

    # FIX 7: H2 Tank cost extraction.
    # h2_storage_capex is in USD/kWh_H2 → annual cost = h2t_kwh * auc (USD/kWh/yr).
    # Old code was:  h2t_kwh * auc(...) * LHV_H2
    # That multiplied by LHV_H2 a second time, inflating the cost by ~33×.
    #
    # NOTE: If h2_storage_capex is in USD/kg_H2, use instead:
    #   c_h2t = h2t_kg * auc(params.h2_storage_capex, ...)
    c_h2t = h2t_kwh * auc(params.h2_storage_capex, params.h2_storage_opex_pct, params.h2_storage_lifetime)
    #                                                                            FIX 7: removed * LHV_H2

    total   = c_solar + c_wind + c_elec + c_h2t + c_bat + c_grid

    lcoh_usd = total / h2_kg if h2_kg > 0 else float("inf")
    lcoh_dzd = lcoh_usd * params.dzd_to_usd

    solar_flh = solar_kwh / solar_cap if solar_cap > 0 else 0.0
    elec_util = elec_input_kwh / (elec_cap * 8760) * 100 if elec_cap > 0 else 0.0
    pct_ren   = min((solar_kwh + wind_kwh) / (elec_input_kwh + 1e-9) * 100, 100.0)

    return LCOHResult(
        status="optimal",
        lcoh_usd_per_kg=lcoh_usd,
        lcoh_dzd_per_kg=lcoh_dzd,
        solar_capacity_kw=solar_cap,
        wind_capacity_kw=wind_cap,
        electrolyzer_capacity_kw=elec_cap,
        battery_capacity_kwh=bat_cap_kwh,
        h2_storage_capacity_kg=h2t_kg,
        solar_generation_kwh=solar_kwh,
        wind_generation_kwh=wind_kwh,
        grid_import_kwh=grid_kwh,
        h2_produced_kg=h2_kg,
        electrolyzer_utilization_pct=elec_util,
        cost_solar=c_solar,
        cost_wind=c_wind,
        cost_electrolyzer=c_elec,
        cost_h2_storage=c_h2t,
        cost_battery=c_bat,
        cost_grid=c_grid,
        total_cost_usd=total,
        solar_full_load_hours=solar_flh,
        wind_cf_mean=float(wind_cf.mean()),
        pct_renewable=pct_ren,
    )



def run_lcoh_from_arrays(
    solar_cf:               np.ndarray,
    wind_cf:                np.ndarray,
    params:                 TechEconomicParams = DEFAULT_PARAMS,
    h2_demand_kg_per_year:  float = 1000.0,
    allow_wind:             bool  = True,
    allow_grid:             bool  = True,
    representative_weeks:   int   = 4,
    solver:                 str   = "highs",
) -> LCOHResult:
    """
    Main public function. Takes CF arrays directly from atlite.
    No file I/O.
    """
    net    = build_network(solar_cf, wind_cf, params,
                           h2_demand_kg_per_year, allow_wind, allow_grid,
                           representative_weeks)
    status = solve_network(net, solver_name=solver)

    if status != "optimal":
        return LCOHResult(
            status=status,
            lcoh_usd_per_kg=float("inf"), lcoh_dzd_per_kg=float("inf"),
            solar_capacity_kw=0, wind_capacity_kw=0,
            electrolyzer_capacity_kw=0, battery_capacity_kwh=0,
            h2_storage_capacity_kg=0, solar_generation_kwh=0,
            wind_generation_kwh=0, grid_import_kwh=0,
            h2_produced_kg=0, electrolyzer_utilization_pct=0,
            cost_solar=0, cost_wind=0, cost_electrolyzer=0,
            cost_h2_storage=0, cost_battery=0, cost_grid=0,
            total_cost_usd=0, solar_full_load_hours=0,
            wind_cf_mean=float(wind_cf.mean()), pct_renewable=0,
        )

    return extract_results(net, params, wind_cf)



if __name__ == "__main__":
    print("pypsa_model.py — self test with dummy CF arrays")
    rng      = __import__("numpy").random.default_rng(42)
    solar_cf = rng.uniform(0, 0.6, 8760) * (rng.random(8760) > 0.5)
    wind_cf  = rng.uniform(0.1, 0.4, 8760)
    res = run_lcoh_from_arrays(solar_cf, wind_cf, representative_weeks=2)
    print(f"  LCOH: ${res.lcoh_usd_per_kg:.3f}/kg  |  status: {res.status}")
    print("✓ pypsa_model.py OK")
