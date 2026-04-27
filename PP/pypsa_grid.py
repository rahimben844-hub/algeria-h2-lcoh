"""
pypsa_grid.py — Run PyPSA LCOH optimization for every Algeria grid cell.

Reads:  data/grid_cells.csv, data/solar_cf_grid.csv, data/wind_cf_grid.csv
Writes: data/lcoh_grid.csv   (one row per cell, columns = LCOH + all cost components)

This is the offline precomputation step.
Typical runtime: ~9 minutes for 281 cells at 4-week resolution.

Usage:
    python pypsa_grid.py                        # default params
    python pypsa_grid.py --weeks 8              # more accurate, slower
    python pypsa_grid.py --h2-demand 5000       # scale to 5 t/yr plant
    python pypsa_grid.py --no-wind              # solar-only scenario
    python pypsa_grid.py --resume               # skip already-solved cells
"""

from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data"
CELLS_PATH   = DATA_DIR / "grid_cells.csv"
SOLAR_PATH   = DATA_DIR / "solar_cf_grid.csv"
WIND_PATH    = DATA_DIR / "wind_cf_grid.csv"
OUTPUT_PATH  = DATA_DIR / "lcoh_grid.csv"

# ── Lazy imports so the module loads fast ────────────────────────────────────
def _get_model():
    from utils import DEFAULT_PARAMS, TechEconomicParams
    from pypsa_model import run_lcoh_from_arrays, LCOHResult
    return DEFAULT_PARAMS, TechEconomicParams, run_lcoh_from_arrays, LCOHResult


def load_grid_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load grid cells and pre-read all CF arrays into memory."""
    cells = pd.read_csv(CELLS_PATH)
    print(f"✓ Loaded {len(cells)} grid cells")

    solar_df = pd.read_csv(SOLAR_PATH, index_col=0, parse_dates=True)
    wind_df  = pd.read_csv(WIND_PATH,  index_col=0, parse_dates=True)
    print(f"✓ Loaded CF arrays: {solar_df.shape[0]} hours × {solar_df.shape[1]} cells")
    return cells, solar_df, wind_df


def solve_all_cells(
    cells:             pd.DataFrame,
    solar_df:          pd.DataFrame,
    wind_df:           pd.DataFrame,
    params,
    h2_demand:         float = 1000.0,
    allow_wind:        bool  = True,
    allow_grid:        bool  = True,
    representative_weeks: int = 4,
    resume:            bool  = False,
    solver:            str   = "highs",
) -> pd.DataFrame:
    """
    Loop over every grid cell, run PyPSA, collect results into a DataFrame.

    Parameters
    ----------
    cells        : grid_cells DataFrame (cell_id, lat, lon)
    solar_df     : solar CF (8760 × n_cells)
    wind_df      : wind CF  (8760 × n_cells)
    params       : TechEconomicParams
    h2_demand    : Annual H2 demand per cell in kg
    allow_wind   : Include wind generator
    allow_grid   : Allow grid electricity backup
    representative_weeks : Weeks to sample (4 = fast, 52 = full year)
    resume       : If True, skip cells already in OUTPUT_PATH
    """
    from utils import annuity_factor, LHV_H2
    import pypsa
    from pypsa_model import build_network, solve_network, extract_results

    # Load existing results for resume mode
    done_cells = set()
    existing_rows = []
    if resume and OUTPUT_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_PATH)
        done_cells  = set(existing_df["cell_id"].values)
        existing_rows = existing_df.to_dict("records")
        print(f"  Resume mode: {len(done_cells)} cells already solved, "
              f"{len(cells) - len(done_cells)} remaining")

    results = list(existing_rows)
    n_total = len(cells)
    n_todo  = n_total - len(done_cells)
    t_start = time.time()

    for i, row in cells.iterrows():
        cell_id = row["cell_id"]

        if cell_id in done_cells:
            continue

        # Progress
        done_so_far = i + 1 - len(done_cells)
        elapsed     = time.time() - t_start
        if done_so_far > 1:
            rate    = elapsed / (done_so_far - 1)
            eta_s   = rate * (n_todo - done_so_far + 1)
            eta_str = f"ETA {eta_s/60:.1f} min"
        else:
            eta_str = "calculating..."

        print(f"  [{done_so_far:3d}/{n_todo}] cell {cell_id} "
              f"(lat={row['lat']:.1f}, lon={row['lon']:.1f})  {eta_str}",
              end="\n", flush=True)

        try:
            # Pull CF arrays for this cell
            solar_cf = solar_df[cell_id].values
            wind_cf  = wind_df[cell_id].values

            # Build + solve network
            net = build_network(
                solar_cf=solar_cf,
                wind_cf=wind_cf,
                params=params,
                h2_demand_kg_per_year=h2_demand,
                allow_grid=allow_grid,
                allow_wind=allow_wind,
                representative_weeks=representative_weeks if representative_weeks < 52 else None,
            )
            status = solve_network(net, solver_name=solver)

            if status == "optimal":
                res = extract_results(net, params, wind_cf)
                record = res.to_dict()
            else:
                record = {"status": status, "lcoh_usd_per_kg": np.nan}

            record["cell_id"] = cell_id
            record["lat"]     = row["lat"]
            record["lon"]     = row["lon"]
            results.append(record)

        except Exception as e:
            print(f"\n  ✗ Failed cell {cell_id}: {e}")
            results.append({
                "cell_id": cell_id, "lat": row["lat"], "lon": row["lon"],
                "status": "error", "lcoh_usd_per_kg": np.nan,
            })

        # Save checkpoint every 20 cells
        if len(results) % 20 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print()  # newline after \r progress

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


def print_summary(df: pd.DataFrame):
    valid = df[df["lcoh_usd_per_kg"].notna() & (df["lcoh_usd_per_kg"] < 100)]
    print(f"\n{'='*60}")
    print(f"  LCOH Grid Results — {len(valid)}/{len(df)} cells solved")
    print(f"{'='*60}")
    print(f"  Min LCOH : ${valid['lcoh_usd_per_kg'].min():.3f} / kg H₂")
    print(f"  Max LCOH : ${valid['lcoh_usd_per_kg'].max():.3f} / kg H₂")
    print(f"  Mean LCOH: ${valid['lcoh_usd_per_kg'].mean():.3f} / kg H₂")

    best = valid.loc[valid["lcoh_usd_per_kg"].idxmin()]
    worst= valid.loc[valid["lcoh_usd_per_kg"].idxmax()]
    print(f"\n  Best  cell: lat={best['lat']:.1f}, lon={best['lon']:.1f} "
          f"→ ${best['lcoh_usd_per_kg']:.3f}/kg")
    print(f"  Worst cell: lat={worst['lat']:.1f}, lon={worst['lon']:.1f} "
          f"→ ${worst['lcoh_usd_per_kg']:.3f}/kg")


def run(
    h2_demand:  float = 1000.0,
    weeks:      int   = 4,
    allow_wind: bool  = True,
    allow_grid: bool  = True,
    resume:     bool  = False,
    solver:     str   = "highs",
):
    print("=" * 60)
    print("  Algeria LCOH Grid Precomputation")
    print("=" * 60)

    DEFAULT_PARAMS, TechEconomicParams, _, _ = _get_model()

    # Load data
    cells, solar_df, wind_df = load_grid_data()

    print(f"\nSettings:")
    print(f"  H₂ demand      : {h2_demand:,.0f} kg/yr per cell")
    print(f"  Time resolution: {weeks} representative weeks")
    print(f"  Wind enabled   : {allow_wind}")
    print(f"  Grid backup    : {allow_grid}")
    print(f"  Est. runtime   : ~{len(cells) * 2 / 60:.0f} min\n")

    t0  = time.time()
    df  = solve_all_cells(
        cells=cells, solar_df=solar_df, wind_df=wind_df,
        params=DEFAULT_PARAMS,
        h2_demand=h2_demand, allow_wind=allow_wind,
        allow_grid=allow_grid, representative_weeks=weeks,
        resume=resume, solver=solver,
    )
    elapsed = time.time() - t0

    print_summary(df)
    print(f"\n  Total time: {elapsed/60:.1f} min")
    print(f"  Saved to  : {OUTPUT_PATH}")
    print(f"\n✅ Precomputation complete — run app.py next")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute LCOH for every Algeria grid cell using PyPSA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python pypsa_grid.py                          # default (4 weeks, 1000 kg/yr)
          python pypsa_grid.py --weeks 8                # more accurate
          python pypsa_grid.py --h2-demand 5000         # larger plant scale
          python pypsa_grid.py --no-wind                # solar-only
          python pypsa_grid.py --resume                 # continue interrupted run
        """) if False else ""
    )
    parser.add_argument("--h2-demand",  type=float, default=1000.0)
    parser.add_argument("--weeks",      type=int,   default=4)
    parser.add_argument("--no-wind",    action="store_true")
    parser.add_argument("--no-grid",    action="store_true")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--solver",     type=str,   default="highs")
    args = parser.parse_args()

    import textwrap  # needed in __main__
    run(
        h2_demand  = args.h2_demand,
        weeks      = args.weeks,
        allow_wind = not args.no_wind,
        allow_grid = not args.no_grid,
        resume     = args.resume,
        solver     = args.solver,
    )
