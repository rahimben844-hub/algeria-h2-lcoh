"""
atlite_grid.py — Merge manually downloaded ERA5 files and extract
                 solar/wind capacity factors for every 2 degree grid cell in Algeria by default.

Your data/ folder should contain:
    era5_p1_accum.nc      <- Jan-Apr  solar radiation (accumulated variables)
    era5_p1_instant.nc    <- Jan-Apr  wind + temperature (instantaneous variables)
    era5_p2_accum.nc      <- May-Aug  solar radiation
    era5_p2_instant.nc    <- May-Aug  wind + temperature
    era5_p3_accum.nc      <- Sep-Dec  solar radiation
    era5_p3_instant.nc    <- Sep-Dec  wind + temperature

Usage:
    python atlite_grid.py
    python atlite_grid.py --year 2024
    python atlite_grid.py --force    # rebuild cutout even if cache exists

Output:
    data/grid_cells.csv          - cell coordinates
    data/solar_cf_grid.csv       - hourly solar CF  (8760 x n_cells)
    data/wind_cf_grid.csv        - hourly wind CF   (8760 x n_cells)
    data/grid_cf_stats.csv       - annual stats per cell
"""

from __future__ import annotations
import argparse
import sys
import textwrap
import warnings                                           # FIX 1a
warnings.filterwarnings("ignore", category=FutureWarning) # FIX 1b: suppress atlite 'H' deprecation
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon


ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

GRID_CELLS_PATH = DATA_DIR / "grid_cells.csv"
ATLITE_NC       = DATA_DIR / "algeria_era5_cutout.nc"
STATS_PATH      = DATA_DIR / "grid_cf_stats.csv"


FILE_PAIRS = [
    ("era5_p1_accum.nc",  "era5_p1_instant.nc",  "Jan-Apr"),
    ("era5_p2_accum.nc",  "era5_p2_instant.nc",  "May-Aug"),
    ("era5_p3_accum.nc",  "era5_p3_instant.nc",  "Sep-Dec"),
]


ALGERIA_BORDER = Polygon([
    (-1.79, 35.84), (-1.21, 35.60), (-0.63, 35.73), (0.00, 35.93),
    (0.57,  36.07), (1.33,  36.17), (2.45,  36.59), (3.06,  36.74),
    (3.48,  36.76), (4.05,  36.72), (4.76,  36.07), (5.08,  36.75),
    (5.41,  36.19), (5.77,  36.82), (6.26,  36.45), (6.61,  36.37),
    (7.43,  36.46), (7.77,  36.90), (8.12,  35.40),
    (8.57,  34.50), (9.05,  33.50), (9.52,  30.23),
    (9.84,  26.50), (9.94,  24.00), (9.36,  23.00),
    (8.57,  21.50), (5.83,  19.45),
    (4.23,  19.14), (3.12,  19.14), (1.16,  20.73),
    (0.16,  14.99),
    (-4.82, 14.99), (-5.40, 15.49), (-5.65, 16.57),
    (-8.67, 27.66), (-8.67, 28.70),
    (-8.00, 32.50), (-6.36, 34.01), (-4.35, 35.17),
    (-3.16, 35.24), (-1.79, 35.84),
])


# =============================================================================
# 1. Grid cells
# =============================================================================
def build_grid_cells(resolution: float = 1.0) -> pd.DataFrame:
    lons = np.arange(-8.5, 10.5 + resolution, resolution)
    lats = np.arange(19.5, 37.5 + resolution, resolution)
    cells = []
    for lat in lats:
        for lon in lons:
            if ALGERIA_BORDER.contains(Point(lon, lat)):
                cells.append({
                    "cell_id": f"{lat:.1f}_{lon:.1f}",
                    "lat":     round(float(lat), 1),
                    "lon":     round(float(lon), 1),
                })
    df = pd.DataFrame(cells)
    print(f"+ {len(df)} grid cells inside Algeria at {resolution} resolution")
    return df


# =============================================================================
# 2. Check input files
# =============================================================================
def check_input_files() -> list[tuple[Path, Path, str]]:
    missing = []
    pairs   = []
    for accum_name, instant_name, label in FILE_PAIRS:
        accum_path   = DATA_DIR / accum_name
        instant_path = DATA_DIR / instant_name
        if not accum_path.exists():   missing.append(str(accum_path))
        if not instant_path.exists(): missing.append(str(instant_path))
        pairs.append((accum_path, instant_path, label))

    if missing:
        print("\n  Missing input files:")
        for f in missing:
            print(f"   {f}")
        sys.exit(1)

    print("+ All 6 ERA5 input files found")
    return pairs


# =============================================================================
# 3. Merge helpers
# =============================================================================
def _drop_problem_dims(ds):
    """
    Handle 'valid_time' correctly depending on what it is in the file:

      Case A — valid_time is the ONLY time dimension (your files):
               rename it to 'time' so xr.concat(dim='time') works.

      Case B — valid_time is a secondary dimension alongside 'time':
               drop it entirely (it's redundant).

      Case C — valid_time is just a plain coordinate:
               drop it.

    Also drops other scalar extras that break concat alignment.

    NOTE (FIX 2): This function ONLY handles dimension/coordinate cleanup.
    Roughness and wind variable handling was incorrectly placed here before —
    at this stage only accum OR instant vars exist (never both), so wnd100m
    does not exist yet. All variable renaming lives in STAGE 7 of merge_era5_files,
    after xr.merge([ds_accum, ds_instant]).
    """
    EXTRA_VARS = ("expver", "number", "realization", "surface")

    if "valid_time" in ds.dims:
        if "time" not in ds.dims:
            # Case A: valid_time IS the time axis — just rename it
            ds = ds.rename({"valid_time": "time"})
        else:
            # Case B: redundant secondary dimension alongside time
            ds = ds.drop_dims("valid_time")
    elif "valid_time" in ds.coords:
        # Case C: plain auxiliary coordinate
        ds = ds.drop_vars("valid_time")

    for name in EXTRA_VARS:
        ds = ds.drop_vars(name, errors="ignore")

    return ds


def _harmonise_grid(datasets: list, decimals: int = 4) -> list:
    """
    Reindex every dataset onto the first dataset's lat/lon grid.
    Fixes AlignmentError when three separate CDS jobs return grids
    that differ by one cell (e.g. 225 vs 224 latitude points).
    """
    if len(datasets) <= 1:
        return datasets

    ref      = datasets[0]
    lat_name = next((c for c in ("latitude", "lat", "y") if c in ref.coords), None)
    lon_name = next((c for c in ("longitude", "lon", "x") if c in ref.coords), None)

    if lat_name is None or lon_name is None:
        print("  WARNING: cannot detect lat/lon coord names - skipping harmonise")
        return datasets

    ref_lat = np.round(ref[lat_name].values, decimals)
    ref_lon = np.round(ref[lon_name].values, decimals)

    out = []
    for i, ds in enumerate(datasets):
        ds = ds.assign_coords({
            lat_name: np.round(ds[lat_name].values, decimals),
            lon_name: np.round(ds[lon_name].values, decimals),
        })
        ds = ds.reindex(
            {lat_name: ref_lat, lon_name: ref_lon},
            method="nearest",
            tolerance=0.5,
        )
        out.append(ds)
        print(f"    Part {i + 1}: {len(ref_lat)} lat x {len(ref_lon)} lon")

    return out


def _deaccumulate(ds, variables: list[str]):
    """
    Convert ERA5 accumulated fields (J/m2) to mean power (W/m2).
    Accumulations reset at 00:00 UTC each day; differencing consecutive
    hourly values gives per-hour energy, then divide by 3600 s.
    """
    ds_out = ds.copy(deep=True)
    for var in variables:
        if var not in ds_out:
            continue
        data  = ds_out[var].values.copy()
        deacc = np.zeros_like(data)
        deacc[0] = data[0]
        for t in range(1, data.shape[0]):
            diff     = data[t] - data[t - 1]
            deacc[t] = np.where(diff >= 0, diff, data[t])
        ds_out[var].values[:] = np.clip(deacc / 3600.0, 0, None)
    return ds_out


# =============================================================================
# 4. Main merge function
# =============================================================================
def merge_era5_files(
    pairs: list[tuple[Path, Path, str]],
    year:  int,
    force: bool = False,
):
    """
    Load 6 raw ERA5 NetCDF files, merge them into one atlite-compatible
    cutout file, and return the atlite.Cutout object.
    """
    import json
    import atlite
    import xarray as xr

    # Fast path: reuse a previously built cutout
    if ATLITE_NC.exists() and not force:
        print(f"  Loading cached cutout: {ATLITE_NC.name}")
        cutout = atlite.Cutout(path=str(ATLITE_NC))
        print(f"+ Cutout loaded - {len(cutout.coords['time'])} time steps")
        return cutout

    # STAGE 1: Load raw files
    print("  Loading ERA5 part files ...")
    accum_parts, instant_parts = [], []

    for accum_path, instant_path, label in pairs:
        print(f"    {label} ...", end=" ", flush=True)
        ds_acc = xr.open_dataset(str(accum_path),   engine="netcdf4")
        ds_ins = xr.open_dataset(str(instant_path), engine="netcdf4")
        # FIX 2: _drop_problem_dims is now coord/dim cleanup ONLY.
        # No roughness or wind variable code belongs here — wnd100m
        # doesn't exist yet at this point (accum files have ssrd/fdir,
        # instant files have u100/v100/t2m). All variable renaming
        # happens in STAGE 7 after the merge.
        ds_acc = _drop_problem_dims(ds_acc)
        ds_ins = _drop_problem_dims(ds_ins)
        accum_parts.append(ds_acc)
        instant_parts.append(ds_ins)
        print("ok")

    # STAGE 2: Harmonise grids (fixes size-mismatch AlignmentError)
    print("  Harmonising spatial grids ...")
    print("  accum:")
    accum_parts   = _harmonise_grid(accum_parts)
    print("  instant:")
    instant_parts = _harmonise_grid(instant_parts)

    # STAGE 3: Concatenate along time
    print("  Concatenating along time ...")
    ds_accum   = xr.concat(accum_parts,   dim="time").sortby("time")
    ds_instant = xr.concat(instant_parts, dim="time").sortby("time")

    # STAGE 4: Deaccumulate radiation (J/m2 -> W/m2)
    accum_vars = [
        v for v in ds_accum.data_vars
        if any(k in v for k in ("ssrd", "fdir", "tisr", "ssr", "str"))
    ]
    if accum_vars:
        print(f"  De-accumulating: {accum_vars}")
        ds_accum = _deaccumulate(ds_accum, accum_vars)
    else:
        print("  WARNING: no accumulated radiation variables found - skipping")

    # STAGE 5: Merge accum + instant
    ds = xr.merge([ds_accum, ds_instant], join="override")

    # STAGE 6: Rename coordinates -> atlite convention (latitude->y, longitude->x)
    coord_rename = {}
    if "longitude" in ds.coords: coord_rename["longitude"] = "x"
    if "latitude"  in ds.coords: coord_rename["latitude"]  = "y"
    if coord_rename:
        ds = ds.rename(coord_rename)

    ds = ds.assign_coords(
        lon=ds["x"],
        lat=ds["y"],
    )

    # STAGE 7: Rename variables -> atlite internal names
    # All variable renaming happens here — AFTER the accum+instant merge,
    # so every variable (ssrd, fdir, u100, v100, t2m, fsr, etc.) is present.
    var_rename: dict[str, str] = {}

    if "tisr" in ds:
        var_rename["tisr"] = "influx_toa"
    if "fdir" in ds:
        var_rename["fdir"] = "influx_direct"
    if "ssrd" in ds and "fdir" in ds:
        ds["influx_diffuse"] = (ds["ssrd"] - ds["fdir"]).clip(min=0.0)
        ds = ds.drop_vars("ssrd")
    elif "ssrd" in ds:
        var_rename["ssrd"] = "influx_diffuse"
    if "t2m" in ds:
        var_rename["t2m"] = "temperature"

    ds = ds.rename({k: v for k, v in var_rename.items() if k in ds})

    if "influx_toa" not in ds and "influx_direct" in ds and "influx_diffuse" in ds:
        print("  ⚠  influx_toa missing — estimating from direct + diffuse (re-download tisr for accuracy)")
        ds["influx_toa"] = (ds["influx_direct"] + ds["influx_diffuse"]) * 1.20
        ds["influx_toa"] = ds["influx_toa"].clip(min=0.0)

    # ── Build wnd100m from wind components ────────────────────────────────────
    if "u100" in ds and "v100" in ds:
        ds["wnd100m"] = np.sqrt(ds["u100"] ** 2 + ds["v100"] ** 2).astype("float32")
        ds = ds.drop_vars(["u100", "v100"])
    elif "u10" in ds and "v10" in ds:
        ds["wnd100m"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2).astype("float32")
        ds = ds.drop_vars(["u10", "v10"])
        print("  WARNING: 100 m wind not found — using 10 m wind as fallback")
    else:
        print("  WARNING: no wind components found — wind CF will be unavailable")

    # ── FIX 3: Build roughness from fsr (MUST be after wnd100m is built) ──────
    # Previously this block was wrongly inside _drop_problem_dims, where it
    # crashed because wnd100m (used as a shape template for the fallback) only
    # exists after the accum+instant merge above.
    #
    # Priority order:
    #   1. fsr in dataset  → rename to roughness (best: actual surface roughness)
    #   2. roughness already named → skip
    #   3. Neither present → constant 2e-4 m fallback (open desert terrain)
    if "fsr" in ds:
        ds = ds.rename({"fsr": "roughness"})
        ds["roughness"] = ds["roughness"].clip(min=1e-5)
        print("  ✓  roughness loaded from fsr")
    elif "roughness" not in ds:
        if "wnd100m" in ds:
            print("  ⚠  roughness (fsr) missing — using constant 2e-4 m (open terrain fallback)")
            ds["roughness"] = xr.DataArray(
                np.full_like(ds["wnd100m"].values, 2e-4, dtype="float32"),
                dims=ds["wnd100m"].dims,
                coords=ds["wnd100m"].coords,
                attrs={"long_name": "surface roughness length (constant)", "units": "m"},
            )
        else:
            print("  WARNING: cannot create roughness fallback — wnd100m not found")

    # ── Build albedo if missing ────────────────────────────────────────────────
    if "albedo" not in ds and "influx_direct" in ds:
        print("  ⚠  albedo missing — using constant 0.28 (Sahara desert average)")
        ds["albedo"] = xr.DataArray(
            np.full_like(ds["influx_direct"].values, 0.28, dtype="float32"),
            dims=ds["influx_direct"].dims,
            coords=ds["influx_direct"].coords,
            attrs={"long_name": "surface albedo (constant approximation)", "units": "1"},
        )

    # STAGE 8: Set mandatory atlite metadata
    # FIX 4: 'roughness' added to wind feature requirements so atlite
    # marks 'wind' as a prepared feature. Without it, atlite silently
    # skips wind even when wnd100m is present.
    FEATURE_MAP = {
        "influx":      ["influx_toa", "influx_direct", "influx_diffuse"],
        "wind":        ["wnd100m", "roughness"],   # FIX 4: roughness required for log extrapolation
        "temperature": ["temperature"],
    }
    prepared = [
        feat for feat, needed in FEATURE_MAP.items()
        if all(v in ds for v in needed)
    ]
    ds.attrs.update({
        "module":            "era5",
        "prepared_features": json.dumps(prepared),
        "history":           f"Merged manually downloaded ERA5 files, year={year}",
    })

    print(f"\n  -- Dataset summary --")
    print(f"  Variables : {sorted(ds.data_vars)}")
    print(f"  Time steps: {len(ds['time'])}")
    print(f"  Lat range : {float(ds['y'].min()):.2f} - {float(ds['y'].max()):.2f}")
    print(f"  Lon range : {float(ds['x'].min()):.2f} - {float(ds['x'].max()):.2f}")
    print(f"  Features  : {prepared}\n")

    # STAGE 9: Write to disk + load as atlite Cutout
    print(f"  Writing {ATLITE_NC.name} ...", flush=True)
    ds.to_netcdf(str(ATLITE_NC))
    print(f"+ Saved ({ATLITE_NC.stat().st_size / 1e6:.0f} MB)")

    cutout = atlite.Cutout(path=str(ATLITE_NC))
    print(f"+ Cutout initialized - {len(cutout.coords['time'])} time steps")
    return cutout


# =============================================================================
# 5. Extract capacity factors
# =============================================================================
def extract_cf_for_grid(
    cutout,
    cells:   pd.DataFrame,
    panel:   str = "CSi",
    turbine: str = "Vestas_V112_3MW",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import xarray as xr

    cutout_lats = cutout.coords["y"].values
    cutout_lons = cutout.coords["x"].values
    n_times     = len(cutout.coords["time"].values)

    print(f"\n  ERA5 grid: {len(cutout_lats)} lat x {len(cutout_lons)} lon, "
          f"{n_times} time steps")
    print(f"  Extracting CFs for {len(cells)} cells...")

    solar_parts, wind_parts = [], []

    for idx, row in cells.iterrows():
        i_lat = int(np.argmin(np.abs(cutout_lats - row["lat"])))
        i_lon = int(np.argmin(np.abs(cutout_lons - row["lon"])))

        layout_pt = xr.DataArray(
            np.zeros((len(cutout_lats), len(cutout_lons))),
            coords={"y": cutout_lats, "x": cutout_lons},
            dims=["y", "x"],
        )
        layout_pt.values[i_lat, i_lon] = 1.0

        tilt = float(abs(row["lat"])) * 0.76
        cf_solar = cutout.pv(
            panel=panel,
            orientation={"slope": tilt, "azimuth": 180.0},
            layout=layout_pt,
            per_unit=True,
        )
        solar_parts.append(cf_solar.values.ravel())

        # FIX 5: Changed from "power" to "logarithmic".
        # "power" law requires wnd_shear_exp, which atlite derives only when
        # BOTH wnd10m and wnd100m are downloaded (needs u10/v10 + u100/v100).
        # "logarithmic" law uses roughness (fsr), which we now keep in STAGE 7.
        # To use power law in the future: add u10/v10 to your CDS download —
        # atlite will then compute wnd_shear_exp automatically.
        cf_wind = cutout.wind(
            turbine=turbine,
            interpolation_method="logarithmic",  # FIX 5: was "power"
            layout=layout_pt,
            per_unit=True,
        )
        wind_parts.append(cf_wind.values.ravel())

        if (idx + 1) % 20 == 0 or (idx + 1) == len(cells):
            pct = 100 * (idx + 1) / len(cells)
            print(f"    [{idx + 1:3d}/{len(cells)}] {pct:.0f}%", flush=True)

    solar_arr  = np.column_stack(solar_parts)
    wind_arr   = np.column_stack(wind_parts)
    timestamps = cutout.coords["time"].values

    solar_df = pd.DataFrame(solar_arr, index=timestamps, columns=cells["cell_id"].values)
    wind_df  = pd.DataFrame(wind_arr,  index=timestamps, columns=cells["cell_id"].values)
    solar_df.index.name = "timestamp"
    wind_df.index.name  = "timestamp"

    return solar_df, wind_df


# =============================================================================
# 6. Statistics
# =============================================================================
def compute_stats(
    cells:    pd.DataFrame,
    solar_df: pd.DataFrame,
    wind_df:  pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, row in cells.iterrows():
        cid = row["cell_id"]
        scf = solar_df[cid]
        wcf = wind_df[cid]
        rows.append({
            "cell_id":               cid,
            "lat":                   row["lat"],
            "lon":                   row["lon"],
            "solar_cf_mean":         round(float(scf.mean()), 4),
            "solar_full_load_hours": round(float(scf.sum()),  1),
            "wind_cf_mean":          round(float(wcf.mean()), 4),
            "wind_full_load_hours":  round(float(wcf.sum()),  1),
        })
    return pd.DataFrame(rows).set_index("cell_id")


# =============================================================================
# 7. Save outputs
# =============================================================================
def save_outputs(
    cells:    pd.DataFrame,
    solar_df: pd.DataFrame,
    wind_df:  pd.DataFrame,
    stats_df: pd.DataFrame,
):
    cells.to_csv(GRID_CELLS_PATH, index=False)
    solar_df.to_csv(DATA_DIR / "solar_cf_grid.csv")
    wind_df.to_csv( DATA_DIR / "wind_cf_grid.csv")
    stats_df.to_csv(STATS_PATH)

    print(f"\n+ Saved:")
    print(f"  {GRID_CELLS_PATH}              ({len(cells)} cells)")
    print(f"  {DATA_DIR / 'solar_cf_grid.csv'}  {solar_df.shape}")
    print(f"  {DATA_DIR / 'wind_cf_grid.csv'}   {wind_df.shape}")
    print(f"  {STATS_PATH}")

    print("\nTop 10 cells by solar full-load hours:")
    print(
        stats_df
        .sort_values("solar_full_load_hours", ascending=False)
        [["lat", "lon", "solar_cf_mean", "solar_full_load_hours", "wind_cf_mean"]]
        .head(10)
        .to_string()
    )


# =============================================================================
# Main
# =============================================================================
def run(year: int = 2024, resolution: float = 2.0, force: bool = False):
    print("=" * 60)
    print("  Algeria LCOH Grid - ERA5 Capacity Factor Extraction")
    print("=" * 60)

    cells = build_grid_cells(resolution)
    cells.to_csv(GRID_CELLS_PATH, index=False)

    pairs = check_input_files()

    cutout = merge_era5_files(pairs, year=year, force=force)

    solar_df, wind_df = extract_cf_for_grid(cutout, cells)

    stats_df = compute_stats(cells, solar_df, wind_df)

    save_outputs(cells, solar_df, wind_df, stats_df)

    print("\n  atlite_grid.py complete - run pypsa_grid.py next")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge downloaded ERA5 files and extract CFs for Algeria grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Required files in data/ folder:
          era5_p1_accum.nc    era5_p1_instant.nc    (Jan-Apr)
          era5_p2_accum.nc    era5_p2_instant.nc    (May-Aug)
          era5_p3_accum.nc    era5_p3_instant.nc    (Sep-Dec)

        Examples:
          python atlite_grid.py
          python atlite_grid.py --year 2024
          python atlite_grid.py --force
        """),
    )
    parser.add_argument("--year",       type=int,   default=2024)
    parser.add_argument("--resolution", type=float, default=2.0)
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()
    run(year=args.year, resolution=args.resolution, force=args.force)
