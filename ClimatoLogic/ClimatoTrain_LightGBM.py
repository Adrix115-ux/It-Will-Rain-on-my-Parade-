# ClimatoTrain_LightGBM.py
# ASCII only

import os
import math
import json
import time
import random
import zipfile
from typing import Dict, List, Tuple, Optional

import requests
import numpy as np
import pandas as pd

import lightgbm as lgb
import geopandas as gpd
from shapely.geometry import Point

try:
    from tqdm import tqdm
except Exception:
    class _DummyBar:
        def __init__(self, total=None, desc="", unit="", position=0, leave=True):
            self.total = total; self.n = 0
        def update(self, n=1): self.n += n
        def set_postfix(self, **kwargs): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, a,b,c): pass
    def tqdm(*args, **kwargs): return _DummyBar(*args, **kwargs)

# --------------------------- Paths and constants ---------------------------

BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
COMMUNITY = "AG"
MAX_PARAMS_PER_REQUEST = 18
REQUEST_TIMEOUT_S = 45
MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.5
PAUSE_BETWEEN_REQUESTS = 0.4
EARTH_R_KM = 6371.0

TRAIN_BASE = r"C:\Users\ENVY\Documents\It-Will-Rain-on-my-Parade\DatosEntrenaMiento"
DATA_DIR = os.path.join(TRAIN_BASE, "data_cache")
CKPT_DIR = os.path.join(TRAIN_BASE, "checkpoints")
MODELS_DIR = os.path.join(TRAIN_BASE, "models")
GEODATA_DIR = os.path.join(TRAIN_BASE, "geodata")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GEODATA_DIR, exist_ok=True)

START_YEAR = 1990
END_YEAR = 2024
PREDICT_HORIZON_MONTHS = 1

N_LAND_POINTS = 20
NEIGHBOR_RADIUS_KM = 50.0
NEIGH_BEARINGS = [0.0, 120.0, 240.0]

LAG_MONTHS = [1, 2, 3]
USE_NEIGH_DIFFS = True

# --------------------------- Parameters ---------------------------

PARAMETERS_ALL = [
    "T2M","T2M_MAX","T2M_MIN","T2MDEW","T2MWET",
    "ALLSKY_SFC_SW_DWN","CLRSKY_SFC_SW_DWN",
    "CLOUD_AMT","RH2M","QV2M","PRECTOTCORR","PS",
    "WS2M","WS10M","WD2M","WD10M",
    "GWETTOP","GWETROOT",
    "ALLSKY_SFC_SW_DIFF","ALLSKY_SRF_ALB","TOA_SW_DWN",
    "TS","ALLSKY_SFC_PAR_TOT","CLRSKY_SFC_PAR_TOT",
    "WS2M_MAX","WS2M_MIN","WS10M_MAX","WS10M_MIN"
]

# Core vars used to align months across main and neighbors
CORE_VARS = ["T2M", "PRECTOTCORR", "RH2M", "WS2M"]

# Use full set
PARAMS = PARAMETERS_ALL

# --------------------------- Weights and LGB params ---------------------------

TARGET_WEIGHTS = {
    "T2M": 1.2, "T2M_MAX": 1.0, "T2M_MIN": 1.0, "T2MDEW": 0.8, "T2MWET": 0.8,
    "ALLSKY_SFC_SW_DWN": 1.0, "CLRSKY_SFC_SW_DWN": 0.7,
    "CLOUD_AMT": 0.9, "RH2M": 0.9, "QV2M": 0.9, "PRECTOTCORR": 1.3, "PS": 0.6,
    "WS2M": 0.9, "WS10M": 0.9, "WD2M": 0.5, "WD10M": 0.5,
    "GWETTOP": 0.8, "GWETROOT": 0.8,
    "ALLSKY_SFC_SW_DIFF": 0.7, "ALLSKY_SRF_ALB": 0.4, "TOA_SW_DWN": 0.7,
    "TS": 1.0, "ALLSKY_SFC_PAR_TOT": 0.9, "CLRSKY_SFC_PAR_TOT": 0.7,
    "WS2M_MAX": 0.9, "WS2M_MIN": 0.9, "WS10M_MAX": 0.9, "WS10M_MIN": 0.9
}

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_data_in_leaf": 20,
    "verbosity": -1,
    "num_threads": max(1, (os.cpu_count() or 4) - 1)
}
NUM_BOOST_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 100
EVAL_LOG_PERIOD = 200

CKPT_EVERY_POINTS = 25
MIN_NON_NA_FRACTION = 0.70
MIN_ROWS_PER_TARGET = 120

PARAM_ALIASES = {
    "PRECTOT": "PRECTOTCORR"
}

# --------------------------- Geo helpers ---------------------------

def get_series(block, var):
    real = var if var in block else PARAM_ALIASES.get(var, var)
    return block.get(real, {})

def dest_point(lat_deg: float, lon_deg: float, bearing_deg: float, distance_km: float) -> Tuple[float, float]:
    phi1 = math.radians(lat_deg)
    lam1 = math.radians(lon_deg)
    theta = math.radians(bearing_deg)
    delta = distance_km / EARTH_R_KM
    sinphi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    phi2 = math.asin(max(-1.0, min(1.0, sinphi2)))
    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    lam2 = lam1 + math.atan2(y, x)
    lat2 = math.degrees(phi2)
    lon2 = (math.degrees(lam2) + 540.0) % 360.0 - 180.0
    return round(lat2, 6), round(lon2, 6)

def _download_natural_earth_land_zip(dest_zip_path: str) -> None:
    urls = [
        "https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_land.zip",
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/110m/physical/ne_110m_land.zip",
    ]
    last_err = None
    for u in urls:
        try:
            r = requests.get(u, timeout=120)
            r.raise_for_status()
            with open(dest_zip_path, "wb") as f: f.write(r.content)
            return
        except Exception as e:
            last_err = e; time.sleep(1)
    raise RuntimeError(f"Could not download Natural Earth land. Last error: {last_err}")

def _ensure_land_shapefile() -> str:
    shp_name = "ne_110m_land.shp"
    shp_path = os.path.join(GEODATA_DIR, shp_name)
    if os.path.exists(shp_path): return shp_path
    zip_path = os.path.join(GEODATA_DIR, "ne_110m_land.zip")
    if not os.path.exists(zip_path): _download_natural_earth_land_zip(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf: zf.extractall(GEODATA_DIR)
    if os.path.exists(shp_path): return shp_path
    for root,_,files in os.walk(GEODATA_DIR):
        for fn in files:
            if fn.lower() == "ne_110m_land.shp": return os.path.join(root, fn)
    raise RuntimeError("ne_110m_land.shp not found after extract")

def _load_land_union():
    shp_path = _ensure_land_shapefile()
    gdf = gpd.read_file(shp_path)
    # Use union_all if available (newer GeoPandas), otherwise fallback to unary_union
    if hasattr(gdf, "union_all"):
        return gdf.union_all()
    return gdf.unary_union

def sample_land_points(n_points: int, random_seed: int = 13) -> List[Tuple[float, float]]:
    random.seed(random_seed)
    land_union = _load_land_union()
    points, trials = [], 0
    lat_min, lat_max = -55.0, 75.0
    while len(points) < n_points and trials < n_points * 300:
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(-180.0, 180.0)
        if land_union.contains(Point(lon, lat)):
            points.append((round(lat, 6), round(lon, 6)))
        trials += 1
    if len(points) < n_points:
        print("Warning: only sampled", len(points), "land points")
    else:
        print("Sampled", len(points), "land points.")
    return points

# --------------------------- NASA POWER helpers ---------------------------

def chunk_list(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def build_url(lat: float, lon: float, start: int, end: int, params_list: List[str]) -> str:
    params_str = ",".join(params_list)
    return (f"{BASE_URL}?parameters={params_str}"
            f"&community={COMMUNITY}"
            f"&longitude={lon}&latitude={lat}"
            f"&start={start}&end={end}"
            f"&format=JSON")

def sanitize_name(s: str) -> str:
    return str(s).replace(".", "p").replace("-", "m").replace(" ", "")

def request_with_retries(url: str, timeout: int = REQUEST_TIMEOUT_S) -> requests.Response:
    backoff = RETRY_BACKOFF_S
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=timeout)
            return r
        except requests.exceptions.RequestException as e:
            last_exc = e
            time.sleep(backoff)
            backoff *= 1.8
    raise RuntimeError(f"Request failed after retries: {last_exc}")

def fetch_power_series(lat: float, lon: float, start_year: int, end_year: int,
                       params: List[str], role: str = "Main", sec_idx: Optional[int] = None,
                       pb: Optional[tqdm] = None) -> Dict[str, Dict[str, float]]:
    role_tag = f"{role}" if sec_idx is None else f"Sec{sec_idx}"
    cache_key = f"{role_tag}_lat{sanitize_name(lat)}_lon{sanitize_name(lon)}_{start_year}-{end_year}.json"
    cache_path = os.path.join(DATA_DIR, cache_key)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    merged: Dict[str, Dict[str, float]] = {}
    chunks = chunk_list(params, MAX_PARAMS_PER_REQUEST)
    for ci, chunk in enumerate(chunks, start=1):
        if pb is not None:
            pb.set_postfix(point_role=role_tag, chunk=f"{ci}/{len(chunks)}", refresh=True)
        url = build_url(lat, lon, start_year, end_year, chunk)
        resp = request_with_retries(url)
        if resp.status_code != 200:
            raise RuntimeError(f"POWER HTTP {resp.status_code} {resp.text[:140]}")
        data = resp.json()
        block = (data.get("properties", {}) or {}).get("parameter", {}) or data.get("parameters", {}) or {}

        for var, series in block.items():
            clean = {}
            for k, v in (series or {}).items():
                if not k or len(k) != 6 or k.endswith("13"):
                    continue
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if fv == -999.0:
                    continue
                clean[str(k)] = fv
            merged[var] = clean
        time.sleep(PAUSE_BETWEEN_REQUESTS)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=True, indent=2)
    return merged

# --------------------------- Dataset builder ---------------------------

def aligned_months_core(main_series: Dict[str, Dict[str, float]],
                        neigh_list: List[Dict[str, Dict[str, float]]]) -> List[str]:
    sets = []
    for v in CORE_VARS:
        mv = main_series.get(v, {})
        if not mv: return []
        sets.append(set(mv.keys()))
    common = set.intersection(*sets) if sets else set()
    if not common: return []

    def neigh_ok(neigh, months):
        for v in CORE_VARS:
            sv = neigh.get(v, {})
            if not sv or not months.issubset(sv.keys()):
                return False
        return True

    for neigh in neigh_list:
        if not neigh_ok(neigh, common):
            for v in CORE_VARS:
                common = common & set(neigh.get(v, {}).keys())
            if not common: return []
    return sorted(common)

def make_point_frame(series_main: Dict[str, Dict[str, float]],
                     neigh_list: List[Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    months = aligned_months_core(series_main, neigh_list)
    if not months:
        return pd.DataFrame()

    df = pd.DataFrame({"yyyymm": months})
    df["yyyymm"] = df["yyyymm"].astype(str)
    yyyymm_list = df["yyyymm"].tolist()

    # main vars block
    main_cols = {}
    for var in PARAMS:
        ser = series_main.get(var, {})
        main_cols[f"{var}__main"] = [ser.get(m, np.nan) for m in yyyymm_list]
    main_df = pd.DataFrame(main_cols)

    # neighbor diffs blocks
    diff_blocks = []
    if USE_NEIGH_DIFFS:
        for i, neigh in enumerate(neigh_list, start=1):
            nd = {}
            for var in PARAMS:
                nser = neigh.get(var, {})
                nvals = [nser.get(m, np.nan) for m in yyyymm_list]
                nd[f"{var}__diff_n{i}"] = np.array(main_cols[f"{var}__main"], dtype="float32") - np.array(nvals, dtype="float32")
            diff_blocks.append(pd.DataFrame(nd))

    # concat blocks
    blocks = [df, main_df] + diff_blocks
    df = pd.concat(blocks, axis=1)

    # lags block
    lag_cols = {}
    for var in PARAMS:
        base = f"{var}__main"
        base_ser = pd.Series(main_df[base])
        for L in LAG_MONTHS:
            lag_cols[f"{base}__lag{L}"] = base_ser.shift(L).values
    if lag_cols:
        df = pd.concat([df, pd.DataFrame(lag_cols)], axis=1)

    # targets block
    tgt_cols = {}
    for var in PARAMS:
        tgt_cols[f"y_{var}"] = pd.Series(main_df[f"{var}__main"]).shift(-PREDICT_HORIZON_MONTHS).values
    df = pd.concat([df, pd.DataFrame(tgt_cols)], axis=1)

    # typing
    numeric_cols = [c for c in df.columns if c != "yyyymm"]
    df[numeric_cols] = df[numeric_cols].astype("float32")

    # minimal must-have columns
    must_have = []
    for v in CORE_VARS:
        must_have.append(f"{v}__main")
        for L in LAG_MONTHS:
            must_have.append(f"{v}__main__lag{L}")
        if USE_NEIGH_DIFFS:
            for i in range(1, 4):
                must_have.append(f"{v}__diff_n{i}")
        must_have.append(f"y_{v}")

    df = df.dropna(subset=must_have).reset_index(drop=True)
    return df

def build_training_dataset(n_points: int) -> pd.DataFrame:
    pts = sample_land_points(n_points)
    frames = []

    with tqdm(total=len(pts), desc="Building dataset (points)", unit="pt", position=0, leave=True) as pb:
        for idx, (lat, lon) in enumerate(pts, start=1):
            neigh_coords = [dest_point(lat, lon, b, NEIGHBOR_RADIUS_KM) for b in NEIGH_BEARINGS]
            try:
                main_series = fetch_power_series(lat, lon, START_YEAR, END_YEAR, PARAMS, role="Main", pb=pb)
                neigh_series = []
                for si, (la, lo) in enumerate(neigh_coords, start=1):
                    ns = fetch_power_series(la, lo, START_YEAR, END_YEAR, PARAMS, role="Sec", sec_idx=si, pb=pb)
                    neigh_series.append(ns)
                dfp = make_point_frame(main_series, neigh_series)
                if not dfp.empty:
                    dfp["lat"] = lat; dfp["lon"] = lon
                    frames.append(dfp)
            except Exception as e:
                tqdm.write(f"Point failed lat={lat} lon={lon}: {e}")

            if idx % CKPT_EVERY_POINTS == 0 and frames:
                ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{idx}.csv")
                try:
                    pd.concat(frames, ignore_index=True).to_csv(ckpt_path, index=False)
                    tqdm.write(f"Checkpoint written: {ckpt_path}")
                except Exception as e:
                    tqdm.write(f"Checkpoint failed at {idx}: {e}")

            pb.update(1)
            pb.set_postfix(done=f"{idx}/{len(pts)}", point_role="idle", chunk="-", refresh=True)

    if not frames:
        raise RuntimeError("No training rows built")
    full = pd.concat(frames, ignore_index=True)

    non_na_frac = full.notna().mean()
    keep_cols = non_na_frac[non_na_frac >= MIN_NON_NA_FRACTION].index.tolist()
    if "yyyymm" not in keep_cols:
        keep_cols = ["yyyymm"] + keep_cols
    full = full[keep_cols].copy()

    TARGETS_USED = []
    for v in PARAMS:
        ycol = f"y_{v}"
        if ycol in full.columns and full[ycol].notna().sum() >= MIN_ROWS_PER_TARGET:
            TARGETS_USED.append(v)

    if not TARGETS_USED:
        raise RuntimeError("No usable targets after pruning; adjust thresholds or add points/vars.")

    used_path = os.path.join(CKPT_DIR, "targets_used.json")
    snap_path = os.path.join(CKPT_DIR, "dataset_full.csv")

    try:
        full.to_csv(snap_path, index=False)
    except Exception as e:
        print(f"Warning: could not write snapshot CSV: {e}")

    try:
        with open(used_path, "w", encoding="utf-8") as f:
            json.dump({"targets_used": TARGETS_USED, "params": PARAMS}, f, ensure_ascii=True, indent=2)
    except Exception as e:
        print(f"Warning: could not write targets_used.json: {e}")

    print(f"Full dataset snapshot: {snap_path}  rows={len(full)}  cols={len(full.columns)}")
    print(f"Targets used ({len(TARGETS_USED)}): {TARGETS_USED}")
    return full, TARGETS_USED

# --------------------------- Training per target ---------------------------

def _best_valid_rmse_from_scores(scores_dict: dict) -> Optional[float]:
    if not isinstance(scores_dict, dict):
        return None
    best = None
    for k, v in scores_dict.items():
        if not isinstance(v, dict):
            continue
        for metric_name in ("rmse", "l2"):
            if metric_name in v:
                try:
                    val = float(v[metric_name])
                    best = val if (best is None or val < best) else best
                except Exception:
                    pass
    return best

def train_models(df: pd.DataFrame, targets: List[str]):
    work = df.copy()
    work["yyyymm_int"] = work["yyyymm"].astype(int)
    work = work.sort_values("yyyymm_int").reset_index(drop=True)

    cutoff = np.percentile(work["yyyymm_int"], 90)
    train_df = work[work["yyyymm_int"] < cutoff]
    valid_df = work[work["yyyymm_int"] >= cutoff]

    drop_cols = set(["yyyymm", "yyyymm_int"])
    feature_cols = [c for c in work.columns if (c not in drop_cols) and (not c.startswith("y_"))]

    X_train = train_df[feature_cols]
    X_valid = valid_df[feature_cols]

    models_registry = {
        "features": feature_cols,
        "lags": LAG_MONTHS,
        "neighbor_radius_km": NEIGHBOR_RADIUS_KM,
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "predict_horizon_months": PREDICT_HORIZON_MONTHS,
        "targets": {}
    }

    cb_list = []
    if hasattr(lgb, "early_stopping"):
        cb_list.append(lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False))
    if hasattr(lgb, "log_evaluation") and EVAL_LOG_PERIOD:
        cb_list.append(lgb.log_evaluation(EVAL_LOG_PERIOD))

    for tgt in targets:
        y_tr = train_df[f"y_{tgt}"]
        y_va = valid_df[f"y_{tgt}"]

        if len(y_tr) < MIN_ROWS_PER_TARGET or len(y_va) < max(24, int(0.05 * len(y_tr))):
            print(f"Skip target {tgt}: not enough rows train={len(y_tr)} valid={len(y_va)}")
            continue

        w_tr = np.ones(len(y_tr), dtype="float32") * float(TARGET_WEIGHTS.get(tgt, 1.0))
        w_va = np.ones(len(y_va), dtype="float32")

        dtrain = lgb.Dataset(X_train, label=y_tr, weight=w_tr, free_raw_data=False)
        dvalid = lgb.Dataset(X_valid, label=y_va, weight=w_va, reference=dtrain, free_raw_data=False)

        print(f"Training LightGBM for target {tgt}  rows train/valid = {len(y_tr)}/{len(y_va)}")
        booster = lgb.train(
            params=LGB_PARAMS,
            train_set=dtrain,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            num_boost_round=NUM_BOOST_ROUNDS,
            callbacks=cb_list
        )

        model_path = os.path.join(MODELS_DIR, f"lgb_{tgt}.txt")
        booster.save_model(model_path, num_iteration=booster.best_iteration or booster.current_iteration())

        best_rmse = _best_valid_rmse_from_scores(booster.best_score)

        models_registry["targets"][tgt] = {
            "model_path": model_path,
            "best_iteration": int(booster.best_iteration or 0),
            "rmse_valid": best_rmse
        }

    reg_path = os.path.join(MODELS_DIR, "models_registry.json")
    try:
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(models_registry, f, ensure_ascii=True, indent=2)
        print(f"Models saved to: {MODELS_DIR}")
        print(f"Registry written: {reg_path}")
    except Exception as e:
        print(f"Warning: could not write models_registry.json: {e}")

# --------------------------- Main ---------------------------

if __name__ == "__main__":
    print(f"Building training dataset with {N_LAND_POINTS} land points ...")
    dataset, targets_used = build_training_dataset(N_LAND_POINTS)
    print("Training LightGBM models (one per target) ...")
    train_models(dataset, targets_used)
    print("Done.")
