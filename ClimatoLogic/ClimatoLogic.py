# ClimatoPredict_LightGBM_filtered.py
# ASCII only - relative paths, single output python-java.json
# Adds optional versioned JSON and CSV audit controlled by java-python.json

import os
import re
import math
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import lightgbm as lgb
import requests

# --------------------------- Paths and constants (relative) ----------------

THIS_DIR = Path(__file__).resolve().parent

def detect_project_root(start: Path) -> Path:
    candidates = [start, start.parent, start.parent.parent]
    for p in candidates:
        if (p / "Comunicacion").exists() and (p / "nasa_power_output").exists():
            return p
    return start

PROJECT_ROOT = detect_project_root(THIS_DIR)

IO_BASE = PROJECT_ROOT / "Comunicacion"
OUT_BASE = PROJECT_ROOT / "nasa_power_output"
TRAIN_BASE = PROJECT_ROOT / "DatosEntrenaMiento"
MODELS_DIR = TRAIN_BASE / "models"
CKPT_DIR = TRAIN_BASE / "checkpoints"
REGISTRY_PATH = MODELS_DIR / "models_registry.json"

JAVA_PYTHON_IN = IO_BASE / "java-python.json"
PYTHON_JAVA_OUT = IO_BASE / "python-java.json"
PYTHON_JAVA_OUT_DIR = IO_BASE

for d in [IO_BASE, OUT_BASE, TRAIN_BASE, MODELS_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("PROJECT_ROOT:", str(PROJECT_ROOT))
print("IO_BASE:", str(IO_BASE))
print("OUT_BASE:", str(OUT_BASE))
print("TRAIN_BASE:", str(TRAIN_BASE))
print("MODELS_DIR:", str(MODELS_DIR))
print("CKPT_DIR:", str(CKPT_DIR))
print("Will write to:", str(PYTHON_JAVA_OUT))

# --------------------------- Fixed constants ------------------------------

BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
COMMUNITY = "AG"

START_YEAR = 1990
END_YEAR = 2025
MAX_PARAMS_PER_REQUEST = 20
PAUSE_BETWEEN_REQUESTS = 0.80
MAX_RETRIES = 3
EARTH_R_KM = 6371.0

# KEEP_VARS: variables exactly as in your DTO
KEEP_VARS = [
    "T2M",              # Temperatura a 2 metros
    "T2M_MAX",          # Temperatura maxima 2m
    "T2M_MIN",          # Temperatura minima 2m
    "T2MDEW",           # Punto de rocio
    "T2MWET",           # Humedad relativa (si POWER lo entrega)
    "ALLSKY_SFC_SW_DWN",# Radiacion total
    "CLOUD_AMT",        # Nubosidad
    "PRECTOTCORR",      # Precipitacion corregida
    "PS",               # Presion superficial
    "WS10M"             # Velocidad viento 10m
]

# Core vars used to align months across main and neighbors.
# Must be subset of KEEP_VARS. Keep minimal but stable.
CORE_VARS = ["T2M", "PRECTOTCORR", "WS10M", "PS"]

PARAM_ALIASES = {"PRECTOT": "PRECTOTCORR"}  # alias handling
USE_NEIGH_DIFFS_DEFAULT = True

# Optional outputs (defaults keep "no duplicate" behavior)
DEFAULT_SAVE_VERSIONED = False
DEFAULT_SAVE_CSV = False

# --------------------------- Utils ----------------------------------------

def yyyymm_from_yyyymmdd(s: str) -> str:
    s = str(s).strip()
    if not re.fullmatch(r"\d{8}", s):
        raise ValueError("target_date must be YYYYMMDD")
    return s[:6]

def dedupe(lst: List[str]) -> List[str]:
    seen, out = set(), []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def safe_coord(x: float) -> str:
    s = f"{x:.4f}"
    return s.replace("-", "m").replace(".", "p")

def dest_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> Tuple[float, float]:
    phi1, lam1 = math.radians(lat), math.radians(lon)
    theta, delta = math.radians(bearing_deg), distance_km / EARTH_R_KM
    sinphi2 = math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    phi2 = math.asin(max(-1.0, min(1.0, sinphi2)))
    y = math.sin(theta) * math.sin(delta) * math.cos(phi1)
    x = math.cos(delta) - math.sin(phi1) * math.sin(phi2)
    lam2 = lam1 + math.atan2(y, x)
    lat2 = math.degrees(phi2)
    lon2 = (math.degrees(lam2) + 540.0) % 360.0 - 180.0
    return round(lat2, 6), round(lon2, 6)

def build_url(lat: float, lon: float, start: int, end: int, params: List[str]) -> str:
    params_str = ",".join(params)
    return (
        f"{BASE_URL}?parameters={params_str}"
        f"&community={COMMUNITY}"
        f"&longitude={lon}&latitude={lat}"
        f"&start={start}&end={end}&format=JSON"
    )

def request_with_retries(url: str, timeout: int = 120) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return requests.get(url, timeout=timeout)
        except Exception as e:
            last_exc = e
            time.sleep(1.0 * (2 ** (attempt - 1)))
    raise RuntimeError(f"Network failed after retries: {last_exc}")

def chunk_list(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i:i+n] for i in range(0, len(lst), n)]

def json_nan_to_none(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_nan_to_none(v) for v in obj]
    return obj

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = json_nan_to_none(data)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=True, indent=2)

def load_json(path: Path) -> dict:
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)

def month_add(yyyymm: str, k: int) -> str:
    y = int(yyyymm[:4]); m = int(yyyymm[4:])
    m += k
    while m > 12:
        y += 1; m -= 12
    while m <= 0:
        y -= 1; m += 12
    return f"{y}{m:02d}"

def month_diff(m2: str, m1: str) -> int:
    y2, m2i = int(m2[:4]), int(m2[4:])
    y1, m1i = int(m1[:4]), int(m1[4:])
    return (y2 - y1) * 12 + (m2i - m1i)

def prev_month(yyyymm: str, k: int) -> str:
    return month_add(yyyymm, -k)

# --------------------------- POWER IO -------------------------------------

def normalize_power_parameters(raw: dict) -> Dict[str, Dict[str, float]]:
    block = (raw.get("properties", {}) or {}).get("parameter", {}) or raw.get("parameters", {}) or {}
    clean: Dict[str, Dict[str, float]] = {}
    for var, series in (block or {}).items():
        real_var = PARAM_ALIASES.get(var, var)
        if real_var not in KEEP_VARS:
            continue
        d = {}
        for k, v in (series or {}).items():
            if not k or len(k) != 6 or k.endswith("13"):
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if fv == -999.0:
                continue
            d[str(k)] = fv
        clean[real_var] = d
    return clean

def download_point_to_json(lat: float, lon: float, params: List[str], out_file: Path) -> None:
    chunks = chunk_list(params, MAX_PARAMS_PER_REQUEST)
    merged = {}
    for i, ch in enumerate(chunks, start=1):
        url = build_url(lat, lon, START_YEAR, END_YEAR, ch)
        print(f"Chunk {i}/{len(chunks)} for {lat},{lon}")
        resp = request_with_retries(url)
        if resp.status_code != 200:
            print("Failed:", resp.status_code, resp.text[:140])
            continue
        try:
            d = resp.json()
            props = d.get("properties", {})
            pset = props.get("parameter", {}) or d.get("parameters", {})
            for k, v in (pset or {}).items():
                merged.setdefault(k, {}).update(v or {})
        except Exception as e:
            print("Parse failed:", e)
        time.sleep(PAUSE_BETWEEN_REQUESTS)
    save_json({"parameters": merged}, out_file)

# ---------------------- Feature reconstruction (as training) ---------------

def aligned_months_core(series_main: Dict[str, Dict[str, float]],
                        neigh_list: List[Dict[str, Dict[str, float]]],
                        core_vars: List[str]) -> List[str]:
    sets = []
    for v in core_vars:
        mv = series_main.get(v, {})
        if not mv:
            return []
        sets.append(set(mv.keys()))
    common = set.intersection(*sets) if sets else set()
    if not common:
        return []
    for neigh in neigh_list:
        for v in core_vars:
            sv = neigh.get(v, {})
            if not sv:
                return []
            common = common & set(sv.keys())
            if not common:
                return []
    return sorted(common)

def build_feature_row_for_month(
    month_key: str,
    series_main: Dict[str, Dict[str, float]],
    neigh_list: List[Dict[str, Dict[str, float]]],
    feature_cols: List[str],
    all_params: List[str],
    lags: List[int],
    use_neigh_diffs: bool,
    lat: float,
    lon: float
) -> pd.DataFrame:
    row = {}
    for var in all_params:
        row[f"{var}__main"] = series_main.get(var, {}).get(month_key, np.nan)
    if use_neigh_diffs:
        for i, neigh in enumerate(neigh_list, start=1):
            for var in all_params:
                mv = series_main.get(var, {}).get(month_key, np.nan)
                nv = neigh.get(var, {}).get(month_key, np.nan)
                row[f"{var}__diff_n{i}"] = (mv - nv) if (mv == mv and nv == nv) else np.nan
    for var in all_params:
        base_series = series_main.get(var, {})
        for L in lags:
            lag_key = prev_month(month_key, L)
            row[f"{var}__main__lag{L}"] = base_series.get(lag_key, np.nan)
    row["lat"] = float(lat)
    row["lon"] = float(lon)
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in feature_cols}])
    for c in df.columns:
        if c != "yyyymm":
            df[c] = df[c].astype("float32")
    return df

# --------------------------- Registry and prediction -----------------------

def load_models_registry(registry_path: Path) -> dict:
    with open(str(registry_path), "r", encoding="utf-8") as f:
        return json.load(f)

def infer_lat_lon_from_folder(folder_path: Path) -> Tuple[float, float]:
    name = folder_path.name
    try:
        parts = name.split("_")
        latp = [p for p in parts if p.startswith("lat")][0].replace("lat", "")
        lonp = [p for p in parts if p.startswith("lon")][0].replace("lon", "")
        lat = float(latp.replace("m", "-").replace("p", "."))
        lon = float(lonp.replace("m", "-").replace("p", "."))
        return lat, lon
    except Exception:
        return np.nan, np.nan

def ensure_download_session(
    main_lat: float,
    main_lon: float,
    radius_km: float,
    bearings: List[float],
    parameters: List[str],
    out_base: Path
) -> Tuple[Path, List[Path]]:
    folder_name = f"lat{safe_coord(main_lat)}_lon{safe_coord(main_lon)}_{START_YEAR}-{END_YEAR}"
    session_folder = out_base / folder_name
    session_folder.mkdir(parents=True, exist_ok=True)

    points = [(main_lat, main_lon)]
    for b in bearings:
        points.append(dest_point(main_lat, main_lon, b, radius_km))

    json_paths: List[Path] = []
    for i, (la, lo) in enumerate(points, start=1):
        tag = "main_point" if i == 1 else f"secondary_{i-1}"
        out_file = session_folder / f"{tag}.json"
        if not out_file.exists():
            print(f"Downloading {tag} at {la},{lo} ...")
            download_point_to_json(la, lo, parameters, out_file)
        else:
            print(f"Using cached {tag}: {str(out_file)}")
        json_paths.append(out_file)

    return session_folder, json_paths

def next_valid_feature_month(series_main: Dict[str, Dict[str, float]], neigh_list: List[Dict[str, Dict[str, float]]]) -> str:
    months = aligned_months_core(series_main, neigh_list, CORE_VARS)
    if not months:
        raise RuntimeError("No common months across core vars and neighbors")
    return months[-1]

def predict_for_month(
    feature_month: str,
    series_main: Dict[str, Dict[str, float]],
    neigh_list: List[Dict[str, Dict[str, float]]],
    models_dir: Path,
    registry: dict,
    lat: float,
    lon: float
) -> Dict[str, float]:
    feature_cols: List[str] = registry.get("features", [])
    lags: List[int] = registry.get("lags", [1, 2, 3])
    use_neigh_diffs: bool = registry.get("use_neigh_diffs", USE_NEIGH_DIFFS_DEFAULT)
    targets_dict = registry.get("targets", {}) or {}

    params_in_features = sorted(set([c.split("__")[0] for c in feature_cols if ("__main" in c or "__diff_n" in c)]))
    # only keep params that are in KEEP_VARS
    params_in_features = [p for p in params_in_features if p in KEEP_VARS]
    X = build_feature_row_for_month(
        feature_month,
        series_main,
        neigh_list,
        feature_cols,
        params_in_features,
        lags,
        use_neigh_diffs,
        lat,
        lon
    )

    yhat_step: Dict[str, float] = {}
    # if models exist for targets, use them; else return nan for modelled targets
    for tgt, info in targets_dict.items():
        if tgt not in KEEP_VARS:
            # we discard targets not in KEEP_VARS
            continue
        model_path = Path(info.get("model_path") or (models_dir / f"lgb_{tgt}.txt"))
        if not model_path.exists():
            yhat_step[tgt] = float("nan")
            continue
        booster = lgb.Booster(model_file=str(model_path))
        try:
            yval = float(booster.predict(X)[0])
        except Exception:
            yval = float("nan")
        yhat_step[tgt] = yval
    return yhat_step

def roll_forecast_to_month(
    target_month: str,
    series_main: Dict[str, Dict[str, float]],
    neigh_list: List[Dict[str, Dict[str, float]]],
    models_dir: Path,
    registry: dict,
    lat: float,
    lon: float
) -> Tuple[str, Dict[str, float], List[dict]]:
    horizon: int = int(registry.get("predict_horizon_months", 1))

    current_feature_month = next_valid_feature_month(series_main, neigh_list)
    steps = max(0, (month_diff(target_month, current_feature_month) + horizon - 1) // horizon)
    print("[DEBUG] current_feature_month (start):", current_feature_month)
    print("[DEBUG] steps to do:", steps, "horizon:", horizon, "-> target:", target_month)

    last_pred_month = None
    last_yhat: Dict[str, float] = {}
    trace: List[dict] = []

    for step in range(steps):
        predict_month = month_add(current_feature_month, horizon)
        print(f"[DEBUG] step {step+1}/{steps} -> predict {predict_month}")

        yhat_step = predict_for_month(
            feature_month=current_feature_month,
            series_main=series_main,
            neigh_list=neigh_list,
            models_dir=models_dir,
            registry=registry,
            lat=lat,
            lon=lon
        )

        # persist synthetic targets only if they are in KEEP_VARS
        for tgt, val in yhat_step.items():
            if tgt in KEEP_VARS:
                series_main.setdefault(tgt, {})[predict_month] = val

        trace.append({"feature_month": current_feature_month, "predicted_month": predict_month, "yhat": yhat_step})

        last_pred_month = predict_month
        last_yhat = yhat_step
        current_feature_month = month_add(current_feature_month, horizon)

    # if no steps but we could need a single predict to align to target
    if steps == 0:
        predict_month = month_add(current_feature_month, horizon)
        while predict_month < target_month:
            print(f"[DEBUG] preroll to {predict_month} (advance to reach target {target_month})")
            yhat_step = predict_for_month(
                feature_month=current_feature_month,
                series_main=series_main,
                neigh_list=neigh_list,
                models_dir=models_dir,
                registry=registry,
                lat=lat,
                lon=lon
            )
            for tgt, val in yhat_step.items():
                if tgt in KEEP_VARS:
                    series_main.setdefault(tgt, {})[predict_month] = val
            trace.append({"feature_month": current_feature_month, "predicted_month": predict_month, "yhat": yhat_step})
            last_pred_month = predict_month
            last_yhat = yhat_step
            current_feature_month = month_add(current_feature_month, horizon)
            predict_month = month_add(current_feature_month, horizon)

        if predict_month == target_month:
            yhat_step = predict_for_month(
                feature_month=current_feature_month,
                series_main=series_main,
                neigh_list=neigh_list,
                models_dir=models_dir,
                registry=registry,
                lat=lat,
                lon=lon
            )
            for tgt, val in yhat_step.items():
                if tgt in KEEP_VARS:
                    series_main.setdefault(tgt, {})[predict_month] = val
            trace.append({"feature_month": current_feature_month, "predicted_month": predict_month, "yhat": yhat_step})
            last_pred_month = predict_month
            last_yhat = yhat_step

    return last_pred_month or current_feature_month, last_yhat, trace

# --------------------------- Main -----------------------------------------

if __name__ == "__main__":
    main_lat = -17.6000
    main_lon = -62.9200
    radius_km = 50.0
    bearings = [0.0, 120.0, 240.0]

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"models_registry.json not found at: {str(REGISTRY_PATH)}")
    registry = load_models_registry(REGISTRY_PATH)

    # always request only KEEP_VARS (apply aliases if any)
    params_needed = dedupe([PARAM_ALIASES.get(p, p) for p in KEEP_VARS])
    for core in CORE_VARS:
        if core not in params_needed:
            params_needed.append(core)
    print("[DEBUG] params_needed (forced KEEP_VARS):", params_needed)

    session_folder, json_paths = ensure_download_session(
        main_lat=main_lat,
        main_lon=main_lon,
        radius_km=radius_km,
        bearings=bearings,
        parameters=params_needed,
        out_base=OUT_BASE
    )
    main_json = json_paths[0]
    sec_jsons = json_paths[1:]

    # normalize and then filter to KEEP_VARS (normalize already filters)
    series_main = normalize_power_parameters(load_json(main_json))
    neigh_list = [normalize_power_parameters(load_json(p)) for p in sec_jsons]

    # infer lat lon from folder if possible
    lat_infer, lon_infer = infer_lat_lon_from_folder(main_json.parent)
    if lat_infer == lat_infer and lon_infer == lon_infer:
        lat_use, lon_use = lat_infer, lon_infer
    else:
        lat_use, lon_use = main_lat, main_lon

    # Read control flags and inputs from java-python.json
    save_versioned = DEFAULT_SAVE_VERSIONED
    save_csv = DEFAULT_SAVE_CSV
    try:
        if not JAVA_PYTHON_IN.exists():
            raise FileNotFoundError(str(JAVA_PYTHON_IN) + " not found")
        jpin = load_json(JAVA_PYTHON_IN)
        lat_use = float(jpin.get("latitude", lat_use))
        lon_use = float(jpin.get("longitude", lon_use))
        target_month = yyyymm_from_yyyymmdd(jpin["target_date"]) if "target_date" in jpin else None
        save_versioned = bool(jpin.get("save_versioned", save_versioned))
        save_csv = bool(jpin.get("save_csv", save_csv))
        print("[DEBUG] Read from java-python.json:", lat_use, lon_use, target_month, "flags:", save_versioned, save_csv)
    except Exception as e:
        print("WARN reading java-python.json, fallback to folder coords + next month:", e)
        lat_use, lon_use = infer_lat_lon_from_folder(main_json.parent)
        if not (lat_use == lat_use and lon_use == lon_use):
            lat_use, lon_use = main_lat, main_lon
        target_month = None

    out_payload: Dict[str, Any] = {
        "requested_month": None,
        "predicted_month": None,
        "horizon_months": int(registry.get("predict_horizon_months", 1)),
        "lat": lat_use,
        "lon": lon_use,
        "data": None  # will contain only KEEP_VARS values at predicted_month
    }

    try:
        last_common = next_valid_feature_month(series_main, neigh_list)
        horizon = int(registry.get("predict_horizon_months", 1))
        if not target_month:
            target_month = month_add(last_common, horizon)
        if month_diff(target_month, last_common) < horizon:
            target_month = month_add(last_common, horizon)
        print("[DEBUG] last_common:", last_common, "target_month:", target_month, "horizon:", horizon)

        final_month, yhat, trace = roll_forecast_to_month(
            target_month=target_month,
            series_main=series_main,
            neigh_list=neigh_list,
            models_dir=MODELS_DIR,
            registry=registry,
            lat=lat_use,
            lon=lon_use
        )
        print("[DEBUG] final_month:", final_month, "pred_count:", len(yhat or {}))

        # ---------- REEMPLAZADO: construccion de data_at_month con fallback y lag ----------
        # funciones auxiliares
        def yyyymm_to_index(yyyymm: str) -> int:
            y = int(yyyymm[:4]); m = int(yyyymm[4:]); return y * 12 + m

        def clamp(val: float, lo: Optional[float], hi: Optional[float]) -> Optional[float]:
            if val is None:
                return None
            try:
                v = float(val)
            except Exception:
                return None
            if lo is not None and v < lo:
                return lo
            if hi is not None and v > hi:
                return hi
            return v

        VAR_BOUNDS = {
            "T2M": (-90.0, 60.0),
            "T2M_MAX": (-90.0, 60.0),
            "T2M_MIN": (-90.0, 60.0),
            "T2MDEW": (-90.0, 60.0),
            "T2MWET": (-90.0, 60.0),
            "ALLSKY_SFC_SW_DWN": (0.0, None),
            "CLOUD_AMT": (0.0, 1.0),
            "PRECTOTCORR": (0.0, None),
            "PS": (300.0, 1100.0),
            "WS10M": (0.0, 100.0)
        }

        def linear_extrapolate_months(months: List[str], values: List[float], target_yyyymm: str) -> Optional[float]:
            try:
                idx = np.array([yyyymm_to_index(m) for m in months], dtype=float)
                y = np.array(values, dtype=float)
                if len(idx) == 0:
                    return None
                if np.allclose(y, y[0]):
                    return float(y[0])
                if len(idx) == 1:
                    return float(y[0])
                coef = np.polyfit(idx, y, 1)  # [slope, intercept]
                target_idx = float(yyyymm_to_index(target_yyyymm))
                pred = np.polyval(coef, target_idx)
                if np.isnan(pred) or np.isinf(pred):
                    return None
                return float(pred)
            except Exception:
                return None

        data_at_month = {}
        for v in KEEP_VARS:
            entry = {"value": None, "source_yyyymm": None, "method": None, "used_months": []}
            # 1) valor exacto en final_month (observado o sintetico ya injectado en series_main)
            sval = series_main.get(v, {}).get(final_month)
            if sval is not None:
                try:
                    vv = float(sval)
                    if not (math.isnan(vv) or math.isinf(vv)):
                        entry["value"] = clamp(vv, *VAR_BOUNDS.get(v, (None, None)))
                        entry["source_yyyymm"] = final_month
                        entry["method"] = "observed_or_synthetic"
                        entry["used_months"] = [final_month]
                        data_at_month[v] = entry
                        continue
                except Exception:
                    pass

            # 2) extrapolar desde los ultimos meses observados (hasta 6)
            months_avail = sorted(series_main.get(v, {}).keys())
            if months_avail:
                take_n = min(6, len(months_avail))
                used = months_avail[-take_n:]
                used_vals = []
                for m in used:
                    try:
                        vv = series_main.get(v, {}).get(m)
                        vv = float(vv)
                        if math.isnan(vv) or math.isinf(vv):
                            used_vals.append(None)
                        else:
                            used_vals.append(vv)
                    except Exception:
                        used_vals.append(None)
                valid_pairs = [(m, val) for m, val in zip(used, used_vals) if val is not None]
                if len(valid_pairs) >= 2:
                    ums, uvals = zip(*valid_pairs)
                    pred = linear_extrapolate_months(list(ums), list(uvals), final_month)
                    if pred is not None:
                        entry["value"] = clamp(pred, *VAR_BOUNDS.get(v, (None, None)))
                        entry["source_yyyymm"] = ums[-1]
                        entry["method"] = f"linear_extrapolation_from_last_{len(ums)}"
                        entry["used_months"] = list(ums)
                        data_at_month[v] = entry
                        continue
                if len(valid_pairs) == 1:
                    mlast, vlast = valid_pairs[0]
                    # buscar previa valida para delta
                    prev_val = None
                    prev_month = None
                    idx_last = months_avail.index(mlast)
                    for j in range(idx_last - 1, -1, -1):
                        try:
                            pv = series_main.get(v, {}).get(months_avail[j])
                            pvf = float(pv)
                            if not (math.isnan(pvf) or math.isinf(pvf)):
                                prev_val = pvf
                                prev_month = months_avail[j]
                                break
                        except Exception:
                            continue
                    if prev_val is not None:
                        delta = vlast - prev_val
                        months_forward = yyyymm_to_index(final_month) - yyyymm_to_index(mlast)
                        pred = vlast + delta * months_forward
                        entry["value"] = clamp(pred, *VAR_BOUNDS.get(v, (None, None)))
                        entry["source_yyyymm"] = mlast
                        entry["method"] = "delta_extrapolation"
                        entry["used_months"] = [prev_month, mlast]
                        data_at_month[v] = entry
                        continue
                    else:
                        entry["value"] = clamp(vlast, *VAR_BOUNDS.get(v, (None, None)))
                        entry["source_yyyymm"] = mlast
                        entry["method"] = "persistence_last_value"
                        entry["used_months"] = [mlast]
                        data_at_month[v] = entry
                        continue

            # 3) si no hay observaciones pero hay prediccion modelo (yhat), usarla
            if yhat and (v in yhat) and (yhat.get(v) == yhat.get(v)):
                try:
                    vv = float(yhat.get(v))
                    entry["value"] = clamp(vv, *VAR_BOUNDS.get(v, (None, None)))
                    entry["source_yyyymm"] = final_month
                    entry["method"] = "model_prediction"
                    entry["used_months"] = []
                    data_at_month[v] = entry
                    continue
                except Exception:
                    pass

            # 4) sin datos
            entry["value"] = None
            entry["source_yyyymm"] = None
            entry["method"] = "no_data"
            entry["used_months"] = []
            data_at_month[v] = entry

        # asignar al payload
        out_payload["requested_month"] = target_month
        out_payload["predicted_month"] = final_month
        out_payload["data"] = data_at_month
        # ---------- FIN DEL BLOQUE REEMPLAZADO ----------

    except Exception as e:
        out_payload["requested_month"] = target_month
        out_payload["error"] = f"{type(e).__name__}: {e}"
        print("[ERROR] forecast failed:", e)

    # Always save single JSON for Java (only KEEP_VARS in data)
    try:
        save_json(out_payload, PYTHON_JAVA_OUT)
        print("Saved JSON for Java:", str(PYTHON_JAVA_OUT))
    except Exception as e:
        print("ERROR saving python-java.json:", e)

    # Optional versioned copy (controlled by flag)
    if save_versioned:
        try:
            tag_month = out_payload.get("predicted_month") or out_payload.get("requested_month") or "unknown"
            out_json_versioned = PYTHON_JAVA_OUT_DIR / f"predictions_{tag_month}.json"
            save_json(out_payload, out_json_versioned)
            print("Saved versioned JSON:", str(out_json_versioned))
        except Exception as e:
            print("Warning saving versioned JSON:", e)

    # Optional CSV audit (controlled by flag)
    if save_csv:
        try:
            # save single-line csv with KEEP_VARS values (compact)
            data = out_payload.get("data") or {}
            rows = []
            for k, v in data.items():
                if isinstance(v, dict):
                    rows.append({"variable": k, "value": v.get("value"), "method": v.get("method"), "source_yyyymm": v.get("source_yyyymm")})
                else:
                    rows.append({"variable": k, "value": v})
            df = pd.DataFrame(rows)
            out_csv_path = PYTHON_JAVA_OUT_DIR / f"predictions_{out_payload.get('predicted_month','unknown')}.csv"
            df.to_csv(out_csv_path, index=False)
            print("CSV saved:", str(out_csv_path))
        except Exception as e:
            print("CSV save failed:", e)
