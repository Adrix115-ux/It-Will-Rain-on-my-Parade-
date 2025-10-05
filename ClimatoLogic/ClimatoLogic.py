# ClimatoPredict_LightGBM.py
# ASCII only

import os
import re
import math
import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import requests

# --------------------------- Paths and constants ---------------------------

BASE_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"
COMMUNITY = "AG"

OUT_BASE = r"C:\Users\ENVY\Documents\It-Will-Rain-on-my-Parade\nasa_power_output"

TRAIN_BASE = r"C:\Users\ENVY\Documents\It-Will-Rain-on-my-Parade\DatosEntrenaMiento"
MODELS_DIR = os.path.join(TRAIN_BASE, "models")
CKPT_DIR = os.path.join(TRAIN_BASE, "checkpoints")
REGISTRY_PATH = os.path.join(MODELS_DIR, "models_registry.json")

IO_BASE = r"C:\Users\ENVY\Documents\It-Will-Rain-on-my-Parade\Comunicacion"
JAVA_PYTHON_IN = os.path.join(IO_BASE, "java-python.json")
PYTHON_JAVA_OUT = os.path.join(IO_BASE, "python-java.json")
PYTHON_JAVA_OUT_DIR = IO_BASE
os.makedirs(IO_BASE, exist_ok=True)
print("IO_BASE set to:", os.path.abspath(IO_BASE))
print("Will write to:", os.path.abspath(PYTHON_JAVA_OUT))

START_YEAR = 1990
END_YEAR = 2025
MAX_PARAMS_PER_REQUEST = 20
PAUSE_BETWEEN_REQUESTS = 0.80
MAX_RETRIES = 3
EARTH_R_KM = 6371.0

# Must match training
CORE_VARS = ["T2M", "PRECTOTCORR", "RH2M", "WS2M"]
PARAM_ALIASES = {"PRECTOT": "PRECTOTCORR"}
USE_NEIGH_DIFFS_DEFAULT = True

# --------------------------- Utils ---------------------------

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

def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
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

# --------------------------- POWER IO ---------------------------

def normalize_power_parameters(raw: dict) -> Dict[str, Dict[str, float]]:
    block = (raw.get("properties", {}) or {}).get("parameter", {}) or raw.get("parameters", {}) or {}
    clean: Dict[str, Dict[str, float]] = {}
    for var, series in (block or {}).items():
        real_var = PARAM_ALIASES.get(var, var)
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

def download_point_to_json(lat: float, lon: float, params: List[str], out_file: str) -> None:
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

# --------------------------- Feature reconstruction (as training) ----------

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
    # no dropna here; upstream alignment ensures feasibility
    return df

# --------------------------- Registry and prediction -----------------------

def load_models_registry(registry_path: str) -> dict:
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_lat_lon_from_folder(folder_path: str) -> Tuple[float, float]:
    name = os.path.basename(folder_path)
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
    out_base: str
) -> Tuple[str, List[str]]:
    folder_name = f"lat{safe_coord(main_lat)}_lon{safe_coord(main_lon)}_{START_YEAR}-{END_YEAR}"
    session_folder = os.path.join(out_base, folder_name)
    os.makedirs(session_folder, exist_ok=True)

    points = [(main_lat, main_lon)]
    for b in bearings:
        points.append(dest_point(main_lat, main_lon, b, radius_km))

    json_paths = []
    for i, (la, lo) in enumerate(points, start=1):
        tag = "main_point" if i == 1 else f"secondary_{i-1}"
        out_file = os.path.join(session_folder, f"{tag}.json")
        if not os.path.exists(out_file):
            print(f"Downloading {tag} at {la},{lo} ...")
            download_point_to_json(la, lo, parameters, out_file)
        else:
            print(f"Using cached {tag}: {out_file}")
        json_paths.append(out_file)

    return session_folder, json_paths

def next_valid_feature_month(series_main: Dict[str, Dict[str, float]],
                             neigh_list: List[Dict[str, Dict[str, float]]]) -> str:
    months = aligned_months_core(series_main, neigh_list, CORE_VARS)
    if not months:
        raise RuntimeError("No common months across core vars and neighbors")
    return months[-1]

def roll_forecast_to_month(
    target_month: str,
    series_main: Dict[str, Dict[str, float]],
    neigh_list: List[Dict[str, Dict[str, float]]],
    models_dir: str,
    registry: dict,
    lat: float,
    lon: float
) -> Tuple[str, Dict[str, float]]:
    feature_cols: List[str] = registry.get("features", [])
    lags: List[int] = registry.get("lags", [1, 2, 3])
    use_neigh_diffs: bool = registry.get("use_neigh_diffs", USE_NEIGH_DIFFS_DEFAULT)
    horizon: int = int(registry.get("predict_horizon_months", 1))
    targets_dict = registry.get("targets", {})

    params_in_features = sorted(set([
        c.split("__")[0] for c in feature_cols if ("__main" in c or "__diff_n" in c)
    ]))

    current_feature_month = next_valid_feature_month(series_main, neigh_list)

    steps = max(0, (month_diff(target_month, current_feature_month) + horizon - 1) // horizon)
    print("[DEBUG] current_feature_month (start):", current_feature_month)
    print("[DEBUG] steps to do:", steps, "horizon:", horizon, "-> target:", target_month)

    last_pred_month = None
    last_yhat: Dict[str, float] = {}

    for step in range(steps):
        predict_month = month_add(current_feature_month, horizon)
        print(f"[DEBUG] step {step+1}/{steps} -> predict {predict_month}")

        X = build_feature_row_for_month(
            current_feature_month,
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
        for tgt, info in (targets_dict or {}).items():
            model_path = info.get("model_path") or os.path.join(models_dir, f"lgb_{tgt}.txt")
            if not os.path.exists(model_path):
                continue
            booster = lgb.Booster(model_file=model_path)
            try:
                yval = float(booster.predict(X)[0])
            except Exception:
                yval = float("nan")
            yhat_step[tgt] = yval

        for tgt, val in yhat_step.items():
            series_main.setdefault(tgt, {})[predict_month] = val

        last_pred_month = predict_month
        last_yhat = yhat_step
        current_feature_month = month_add(current_feature_month, horizon)

    if steps == 0:
        predict_month = month_add(current_feature_month, horizon)
        while predict_month < target_month:
            print(f"[DEBUG] preroll to {predict_month} (advance to reach target {target_month})")
            X = build_feature_row_for_month(
                current_feature_month,
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
            for tgt, info in (targets_dict or {}).items():
                model_path = info.get("model_path") or os.path.join(models_dir, f"lgb_{tgt}.txt")
                if not os.path.exists(model_path):
                    continue
                booster = lgb.Booster(model_file=model_path)
                try:
                    yval = float(booster.predict(X)[0])
                except Exception:
                    yval = float("nan")
                yhat_step[tgt] = yval
            for tgt, val in yhat_step.items():
                series_main.setdefault(tgt, {})[predict_month] = val
            last_pred_month = predict_month
            last_yhat = yhat_step
            current_feature_month = month_add(current_feature_month, horizon)
            predict_month = month_add(current_feature_month, horizon)

        if predict_month == target_month:
            print(f"[DEBUG] final predict to match target {target_month}")
            X = build_feature_row_for_month(
                current_feature_month,
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
            for tgt, info in (targets_dict or {}).items():
                model_path = info.get("model_path") or os.path.join(models_dir, f"lgb_{tgt}.txt")
                if not os.path.exists(model_path):
                    continue
                booster = lgb.Booster(model_file=model_path)
                try:
                    yval = float(booster.predict(X)[0])
                except Exception:
                    yval = float("nan")
                yhat_step[tgt] = yval
            for tgt, val in yhat_step.items():
                series_main.setdefault(tgt, {})[predict_month] = val
            last_pred_month = predict_month
            last_yhat = yhat_step

    return last_pred_month or current_feature_month, last_yhat

# --------------------------- Main ---------------------------

if __name__ == "__main__":
    main_lat = -17.6000
    main_lon = -62.9200
    radius_km = 50.0
    bearings = [0.0, 120.0, 240.0]

    if not os.path.exists(REGISTRY_PATH):
        raise FileNotFoundError(f"models_registry.json not found at: {REGISTRY_PATH}")
    registry = load_models_registry(REGISTRY_PATH)

    feature_cols: List[str] = registry.get("features", [])
    params_needed = sorted(set([c.split("__")[0] for c in feature_cols if ("__main" in c or "__diff_n" in c)]))
    params_needed = dedupe([PARAM_ALIASES.get(p, p) for p in params_needed])
    # Ensure core vars are present for alignment
    for core in CORE_VARS:
        if core not in params_needed:
            params_needed.append(core)
    print("[DEBUG] params_needed:", params_needed)

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

    series_main = normalize_power_parameters(load_json(main_json))
    neigh_list = [normalize_power_parameters(load_json(p)) for p in sec_jsons]

    lat_infer, lon_infer = infer_lat_lon_from_folder(os.path.dirname(main_json))
    if lat_infer == lat_infer and lon_infer == lon_infer:
        lat_use, lon_use = lat_infer, lon_infer
    else:
        lat_use, lon_use = main_lat, main_lon

    # Read latitude, longitude, target_date (YYYYMMDD) from java-python.json
    try:
        os.makedirs(PYTHON_JAVA_OUT_DIR, exist_ok=True)
        if not os.path.exists(JAVA_PYTHON_IN):
            raise FileNotFoundError(JAVA_PYTHON_IN + " not found")
        jpin = load_json(JAVA_PYTHON_IN)
        lat_use = float(jpin["latitude"])
        lon_use = float(jpin["longitude"])
        target_month = yyyymm_from_yyyymmdd(jpin["target_date"])
        print("[DEBUG] Read from java-python.json:", lat_use, lon_use, target_month)
    except Exception as e:
        print("ERROR reading java-python.json, falling back to folder coords + next month:", e)
        lat_use, lon_use = infer_lat_lon_from_folder(os.path.dirname(main_json))
        if not (lat_use == lat_use and lon_use == lon_use):
            lat_use, lon_use = main_lat, main_lon
        target_month = None

    # Compute base and enforce reachability
    out_payload = {
        "requested_month": None,
        "predicted_month": None,
        "horizon_months": int(registry.get("predict_horizon_months", 1)),
        "lat": lat_use,
        "lon": lon_use,
        "predictions": None
    }

    try:
        last_common = next_valid_feature_month(series_main, neigh_list)
        horizon = int(registry.get("predict_horizon_months", 1))
        if not target_month:
            target_month = month_add(last_common, horizon)
        if month_diff(target_month, last_common) < horizon:
            target_month = month_add(last_common, horizon)
        print("[DEBUG] last_common:", last_common, "target_month:", target_month, "horizon:", horizon)

        final_month, yhat = roll_forecast_to_month(
            target_month=target_month,
            series_main=series_main,
            neigh_list=neigh_list,
            models_dir=MODELS_DIR,
            registry=registry,
            lat=lat_use,
            lon=lon_use
        )
        print("[DEBUG] final_month:", final_month, "pred_count:", len(yhat or {}))

        out_payload["requested_month"] = target_month
        out_payload["predicted_month"] = final_month
        out_payload["horizon_months"] = horizon
        out_payload["predictions"] = yhat

    except Exception as e:
        out_payload["requested_month"] = target_month
        out_payload["error"] = f"{type(e).__name__}: {e}"
        print("[ERROR] forecast failed:", e)

    # Always save JSON for Java
    try:
        save_json(out_payload, PYTHON_JAVA_OUT)
        print("Saved JSON for Java:", PYTHON_JAVA_OUT)
    except Exception as e:
        print("ERROR saving python-java.json:", e)

    # Versioned copy
    try:
        tag_month = out_payload.get("predicted_month") or out_payload.get("requested_month") or "unknown"
        out_json_versioned = os.path.join(PYTHON_JAVA_OUT_DIR, f"predictions_{tag_month}.json")
        save_json(out_payload, out_json_versioned)
        print("Saved versioned JSON:", out_json_versioned)
    except Exception as e:
        print("Warning saving versioned JSON:", e)

    # Optional CSV for audit
    try:
        if isinstance(out_payload.get("predictions"), dict):
            rows = [{"target": k, "yhat": v} for k, v in (out_payload["predictions"] or {}).items()]
            df = pd.DataFrame(rows).sort_values("target")
            out_csv_path = os.path.join(PYTHON_JAVA_OUT_DIR, f"predictions_{out_payload.get('predicted_month','unknown')}.csv")
            df.to_csv(out_csv_path, index=False)
            print("CSV saved:", out_csv_path)
    except Exception as e:
        print("CSV save failed:", e)
