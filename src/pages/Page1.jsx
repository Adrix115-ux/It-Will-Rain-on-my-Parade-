// ASCII only
// src/pages/Page1.jsx
import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import api from "../utils/api";
import DataCard from "../components/DataCard";

// helper: date -> YYYYMMDD
const dateToYYYYMMDD = (d) => {
    const yyyy = d.getFullYear();
    const mm = String(d.getMonth() + 1).padStart(2, "0");
    const dd = String(d.getDate()).padStart(2, "0");
    return `${yyyy}${mm}${dd}`;
};

// sanitize helpers
const MISSING_SENTINELS = new Set([-999, -8888, -99, -9]);
const isMissing = (v) =>
    v === null ||
    v === undefined ||
    Number.isNaN(v) ||
    (typeof v === "number" && MISSING_SENTINELS.has(Math.trunc(v)));

const toFixedOrDash = (v, digits = 2) =>
    typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "—";

const clamp = (v, min, max) =>
    typeof v === "number" ? Math.min(Math.max(v, min), max) : v;

const avg = (arr) => (arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null);

const cleanList = (rows, key, opts = {}) => {
    // opts: { nonNegative: bool, clamp01: [min,max], convert: fn }
    let list = rows
        .map((r) => (typeof r?.[key] === "number" ? r[key] : null))
        .filter((v) => !isMissing(v));
    if (opts.nonNegative) list = list.map((v) => (v < 0 ? 0 : v));
    if (opts.clamp01) {
        const [mn, mx] = opts.clamp01;
        list = list.map((v) => clamp(v, mn, mx));
    }
    if (opts.convert) list = list.map((v) => opts.convert(v));
    return list;
};

// normalize numbers from API (string -> number, invalid -> null)
const normalizeValue = (v) => {
    if (v === null || v === undefined) return null;
    if (typeof v === "number") return Number.isFinite(v) ? v : null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
};

export default function Page1() {
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [rows, setRows] = useState([]); // 7 rows (daily)
    const [openTable, setOpenTable] = useState(false);

    // resumen para tarjetas
    const summary = useMemo(() => {
        const lastValidT2M = [...rows].reverse().map((r) => r?.T2M).find((v) => !isMissing(v));
        const t2mCurrent = toFixedOrDash(lastValidT2M);

        const rhAvg = avg(cleanList(rows, "RH2M", { clamp01: [0, 100] }));
        const relativeHumidityAvg = toFixedOrDash(rhAvg);

        const prcpSum = cleanList(rows, "PRECTOTCORR", { nonNegative: true }).reduce((a, b) => a + b, 0);
        const precipitation7d = toFixedOrDash(prcpSum);

        const windAvgKmH = avg(cleanList(rows, "WS10M", { nonNegative: true, convert: (v) => v * 3.6 }));
        const windKmHAvg = toFixedOrDash(windAvgKmH);

        return {
            t2mCurrent,
            relativeHumidityAvg,
            precipitation7d,
            windKmHAvg,
            // extras por si sigues usando mas tarjetas
            t2mMaxAvg: toFixedOrDash(avg(cleanList(rows, "T2M_MAX"))),
            t2mMinAvg: toFixedOrDash(avg(cleanList(rows, "T2M_MIN"))),
            dewAvg: toFixedOrDash(avg(cleanList(rows, "T2MDEW"))),
            pressureAvg: toFixedOrDash(avg(cleanList(rows, "PS"))),
            swDownAvg: toFixedOrDash(avg(cleanList(rows, "ALLSKY_SFC_SW_DWN"))),
        };
    }, [rows]);

    useEffect(() => {
        let cancelled = false;

        async function loadWeek() {
            setLoading(true);
            setError("");
            try {
                // default coords Santa Cruz de la Sierra
                const lat = -17.7833;
                const lon = -63.1833;

                const today = new Date();
                const start = new Date(today);
                start.setDate(today.getDate() - 6);
                const startS = dateToYYYYMMDD(start);

                // pedir set diario estable (coincide con backend DEFAULT_VARS)
                const params = [
                    // Radiacion
                    "ALLSKY_SFC_SW_DWN","CLRSKY_SFC_SW_DWN","ALLSKY_SFC_SW_DIFF",
                    "TOA_SW_DWN","ALLSKY_SRF_ALB","ALLSKY_SFC_PAR_TOT","CLRSKY_SFC_PAR_TOT",
                    // Temperatura
                    "T2M","T2M_MAX","T2M_MIN","T2MDEW","T2MWET","TS",
                    // Humedad / precip
                    "RH2M","QV2M","PRECTOTCORR",
                    // Viento / presion
                    "WS10M","WD10M","PS"
                ].join(",");

                // ajusta a "/ClimateLogic/weekly" si tu api ya antepone /api
                const { data: payload } = await api.get("/api/ClimateLogic/weekly", {
                    params: {
                        latitude: lat,
                        longitude: lon,
                        start: startS,
                        days: 7,
                        parametersCsv: params,
                    },
                });

                if (cancelled) return;

                const raw = Array.isArray(payload?.data) ? payload.data : [];
                // normaliza: numbers consistentes y date estandarizada
                const r = raw.map((row) => {
                    const out = { ...row };
                    Object.keys(out).forEach((k) => {
                        if (k === "date" || k === "DATE") return;
                        out[k] = normalizeValue(out[k]);
                    });
                    out.date = row.date || row.DATE || null;
                    return out;
                });
                setRows(r);
            } catch (e) {
                console.error("weekly request failed:", {
                    message: e?.message,
                    code: e?.code,
                    url: (e?.config?.baseURL ?? "") + (e?.config?.url ?? ""),
                    status: e?.response?.status,
                    data: e?.response?.data,
                });
                setError("No se pudo obtener la semana. Revisa backend o red.");
            } finally {
                if (!cancelled) setLoading(false);
            }
        }

        loadWeek();
        return () => {
            cancelled = true;
        };
    }, []);

    return (
        <div className="page-container">
            <h1>Datos meteorologicos (ultima semana)</h1>

            {loading && <p>Cargando...</p>}
            {error && <p style={{ color: "red" }}>{error}</p>}

            {/* cuatro tarjetas clave */}
            <div
                className="data-cards"
                style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(220px,1fr))", gap: 12 }}
            >
                <DataCard title="Temp actual" value={summary.t2mCurrent} unit="C" color="#3498db" />
                <DataCard title="Humedad prom" value={summary.relativeHumidityAvg} unit="%" color="#f39c12" />
                <DataCard title="Lluvia 7d" value={summary.precipitation7d} unit="mm" color="#1abc9c" />
                <DataCard title="Viento prom" value={summary.windKmHAvg} unit="km/h" color="#e74c3c" />
            </div>

            {/* boton para desplegar tabla completa */}
            <div style={{ marginTop: 14 }}>
                <button className="btn" onClick={() => setOpenTable((s) => !s)}>
                    {openTable ? "Ocultar tabla semanal" : "Ver tabla semanal completa"}
                </button>
                <Link to="/page3" className="btn" style={{ marginLeft: 8 }}>
                    Ver datos avanzados
                </Link>
            </div>

            {/* tabla de todos los datos */}
            {openTable && (
                <div className="table-card" style={{ marginTop: 12 }}>
                    <div className="table-scroll">
                        <table className="result-table" aria-label="Tabla semanal">
                            <thead>
                            <tr>
                                <th>Fecha</th>
                                <th>T2M</th>
                                <th>T2M_MAX</th>
                                <th>T2M_MIN</th>
                                <th>Dew</th>
                                <th>RH2M</th>
                                <th>PRECTOTCORR</th>
                                <th>WS10M</th>
                                <th>PS</th>
                                <th>SW down</th>
                            </tr>
                            </thead>
                            <tbody>
                            {rows.map((r, i) => (
                                <tr key={i}>
                                    <td>{r.date || "-"}</td>
                                    <td>{toFixedOrDash(r.T2M)}</td>
                                    <td>{toFixedOrDash(r.T2M_MAX)}</td>
                                    <td>{toFixedOrDash(r.T2M_MIN)}</td>
                                    <td>{toFixedOrDash(r.T2MDEW)}</td>
                                    <td>{toFixedOrDash(clamp(r.RH2M, 0, 100))}</td>
                                    <td>{toFixedOrDash(Math.max(0, r.PRECTOTCORR ?? 0))}</td>
                                    <td>{toFixedOrDash(r.WS10M)}</td>
                                    <td>{toFixedOrDash(r.PS)}</td>
                                    <td>{toFixedOrDash(r.ALLSKY_SFC_SW_DWN)}</td>
                                </tr>
                            ))}
                            {rows.length === 0 && (
                                <tr>
                                    <td colSpan={10} style={{ textAlign: "center", opacity: 0.7 }}>
                                        Sin datos
                                    </td>
                                </tr>
                            )}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
}
