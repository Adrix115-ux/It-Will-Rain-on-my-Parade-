// src/components/PredictionForm.jsx
import React, { useState, useRef } from "react";
import api from "../utils/api";

const formatDateToYYYYMMDD = (isoDate) => {
    if (!isoDate) return null;
    const parts = isoDate.split("-");
    if (parts.length < 3) return null;
    return `${parts[0]}${parts[1].padStart(2, "0")}${parts[2].padStart(2, "0")}`;
};

const validateCoords = (lat, lon) => {
    if (Number.isNaN(lat) || Number.isNaN(lon)) return "Latitud/Longitud deben ser números";
    if (lat < -90 || lat > 90) return "Latitud fuera de rango (-90..90)";
    if (lon < -180 || lon > 180) return "Longitud fuera de rango (-180..180)";
    return null;
};

export default function PredictionForm() {
    const [latitud, setLatitud] = useState("");
    const [longitud, setLongitud] = useState("");
    const [fecha, setFecha] = useState("");
    const [prediccion, setPrediccion] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const abortRef = useRef(null);

    // Exponential backoff retries (axios + AbortController compatible)
    async function postWithRetries(url, payload, { maxRetries = 2, baseDelay = 400 } = {}) {
        let attempt = 0;
        while (true) {
            attempt++;
            const controller = new AbortController();
            abortRef.current = controller;
            try {
                const resp = await api.post(url, payload, { signal: controller.signal });
                abortRef.current = null;
                return resp;
            } catch (err) {
                abortRef.current = null;
                // if aborted, rethrow immediately so caller can handle cancellation
                if (err?.name === "CanceledError" || err?.message === "canceled") throw err;
                if (attempt > maxRetries) throw err;
                const wait = baseDelay * Math.pow(2, attempt - 1);
                await new Promise((r) => setTimeout(r, wait));
            }
        }
    }

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");
        setPrediccion(null);

        const latNum = Number(latitud);
        const lonNum = Number(longitud);
        const v = validateCoords(latNum, lonNum);
        if (v) {
            setError(v);
            return;
        }
        const target_date = formatDateToYYYYMMDD(fecha);
        if (!target_date) {
            setError("Fecha inválida. Use el selector de fecha.");
            return;
        }

        const payload = {
            latitude: latNum,
            longitude: lonNum,
            target_date: target_date,
        };

        try {
            setLoading(true);
            // endpoint relativo (vite proxy -> /api -> backend)
            const resp = await postWithRetries("/api/ClimateLogic/prediccion", payload, { maxRetries: 2 });
            // Axios returns resp.data
            setPrediccion(resp.data ?? null);
        } catch (err) {
            if (err?.name === "CanceledError" || err?.message === "canceled") {
                setError("Petición cancelada por el usuario.");
            } else {
                console.error("Prediction error:", err);
                setError(
                    err?.response?.data?.message ||
                    err?.response?.statusText ||
                    "Error al obtener la predicción (revisa la consola)."
                );
            }
        } finally {
            setLoading(false);
        }
    };

    const cancelRequest = () => {
        try {
            if (abortRef.current) abortRef.current.abort();
        } catch (e) {}
    };

    return (
        <div className="prediction-form">
            <h2>Obtener Predicción de Clima</h2>
            <form onSubmit={handleSubmit} aria-busy={loading}>
                <div>
                    <label>Latitud:</label>
                    <input
                        type="number"
                        step="any"
                        value={latitud}
                        onChange={(e) => setLatitud(e.target.value)}
                        placeholder="-17.6"
                        required
                    />
                </div>
                <div>
                    <label>Longitud:</label>
                    <input
                        type="number"
                        step="any"
                        value={longitud}
                        onChange={(e) => setLongitud(e.target.value)}
                        placeholder="-62.92"
                        required
                    />
                </div>
                <div>
                    <label>Fecha:</label>
                    <input type="date" value={fecha} onChange={(e) => setFecha(e.target.value)} required />
                </div>

                <div style={{ marginTop: 12 }}>
                    <button type="submit" disabled={loading}>
                        {loading ? "Cargando..." : "Obtener Predicción"}
                    </button>
                    {loading && (
                        <button type="button" onClick={cancelRequest} style={{ marginLeft: 8 }}>
                            Cancelar
                        </button>
                    )}
                </div>
            </form>

            {error && <p style={{ color: "red" }}>{error}</p>}

            {prediccion && (
                <div className="prediction-result" style={{ marginTop: 12 }}>
                    <h3>Predicción Recibida:</h3>
                    <pre style={{ maxHeight: 400, overflow: "auto" }}>{JSON.stringify(prediccion, null, 2)}</pre>
                </div>
            )}
        </div>
    );
}
