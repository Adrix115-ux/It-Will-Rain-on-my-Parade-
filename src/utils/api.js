// src/utils/api.js
import axios from "axios";

const BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");

// axios instancia
const api = axios.create({
    baseURL: "", // dejamos vacío porque usamos rutas absolutas relativas con vite proxy (/api/...)
    timeout: 30_000,
    headers: {
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    },
});

// opcional: agregar token si existe (ejemplo de seguridad)
api.interceptors.request.use((config) => {
    try {
        const token = localStorage.getItem("auth_token");
        if (token) {
            config.headers = config.headers || {};
            config.headers.Authorization = `Bearer ${token}`;
        }
    } catch (e) {
        // noop
    }
    // si quieres que el cliente use BASE prefijo, podrías:
    // config.url = (config.url || "").startsWith("/api") ? config.url : `${BASE}${config.url}`;
    return config;
});

export default api;
