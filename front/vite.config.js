// vite.config.js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // puerto del dev server (WebStorm lo arrancará aquí)
    host: true, // opcional si quieres acceder desde LAN (no obligatorio)
    proxy: {
      // redirige /api/* a tu backend en localhost:8080 (evita CORS)
      '^/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        secure: false,
        rewrite: (path) => path // leave /api prefix
      }
    }
  }
});
