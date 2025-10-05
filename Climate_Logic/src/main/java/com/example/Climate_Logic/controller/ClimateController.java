// src/main/java/com/example/Climate_Logic/controller/ClimateController.java
package com.example.Climate_Logic.controller;

import com.example.Climate_Logic.dto.ClimateResponseDto;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.*;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

@RestController
@RequestMapping("/api/ClimateLogic")
public class ClimateController {

    private static final Logger log = LoggerFactory.getLogger(ClimateController.class);

    @PostMapping("/prediccion")

    public ResponseEntity<ClimateResponseDto> predict(@RequestBody Map<String, Object> dtoMap) {
        try {
            // === 1. Rutas (ajusta si es necesario) ===
            String baseDir = "C:\\Users\\ENVY\\Documents\\It-Will-Rain-on-my-Parade";
            String comunicacionDir = baseDir + "\\Comunicacion";
            String jsonEntrada = comunicacionDir + "\\java-python.json";
            String jsonSalida = comunicacionDir + "\\python-java.json";
            String batPath = baseDir + "\\Climate_Logic\\src\\main\\java\\com\\example\\Climate_Logic\\EjecutarPython.bat";

            ObjectMapper mapper = new ObjectMapper();

            // === 2. Extraer valores del request map (no dependemos de getters) ===
            // Esperamos keys: "latitude", "longitude", "target_date" (YYYYMMDD)
            Object latObj = dtoMap.getOrDefault("latitude", dtoMap.get("latitud"));
            Object lonObj = dtoMap.getOrDefault("longitude", dtoMap.get("longitud"));
            Object dateObj = dtoMap.getOrDefault("target_date", dtoMap.get("fecha"));

            if (latObj == null || lonObj == null || dateObj == null) {
                log.error("Faltan parametros required en el request: latitude/longitude/target_date");
                return ResponseEntity.badRequest().build();
            }

            // garantizar tipos primitivos compatibles
            Double latitude = null;
            Double longitude = null;
            String targetDate = null;
            try {
                latitude = (latObj instanceof Number) ? ((Number) latObj).doubleValue() : Double.parseDouble(latObj.toString());
                longitude = (lonObj instanceof Number) ? ((Number) lonObj).doubleValue() : Double.parseDouble(lonObj.toString());
                targetDate = dateObj.toString().trim();
            } catch (Exception ex) {
                log.error("Error parseando parametros de entrada: {}", ex.getMessage());
                return ResponseEntity.badRequest().build();
            }

            // === 3. Escribir java-python.json con las claves que espera Python ===
            Map<String, Object> inputMap = Map.of(
                    "latitude", latitude,
                    "longitude", longitude,
                    "target_date", targetDate
            );
            mapper.writerWithDefaultPrettyPrinter().writeValue(new File(jsonEntrada), inputMap);
            log.info("Archivo java-python.json escrito: {}", jsonEntrada);

            // === 4. Ejecutar BAT que llama a Python ===
            ProcessBuilder pb = new ProcessBuilder("cmd.exe", "/c", batPath);
            pb.directory(new File(comunicacionDir));
            pb.redirectErrorStream(true);
            Process process = pb.start();

            // leer salida del proceso (stdout + stderr)
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                log.info("--- Salida del BAT (start) ---");
                while ((line = reader.readLine()) != null) {
                    log.info(line);
                }
                log.info("--- Salida del BAT (end) ---");
            }

            int exitCode = process.waitFor();
            log.info("Codigo de salida del BAT: {}", exitCode);
            if (exitCode != 0) {
                log.warn("BAT devolvio codigo distinto de 0. Revisar salida. ExitCode={}", exitCode);
            }

            // === 5. Leer python-java.json de salida ===
            Path salidaPath = Paths.get(jsonSalida);
            if (!Files.exists(salidaPath)) {
                log.error("No se encontro python-java.json en: {}", jsonSalida);
                return ResponseEntity.internalServerError().build();
            }

            @SuppressWarnings("unchecked")
            Map<String, Object> resultData = mapper.readValue(new File(jsonSalida), Map.class);
            log.info("Archivo python-java.json leido correctamente.");

            Object dataObj = resultData.get("data");

            // === 6. Construir ClimateResponseDto y poblar campos por reflection ===
            ClimateResponseDto response = new ClimateResponseDto();

            // lista de campos que esperamos en el DTO (coincide con KEEP_VARS)
            String[] keys = new String[] {
                    "T2M", "T2M_MAX", "T2M_MIN", "T2MDEW", "T2MWET",
                    "ALLSKY_SFC_SW_DWN", "CLOUD_AMT", "PRECTOTCORR", "PS", "WS10M"
            };

            for (String key : keys) {
                Double val = extractDoubleFromData(dataObj, key);
                if (val == null) {
                    // si no hay valor, dejamos null
                    continue;
                }
                // usar reflection para setear el campo directamente (evita necesidad de setters)
                try {
                    Field f = ClimateResponseDto.class.getDeclaredField(key);
                    f.setAccessible(true);
                    f.set(response, val);
                } catch (NoSuchFieldException nsf) {
                    // campo no existe en DTO (posible mismatch de nombres) -> log y continuar
                    log.warn("Campo {} no encontrado en ClimateResponseDto (skip).", key);
                } catch (Exception ex) {
                    log.error("Error seteando campo {} en DTO: {}", key, ex.getMessage());
                }
            }

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            log.error("Error en predict(): {}", e.getMessage(), e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Extrae un Double desde el objeto 'data' leido del JSON de python.
     * Soporta dos formas:
     *  - data = { "T2M": {"value": 23.5, ...}, ... }
     *  - data = { "T2M": 23.5, ... }
     */
    private Double extractDoubleFromData(Object dataObj, String key) {
        try {
            if (dataObj == null) return null;
            if (!(dataObj instanceof Map)) return null;
            @SuppressWarnings("unchecked")
            Map<String, Object> dataMap = (Map<String, Object>) dataObj;
            Object node = dataMap.get(key);
            if (node == null) return null;
            if (node instanceof Map) {
                Object val = ((Map) node).get("value");
                if (val == null) return null;
                if (val instanceof Number) return ((Number) val).doubleValue();
                String s = val.toString().trim();
                if (s.isEmpty() || s.equalsIgnoreCase("null")) return null;
                return Double.parseDouble(s);
            } else if (node instanceof Number) {
                return ((Number) node).doubleValue();
            } else {
                String s = node.toString().trim();
                if (s.isEmpty() || s.equalsIgnoreCase("null")) return null;
                return Double.parseDouble(s);
            }
        } catch (Exception ex) {
            log.warn("No se pudo extraer double para {} : {}", key, ex.getMessage());
            return null;
        }
    }
}
