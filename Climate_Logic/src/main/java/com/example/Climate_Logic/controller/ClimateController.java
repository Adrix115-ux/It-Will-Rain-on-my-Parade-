package com.example.Climate_Logic.controller;


import com.example.Climate_Logic.dto.ClimateRequestDto;
import com.example.Climate_Logic.dto.ClimateResponseDto;
import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.List;

@RestController
@RequestMapping("/api/ClimateLogic")
@AllArgsConstructor
public class ClimateController {

    @PostMapping("/prediccion")
    public ResponseEntity<ClimateResponseDto> predict(@RequestBody ClimateRequestDto dto) {
        try {
            //cosa
            // Parámetros que quieres pasar al Python

            // Ruta completa del archivo BAT
            String batPath = "C:\\Users\\Alejandra\\Documents\\Personal\\Climate_Logic\\Climate_Logic\\src\\main\\java\\com\\example\\Climate_Logic\\EjecutarPython.bat";

            // Crear el proceso
            ProcessBuilder pb = new ProcessBuilder("cmd.exe", "/c", batPath, String.valueOf(dto.getLongitud()), String.valueOf(dto.getLatitud()), String.valueOf(dto.getFecha()));

            pb.redirectErrorStream(true); // combinar salida y errores
            Process process = pb.start();

            // Leer salida del script

            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream())
            );

            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }

            int exitCode = process.waitFor();
            System.out.println("Código de salida: " + exitCode);

            return ResponseEntity.ok(new ClimateResponseDto());
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

}
