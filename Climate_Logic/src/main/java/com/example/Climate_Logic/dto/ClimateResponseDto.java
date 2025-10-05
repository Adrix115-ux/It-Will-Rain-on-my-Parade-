package com.example.Climate_Logic.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.*;

import java.time.LocalDate;

@AllArgsConstructor
@NoArgsConstructor
@Builder
@Getter
@Setter
public class ClimateResponseDto {
    // ==== Temperatura ====
    private Double T2M;       // Temperatura a 2 metros
    private Double T2M_MAX;    // Temperatura máxima a 2 metros
    private Double T2M_MIN;    // Temperatura mínima a 2 metros

    // ==== Humedad ====
    private Double T2MDEW;          // Punto de rocío o helada
    private Double T2MWET;  // Humedad relativa (%)

    // ==== Radiación ====
    private Double ALLSKY_SFC_SW_DWN;    // Radiación total (con nubes)

    // ==== Nubosidad ====
    private Double CLOUD_AMT;       // Fracción de nubosidad (0–1)

    // ==== Precipitación ====
    private Double PRECTOTCORR;  // Precipitación promedio

    // ==== Viento y presión ====
    private Double PS;   // Presión superficial (hPa)
    private Double WS10M;             // Velocidad del viento a 10 m
}
