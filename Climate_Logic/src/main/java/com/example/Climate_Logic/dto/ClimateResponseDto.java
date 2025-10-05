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
    private Double t2m;       // Temperatura a 2 metros
    private Double t2mMax;    // Temperatura máxima a 2 metros
    private Double t2mMin;    // Temperatura mínima a 2 metros

    // ==== Humedad ====
    private Double dewPoint;          // Punto de rocío o helada
    private Double relativeHumidity;  // Humedad relativa (%)

    // ==== Radiación ====
    private Double allskySfcSwDwn;    // Radiación total (con nubes)

    // ==== Nubosidad ====
    private Double cloudAmount;       // Fracción de nubosidad (0–1)

    // ==== Precipitación ====
    private Double precipitationAvg;  // Precipitación promedio

    // ==== Viento y presión ====
    private Double surfacePressure;   // Presión superficial (hPa)
    private Double ws10m;             // Velocidad del viento a 10 m
}
