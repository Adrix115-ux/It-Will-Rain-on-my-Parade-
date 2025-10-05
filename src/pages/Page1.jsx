// src/pages/Page1.jsx
import React, { useEffect, useState } from 'react';
import axios from 'axios';
import DataCard from '../components/DataCard';
import AdvancedDataSection from '../components/AdvancedDataSection';

const Page1 = () => {
    const [historicalData, setHistoricalData] = useState({});
    const [futureData, setFutureData] = useState({});

    useEffect(() => {
        // Datos históricos
        axios.get('https://api.tuservicio.com/historical')
            .then(response => setHistoricalData(response.data))
            .catch(error => console.error(error));

        // Datos futuros
        axios.get('https://api.tuservicio.com/future')
            .then(response => setFutureData(response.data))
            .catch(error => console.error(error));
    }, []);

    return (
        <div className="page-container">
            <h1>Datos Meteorológicos</h1>

            <div className="data-cards">
                <DataCard
                    title="Temperatura Actual"
                    value={historicalData.t2m || 'Cargando...'}
                    unit="°C"
                    color="#3498db"
                />
                <DataCard
                    title="Precipitación Promedio"
                    value={historicalData.precipitationAvg || 'Cargando...'}
                    unit="mm"
                    color="#1abc9c"
                />
                <DataCard
                    title="Humedad Relativa"
                    value={historicalData.relativeHumidity || 'Cargando...'}
                    unit="%"
                    color="#f39c12"
                />
                <DataCard
                    title="Velocidad del Viento"
                    value={historicalData.ws2m || 'Cargando...'}
                    unit="km/h"
                    color="#e74c3c"
                />
            </div>

            <AdvancedDataSection
                data={[
                    { title: 'Temperatura Máxima', value: futureData.t2mMax, unit: '°C' },
                    { title: 'Temperatura Mínima', value: futureData.t2mMin, unit: '°C' },
                    { title: 'Punto de Rocío', value: futureData.dewPoint, unit: '°C' },
                    { title: 'Radiación Total', value: futureData.allskySfcSwDwn, unit: 'W/m²' },
                    { title: 'Presión Superficial', value: futureData.surfacePressure, unit: 'hPa' },
                ]}
            />
        </div>
    );
};

export default Page1;
