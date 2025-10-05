// src/components/PredictionForm.jsx
import React, { useState } from 'react';
import axios from 'axios';

const PredictionForm = () => {
    const [latitud, setLatitud] = useState('');
    const [longitud, setLongitud] = useState('');
    const [fecha, setFecha] = useState('');
    const [prediccion, setPrediccion] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setPrediccion(null);

        try {
            const response = await axios.post('http://localhost:8080/api/ClimateLogic/prediccion', {
                latitud: Number(latitud),
                longitud: Number(longitud),
                fecha: fecha
            });

            setPrediccion(response.data);  // Guardamos la predicción recibida
        } catch (err) {
            setError('Error al obtener la predicción.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="prediction-form">
            <h2>Obtener Predicción de Clima</h2>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>Latitud:</label>
                    <input
                        type="number"
                        value={latitud}
                        onChange={(e) => setLatitud(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label>Longitud:</label>
                    <input
                        type="number"
                        value={longitud}
                        onChange={(e) => setLongitud(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label>Fecha:</label>
                    <input
                        type="date"
                        value={fecha}
                        onChange={(e) => setFecha(e.target.value)}
                        required
                    />
                </div>
                <button type="submit" disabled={loading}>
                    {loading ? 'Cargando...' : 'Obtener Predicción'}
                </button>
            </form>

            {error && <p style={{ color: 'red' }}>{error}</p>}

            {prediccion && (
                <div className="prediction-result">
                    <h3>Predicción Recibida:</h3>
                    <pre>{JSON.stringify(prediccion, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default PredictionForm;
