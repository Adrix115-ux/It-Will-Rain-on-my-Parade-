// src/components/DataTable.jsx
import React from 'react';

const DataTable = ({ title, data }) => {
    return (
        <div className="data-table">
            <h2>{title}</h2>
            <table>
                <thead>
                <tr>
                    <th>Fecha</th>
                    <th>Probabilidad de Lluvia (%)</th>
                    <th>Temperatura (°C)</th>
                </tr>
                </thead>
                <tbody>
                {data && data.length > 0 ? (
                    data.map((item, index) => (
                        <tr key={index}>
                            <td>{item.date}</td>
                            <td>{item.rainProbability}</td>
                            <td>{item.temperature}</td>
                        </tr>
                    ))
                ) : (
                    <tr>
                        <td colSpan="3">No hay datos disponibles</td>
                    </tr>
                )}
                </tbody>
            </table>
        </div>
    );
};

export default DataTable;
