// src/components/DataCard.jsx
import React from 'react';

const DataCard = ({ title, value, unit, color }) => {
    return (
        <div className="data-card" style={{ borderLeft: `5px solid ${color}` }}>
            <h3>{title}</h3>
            <p>{value} {unit}</p>
        </div>
    );
};

export default DataCard;
