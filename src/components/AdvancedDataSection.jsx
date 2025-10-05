// src/components/AdvancedDataSection.jsx
import React, { useState } from 'react';

const AdvancedDataSection = ({ data }) => {
    const [isOpen, setIsOpen] = useState(false);

    const toggleSection = () => setIsOpen(!isOpen);

    return (
        <div className="advanced-section">
            <button onClick={toggleSection}>
                {isOpen ? "Ocultar Datos Avanzados" : "Ver Datos Avanzados"}
            </button>
            {isOpen && (
                <div className="advanced-data">
                    {data.map((item, index) => (
                        <div key={index} className="data-item">
                            <h4>{item.title}</h4>
                            <p>{item.value} {item.unit}</p>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export default AdvancedDataSection;
