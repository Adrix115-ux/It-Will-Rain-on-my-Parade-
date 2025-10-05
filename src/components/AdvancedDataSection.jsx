// src/components/AdvancedDataSection.jsx (mejorada)
import React, { useState } from "react";

const AdvancedDataSection = ({ data = [] }) => {
    const [isOpen, setIsOpen] = useState(false);
    if (!Array.isArray(data)) data = [];

    return (
        <div className="advanced-section">
            <button onClick={() => setIsOpen((s) => !s)}>
                {isOpen ? "Ocultar Datos Avanzados" : "Ver Datos Avanzados"}
            </button>
            {isOpen && (
                <div className="advanced-data">
                    {data.length === 0 ? (
                        <div>No hay datos avanzados</div>
                    ) : (
                        data.map((item, idx) => (
                            <div className="data-item" key={idx}>
                                <h4>{item.title}</h4>
                                <p>{item.value ?? "—"} {item.unit ?? ""}</p>
                            </div>
                        ))
                    )}
                </div>
            )}
        </div>
    );
};

export default AdvancedDataSection;
