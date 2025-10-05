// src/components/MapComponent.jsx
import React from "react";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";

const MapComponent = ({ position = [-17.7863, -63.1812], height = 450 }) => {
    return (
        <div id="map" style={{ width: "100%", height }}>
            <MapContainer center={position} zoom={10} style={{ height: "100%", width: "100%" }}>
                <TileLayer
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    attribution='&copy; OpenStreetMap contributors'
                />
                <Marker position={position}>
                    <Popup>Ubicación seleccionada</Popup>
                </Marker>
            </MapContainer>
        </div>
    );
};

export default MapComponent;
