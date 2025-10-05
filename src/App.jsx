// src/App.jsx (pequeña modificación hacia arriba)
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Page1 from "./pages/Page1";
import Page2 from "./pages/Page2";
import Home from "./pages/Home";
import Navbar from "./components/Navbar";
import CustomCursor from "./components/CustomCursor";
import "./App.css";
import "./components/WeatherData.css";

const App = () => (
    <Router>
        {/* cursor global: tamaño 20 (tiny). Cambia size={16} si quieres aún más pequeño */}
        <CustomCursor enabled={true} size={20} />
        <div className="app-container">
            <Navbar />
            <main className="main-content" style={{ marginTop: 100 /* evitar solapamiento con navbar fija */ }}>
                <Routes>
                    <Route path="/" element={<Home />} />
                    <Route path="/page1" element={<Page1 />} />
                    <Route path="/page2" element={<Page2 />} />
                </Routes>
            </main>
        </div>
    </Router>
);

export default App;
