// src/components/Navbar.jsx
import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
    return (
        <nav className="navbar">
            <div className="nav-links">
                <Link to="/">Inicio</Link>
                <Link to="/page1">Datos Meteorológicos</Link>
                <Link to="/page2">Predicciones</Link>
            </div>
        </nav>
    );
};

export default Navbar;