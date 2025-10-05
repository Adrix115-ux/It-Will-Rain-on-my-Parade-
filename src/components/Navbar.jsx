import React from 'react';
import LinkCard from './LinkCard';

const Navbar = () => {
    return (
        <nav className="navbar">
            <div className="nav-cards">
                <LinkCard
                    title="Inicio"
                    description="Página principal"
                    to="/"
                />
                <LinkCard
                    title="Datos Meteorológicos"
                    description="Ver datos actuales"
                    to="/page1"
                />
                <LinkCard
                    title="Predicciones"
                    description="Ver pronósticos"
                    to="/page2"
                />
            </div>
        </nav>
    );
};

export default Navbar;