// src/components/Navbar.js
import React from 'react';
import { Link } from 'react-router-dom';  // Usamos React Router para navegación

const Navbar = () => {
    return (
        <nav>
            <ul>
                <li>
                    <Link to="/">Home</Link>
                </li>
                <li>
                    <Link to="/page1">Page 1</Link>
                </li>
                <li>
                    <Link to="/page2">Page 2</Link>
                </li>
            </ul>
        </nav>
    );
};

export default Navbar;
