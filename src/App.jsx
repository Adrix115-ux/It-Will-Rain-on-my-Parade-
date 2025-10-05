import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Page1 from './pages/Page1';
import Page2 from './pages/Page2';
import Home from './pages/Home';
import './App.css'; // Importa el archivo CSS para aplicar los estilos
import './index.css'

const App = () => (
    <Router>
        <Routes>
            <Route path="/" element={<Home />}></Route>
            <Route path="/page1" element={<Page1 />} />
            <Route path="/page2" element={<Page2 />} />
        </Routes>
    </Router>
);

export default App;