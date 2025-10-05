import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Page1 from './pages/Page1';
import Page2 from './pages/Page2';
import Home from './pages/Home';
import Navbar from './components/Navbar';
import './App.css';
//import './index.css';
import './components/WeatherData.css';

const App = () => (
    <Router>
        <div className="app-container">
            <Navbar />
            <main className="main-content">
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