// src/pages/Home.js
import React from 'react';
import MapComponent from '../components/MapComponent';
import LinkCard from '../components/LinkCard';

const Home = () => {
    return (
        <div>
            <h1>Welcome to the Rain Prediction App</h1>
            <MapComponent />
            <div className="link-cards">
                <LinkCard
                    title="Page 1"
                    description="Description of Page 1"
                    to="/page1"
                />
                <LinkCard
                    title="Page 2"
                    description="Description of Page 2"
                    to="/page2"
                />
            </div>
        </div>
    );
};

export default Home;
