// src/components/LinkCard.js
import React from 'react';
import { Link } from 'react-router-dom';

const LinkCard = ({ title, description, to }) => {
    return (
        <div className="card">
            <h3>{title}</h3>
            <p>{description}</p>
            <Link to={to}>Go to {title}</Link>
        </div>
    );
};

export default LinkCard;
