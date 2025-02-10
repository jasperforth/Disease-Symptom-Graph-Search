import React, { useState } from 'react';

const SearchForm = ({ onSearch }) => {
    const [symptom, setSymptom] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (symptom.trim() === '') {
            alert('Please enter a symptom');
            return; 
        }
        onSearch(symptom.trim());
    };

    return (
        <form onSubmit={handleSubmit} className="search-form">  
            <label htmlFor="symptom-input">Enter a Symptom:</label>
            <input
                type="text"
                id="symptom-input"
                value={symptom}
                onChange={(e) => setSymptom(e.target.value)}
                placeholder="e.g., Fever"
            />
            <button type="submit">Search</button>
        </form>
    );
};

export default SearchForm;