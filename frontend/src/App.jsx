// ./frontend/src/App.jsx
// Code for the main App component, which renders the entire application.

import React, { useState, useEffect } from 'react';
import MultiSymptomInput from './components/MultiSymptomInput';
import DiseaseList from './components/DiseaseList';
import { searchDiseaseMultiple } from './utils/api';
import { FaCog } from 'react-icons/fa';
import './styles/App.css';

function App() {
  const [diseases, setDiseases] = useState([]);
  const [relatedSymptoms, setRelatedSymptoms] = useState([]);
  const [loading, setLoading] = useState(false);
  const [showDiseases, setShowDiseases] = useState(false); // Control visibility
  const [isSettingsOpen, setIsSettingsOpen] = useState(false); // Control settings modal
  const [isInfoOpen, setIsInfoOpen] = useState(false); // Control info modal
  const [isDarkMode, setIsDarkMode] = useState(false);

  const toggleDarkMode = () => setIsDarkMode((prevMode) => !prevMode);

  // Persist theme preference using localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('isDarkMode');
    if (savedTheme) {
      setIsDarkMode(JSON.parse(savedTheme));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('isDarkMode', isDarkMode);
  }, [isDarkMode]);

  useEffect(() => {
    document.body.style.backgroundColor = isDarkMode ? '#121212' : '#f8f9fa';
    document.body.style.color = isDarkMode ? '#e0e0e0' : '#333';
  }, [isDarkMode]);


  const handleSearch = async (symptoms) => {
    if (symptoms.length === 0) {
      // If no symptoms, clear results
      setDiseases([]);
      setRelatedSymptoms([]); // Clear related symptoms
      setShowDiseases(false); // Reset visibility
      return;
    }
    setLoading(true);
    try {
      const response = await searchDiseaseMultiple(symptoms);
      // "response" = { diseases: [...], related_symptoms: [...] }
      const { diseases: newDiseases, related_symptoms: newRelated } = response;

      setDiseases(newDiseases || []);
      setRelatedSymptoms(newRelated || []);
      setShowDiseases(true); // Trigger fade-in
    } catch (error) {
      console.error('Error fetching disease results:', error);
      setDiseases([]);
      setRelatedSymptoms([]); // Clear related symptoms
      setShowDiseases(false); // Reset visibility
    } finally {
      setLoading(false);
    }
  };

  const toggleSettings = () => setIsSettingsOpen(!isSettingsOpen);
  const toggleInfo = () => setIsInfoOpen(!isInfoOpen);

  return (
    /* apply 'dark-mode' class based on isDarkMode state */
    <div className={`app-container ${isDarkMode ? 'dark-mode' : ''}`}>
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
        <h1>Disease-Symptom Analysis</h1>
        </div>

        <span className="app-description" onClick={toggleInfo}>
          <p> Disease-symptom scoring based on PubMed data. </p>

        </span>

        <button className="settings-button" onClick={toggleSettings}>
          <FaCog size={15}/>
        </button>
      </header>

      {/* Info Popover */}
      {isInfoOpen && (
        <>
          <div className="info-backdrop" onClick={toggleInfo} />
          {/* inside the info-popover */}
          <div className="info-popover" onClick={toggleInfo}>
            <h2>About This App</h2>
            {/* <p>
              This project builds an interactive tool that uses PubMed data, Neo4j, and TF-IDF 
              scoring to identify likely diseases from user-input symptoms.
            </p>
            <ul>
              <li>Import disease/symptom occurrence datasets and construct a weighted graph.</li>
              <li>Rank diseases based on symptom co-occurrence, TF-IDF, and optional NPMI scores.</li>
              <li>Provide fuzzy symptom suggestions and related-symptom hints.</li>
              <li>Return a top-10 list of best-matching diseases for the entered symptoms.</li>
            </ul> */}

          <h3>Purpose</h3>
            <p>
              This application analyzes user-input symptoms and matches them to 
              the most relevant diseases using data from PubMed publications.
            </p>

            <h3>How It Works</h3>
            <ul>
              <li>Enter one or more symptoms.</li>
              <li>Receive a ranked list of the top 10 matching diseases.</li>
              <li>Suggestions for related symptoms refine as you enter more data.</li>
            </ul>

            <h3>Methodology</h3>
            <p>
              The matching is based on a weighted scoring system using:
            </p>
            <ul>
              <li>Co-occurrence frequency in PubMed articles.</li>
              <li>TF-IDF to prioritize unique symptom-disease relationships.</li>
              <li>NPMI (Normalized Pointwise Mutual Information) to adjust for statistical significance.</li>
            </ul>

            <h3>Limitations</h3>
            <p>
              This tool is for academic and informational purposes only. 
              It does not provide medical advice or diagnoses. 
            </p>
            <p>
              Always consult a healthcare professional for medical concerns.
            </p>
          </div>
        </>
      )}

      {/* Settings Popover */}
      {isSettingsOpen && (
        <>
          <div className="settings-backdrop" onClick={toggleSettings} />

          <div className="settings-popover">
            <h2>Settings</h2>
            {/* Dark Mode Toggle */}
            <div className="theme-toggle">
              <label htmlFor="dark-mode-toggle" className="toggle-label">
                Dark Mode
              </label>
              <input
                type="checkbox"
                id="dark-mode-toggle"
                checked={isDarkMode}
                onChange={toggleDarkMode}
              />
            </div>
            <button onClick={toggleSettings} className="close-settings-button">
              Close
            </button>
          </div>
        </>
      )}

      {/* Main content */}
      <div className="main-content">
        {/* Left column symptom input */}
        <div className="left-column">
          <MultiSymptomInput onLiveUpdate={handleSearch} relatedSymptoms={relatedSymptoms}/>
        </div>

        {/* Right column disease list */}
        <div className="right-column">
          {loading ? (
            <p>Loading...</p>
          ) : (
            <div className={`disease-container ${showDiseases ? 'fade-in' : ''}`}>
              <DiseaseList diseases={diseases} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
