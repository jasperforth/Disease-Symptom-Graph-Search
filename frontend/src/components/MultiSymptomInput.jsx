// ./frontend/src/components/MultiSymptomInput.jsx

import React, { useState, useEffect, useRef } from 'react';
import { FaPlus, FaTrash, FaUndo } from 'react-icons/fa';
import { getSymptomSuggestions } from '../utils/api';
import { debounce } from 'lodash'
import './MultiSymptomInput.css';

const MultiSymptomInput = ({ onLiveUpdate, relatedSymptoms }) => {
  const [symptoms, setSymptoms] = useState([
    { name: '', severity: 'Medium', suggestions: []},
  ]);
  const [overlayVisible, setOverlayVisible] = useState(false);
  const inputRefs = useRef([]);
  const [isSuggestionOpen, setIsSuggestionOpen] = useState(false); 

  // Show or hide the overlay based on relatedSymptoms prop
  useEffect(() => {
    if (Array.isArray(relatedSymptoms) && relatedSymptoms.length > 0) {
      setOverlayVisible(true);
    } else {
      setOverlayVisible(false);
      setIsSuggestionOpen(false); // Close suggestions
    }
  }, [relatedSymptoms]);

  // Debounced fetchSuggestions
  const fetchSuggestions = async (index, query) => {
    try {
      const suggestions = await getSymptomSuggestions(query);
      setSymptoms((prevSymptoms) => {
        const updated = [...prevSymptoms];
        updated[index].suggestions = Array.isArray(suggestions) ? suggestions : [];
        return updated;
      });
    } catch (error) {
      console.error('Error fetching symptom suggestions:', error);
      setSymptoms((prevSymptoms) => {
        const updated = [...prevSymptoms];
        updated[index].suggestions = [];
        return updated;
      });
    }
  };

  const debouncedFetchSuggestions = useRef(
    debounce((index, query) => {
      fetchSuggestions(index, query);
    }, 300)
  ).current;

  useEffect(() => {
    return () => {
      debouncedFetchSuggestions.cancel();
    };
  }, [debouncedFetchSuggestions]);

  const updateSymptom = (index, field, value) => {
    const newSymptoms = [...symptoms];
    newSymptoms[index][field] = value;

    // Keep fuzzy suggestions for user input
    if (field === 'name') {
      const query = value.trim();
      if (query.length > 0) {
        debouncedFetchSuggestions(index, query);
      } else {
        newSymptoms[index].suggestions = [];
        setSymptoms(newSymptoms);
      }
    }

    setSymptoms(newSymptoms);
  };

  // Handle relatedSymptoms from the parent
  const handleRelatedSuggestionClick = (suggestion) => {
    setSymptoms([
      ...symptoms,
      {
        name: suggestion,
        severity: 'Medium',
        // Same idea: store the chosen suggestion so it’s recognized as valid
        suggestions: [suggestion],
      },
    ]);
    setOverlayVisible(false);
    setIsSuggestionOpen(false);
  };

  const handleNoOtherSymptoms = () => {
    setOverlayVisible(false);
    setIsSuggestionOpen(false);
    // Do not trigger live update to avoid re-triggering the fade
    // Additional actions can be added here if necessary
  };

  const handleSuggestionClick = (index, suggestion) => {
    const newSymptoms = [...symptoms];
    newSymptoms[index].name = suggestion;
    // Keep the chosen suggestion so isValidSymptom() recognizes it.
    newSymptoms[index].suggestions = [suggestion]; 
    setSymptoms(newSymptoms);
  };

  const lastPayloadRef = useRef(null);

  useEffect(() => {
    // Only consider a symptom valid if it is non-empty, has length >= 3, 
    // and exactly matches one of its suggestions.
    const allValid = symptoms.every((sym) => {
      return sym.name.trim() !== '' && isValidSymptom(sym.name, sym.suggestions);
    });
  
    const payload = allValid
      ? symptoms.map((sym) => ({
          name: sym.name.trim(),
          severity: sym.severity.toLowerCase(),
        }))
      : [];
  
    const payloadString = JSON.stringify(payload);
  
    if (payloadString !== lastPayloadRef.current) {
      onLiveUpdate(payload);
      lastPayloadRef.current = payloadString;
    }
  }, [symptoms, onLiveUpdate]);
  

  const isValidSymptom = (symptomName, suggestions) => {
    const input = symptomName.trim().toLowerCase();
    if (input.length < 3) return false;
    // Ensure suggestions is an array before using map
    const suggestionsArray = Array.isArray(suggestions) ? suggestions : [];
    return suggestionsArray.map(s => s.toLowerCase()).includes(input);
  };
  

  const addSymptom = () => {
    setSymptoms([
      ...symptoms,
      { name: '', severity: 'Medium', suggestions: [] },
    ]);
  };

  const removeSymptom = (index) => {
    const newSymptoms = symptoms.filter((_, i) => i !== index);
    setSymptoms(newSymptoms);
  };

  const resetAllSymptoms = () => {
    setSymptoms([{ name: '', severity: 'Medium', suggestions: [] }]);
  };
  
  const resetSymptom = (index) => {
    const newSymptoms = [...symptoms];
    newSymptoms[index] = {
      name: '',
      severity: 'Medium',
      suggestions: []
    };
    setSymptoms(newSymptoms);
  };
  

  // Toggle function for suggestion window
  const toggleSuggestionWindow = () => {
    setIsSuggestionOpen(!isSuggestionOpen);
  };

  return (
    <div className="multi-symptom-form">
      <h2>Enter Symptoms:</h2>
      <div className="symptoms-container">
        {symptoms.map((symptom, index) => (
          <div key={index} className="symptom-entry">
            <input
              type="text"
              placeholder="e.g., Headache"
              value={symptom.name}
              onChange={(e) => updateSymptom(index, 'name', e.target.value)}
              // ref={(el) => (inputRefs.current[index] = el)}
              className={
                symptom.name.trim() !== '' && !isValidSymptom(symptom.name, symptom.suggestions)
                  ? 'invalid-input'
                  : ''
              }
            />
            <select
              value={symptom.severity}
              onChange={(e) => updateSymptom(index, 'severity', e.target.value)}
            >
              <option value="Low">Low</option>
              <option value="Medium">Medium</option>
              <option value="High">High</option>
            </select>
            {/* Button: Reset for first symptom, Remove for subsequent */}
            {index === 0 ? (
              <button
                type="button"
                onClick = {resetAllSymptoms}
                className="remove-button"
                title="Reset (all) Symptom input(s)"
              >
                <FaUndo />
              </button>
            ) : (
              <button
                type="button"
                onClick={() => removeSymptom(index)}
                className="remove-button"
                title="Remove Symptom Input"
              >
                <FaTrash />
              </button>
            )}
            {symptom.suggestions.length > 0 && (
                // If there's exactly one suggestion that matches the trimmed input, hide suggestions
                (symptom.suggestions.length === 1 &&
                  symptom.suggestions[0].toLowerCase() === symptom.name.trim().toLowerCase())
                  ? null
                  : (
                      <ul className="suggestions">
                        {symptom.suggestions.map((suggestion, sIndex) => (
                          <li
                            key={sIndex}
                            onClick={() => handleSuggestionClick(index, suggestion)}
                            className="suggestion-item"
                          >
                            {suggestion}
                          </li>
                        ))}
                      </ul>
                    )
              )
            }
            {(() => {
              const input = symptom.name.trim().toLowerCase();
              // Use the new isValidSymptom check.
              const isSymptomValid = isValidSymptom(symptom.name, symptom.suggestions);
              return (input !== '' && !isSymptomValid) ? (
                <span className="validation-error">Invalid symptom name.</span>
              ) : null;
            })()}
          </div>
        ))}
      </div>

      {/* Related Symptoms Toggle Button or Add Symptom Button */}
      <div className="buttons-container">
        {overlayVisible && Array.isArray(relatedSymptoms) && relatedSymptoms.length > 0 ? (
          isSuggestionOpen ? (
            <>
              {/* State 3: Suggestion Window Open */}
              <button
                onClick={toggleSuggestionWindow}
                className="toggle-suggestions-button align-left"
              >
                Hide Related Symptoms
              </button>
              <button
                type="button"
                onClick={addSymptom}
                className="add-button align-right"
                title="Add Symptom"
              >
                <FaPlus />
              </button>
            </>
          ) : (
            /* State 2: Suggestions Available, Window Closed */
            <button
              onClick={toggleSuggestionWindow}
              className="toggle-suggestions-button centered-button"
            >
              Show Related Symptoms
            </button>
          )
        ) : (
          /* State 1: No Suggestions Available */
          <button
            type="button"
            onClick={addSymptom}
            className="add-button centered-button"
            title="Add Symptom"
          >
            <FaPlus />
          </button>
        )}
      </div>


      {/* Related Symptoms Suggestion Window */}
      {isSuggestionOpen && (
        <div className="related-suggestions-window">
          <h4>Top {relatedSymptoms.length} Related Symptoms:</h4>
          <ul className="related-suggestions-list">
            {relatedSymptoms.slice(0, 12).map((suggestion, index) => (
              <li
                key={index}
                onClick={() => handleRelatedSuggestionClick(suggestion)}
                className="related-suggestion-item"
              >
                {suggestion}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default MultiSymptomInput;
