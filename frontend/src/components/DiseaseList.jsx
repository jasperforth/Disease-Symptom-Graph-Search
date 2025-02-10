// ./frontend/src/components/DiseaseList.jsx
import React, { useState, useEffect } from 'react';
import '../styles/App.css';
import './DiseaseList.css';

const DiseaseList = ({ diseases }) => {
  const [fadeIn, setFadeIn] = useState(false);
  const [fadeOut, setFadeOut] = useState(false);

  useEffect(() => {
    if (diseases.length === 0) {
      setFadeIn(false);
      setFadeOut(false);
      return;
    }

    console.log('Triggering fade-in and fade-out for new diseases');

    // Reset fade states
    setFadeIn(false);
    setFadeOut(false);

    // Trigger fade-in after a short delay to allow transition
    const fadeInTimer = setTimeout(() => {
      setFadeIn(true);
      console.log('Fade-in triggered');
    }, 50); // 50ms delay

    // Trigger fade-out after 1 second
    const fadeOutTimer = setTimeout(() => {
      setFadeOut(true);
      console.log('Fade-out triggered');
    }, 1000); // 1 second

    return () => {
      clearTimeout(fadeInTimer);
      clearTimeout(fadeOutTimer);
    };
  }, [diseases]);

  if (!diseases || diseases.length === 0) {
    return <p>No diseases found for the given symptoms.</p>;
  }

  return (
    <div className={`disease-list ${fadeIn ? 'fade-in' : ''}`}>
      <h2>Top Associated Diseases:</h2>
      <table className="disease-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Disease</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>
          {diseases.map((disease, index) => {
            const rank = index + 1;

            // Calculate reduced opacity based on rank
            const reducedOpacity =
              rank === 1 ? 1 : Math.max(0.1, Math.exp(-0.2 * (rank - 1)));

            return (
              <tr
                key={index}
                className="disease-row"
                style={{
                  opacity: fadeOut ? reducedOpacity : 1,
                }}
              >
                <td>{rank}</td>
                <td>{disease.disease}</td>
                <td>{disease.score.toFixed(2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};

export default DiseaseList;
