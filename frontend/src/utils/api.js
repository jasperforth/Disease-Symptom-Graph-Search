const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const searchDisease = async (symptom) => {
    try {
        const response = await fetch(`${API_BASE_URL}/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symptom })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to fetch diseases.');
        }
        
        const data = await response.json();
        return data; 
    } catch (err) {
        console.error('Error fetching diseases:', err);
        throw err;
    }
};

export const searchDiseaseMultiple = async (symptoms) => {
    try {
        const response = await fetch(`${API_BASE_URL}/searchMultiple`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ symptoms })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to fetch diseases.');
        }
        
        const data = await response.json();
        return data; 
    } catch (err) {
        console.error('Error fetching diseases:', err);
        throw err;
    }
};

export const getSymptomSuggestions = async (query) => {
    try {
      const response = await fetch(`${API_BASE_URL}/symptomsSuggestions?query=${encodeURIComponent(query)}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
  
      if (!response.ok) {
        throw new Error('Failed to fetch symptom suggestions.');
      }
  
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching symptom suggestions:', error);
      return [];
    }
  };

  export const getRelatedSymptomSuggestions = async (symptoms) => {
    try {
      const response = await fetch(`${API_BASE_URL}/relatedSymptoms`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ symptoms }),
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch related symptom suggestions.');
      }
  
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching related symptom suggestions:', error);
      return [];
    }
  };
  
  