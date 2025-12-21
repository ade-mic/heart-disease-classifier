// Get form and result elements
const form = document.getElementById('predictionForm');
const resultsDiv = document.getElementById('results');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');
const predictionBox = document.getElementById('predictionBox');
const recommendationBox = document.getElementById('recommendationBox');
const noDiseaseProb = document.getElementById('noDiseaseProb');
const diseaseProb = document.getElementById('diseaseProb');

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Hide previous results and errors
    resultsDiv.classList.add('hidden');
    errorDiv.classList.add('hidden');
    
    // Show loading
    loadingDiv.classList.remove('hidden');
    
    // Collect form data
    const formData = new FormData(form);
    const data = {};
    
    formData.forEach((value, key) => {
        // Convert to number for numeric fields
        data[key] = key === 'oldpeak' ? parseFloat(value) : parseInt(value);
    });
    
    try {
        // Make prediction request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        // Hide loading
        loadingDiv.classList.add('hidden');
        
        if (result.success) {
            displayResults(result);
        } else {
            displayError(result.error);
        }
        
    } catch (error) {
        loadingDiv.classList.add('hidden');
        displayError('Failed to connect to the server. Please try again.');
        console.error('Error:', error);
    }
});

function displayResults(result) {
    // Update prediction box
    const isDisease = result.prediction === 1;
    
    if (isDisease) {
        predictionBox.className = 'prediction-box at-risk';
        predictionBox.innerHTML = `
            ⚠️ HIGH RISK: Heart Disease Detected<br>
            <span style="font-size: 1.2rem;">Confidence: ${result.confidence}%</span>
        `;
        
        recommendationBox.className = 'recommendation-box warning';
        recommendationBox.innerHTML = `
            <strong>Recommendation:</strong> This prediction suggests a high risk of heart disease. 
            Please consult with a healthcare professional immediately for proper evaluation and treatment.
        `;
    } else {
        predictionBox.className = 'prediction-box healthy';
        predictionBox.innerHTML = `
            ✅ LOW RISK: No Heart Disease Detected<br>
            <span style="font-size: 1.2rem;">Confidence: ${result.confidence}%</span>
        `;
        
        recommendationBox.className = 'recommendation-box success';
        recommendationBox.innerHTML = `
            <strong>Recommendation:</strong> While the model suggests low risk, maintaining regular check-ups 
            and a healthy lifestyle is important for long-term cardiovascular health.
        `;
    }
    
    // Update probabilities
    noDiseaseProb.textContent = `${result.probabilities.no_disease}%`;
    diseaseProb.textContent = `${result.probabilities.disease}%`;
    
    // Show results
    resultsDiv.classList.remove('hidden');
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function displayError(message) {
    errorDiv.textContent = `Error: ${message}`;
    errorDiv.classList.remove('hidden');
    
    // Scroll to error
    errorDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Add input validation
const numericInputs = document.querySelectorAll('input[type="number"]');
numericInputs.forEach(input => {
    input.addEventListener('input', (e) => {
        const value = parseFloat(e.target.value);
        const min = parseFloat(e.target.min);
        const max = parseFloat(e.target.max);
        
        if (value < min) {
            e.target.value = min;
        } else if (value > max) {
            e.target.value = max;
        }
    });
});

// Add smooth animations
document.addEventListener('DOMContentLoaded', () => {
    // Add fade-in animation to form sections
    const sections = document.querySelectorAll('.form-section');
    sections.forEach((section, index) => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            section.style.transition = 'all 0.5s ease';
            section.style.opacity = '1';
            section.style.transform = 'translateY(0)';
        }, index * 100);
    });
});
