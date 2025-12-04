/**
 * Glaucoma E-Predictor - Frontend Application
 * Handles image upload, prediction, and admin functions
 */

// API Configuration (loaded from config.js)
const API_BASE_URL = window.API_CONFIG?.BASE_URL || 'http://localhost:8000';

// DOM Elements
const elements = {
    // Upload
    uploadArea: document.getElementById('uploadArea'),
    imageInput: document.getElementById('imageInput'),
    previewContainer: document.getElementById('previewContainer'),
    previewImage: document.getElementById('previewImage'),
    imageInfo: document.getElementById('imageInfo'),
    clearBtn: document.getElementById('clearBtn'),
    predictBtn: document.getElementById('predictBtn'),
    
    // Results
    resultsSection: document.getElementById('resultsSection'),
    resultsTimestamp: document.getElementById('resultsTimestamp'),
    gaugeProgress: document.getElementById('gaugeProgress'),
    gaugeNeedle: document.getElementById('gaugeNeedle'),
    gaugeValue: document.getElementById('gaugeValue'),
    resultBadge: document.getElementById('resultBadge'),
    predictionText: document.getElementById('predictionText'),
    probabilityValue: document.getElementById('probabilityValue'),
    confidenceValue: document.getElementById('confidenceValue'),
    riskLevelValue: document.getElementById('riskLevelValue'),
    recommendationBox: document.getElementById('recommendationBox'),
    recommendationText: document.getElementById('recommendationText'),
    
    // Status
    apiStatus: document.getElementById('apiStatus'),
    
    // Admin
    modelLoaded: document.getElementById('modelLoaded'),
    modelParams: document.getElementById('modelParams'),
    modelLayers: document.getElementById('modelLayers'),
    dataAvailable: document.getElementById('dataAvailable'),
    totalImages: document.getElementById('totalImages'),
    dataClasses: document.getElementById('dataClasses'),
    refreshModelBtn: document.getElementById('refreshModelBtn'),
    refreshDataBtn: document.getElementById('refreshDataBtn'),
    epochsInput: document.getElementById('epochsInput'),
    retrainBtn: document.getElementById('retrainBtn'),
    trainingLabel: document.getElementById('trainingLabel'),
    trainingValue: document.getElementById('trainingValue'),
    progressFill: document.getElementById('progressFill')
};

// State
let selectedFile = null;
let isTraining = false;
let trainingPollInterval = null;

// ============================================
// API Functions
// ============================================

async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        elements.apiStatus.classList.remove('disconnected');
        elements.apiStatus.classList.add('connected');
        elements.apiStatus.querySelector('.status-text').textContent = 
            data.model_ready ? 'Model Ready' : 'API Connected';
        
        return data;
    } catch (error) {
        elements.apiStatus.classList.remove('connected');
        elements.apiStatus.classList.add('disconnected');
        elements.apiStatus.querySelector('.status-text').textContent = 'Disconnected';
        return null;
    }
}

async function predictImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Prediction failed');
    }
    
    return response.json();
}

async function getModelInfo() {
    const response = await fetch(`${API_BASE_URL}/model-info`);
    return response.json();
}

async function getDataStatus() {
    const response = await fetch(`${API_BASE_URL}/data-status`);
    return response.json();
}

async function startRetraining(epochs) {
    const response = await fetch(`${API_BASE_URL}/retrain?epochs=${epochs}`, {
        method: 'POST'
    });
    return response.json();
}

async function getTrainingStatus() {
    const response = await fetch(`${API_BASE_URL}/training-status`);
    return response.json();
}

// ============================================
// UI Functions
// ============================================

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatNumber(num) {
    if (num >= 1000000) {
        return (num / 1000000).toFixed(2) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(2) + 'K';
    }
    return num.toString();
}

function formatTimestamp(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString();
}

function showPreview(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.previewContainer.classList.add('active');
        elements.predictBtn.disabled = false;
        
        // Show image info
        const img = new Image();
        img.onload = () => {
            elements.imageInfo.innerHTML = `
                <span>${img.width} × ${img.height}px</span>
                <span>${formatBytes(file.size)}</span>
                <span>${file.type || 'image'}</span>
            `;
        };
        img.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
}

function clearPreview() {
    selectedFile = null;
    elements.previewImage.src = '';
    elements.previewContainer.classList.remove('active');
    elements.predictBtn.disabled = true;
    elements.imageInput.value = '';
    elements.resultsSection.classList.remove('active');
}

function updateGauge(percentage) {
    // Calculate stroke offset (arc is 251.2 units total)
    const maxOffset = 251.2;
    const offset = maxOffset - (percentage / 100) * maxOffset;
    elements.gaugeProgress.style.strokeDashoffset = offset;
    
    // Calculate needle rotation (-90 to 90 degrees)
    const rotation = -90 + (percentage / 100) * 180;
    elements.gaugeNeedle.style.transform = `rotate(${rotation}deg)`;
    
    // Update value
    elements.gaugeValue.textContent = `${percentage}%`;
}

function displayResults(result) {
    // Show results section
    elements.resultsSection.classList.add('active');
    
    // Update timestamp
    elements.resultsTimestamp.textContent = formatTimestamp(result.timestamp);
    
    // Animate gauge
    setTimeout(() => {
        updateGauge(result.percentage);
    }, 100);
    
    // Update badge
    const isGlaucoma = result.prediction.includes('Glaucoma');
    elements.resultBadge.className = `result-badge ${isGlaucoma ? 'glaucoma' : 'normal'} pulse`;
    elements.predictionText.textContent = result.prediction;
    
    // Update stats
    elements.probabilityValue.textContent = `${result.percentage}%`;
    elements.confidenceValue.textContent = `${result.confidence}%`;
    elements.riskLevelValue.textContent = result.risk_level;
    
    // Color code risk level
    const riskColors = {
        'LOW': '#10B981',
        'MODERATE': '#F59E0B',
        'HIGH': '#EF4444'
    };
    elements.riskLevelValue.style.color = riskColors[result.risk_level] || '#F8FAFC';
    
    // Update recommendation
    elements.recommendationText.textContent = result.recommendation;
    
    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function updateModelStatus() {
    try {
        const data = await getModelInfo();
        
        if (data.success) {
            elements.modelLoaded.textContent = data.model_loaded ? 'Yes' : 'No';
            elements.modelLoaded.style.color = data.model_loaded ? '#10B981' : '#EF4444';
            
            if (data.model_info) {
                elements.modelParams.textContent = formatNumber(data.model_info.total_parameters);
                elements.modelLayers.textContent = data.model_info.layers;
            } else {
                elements.modelParams.textContent = '--';
                elements.modelLayers.textContent = '--';
            }
        }
    } catch (error) {
        console.error('Failed to get model info:', error);
        elements.modelLoaded.textContent = 'Error';
        elements.modelLoaded.style.color = '#EF4444';
    }
}

async function updateDataStatus() {
    try {
        const data = await getDataStatus();
        
        if (data.success) {
            elements.dataAvailable.textContent = data.data_available ? 'Yes' : 'No';
            elements.dataAvailable.style.color = data.data_available ? '#10B981' : '#EF4444';
            
            if (data.data_available) {
                elements.totalImages.textContent = data.total_images || '--';
                elements.dataClasses.textContent = data.classes ? data.classes.join(', ') : '--';
            } else {
                elements.totalImages.textContent = '--';
                elements.dataClasses.textContent = '--';
            }
        }
    } catch (error) {
        console.error('Failed to get data status:', error);
        elements.dataAvailable.textContent = 'Error';
        elements.dataAvailable.style.color = '#EF4444';
    }
}

function updateTrainingProgress(status) {
    elements.trainingValue.textContent = status.message;
    elements.progressFill.style.width = `${status.progress}%`;
    
    if (status.is_training) {
        elements.retrainBtn.disabled = true;
        elements.retrainBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
                <path d="M21 12a9 9 0 11-6.219-8.56"/>
            </svg>
            Training...
        `;
    } else {
        elements.retrainBtn.disabled = false;
        elements.retrainBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 12a9 9 0 11-6.219-8.56"/>
                <polyline points="21 3 21 9 15 9"/>
            </svg>
            Start Retraining
        `;
        
        // Stop polling if training is complete
        if (trainingPollInterval) {
            clearInterval(trainingPollInterval);
            trainingPollInterval = null;
        }
        
        // Refresh model status after training
        if (status.message.includes('completed')) {
            updateModelStatus();
        }
    }
}

async function pollTrainingStatus() {
    try {
        const status = await getTrainingStatus();
        updateTrainingProgress(status);
    } catch (error) {
        console.error('Failed to get training status:', error);
    }
}

// ============================================
// Event Handlers
// ============================================

function handleFileSelect(file) {
    if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
    }
    
    selectedFile = file;
    showPreview(file);
}

async function handlePredict() {
    if (!selectedFile) {
        alert('Please select an image first.');
        return;
    }
    
    // Show loading state
    elements.predictBtn.classList.add('loading');
    elements.predictBtn.disabled = true;
    
    try {
        const result = await predictImage(selectedFile);
        
        if (result.success) {
            displayResults(result);
        } else {
            throw new Error(result.message || 'Prediction failed');
        }
    } catch (error) {
        alert(`Prediction Error: ${error.message}`);
        console.error('Prediction error:', error);
    } finally {
        elements.predictBtn.classList.remove('loading');
        elements.predictBtn.disabled = false;
    }
}

async function handleRetrain() {
    const epochs = parseInt(elements.epochsInput.value) || 20;
    
    if (epochs < 1 || epochs > 100) {
        alert('Please enter a valid number of epochs (1-100).');
        return;
    }
    
    try {
        const result = await startRetraining(epochs);
        
        if (result.success) {
            updateTrainingProgress({ is_training: true, progress: 0, message: 'Starting...' });
            
            // Start polling for status updates
            trainingPollInterval = setInterval(pollTrainingStatus, 2000);
        } else {
            alert(`Failed to start training: ${result.message}`);
        }
    } catch (error) {
        alert(`Training Error: ${error.message}`);
        console.error('Training error:', error);
    }
}

// ============================================
// Initialize
// ============================================

function init() {
    // Check API health on load
    checkApiHealth();
    setInterval(checkApiHealth, 30000); // Check every 30 seconds
    
    // Load initial status
    updateModelStatus();
    updateDataStatus();
    
    // Upload area click
    elements.uploadArea.addEventListener('click', () => {
        elements.imageInput.click();
    });
    
    // File input change
    elements.imageInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });
    
    elements.uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
    });
    
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    // Clear button
    elements.clearBtn.addEventListener('click', clearPreview);
    
    // Predict button
    elements.predictBtn.addEventListener('click', handlePredict);
    
    // Refresh buttons
    elements.refreshModelBtn.addEventListener('click', updateModelStatus);
    elements.refreshDataBtn.addEventListener('click', updateDataStatus);
    
    // Retrain button
    elements.retrainBtn.addEventListener('click', handleRetrain);
    
    // Navigation smooth scroll
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
            
            // Update active state
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
    
    // Check training status on load (in case training is in progress)
    pollTrainingStatus();
    
    console.log('🔬 Glaucoma E-Predictor initialized');
}

// Start the app
document.addEventListener('DOMContentLoaded', init);

