// Dashboard functionality
const API_BASE = 'http://localhost:8277/api';

let allItems = [];
let filteredItems = [];

// DOM Elements
const totalCount = document.getElementById('total-count');
const reviewedCount = document.getElementById('reviewed-count');
const trainedCount = document.getElementById('trained-count');
const pendingCount = document.getElementById('pending-count');
const tableBody = document.getElementById('table-body');
const trainBtn = document.getElementById('train-btn');
const refreshBtn = document.getElementById('refresh-btn');
const resetDbBtn = document.getElementById('reset-db-btn');
const searchInput = document.getElementById('search-input');
const filterReviewed = document.getElementById('filter-reviewed');
const filterTrained = document.getElementById('filter-trained');
const statusMessage = document.getElementById('status-message');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadData();
    setupEventListeners();
});

function setupEventListeners() {
    trainBtn.addEventListener('click', trainModel);
    refreshBtn.addEventListener('click', () => {
        loadStats();
        loadData();
    });
    resetDbBtn.addEventListener('click', resetDatabase);

    searchInput.addEventListener('input', filterTable);
    filterReviewed.addEventListener('change', filterTable);
    filterTrained.addEventListener('change', filterTable);
}

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/data/stats`);
        if (!response.ok) throw new Error('Failed to load stats');
        
        const stats = await response.json();
        
        totalCount.textContent = stats.total;
        reviewedCount.textContent = stats.reviewed;
        trainedCount.textContent = stats.trained;
        pendingCount.textContent = stats.pending_training;
    } catch (error) {
        console.error('Error loading stats:', error);
        showStatus('Failed to load statistics', 'error');
    }
}

async function loadData() {
    try {
        const response = await fetch(`${API_BASE}/data`);
        if (!response.ok) throw new Error('Failed to load data');
        
        allItems = await response.json();
        filteredItems = [...allItems];
        renderTable();
    } catch (error) {
        console.error('Error loading data:', error);
        showStatus('Failed to load training data', 'error');
    }
}

function filterTable() {
    const searchTerm = searchInput.value.toLowerCase();
    const showReviewedOnly = filterReviewed.checked;
    const showTrainedOnly = filterTrained.checked;
    
    filteredItems = allItems.filter(item => {
        const matchesSearch = item.transcript.toLowerCase().includes(searchTerm);
        const matchesReviewed = !showReviewedOnly || item.is_reviewed;
        const matchesTrained = !showTrainedOnly || item.is_trained;
        
        return matchesSearch && matchesReviewed && matchesTrained;
    });
    
    renderTable();
}

function renderTable() {
    tableBody.innerHTML = '';
    
    if (filteredItems.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px; color: var(--text-light);">No items found</td></tr>';
        return;
    }
    
    filteredItems.forEach(item => {
        const row = document.createElement('tr');
        
        const reviewedBadge = item.is_reviewed 
            ? '<span class="status-badge reviewed">✓ Reviewed</span>'
            : '<span style="color: var(--text-light);">-</span>';
        
        const trainedBadge = item.is_trained
            ? '<span class="status-badge trained">✓ Trained</span>'
            : '<span style="color: var(--text-light);">-</span>';
        
        const truncatedTranscript = item.transcript.length > 100 
            ? item.transcript.substring(0, 100) + '...'
            : item.transcript;
        
        row.innerHTML = `
            <td>${item.id}</td>
            <td title="${item.transcript}">${truncatedTranscript}</td>
            <td style="text-align: center;">${reviewedBadge}</td>
            <td style="text-align: center;">${trainedBadge}</td>
            <td>${new Date(item.timestamp).toLocaleString()}</td>
            <td style="text-align: center;">
                <button class="btn btn-danger btn-sm" onclick="deleteItem(${item.id})">Delete</button>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
}

async function trainModel() {
    if (!confirm('Start training with all reviewed items?')) {
        return;
    }
    
    trainBtn.disabled = true;
    trainBtn.textContent = 'Training...';
    
    try {
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) throw new Error('Training failed');
        
        const result = await response.json();
        
        if (result.success) {
            showStatus(`Training completed: ${result.trained_count} items trained`, 'success');
            // Reload stats and data
            setTimeout(() => {
                loadStats();
                loadData();
            }, 1000);
        } else {
            showStatus(result.message || 'Training failed', 'error');
        }
    } catch (error) {
        console.error('Error training model:', error);
        showStatus('Failed to train model', 'error');
    } finally {
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train Model';
    }
}

async function resetDatabase() {
    const confirmation = prompt('WARNING: This will DELETE ALL training data!\n\nType "RESET" to confirm:');
    
    if (confirmation !== 'RESET') {
        showStatus('Database reset cancelled', 'info');
        return;
    }
    
    resetDbBtn.disabled = true;
    resetDbBtn.textContent = 'Resetting...';
    
    try {
        const response = await fetch(`${API_BASE}/database/reset`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) throw new Error('Database reset failed');
        
        const result = await response.json();
        
        if (result.success) {
            showStatus('Database reset successfully', 'success');
            // Reload stats and data
            setTimeout(() => {
                loadStats();
                loadData();
            }, 1000);
        } else {
            showStatus(result.error || 'Database reset failed', 'error');
        }
    } catch (error) {
        console.error('Error resetting database:', error);
        showStatus('Failed to reset database', 'error');
    } finally {
        resetDbBtn.disabled = false;
        resetDbBtn.textContent = 'Reset Database';
    }
}

async function deleteItem(itemId) {
    if (!confirm(`Are you sure you want to delete item ${itemId}?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/data/${itemId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Failed to delete item');
        
        showStatus('Item deleted successfully', 'success');
        
        // Reload data
        setTimeout(() => {
            loadStats();
            loadData();
        }, 500);
    } catch (error) {
        console.error('Error deleting item:', error);
        showStatus('Failed to delete item', 'error');
    }
}

function showStatus(message, type = 'info') {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    statusMessage.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        statusMessage.style.display = 'none';
    }, 5000);
}
