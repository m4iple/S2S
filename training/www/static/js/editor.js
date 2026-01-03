// Editor functionality
const API_BASE = 'http://localhost:8080/api';

let currentIndex = -1;
let items = [];
let currentItem = null;

// DOM Elements
const itemId = document.getElementById('item-id');
const itemTimestamp = document.getElementById('item-timestamp');
const audioPlayer = document.getElementById('audio-player');
const transcriptEditor = document.getElementById('transcript-editor');
const reviewCheckbox = document.getElementById('review-checkbox');
const saveBtn = document.getElementById('save-btn');
const prevBtn = document.getElementById('prev-btn');
const nextBtn = document.getElementById('next-btn');
const skipBtn = document.getElementById('skip-btn');
const deleteBtn = document.getElementById('delete-btn');
const navStatus = document.getElementById('nav-status');
const statusMessage = document.getElementById('status-message');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadItems();
    setupEventListeners();
});

function setupEventListeners() {
    saveBtn.addEventListener('click', saveItem);
    prevBtn.addEventListener('click', previousItem);
    nextBtn.addEventListener('click', nextItem);
    skipBtn.addEventListener('click', skipItem);
    deleteBtn.addEventListener('click', deleteItem);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey || e.metaKey) {
            if (e.key === 's') {
                e.preventDefault();
                saveItem();
            }
        }
        if (e.key === 'ArrowLeft' && e.altKey) {
            previousItem();
        }
        if (e.key === 'ArrowRight' && e.altKey) {
            nextItem();
        }
    });
}

async function loadItems() {
    try {
        const response = await fetch(`${API_BASE}/data?reviewed=false`);
        if (!response.ok) throw new Error('Failed to load items');
        
        items = await response.json();
        
        if (items.length === 0) {
            showStatus('No unreviewed items to edit', 'info');
            navStatus.textContent = 'All items have been reviewed!';
            disableControls();
            return;
        }
        
        currentIndex = 0;
        loadItem(0);
    } catch (error) {
        console.error('Error loading items:', error);
        showStatus('Failed to load training items', 'error');
        navStatus.textContent = 'Error loading items';
        disableControls();
    }
}

async function loadItem(index) {
    if (index < 0 || index >= items.length) {
        return;
    }
    
    currentIndex = index;
    const item = items[index];
    
    try {
        const response = await fetch(`${API_BASE}/data/${item.id}`);
        if (!response.ok) throw new Error('Failed to load item');
        
        currentItem = await response.json();
        
        // Update UI
        itemId.textContent = currentItem.id;
        itemTimestamp.textContent = new Date(currentItem.timestamp).toLocaleString();
        transcriptEditor.value = currentItem.transcript;
        reviewCheckbox.checked = currentItem.is_reviewed;
        
        // Load audio
        if (currentItem.audio_base64) {
            const audioBlob = base64ToBlob(currentItem.audio_base64, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayer.src = audioUrl;
        } else {
            audioPlayer.src = '';
            showStatus('No audio available for this item', 'info');
        }
        
        updateNavigation();
    } catch (error) {
        console.error('Error loading item:', error);
        showStatus('Failed to load item details', 'error');
    }
}

function updateNavigation() {
    const remaining = items.length - currentIndex;
    navStatus.textContent = `Item ${currentIndex + 1} of ${items.length} (${remaining - 1} remaining)`;
    
    prevBtn.disabled = currentIndex === 0;
    nextBtn.disabled = currentIndex === items.length - 1;
}

async function saveItem() {
    if (!currentItem) return;
    
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving...';
    
    try {
        const response = await fetch(`${API_BASE}/data/${currentItem.id}`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                transcript: transcriptEditor.value,
                is_reviewed: reviewCheckbox.checked
            })
        });
        
        if (!response.ok) throw new Error('Failed to save item');
        
        showStatus('Item saved successfully', 'success');
        
        // Move to next item if reviewed
        if (reviewCheckbox.checked && currentIndex < items.length - 1) {
            setTimeout(() => {
                nextItem();
            }, 500);
        }
    } catch (error) {
        console.error('Error saving item:', error);
        showStatus('Failed to save item', 'error');
    } finally {
        saveBtn.disabled = false;
        saveBtn.textContent = 'Save & Next';
    }
}

function previousItem() {
    if (currentIndex > 0) {
        loadItem(currentIndex - 1);
    }
}

function nextItem() {
    if (currentIndex < items.length - 1) {
        loadItem(currentIndex + 1);
    }
}

async function deleteItem() {
    if (!currentItem) return;
    
    if (!confirm(`Are you sure you want to delete item ${currentItem.id}?`)) {
        return;
    }
    
    deleteBtn.disabled = true;
    deleteBtn.textContent = 'Deleting...';
    
    try {
        const response = await fetch(`${API_BASE}/data/${currentItem.id}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Failed to delete item');
        
        showStatus('Item deleted successfully', 'success');
        
        // Remove from items array
        items.splice(currentIndex, 1);
        
        // Load next item or previous if at end
        if (items.length === 0) {
            showStatus('All items have been deleted', 'info');
            navStatus.textContent = 'No items available';
            disableControls();
            currentItem = null;
        } else {
            if (currentIndex >= items.length) {
                currentIndex = items.length - 1;
            }
            loadItem(currentIndex);
        }
    } catch (error) {
        console.error('Error deleting item:', error);
        showStatus('Failed to delete item', 'error');
    } finally {
        deleteBtn.disabled = false;
        deleteBtn.textContent = 'Delete';
    }
}

function skipItem() {
    if (currentIndex < items.length - 1) {
        loadItem(currentIndex + 1);
    } else {
        showStatus('No more items', 'info');
    }
}

function disableControls() {
    transcriptEditor.disabled = true;
    reviewCheckbox.disabled = true;
    saveBtn.disabled = true;
    prevBtn.disabled = true;
    nextBtn.disabled = true;
    skipBtn.disabled = true;
    deleteBtn.disabled = true;
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

function base64ToBlob(base64, mimeType) {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
}
