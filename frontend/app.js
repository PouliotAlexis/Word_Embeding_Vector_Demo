/**
 * Word Embedding Explorer - Main Application
 */

const API_BASE = 'http://localhost:8001/api';

// State management
const state = {
    positiveWords: [],
    negativeWords: [],
    graphWords: [],
    cloudWords: [],
    clusteringWords: [],
    journeyData: null,
    isPlaying: false
};

// DOM Elements
const elements = {
    status: document.getElementById('status'),
    loading: document.getElementById('loading'),
    toasts: document.getElementById('toasts')
};

// ============================================
// API Functions
// ============================================

async function apiCall(endpoint, options = {}) {
    showLoading(true);
    try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API Error');
        }

        return await response.json();
    } catch (error) {
        showToast(error.message, 'error');
        throw error;
    } finally {
        showLoading(false);
    }
}

async function checkHealth() {
    try {
        const data = await apiCall('/health');
        updateStatus(true, `${data.vocab_size.toLocaleString()} words`);
        return true;
    } catch {
        updateStatus(false, 'Offline');
        return false;
    }
}

// ============================================
// UI Helpers
// ============================================

function showLoading(show) {
    elements.loading.classList.toggle('active', show);
}

function updateStatus(connected, text) {
    const dot = elements.status.querySelector('.status-dot');
    const textEl = elements.status.querySelector('.status-text');

    dot.classList.toggle('connected', connected);
    textEl.textContent = text;
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    elements.toasts.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ============================================
// Navigation
// ============================================

function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const section = btn.dataset.section;

            // Update nav buttons
            navButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update sections
            document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
            document.getElementById(`section-${section}`).classList.add('active');
        });
    });
}

// ============================================
// Word Tag Management
// ============================================

function createTag(word, onRemove) {
    const tag = document.createElement('span');
    tag.className = 'word-tag';
    tag.innerHTML = `${word} <span class="remove">×</span>`;
    tag.querySelector('.remove').addEventListener('click', () => {
        tag.remove();
        onRemove(word);
    });
    return tag;
}

function initTagInput(inputId, tagsId, stateKey) {
    const input = document.getElementById(inputId);
    const tagsContainer = document.getElementById(tagsId);

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && input.value.trim()) {
            const word = input.value.trim().toLowerCase();

            if (!state[stateKey].includes(word)) {
                state[stateKey].push(word);

                const tag = createTag(word, (w) => {
                    state[stateKey] = state[stateKey].filter(x => x !== w);
                });
                tagsContainer.appendChild(tag);
            }

            input.value = '';
        }
    });
}

// ============================================
// Arithmetic Section
// ============================================

function initArithmetic() {
    initTagInput('positive-input', 'positive-tags', 'positiveWords');
    initTagInput('negative-input', 'negative-tags', 'negativeWords');

    // Presets
    document.querySelectorAll('.preset-btn[data-positive]').forEach(btn => {
        btn.addEventListener('click', () => {
            state.positiveWords = btn.dataset.positive.split(',');
            state.negativeWords = btn.dataset.negative ? btn.dataset.negative.split(',') : [];

            renderTags('positive-tags', state.positiveWords, 'positiveWords');
            renderTags('negative-tags', state.negativeWords, 'negativeWords');

            performArithmetic();
        });
    });

    document.getElementById('arithmetic-btn').addEventListener('click', performArithmetic);
}

function renderTags(containerId, words, stateKey) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    words.forEach(word => {
        const tag = createTag(word, (w) => {
            state[stateKey] = state[stateKey].filter(x => x !== w);
        });
        container.appendChild(tag);
    });
}

async function performArithmetic() {
    if (state.positiveWords.length === 0) {
        showToast('Add at least one positive word', 'error');
        return;
    }

    try {
        const data = await apiCall('/arithmetic', {
            method: 'POST',
            body: JSON.stringify({
                positive: state.positiveWords,
                negative: state.negativeWords,
                topn: 10
            })
        });

        renderResults('arithmetic-results', data.results);
    } catch (error) {
        console.error('Arithmetic error:', error);
    }
}

function renderResults(containerId, results) {
    const container = document.getElementById(containerId);

    if (results.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); text-align: center;">No results found</p>';
        return;
    }

    const maxScore = Math.max(...results.map(r => r.similarity));

    container.innerHTML = `
        <div class="result-list">
            ${results.map((r, i) => `
                <div class="result-item" style="animation-delay: ${i * 0.05}s">
                    <span class="result-word">${r.word}</span>
                    <span class="result-score">${(r.similarity * 100).toFixed(1)}%</span>
                    <div class="result-bar">
                        <div class="result-bar-fill" style="width: ${(r.similarity / maxScore) * 100}%"></div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// ============================================
// Synonyms Section
// ============================================

function initSynonyms() {
    document.getElementById('synonyms-btn').addEventListener('click', findSynonyms);

    // Allow Enter key to trigger search
    document.getElementById('synonyms-word').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') findSynonyms();
    });

    // Presets
    document.querySelectorAll('.synonyms-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('synonyms-word').value = btn.dataset.word;
            findSynonyms();
        });
    });
}

async function findSynonyms() {
    const word = document.getElementById('synonyms-word').value.trim();
    const filterSameRoot = document.getElementById('synonyms-filter').checked;

    if (!word) {
        showToast('Enter a word to find synonyms', 'error');
        return;
    }

    try {
        const data = await apiCall(`/synonyms?word=${encodeURIComponent(word)}&topn=15&filter_same_root=${filterSameRoot}`);

        renderSynonyms('synonyms-results', word, data.synonyms);
    } catch (error) {
        console.error('Synonyms error:', error);
    }
}

function renderSynonyms(containerId, originalWord, synonyms) {
    const container = document.getElementById(containerId);

    if (synonyms.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); text-align: center;">No synonyms found</p>';
        return;
    }

    const maxScore = Math.max(...synonyms.map(s => s.similarity));

    container.innerHTML = `
        <div class="synonym-header">
            <span class="synonym-original">${originalWord}</span>
            <span class="synonym-arrow">→</span>
        </div>
        <div class="result-list">
            ${synonyms.map((s, i) => `
                <div class="result-item synonym-item" style="animation-delay: ${i * 0.03}s">
                    <span class="result-word">${s.word}</span>
                    <span class="result-score">${(s.similarity * 100).toFixed(1)}%</span>
                    <div class="result-bar">
                        <div class="result-bar-fill" style="width: ${(s.similarity / maxScore) * 100}%"></div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// ============================================
// Neighbors Section
// ============================================

function initNeighbors() {
    const distanceSlider = document.getElementById('neighbors-distance');
    const distanceValue = document.getElementById('distance-value');

    distanceSlider.addEventListener('input', () => {
        distanceValue.textContent = distanceSlider.value;
    });

    document.getElementById('neighbors-btn').addEventListener('click', findNeighbors);

    // Allow Enter key to trigger search
    document.getElementById('neighbors-word').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') findNeighbors();
    });
}

async function findNeighbors() {
    const word = document.getElementById('neighbors-word').value.trim();
    const maxDistance = parseFloat(document.getElementById('neighbors-distance').value);

    if (!word) {
        showToast('Enter a word to explore', 'error');
        return;
    }

    try {
        const data = await apiCall(`/neighbors?word=${encodeURIComponent(word)}&max_distance=${maxDistance}&limit=50`);

        if (data.results.length === 0) {
            showToast('No neighbors found at this distance', 'info');
            return;
        }

        // Prepare words for visualization
        const words = [word, ...data.results.map(r => r.word)];

        // Get 2D coordinates
        const coordData = await apiCall('/reduce', {
            method: 'POST',
            body: JSON.stringify({
                words: words,
                method: 'pca',
                n_components: 2
            })
        });

        renderNeighborsViz(coordData.points, word, data.results);
    } catch (error) {
        console.error('Neighbors error:', error);
    }
}

// ============================================
// Graph Section
// ============================================

function initGraph() {
    initTagInput('graph-input', 'graph-tags', 'graphWords');

    // Presets
    document.querySelectorAll('.graph-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            state.graphWords = btn.dataset.words.split(',');
            renderTags('graph-tags', state.graphWords, 'graphWords');
            buildGraph();
        });
    });

    document.getElementById('graph-btn').addEventListener('click', buildGraph);
}

async function buildGraph() {
    if (state.graphWords.length < 2) {
        showToast('Add at least 2 words', 'error');
        return;
    }

    try {
        const data = await apiCall('/graph', {
            method: 'POST',
            body: JSON.stringify({
                words: state.graphWords,
                max_intermediates: 3
            })
        });

        renderGraph(data.graph);
    } catch (error) {
        console.error('Graph error:', error);
    }
}

// ============================================
// Analogy Section
// ============================================

function initAnalogy() {
    // Presets
    document.querySelectorAll('.analogy-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('analogy-a').value = btn.dataset.a;
            document.getElementById('analogy-b').value = btn.dataset.b;
            document.getElementById('analogy-c').value = btn.dataset.c;
            solveAnalogy();
        });
    });

    document.getElementById('analogy-btn').addEventListener('click', solveAnalogy);

    // Enter key on any input
    ['analogy-a', 'analogy-b', 'analogy-c'].forEach(id => {
        document.getElementById(id).addEventListener('keydown', (e) => {
            if (e.key === 'Enter') solveAnalogy();
        });
    });
}

async function solveAnalogy() {
    const word1 = document.getElementById('analogy-a').value.trim();
    const word2 = document.getElementById('analogy-b').value.trim();
    const word3 = document.getElementById('analogy-c').value.trim();

    if (!word1 || !word2 || !word3) {
        showToast('Fill all three words', 'error');
        return;
    }

    try {
        const data = await apiCall('/analogy', {
            method: 'POST',
            body: JSON.stringify({ word1, word2, word3, topn: 5 })
        });

        if (data.results.length > 0) {
            const resultBox = document.getElementById('analogy-result');
            resultBox.textContent = data.results[0].word;
            resultBox.classList.add('solved');

            // Show all results
            renderResults('analogy-results', data.results);
        }
    } catch (error) {
        console.error('Analogy error:', error);
    }
}

// ============================================
// Journey Section
// ============================================

function initJourney() {
    // Presets
    document.querySelectorAll('.journey-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('journey-start').value = btn.dataset.start;
            document.getElementById('journey-end').value = btn.dataset.end;
            startJourney();
        });
    });

    document.getElementById('journey-btn').addEventListener('click', startJourney);

    // Slider control
    const slider = document.getElementById('journey-slider');
    slider.addEventListener('input', () => {
        updateJourneyStep(parseInt(slider.value));
    });

    // Play button
    document.getElementById('journey-play').addEventListener('click', toggleJourneyPlay);
}

async function startJourney() {
    const start = document.getElementById('journey-start').value.trim();
    const end = document.getElementById('journey-end').value.trim();

    if (!start || !end) {
        showToast('Enter start and end words', 'error');
        return;
    }

    try {
        const data = await apiCall('/journey', {
            method: 'POST',
            body: JSON.stringify({ start, end, steps: 10 })
        });

        state.journeyData = data.journey;

        // Update slider
        const slider = document.getElementById('journey-slider');
        slider.max = data.journey.length - 1;
        slider.value = 0;

        renderJourney(data.journey);
        updateJourneyStep(0);
    } catch (error) {
        console.error('Journey error:', error);
    }
}

function renderJourney(journey) {
    const container = document.getElementById('journey-timeline');

    container.innerHTML = journey.map((step, i) => `
        <div class="journey-step" data-step="${i}">
            <span class="journey-step-word">${step.closest_words[0]?.word || '?'}</span>
            <span class="journey-step-progress">${Math.round(step.t * 100)}%</span>
        </div>
    `).join('');
}

function updateJourneyStep(step) {
    document.querySelectorAll('.journey-step').forEach((el, i) => {
        el.classList.toggle('active', i === step);
    });
}

function toggleJourneyPlay() {
    if (!state.journeyData) return;

    state.isPlaying = !state.isPlaying;
    const btn = document.getElementById('journey-play');
    btn.textContent = state.isPlaying ? '⏸' : '▶';

    if (state.isPlaying) {
        playJourney();
    }
}

function playJourney() {
    if (!state.isPlaying) return;

    const slider = document.getElementById('journey-slider');
    let current = parseInt(slider.value);

    if (current >= state.journeyData.length - 1) {
        current = 0;
    } else {
        current++;
    }

    slider.value = current;
    updateJourneyStep(current);

    if (current < state.journeyData.length - 1) {
        setTimeout(playJourney, 500);
    } else {
        state.isPlaying = false;
        document.getElementById('journey-play').textContent = '▶';
    }
}

// ============================================
// 3D Cloud Section
// ============================================

function initCloud() {
    initTagInput('cloud-input', 'cloud-tags', 'cloudWords');

    // Presets
    document.querySelectorAll('.cloud-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            state.cloudWords = btn.dataset.words.split(',');
            renderTags('cloud-tags', state.cloudWords, 'cloudWords');
            generateCloud();
        });
    });

    document.getElementById('cloud-btn').addEventListener('click', generateCloud);
}

async function generateCloud() {
    if (state.cloudWords.length === 0) {
        showToast('Add seed words for the cloud', 'error');
        return;
    }

    try {
        const data = await apiCall('/cloud', {
            method: 'POST',
            body: JSON.stringify({
                seed_words: state.cloudWords,
                expand: 15
            })
        });

        render3DCloud(data.points, state.cloudWords);
    } catch (error) {
        console.error('Cloud error:', error);
    }
}

// ============================================
// Clustering Section
// ============================================

const clusterColors = ['#6366f1', '#ec4899', '#10b981', '#f59e0b', '#8b5cf6', '#ef4444'];

function initClustering() {
    initTagInput('clustering-input', 'clustering-tags', 'clusteringWords');

    const countSlider = document.getElementById('clustering-count');
    const countValue = document.getElementById('clustering-count-value');

    countSlider.addEventListener('input', () => {
        countValue.textContent = countSlider.value;
    });

    // Presets
    document.querySelectorAll('.clustering-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            state.clusteringWords = btn.dataset.words.split(',');
            renderTags('clustering-tags', state.clusteringWords, 'clusteringWords');
            performClustering();
        });
    });

    document.getElementById('clustering-btn').addEventListener('click', performClustering);
}

async function performClustering() {
    if (state.clusteringWords.length < 4) {
        showToast('Add at least 4 words to cluster', 'error');
        return;
    }

    const nClusters = parseInt(document.getElementById('clustering-count').value);

    try {
        const data = await apiCall('/cluster-labeled', {
            method: 'POST',
            body: JSON.stringify({
                words: state.clusteringWords,
                n_clusters: nClusters
            })
        });

        renderClusters('clustering-results', data.clusters);
    } catch (error) {
        console.error('Clustering error:', error);
    }
}

function renderClusters(containerId, clusters) {
    const container = document.getElementById(containerId);

    container.innerHTML = `
        <div class="clusters-grid">
            ${clusters.map((cluster, i) => `
                <div class="cluster-group" style="--cluster-color: ${clusterColors[cluster.color_index % clusterColors.length]}">
                    <div class="cluster-header">
                        <span class="cluster-label">${cluster.label}</span>
                        <span class="cluster-count">${cluster.words.length} words</span>
                    </div>
                    <div class="cluster-words">
                        ${cluster.words.map(w => `<span class="cluster-word">${w}</span>`).join('')}
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// ============================================
// Semantic Search Section
// ============================================

function initSearch() {
    document.getElementById('search-btn').addEventListener('click', performSearch);

    document.getElementById('search-query').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') performSearch();
    });

    // Presets
    document.querySelectorAll('.search-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('search-query').value = btn.dataset.query;
            performSearch();
        });
    });
}

async function performSearch() {
    const query = document.getElementById('search-query').value.trim();

    if (!query) {
        showToast('Enter a search query', 'error');
        return;
    }

    try {
        const data = await apiCall(`/semantic-search?query=${encodeURIComponent(query)}&topn=20`);

        renderSearchResults('search-results', query, data.results);
    } catch (error) {
        console.error('Search error:', error);
    }
}

function renderSearchResults(containerId, query, results) {
    const container = document.getElementById(containerId);

    if (results.length === 0) {
        container.innerHTML = '<p style="color: var(--text-muted); text-align: center;">No results found</p>';
        return;
    }

    const maxScore = Math.max(...results.map(r => r.similarity));

    container.innerHTML = `
        <div class="search-header">
            <span class="search-query-label">Results for:</span>
            <span class="search-query-value">"${query}"</span>
        </div>
        <div class="result-list">
            ${results.map((r, i) => `
                <div class="result-item" style="animation-delay: ${i * 0.03}s">
                    <span class="result-word">${r.word}</span>
                    <span class="result-score">${(r.similarity * 100).toFixed(1)}%</span>
                    <div class="result-bar">
                        <div class="result-bar-fill" style="width: ${(r.similarity / maxScore) * 100}%"></div>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// ============================================
// Relationships Wheel Section
// ============================================

function initWheel() {
    document.getElementById('wheel-btn').addEventListener('click', showWheel);

    document.getElementById('wheel-word').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') showWheel();
    });

    // Presets
    document.querySelectorAll('.wheel-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('wheel-word').value = btn.dataset.word;
            showWheel();
        });
    });
}

async function showWheel() {
    const word = document.getElementById('wheel-word').value.trim();

    if (!word) {
        showToast('Enter a word to explore', 'error');
        return;
    }

    try {
        const data = await apiCall(`/relationships?word=${encodeURIComponent(word)}&n_per_category=6`);

        renderWheel('wheel-container', data);
    } catch (error) {
        console.error('Wheel error:', error);
    }
}

function renderWheel(containerId, data) {
    const container = document.getElementById(containerId);

    const categories = [
        { key: 'very_similar', label: 'Very Similar', color: '#10b981', icon: '🎯' },
        { key: 'similar', label: 'Similar', color: '#6366f1', icon: '🔗' },
        { key: 'related', label: 'Related', color: '#8b5cf6', icon: '🌐' },
        { key: 'distant', label: 'Distant', color: '#f59e0b', icon: '🌟' }
    ];

    container.innerHTML = `
        <div class="wheel-visual">
            <div class="wheel-center">
                <span class="wheel-center-word">${data.center}</span>
            </div>
            ${categories.map((cat, i) => `
                <div class="wheel-ring" style="--ring-index: ${i}; --ring-color: ${cat.color}">
                    <div class="wheel-ring-label">${cat.icon} ${cat.label}</div>
                    <div class="wheel-ring-words">
                        ${(data[cat.key] || []).map(w => `
                            <span class="wheel-word" style="--word-color: ${cat.color}">${w.word}</span>
                        `).join('')}
                    </div>
                </div>
            `).join('')}
        </div>
    `;
}

// ============================================
// Vector Inspector Section
// ============================================

function initInspector() {
    document.getElementById('inspector-btn').addEventListener('click', inspectVector);

    document.getElementById('inspector-word').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') inspectVector();
    });

    document.getElementById('inspector-compare').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') inspectVector();
    });

    // Presets
    document.querySelectorAll('.inspector-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('inspector-word').value = btn.dataset.word;
            document.getElementById('inspector-compare').value = btn.dataset.compare || '';
            inspectVector();
        });
    });
}

async function inspectVector() {
    const word = document.getElementById('inspector-word').value.trim();
    const compare = document.getElementById('inspector-compare').value.trim();

    if (!word) {
        showToast('Enter a word to inspect', 'error');
        return;
    }

    try {
        let url = `/inspect?word=${encodeURIComponent(word)}&top_n=15`;
        if (compare) {
            url += `&compare=${encodeURIComponent(compare)}`;
        }

        const data = await apiCall(url);
        renderInspector('inspector-results', data);
    } catch (error) {
        console.error('Inspector error:', error);
    }
}

function renderInspector(containerId, data) {
    const container = document.getElementById(containerId);

    const maxAbsValue = Math.max(
        Math.abs(data.min),
        Math.abs(data.max)
    );

    // Render all 100 dimensions as a heatmap/bar chart
    const allValuesHtml = data.all_values.map((v, i) => {
        const normalized = v / maxAbsValue;
        const color = v >= 0 ?
            `rgba(16, 185, 129, ${Math.abs(normalized)})` :
            `rgba(239, 68, 68, ${Math.abs(normalized)})`;
        return `<div class="dim-cell" title="Dim ${i}: ${v.toFixed(4)}" style="background: ${color}"></div>`;
    }).join('');

    // Comparison section
    let comparisonHtml = '';
    if (data.comparison) {
        comparisonHtml = `
            <div class="inspector-section">
                <h4>📊 Comparison with "${data.comparison.word}"</h4>
                <div class="comparison-similarity">
                    Similarity: <span class="similarity-value">${(data.comparison.similarity * 100).toFixed(1)}%</span>
                </div>
                <div class="differences-grid">
                    ${data.comparison.biggest_differences.slice(0, 10).map(d => `
                        <div class="diff-item">
                            <span class="dim-label">Dim ${d.dim}</span>
                            <div class="diff-bar">
                                <div class="diff-bar-word1" style="width: ${50 + (d.word1_value / maxAbsValue) * 50}%">
                                    ${data.word}: ${d.word1_value.toFixed(2)}
                                </div>
                            </div>
                            <div class="diff-bar">
                                <div class="diff-bar-word2" style="width: ${50 + (d.word2_value / maxAbsValue) * 50}%">
                                    ${data.comparison.word}: ${d.word2_value.toFixed(2)}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
    }

    container.innerHTML = `
        <div class="inspector-content">
            <div class="inspector-header">
                <div class="inspector-word-title">"${data.word}"</div>
                <div class="inspector-stats">
                    <span><strong>${data.dimensions}</strong> dimensions</span>
                    <span>Mean: <strong>${data.mean.toFixed(3)}</strong></span>
                    <span>Std: <strong>${data.std.toFixed(3)}</strong></span>
                    <span>Range: <strong>[${data.min.toFixed(2)}, ${data.max.toFixed(2)}]</strong></span>
                </div>
            </div>

            <div class="inspector-section">
                <h4>🎨 Vector Heatmap (hover for values)</h4>
                <p class="section-hint">Each cell = 1 dimension. Green = positive, Red = negative. Intensity = magnitude.</p>
                <div class="vector-heatmap">
                    ${allValuesHtml}
                </div>
            </div>

            <div class="inspector-columns">
                <div class="inspector-section">
                    <h4>📈 Top Positive Dimensions</h4>
                    <div class="dim-list">
                        ${data.top_positive.slice(0, 10).map(d => `
                            <div class="dim-item positive">
                                <div class="dim-header">
                                    <span class="dim-index">Dim ${d.dim}</span>
                                    <span class="dim-value">+${d.value.toFixed(3)}</span>
                                </div>
                                <div class="dim-bar-container">
                                    <div class="dim-bar positive" style="width: ${(d.value / data.max) * 100}%"></div>
                                </div>
                                ${d.interpretation ? `
                                    <div class="dim-concepts">
                                        <span class="concept-label">High:</span> ${d.interpretation.positive_words.slice(0, 3).join(', ')}
                                    </div>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>

                <div class="inspector-section">
                    <h4>📉 Top Negative Dimensions</h4>
                    <div class="dim-list">
                        ${data.top_negative.slice(0, 10).map(d => `
                            <div class="dim-item negative">
                                <div class="dim-header">
                                    <span class="dim-index">Dim ${d.dim}</span>
                                    <span class="dim-value">${d.value.toFixed(3)}</span>
                                </div>
                                <div class="dim-bar-container">
                                    <div class="dim-bar negative" style="width: ${(Math.abs(d.value) / Math.abs(data.min)) * 100}%"></div>
                                </div>
                                ${d.interpretation ? `
                                    <div class="dim-concepts">
                                        <span class="concept-label">Low:</span> ${d.interpretation.negative_words.slice(0, 3).join(', ')}
                                    </div>
                                ` : ''}
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>

            ${comparisonHtml}
        </div>
    `;
}

// ============================================
// Background Particles
// ============================================

function initParticles() {
    const container = document.getElementById('particles');

    for (let i = 0; i < 30; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.top = `${Math.random() * 100}%`;
        particle.style.animationDelay = `${Math.random() * 20}s`;
        particle.style.animationDuration = `${15 + Math.random() * 10}s`;
        container.appendChild(particle);
    }
}

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', async () => {
    initNavigation();
    initParticles();
    initArithmetic();
    initSynonyms();
    initNeighbors();
    initGraph();
    initAnalogy();
    initJourney();
    initCloud();
    initClustering();
    initSearch();
    initWheel();
    initInspector();

    // Check API health
    const healthy = await checkHealth();

    if (!healthy) {
        showToast('Backend server not running. Start it with: uvicorn backend.server:app --reload', 'error');
    }
});
