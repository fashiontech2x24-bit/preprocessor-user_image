/* ═══════════════════════════════════════════════════════════════
   SAM3 Garment Segmentation — Test Console Logic
   ═══════════════════════════════════════════════════════════════ */

// ──────────────────────────── State ─────────────────────────────
let serverUrl = '';
let isConnected = false;
let selectedFile = null;

// ──────────────────────────── Init ──────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupUploadArea();
    setupRangeInputs();
    loadSavedUrl();
});

function loadSavedUrl() {
    const saved = localStorage.getItem('sam3_server_url');
    if (saved) {
        document.getElementById('serverUrl').value = saved;
    }
}

// ──────────────────────────── Upload Area ────────────────────────
function setupUploadArea() {
    const area = document.getElementById('uploadArea');
    const input = document.getElementById('imageInput');
    const preview = document.getElementById('imagePreview');
    const content = document.getElementById('uploadContent');

    area.addEventListener('click', () => input.click());

    area.addEventListener('dragover', (e) => {
        e.preventDefault();
        area.classList.add('dragover');
    });

    area.addEventListener('dragleave', () => {
        area.classList.remove('dragover');
    });

    area.addEventListener('drop', (e) => {
        e.preventDefault();
        area.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    input.addEventListener('change', () => {
        if (input.files.length) {
            handleFile(input.files[0]);
        }
    });
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        log('Not an image file', 'error');
        return;
    }
    selectedFile = file;
    const preview = document.getElementById('imagePreview');
    const content = document.getElementById('uploadContent');

    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        content.style.display = 'none';
    };
    reader.readAsDataURL(file);

    document.getElementById('segmentBtn').disabled = !isConnected;
    log(`Image loaded: ${file.name} (${(file.size / 1024).toFixed(0)} KB)`, 'info');
}

// ──────────────────────────── Range Inputs ───────────────────────
function setupRangeInputs() {
    const ranges = [
        { id: 'greenAlpha', valId: 'greenAlphaVal' },
        { id: 'blurRadius', valId: 'blurRadiusVal' },
        { id: 'borderWidth', valId: 'borderWidthVal' },
        { id: 'threshold', valId: 'thresholdVal' },
    ];

    ranges.forEach(({ id, valId }) => {
        const input = document.getElementById(id);
        const val = document.getElementById(valId);
        input.addEventListener('input', () => {
            val.textContent = input.value;
        });
    });
}

// ──────────────────────────── Card Toggle ────────────────────────
function toggleCard(cardId) {
    const card = document.getElementById(cardId);
    card.classList.toggle('expanded');
}

// ──────────────────────────── Logging ────────────────────────────
function log(message, type = 'info') {
    const logOutput = document.getElementById('logOutput');
    const entry = document.createElement('div');
    const time = new Date().toLocaleTimeString('en-US', { hour12: false });
    entry.className = `log-entry log-entry--${type}`;
    entry.textContent = `[${time}] ${message}`;
    logOutput.appendChild(entry);
    logOutput.scrollTop = logOutput.scrollHeight;
}

function clearLog() {
    document.getElementById('logOutput').innerHTML = '';
    log('Log cleared', 'info');
}

// ──────────────────────────── API Calls ─────────────────────────

function getBaseUrl() {
    const url = document.getElementById('serverUrl').value.trim().replace(/\/+$/, '');
    localStorage.setItem('sam3_server_url', url);
    return url;
}

function setConnected(connected, detail = '') {
    isConnected = connected;
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');

    if (connected) {
        dot.classList.add('connected');
        text.textContent = detail || 'Connected';
        document.getElementById('segmentBtn').disabled = !selectedFile;
    } else {
        dot.classList.remove('connected');
        text.textContent = detail || 'Disconnected';
        document.getElementById('segmentBtn').disabled = true;
    }
}

function showLoading(message = 'Processing...') {
    document.getElementById('loadingText').textContent = message;
    document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingOverlay').style.display = 'none';
}

// ── Health Check ──
async function checkHealth() {
    const url = getBaseUrl();
    log(`Connecting to ${url}...`, 'info');

    try {
        const res = await fetch(`${url}/health`, { method: 'GET' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        setConnected(true, `Connected (${data.device || 'unknown'})`);
        log(`✓ Connected! Device: ${data.device}, Model loaded: ${data.model_loaded}`, 'success');

        // Populate prompts
        if (data.prompts) {
            document.getElementById('promptFull').value = data.prompts.full || '';
            document.getElementById('promptUpper').value = data.prompts.upper || '';
            document.getElementById('promptLower').value = data.prompts.lower || '';
            document.getElementById('promptLayered').value = data.prompts.layered || '';
        }

        // Populate mask config
        if (data.mask_config) {
            const mc = data.mask_config;
            if (mc.green_color) document.getElementById('greenColor').value = mc.green_color.join(',');
            if (mc.green_alpha !== undefined) {
                document.getElementById('greenAlpha').value = mc.green_alpha;
                document.getElementById('greenAlphaVal').textContent = mc.green_alpha;
            }
            if (mc.blur_radius !== undefined) {
                document.getElementById('blurRadius').value = mc.blur_radius;
                document.getElementById('blurRadiusVal').textContent = mc.blur_radius;
            }
            if (mc.border_width !== undefined) {
                document.getElementById('borderWidth').value = mc.border_width;
                document.getElementById('borderWidthVal').textContent = mc.border_width;
            }
            if (mc.threshold !== undefined) {
                document.getElementById('threshold').value = mc.threshold;
                document.getElementById('thresholdVal').textContent = mc.threshold;
            }
        }

        log(`Prompts: ${JSON.stringify(data.prompts)}`, 'data');
    } catch (e) {
        setConnected(false, 'Connection failed');
        log(`✗ Connection failed: ${e.message}`, 'error');
    }
}

// ── Segmentation ──
async function runSegmentation() {
    if (!selectedFile) {
        log('No image selected', 'error');
        return;
    }
    if (!isConnected) {
        log('Not connected to server', 'error');
        return;
    }

    const url = getBaseUrl();
    const type = document.getElementById('typeSelect').value;
    const promptOverride = document.getElementById('promptOverride').value.trim();
    const showRawMask = document.getElementById('showRawMask').checked;

    log(`Running segmentation: type=${type}${promptOverride ? `, prompt="${promptOverride}"` : ''}`, 'info');
    showLoading('Segmenting garment...');

    try {
        // Main segmentation
        const formData = new FormData();
        formData.append('user_image', selectedFile);
        formData.append('type', type);
        if (promptOverride) formData.append('prompt_override', promptOverride);

        const startTime = performance.now();
        const res = await fetch(`${url}/segment`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const blob = await res.blob();
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
        const processingTime = res.headers.get('X-Processing-Time') || elapsed;
        const maskCoverage = res.headers.get('X-Mask-Coverage') || '?';
        const promptUsed = res.headers.get('X-Prompt-Used') || '(default)';

        log(`✓ Segmentation complete in ${processingTime}s | Mask coverage: ${maskCoverage}% | Prompt: "${promptUsed}"`, 'success');

        // Fetch raw mask if requested
        let rawMaskBlob = null;
        if (showRawMask) {
            log('Fetching raw mask...', 'info');
            const maskFormData = new FormData();
            maskFormData.append('user_image', selectedFile);
            maskFormData.append('type', type);
            if (promptOverride) maskFormData.append('prompt_override', promptOverride);

            const maskRes = await fetch(`${url}/segment/raw-mask`, {
                method: 'POST',
                body: maskFormData,
            });

            if (maskRes.ok) {
                rawMaskBlob = await maskRes.blob();
                log('✓ Raw mask received', 'success');
            } else {
                log('⚠ Raw mask request failed', 'warn');
            }
        }

        // Display results
        displayResults(blob, rawMaskBlob, {
            type,
            processingTime,
            maskCoverage,
            promptUsed,
        });
    } catch (e) {
        log(`✗ Segmentation failed: ${e.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function displayResults(maskedBlob, rawMaskBlob, meta) {
    const grid = document.getElementById('resultsGrid');
    const placeholder = document.getElementById('resultPlaceholder');
    if (placeholder) placeholder.remove();

    const item = document.createElement('div');
    item.className = 'result-item';

    const colCount = rawMaskBlob ? 'triple' : '';

    item.innerHTML = `
        <div class="result-item__header">
            <span class="result-item__label">${meta.type}</span>
            <div class="result-item__meta">
                <span>⏱ ${meta.processingTime}s</span>
                <span>◉ ${meta.maskCoverage}%</span>
            </div>
        </div>
        <div class="result-item__images ${colCount}">
            <div class="result-img-wrap">
                <img src="${document.getElementById('imagePreview').src}" alt="Original">
                <span class="result-img-wrap__label">Original</span>
            </div>
            <div class="result-img-wrap">
                <img src="${URL.createObjectURL(maskedBlob)}" alt="Masked">
                <span class="result-img-wrap__label">Masked</span>
            </div>
            ${rawMaskBlob ? `
            <div class="result-img-wrap">
                <img src="${URL.createObjectURL(rawMaskBlob)}" alt="Raw Mask">
                <span class="result-img-wrap__label">Raw Mask</span>
            </div>
            ` : ''}
        </div>
    `;

    grid.insertBefore(item, grid.firstChild);
}

// ── Prompts Config ──
async function savePrompts() {
    const url = getBaseUrl();
    const payload = {
        full: document.getElementById('promptFull').value.trim() || null,
        upper: document.getElementById('promptUpper').value.trim() || null,
        lower: document.getElementById('promptLower').value.trim() || null,
        layered: document.getElementById('promptLayered').value.trim() || null,
    };

    log(`Saving prompts: ${JSON.stringify(payload)}`, 'info');

    try {
        const res = await fetch(`${url}/config/prompts`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        log(`✓ Prompts saved: ${JSON.stringify(data.changes)}`, 'success');
    } catch (e) {
        log(`✗ Failed to save prompts: ${e.message}`, 'error');
    }
}

// ── Mask Config ──
async function saveMaskConfig() {
    const url = getBaseUrl();

    const greenColorStr = document.getElementById('greenColor').value.trim();
    let greenColor = null;
    if (greenColorStr) {
        greenColor = greenColorStr.split(',').map(v => parseInt(v.trim()));
        if (greenColor.length !== 3 || greenColor.some(isNaN)) {
            log('Invalid color format. Use R,G,B (e.g. 0,255,0)', 'error');
            return;
        }
    }

    const payload = {
        green_color: greenColor,
        green_alpha: parseFloat(document.getElementById('greenAlpha').value),
        blur_radius: parseInt(document.getElementById('blurRadius').value),
        border_width: parseInt(document.getElementById('borderWidth').value),
        threshold: parseFloat(document.getElementById('threshold').value),
    };

    log(`Saving mask config: ${JSON.stringify(payload)}`, 'info');

    try {
        const res = await fetch(`${url}/config/mask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        log(`✓ Mask config saved: ${JSON.stringify(data.changes)}`, 'success');
    } catch (e) {
        log(`✗ Failed to save mask config: ${e.message}`, 'error');
    }
}

// ── Reset Config ──
async function resetConfig() {
    const url = getBaseUrl();
    log('Resetting config to defaults...', 'info');

    try {
        const res = await fetch(`${url}/config/reset`, { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // Repopulate UI
        if (data.prompts) {
            document.getElementById('promptFull').value = data.prompts.full || '';
            document.getElementById('promptUpper').value = data.prompts.upper || '';
            document.getElementById('promptLower').value = data.prompts.lower || '';
            document.getElementById('promptLayered').value = data.prompts.layered || '';
        }

        log(`✓ Config reset to defaults`, 'success');
    } catch (e) {
        log(`✗ Reset failed: ${e.message}`, 'error');
    }
}
