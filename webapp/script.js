document.addEventListener('DOMContentLoaded', () => {

    // ============================================
    // STATE TRACKING
    // ============================================
    let capturedImageBlob = null;
    let recordedAudioBlob = null;
    let accelCsvBlob = null;
    let cameraStream = null;
    let mediaRecorder = null;
    let audioChunks = [];
    let recTimerInterval = null;
    let accelTimerInterval = null;
    let accelData = [];
    let accelListening = false;

    // Chart instances
    let metricsChart = null;
    let distributionChart = null;
    let trendsChart = null;

    // Analysis history - load from localStorage
    let analysisHistory = JSON.parse(localStorage.getItem('omniHealthHistory') || '[]');

    // ============================================
    // FILE INPUT LISTENERS
    // ============================================
    const fileImage = document.getElementById('fileImage');
    const fileAudio = document.getElementById('fileAudio');
    const fileAccel = document.getElementById('fileAccel');

    fileImage.addEventListener('change', (e) => {
        capturedImageBlob = null;
        const thumb = document.getElementById('capturedImageThumb');
        thumb.style.display = 'none';
        const name = e.target.files[0] ? e.target.files[0].name : "No file selected";
        document.getElementById('imageFileName').textContent = name;
        if (e.target.files[0]) markCardActive('card-image', true);
        else markCardActive('card-image', false);
    });

    fileAudio.addEventListener('change', (e) => {
        recordedAudioBlob = null;
        document.getElementById('audioPlayback').style.display = 'none';
        const name = e.target.files[0] ? e.target.files[0].name : "No file selected";
        document.getElementById('audioFileName').textContent = name;
        if (e.target.files[0]) markCardActive('card-audio', true);
        else markCardActive('card-audio', false);
    });

    fileAccel.addEventListener('change', (e) => {
        accelCsvBlob = null;
        const name = e.target.files[0] ? e.target.files[0].name : "No file selected";
        document.getElementById('accelFileName').textContent = name;
        if (e.target.files[0]) markCardActive('card-accel', true);
        else markCardActive('card-accel', false);
    });

    // ============================================
    // CAMERA CAPTURE (Image)
    // ============================================
    const btnCaptureImage = document.getElementById('btnCaptureImage');
    const cameraPreviewWrapper = document.getElementById('cameraPreviewWrapper');
    const cameraPreview = document.getElementById('cameraPreview');
    const btnSnap = document.getElementById('btnSnap');
    const btnCancelCam = document.getElementById('btnCancelCam');
    const cameraCanvas = document.getElementById('cameraCanvas');
    const capturedThumb = document.getElementById('capturedImageThumb');

    btnCaptureImage.addEventListener('click', async () => {
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
            cameraPreview.srcObject = cameraStream;
            cameraPreviewWrapper.style.display = 'block';
            btnCaptureImage.disabled = true;
        } catch (err) {
            showToast('Camera access denied or unavailable. Please use file upload.', 'error');
        }
    });

    btnSnap.addEventListener('click', () => {
        const ctx = cameraCanvas.getContext('2d');
        cameraCanvas.width = cameraPreview.videoWidth;
        cameraCanvas.height = cameraPreview.videoHeight;
        ctx.drawImage(cameraPreview, 0, 0);
        cameraCanvas.toBlob((blob) => {
            capturedImageBlob = blob;
            capturedThumb.src = URL.createObjectURL(blob);
            capturedThumb.style.display = 'block';
            document.getElementById('imageFileName').textContent = 'Camera capture ready';
            fileImage.value = '';
            markCardActive('card-image', true);
            stopCamera();
        }, 'image/jpeg', 0.92);
    });

    btnCancelCam.addEventListener('click', () => stopCamera());

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
        }
        cameraPreviewWrapper.style.display = 'none';
        btnCaptureImage.disabled = false;
    }

    // ============================================
    // AUDIO RECORDING (Microphone)
    // ============================================
    const btnRecordAudio = document.getElementById('btnRecordAudio');
    const btnStopRec = document.getElementById('btnStopRec');
    const recordingIndicator = document.getElementById('recordingIndicator');
    const recTimer = document.getElementById('recTimer');
    const audioPlayback = document.getElementById('audioPlayback');

    btnRecordAudio.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioChunks = [];
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
            mediaRecorder.onstop = () => {
                stream.getTracks().forEach(t => t.stop());
                recordedAudioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioPlayback.src = URL.createObjectURL(recordedAudioBlob);
                audioPlayback.style.display = 'block';
                document.getElementById('audioFileName').textContent = 'Microphone recording ready';
                fileAudio.value = '';
                markCardActive('card-audio', true);
                recordingIndicator.style.display = 'none';
                clearInterval(recTimerInterval);
                btnRecordAudio.disabled = false;
            };
            mediaRecorder.start();
            recordingIndicator.style.display = 'flex';
            btnRecordAudio.disabled = true;
            let secs = 0;
            recTimerInterval = setInterval(() => {
                secs++;
                recTimer.textContent = formatTime(secs);
            }, 1000);
        } catch (err) {
            showToast('Microphone access denied or unavailable. Please use file upload.', 'error');
        }
    });

    btnStopRec.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
    });

    // ============================================
    // ACCELEROMETER SENSOR CAPTURE
    // ============================================
    const btnRecordAccel = document.getElementById('btnRecordAccel');
    const btnStopAccel = document.getElementById('btnStopAccel');
    const accelIndicator = document.getElementById('accelIndicator');
    const accelTimerEl = document.getElementById('accelTimer');
    const accelSamplesEl = document.getElementById('accelSamples');

    btnRecordAccel.addEventListener('click', async () => {
        if (!window.DeviceMotionEvent) {
            showToast('Accelerometer sensor not available on this device. Please upload a CSV file.', 'error');
            return;
        }

        if (typeof DeviceMotionEvent.requestPermission === 'function') {
            try {
                const perm = await DeviceMotionEvent.requestPermission();
                if (perm !== 'granted') { showToast('Sensor permission denied.', 'error'); return; }
            } catch (e) { showToast('Sensor permission error.', 'error'); return; }
        }

        accelData = [];
        accelListening = true;
        accelIndicator.style.display = 'flex';
        btnRecordAccel.disabled = true;
        let secs = 0;
        accelTimerInterval = setInterval(() => {
            secs++;
            accelTimerEl.textContent = formatTime(secs);
            accelSamplesEl.textContent = accelData.length + ' samples';
        }, 1000);

        window.addEventListener('devicemotion', handleMotion);
    });

    function handleMotion(e) {
        if (!accelListening) return;
        const a = e.accelerationIncludingGravity;
        if (a) accelData.push({ x: a.x || 0, y: a.y || 0, z: a.z || 0 });
    }

    btnStopAccel.addEventListener('click', () => {
        accelListening = false;
        window.removeEventListener('devicemotion', handleMotion);
        clearInterval(accelTimerInterval);
        accelIndicator.style.display = 'none';
        btnRecordAccel.disabled = false;

        if (accelData.length > 0) {
            let csv = 'x,y,z\n';
            accelData.forEach(r => csv += `${r.x},${r.y},${r.z}\n`);
            accelCsvBlob = new Blob([csv], { type: 'text/csv' });
            document.getElementById('accelFileName').textContent = `Sensor: ${accelData.length} samples captured`;
            fileAccel.value = '';
            markCardActive('card-accel', true);
        } else {
            showToast('No accelerometer data was captured. Try moving your device or upload a CSV file.', 'warning');
        }
    });

    // ============================================
    // CARD ACTIVE STATE VISUAL FEEDBACK
    // ============================================
    function markCardActive(cardId, active) {
        const card = document.getElementById(cardId);
        if (active) card.classList.add('card-active');
        else card.classList.remove('card-active');
    }

    // ============================================
    // TOAST NOTIFICATIONS
    // ============================================
    function showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        const icons = { error: 'fa-circle-xmark', warning: 'fa-triangle-exclamation', info: 'fa-circle-info', success: 'fa-circle-check' };
        toast.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}"></i> ${message}`;
        container.appendChild(toast);
        setTimeout(() => { toast.classList.add('toast-fade'); setTimeout(() => toast.remove(), 400); }, 4000);
    }

    // ============================================
    // UTILITY
    // ============================================
    function formatTime(totalSecs) {
        const m = String(Math.floor(totalSecs / 60)).padStart(2, '0');
        const s = String(totalSecs % 60).padStart(2, '0');
        return `${m}:${s}`;
    }

    // ============================================
    // CHART INITIALIZATION
    // ============================================
    function initCharts() {
        // Metrics Bar Chart
        const metricsCtx = document.getElementById('metricsChart').getContext('2d');
        metricsChart = new Chart(metricsCtx, {
            type: 'bar',
            data: {
                labels: ['Anemia', 'Stress', 'Fatigue', 'Hydration', 'Sleep'],
                datasets: [{
                    label: 'Health Score',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(239, 68, 68, 0.7)',
                        'rgba(245, 158, 11, 0.7)',
                        'rgba(139, 92, 246, 0.7)',
                        'rgba(59, 130, 246, 0.7)',
                        'rgba(16, 185, 129, 0.7)'
                    ],
                    borderColor: [
                        'rgba(239, 68, 68, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(139, 92, 246, 1)',
                        'rgba(59, 130, 246, 1)',
                        'rgba(16, 185, 129, 1)'
                    ],
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(0, 0, 0, 0.04)' },
                        ticks: { callback: (v) => v + '%' }
                    },
                    x: {
                        grid: { display: false }
                    }
                }
            }
        });

        // Distribution Pie Chart
        const distCtx = document.getElementById('distributionChart').getContext('2d');
        distributionChart = new Chart(distCtx, {
            type: 'doughnut',
            data: {
                labels: ['Healthy', 'Attention', 'Critical'],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(245, 158, 11, 0.7)',
                        'rgba(239, 68, 68, 0.7)'
                    ],
                    borderColor: [
                        'rgba(16, 185, 129, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(239, 68, 68, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { padding: 15, usePointStyle: true }
                    }
                },
                cutout: '65%'
            }
        });

        // Trends Line Chart
        const trendsCtx = document.getElementById('trendsChart').getContext('2d');
        trendsChart = new Chart(trendsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Anemia Risk',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Stress Level',
                        data: [],
                        borderColor: '#f59e0b',
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Fatigue',
                        data: [],
                        borderColor: '#8b5cf6',
                        backgroundColor: 'rgba(139, 92, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Hydration',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        tension: 0.4,
                        fill: true
                    },
                    {
                        label: 'Sleep Quality',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4,
                        fill: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: 'rgba(0, 0, 0, 0.04)' },
                        ticks: { callback: (v) => v + '%' }
                    },
                    x: {
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // ============================================
    // UPDATE CHARTS WITH NEW DATA
    // ============================================
    function updateCharts(results) {
        // Convert results to scores (inverse of risk for healthy conditions)
        const scores = {
            anemia: results.anemia.toLowerCase().includes('healthy') ? 85 : results.anemia.toLowerCase().includes('detected') ? 35 : 50,
            stress: results.stress.toLowerCase().includes('healthy') ? 80 : results.stress.toLowerCase().includes('critical') ? 25 : 45,
            fatigue: results.fatigue.toLowerCase().includes('normal') ? 85 : results.fatigue.toLowerCase().includes('severe') ? 30 : 50,
            dehydration: results.dehydration.toLowerCase().includes('optimal') ? 90 : results.dehydration.toLowerCase().includes('dehydrated') ? 35 : 55,
            sleep_disorder: results.sleep_disorder.toLowerCase().includes('normal') ? 85 : results.sleep_disorder.toLowerCase().includes('apnea') ? 30 : 50
        };

        // Update metrics bar chart
        metricsChart.data.datasets[0].data = [
            scores.anemia,
            scores.stress,
            scores.fatigue,
            scores.dehydration,
            scores.sleep_disorder
        ];
        metricsChart.update();

        // Update distribution pie chart
        const healthyCount = Object.values(results).filter(r =>
            r.toLowerCase().includes('healthy') || r.toLowerCase().includes('normal') || r.toLowerCase().includes('none') || r.toLowerCase().includes('optimal')
        ).length;
        const criticalCount = Object.values(results).filter(r =>
            r.toLowerCase().includes('critical') || r.toLowerCase().includes('severe') || r.toLowerCase().includes('apnea') || r.toLowerCase().includes('detected')
        ).length;
        const attentionCount = 5 - healthyCount - criticalCount;

        distributionChart.data.datasets[0].data = [healthyCount, attentionCount, criticalCount];
        distributionChart.update();

        // Update trends chart with history
        updateTrendsChart();
    }

    function updateTrendsChart() {
        if (analysisHistory.length === 0) {
            trendsChart.data.labels = ['No data'];
            trendsChart.data.datasets.forEach(d => d.data = [0]);
        } else {
            // Show last 10 analyses
            const recent = analysisHistory.slice(-10);
            trendsChart.data.labels = recent.map(h => new Date(h.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}));
            trendsChart.data.datasets[0].data = recent.map(h => h.scores.anemia);
            trendsChart.data.datasets[1].data = recent.map(h => h.scores.stress);
            trendsChart.data.datasets[2].data = recent.map(h => h.scores.fatigue);
            trendsChart.data.datasets[3].data = recent.map(h => h.scores.hydration);
            trendsChart.data.datasets[4].data = recent.map(h => h.scores.sleep);
        }
        trendsChart.update();
    }

    // ============================================
    // UPDATE STATS OVERVIEW
    // ============================================
    function updateStatsOverview() {
        document.getElementById('totalAnalyses').textContent = analysisHistory.length;

        if (analysisHistory.length === 0) {
            document.getElementById('healthyMarkers').textContent = '0';
            document.getElementById('attentionNeeded').textContent = '0';
            document.getElementById('overallWellness').textContent = '--%';
            return;
        }

        // Count from latest analysis
        const latest = analysisHistory[analysisHistory.length - 1];
        let healthy = 0, attention = 0, critical = 0;

        Object.values(latest.results).forEach(r => {
            const lower = r.toLowerCase();
            if (lower.includes('healthy') || lower.includes('normal') || lower.includes('none') || lower.includes('optimal')) healthy++;
            else if (lower.includes('critical') || lower.includes('severe') || lower.includes('apnea') || lower.includes('detected')) critical++;
            else attention++;
        });

        document.getElementById('healthyMarkers').textContent = healthy;
        document.getElementById('attentionNeeded').textContent = attention + critical;

        // Calculate overall wellness
        const wellness = Math.round((healthy / 5) * 100);
        document.getElementById('overallWellness').textContent = wellness + '%';
    }

    // ============================================
    // UPDATE HISTORY TABLE
    // ============================================
    function updateHistoryTable() {
        const tbody = document.getElementById('historyBody');

        if (analysisHistory.length === 0) {
            tbody.innerHTML = '<tr class="empty-row"><td colspan="7">No analysis history yet. Run your first health analysis to see results here.</td></tr>';
            return;
        }

        // Show last 10 entries, newest first
        const recent = analysisHistory.slice(-10).reverse();
        tbody.innerHTML = recent.map(h => {
            const statusClass = h.status === 'good' ? 'status-good' : h.status === 'warning' ? 'status-warning' : 'status-critical';
            const statusIcon = h.status === 'good' ? 'fa-circle-check' : h.status === 'warning' ? 'fa-circle-exclamation' : 'fa-circle-xmark';
            const statusText = h.status === 'good' ? 'All Clear' : h.status === 'warning' ? 'Attention' : 'Critical';

            return `<tr>
                <td>${new Date(h.timestamp).toLocaleString()}</td>
                <td>${h.results.anemia}</td>
                <td>${h.results.stress}</td>
                <td>${h.results.fatigue}</td>
                <td>${h.results.dehydration}</td>
                <td>${h.results.sleep_disorder}</td>
                <td><span class="status-badge ${statusClass}"><i class="fa-solid ${statusIcon}"></i> ${statusText}</span></td>
            </tr>`;
        }).join('');
    }

    // ============================================
    // SAVE ANALYSIS TO HISTORY
    // ============================================
    function saveToHistory(results, scores) {
        // Determine overall status
        let criticalCount = 0, attentionCount = 0;
        Object.values(results).forEach(r => {
            const lower = r.toLowerCase();
            if (lower.includes('critical') || lower.includes('severe') || lower.includes('apnea') || lower.includes('detected')) criticalCount++;
            else if (!lower.includes('healthy') && !lower.includes('normal') && !lower.includes('none') && !lower.includes('optimal')) attentionCount++;
        });

        let status = 'good';
        if (criticalCount > 0) status = 'critical';
        else if (attentionCount > 0) status = 'warning';

        const entry = {
            timestamp: new Date().toISOString(),
            results: results,
            scores: scores,
            status: status
        };

        analysisHistory.push(entry);

        // Keep only last 50 entries
        if (analysisHistory.length > 50) {
            analysisHistory = analysisHistory.slice(-50);
        }

        localStorage.setItem('omniHealthHistory', JSON.stringify(analysisHistory));

        updateStatsOverview();
        updateHistoryTable();
    }

    // ============================================
    // EXPORT HISTORY
    // ============================================
    document.getElementById('btnExportHistory').addEventListener('click', () => {
        if (analysisHistory.length === 0) {
            showToast('No history to export.', 'warning');
            return;
        }

        const csvContent = 'Date,Anemia,Stress,Fatigue,Hydration,Sleep,Status\n' +
            analysisHistory.map(h =>
                `${new Date(h.timestamp).toLocaleString()},"${h.results.anemia}","${h.results.stress}","${h.results.fatigue}","${h.results.dehydration}","${h.results.sleep_disorder}",${h.status}`
            ).join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `health_history_${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('History exported successfully!', 'success');
    });

    // ============================================
    // CLEAR HISTORY
    // ============================================
    document.getElementById('btnClearHistory').addEventListener('click', () => {
        if (analysisHistory.length === 0) return;

        if (confirm('Are you sure you want to clear all analysis history? This cannot be undone.')) {
            analysisHistory = [];
            localStorage.removeItem('omniHealthHistory');
            updateStatsOverview();
            updateHistoryTable();
            updateTrendsChart();
            showToast('History cleared successfully.', 'success');
        }
    });

    // ============================================
    // FORM SUBMISSION & VALIDATION
    // ============================================
    const btnFuse = document.getElementById('btnFuse');
    const resultsPanel = document.getElementById('resultsPanel');
    const validationWarning = document.getElementById('validationWarning');
    const validationMsg = document.getElementById('validationMsg');

    btnFuse.addEventListener('click', async () => {
        // ---- INPUT VALIDATION ----
        const hasImage = (fileImage.files && fileImage.files[0]) || capturedImageBlob;
        const hasAudio = (fileAudio.files && fileAudio.files[0]) || recordedAudioBlob;
        const hasAccel = (fileAccel.files && fileAccel.files[0]) || accelCsvBlob;
        const waterRaw = document.getElementById('waterVal').value.trim();
        const sleepRaw = document.getElementById('sleepVal').value.trim();
        const hasWater = waterRaw !== '';
        const hasSleep = sleepRaw !== '';

        if (!hasImage && !hasAudio && !hasAccel && !hasWater && !hasSleep) {
            showValidationWarning('Please provide at least one input (image, audio, sensor data, or vitals) before running diagnostics.');
            showToast('No inputs provided. Upload or capture data first.', 'error');
            return;
        }

        if (hasWater) {
            const w = parseFloat(waterRaw);
            if (isNaN(w) || w < 0 || w > 20) {
                showValidationWarning('Water consumption must be between 0 and 20 liters.');
                showToast('Invalid water consumption value.', 'error');
                return;
            }
        }
        if (hasSleep) {
            const s = parseFloat(sleepRaw);
            if (isNaN(s) || s < 0 || s > 24) {
                showValidationWarning('Sleep duration must be between 0 and 24 hours.');
                showToast('Invalid sleep duration value.', 'error');
                return;
            }
        }

        hideValidationWarning();

        // ---- BUILD FORMDATA ----
        const formData = new FormData();

        if (capturedImageBlob) {
            formData.append('image', capturedImageBlob, 'camera_capture.jpg');
        } else if (fileImage.files && fileImage.files[0]) {
            formData.append('image', fileImage.files[0]);
        }

        if (recordedAudioBlob) {
            formData.append('audio', recordedAudioBlob, 'mic_recording.webm');
        } else if (fileAudio.files && fileAudio.files[0]) {
            formData.append('audio', fileAudio.files[0]);
        }

        if (accelCsvBlob) {
            formData.append('accel', accelCsvBlob, 'sensor_data.csv');
        } else if (fileAccel.files && fileAccel.files[0]) {
            formData.append('accel', fileAccel.files[0]);
        }

        formData.append('water', hasWater ? waterRaw : '');
        formData.append('sleep', hasSleep ? sleepRaw : '');

        // ---- SEND REQUEST ----
        btnFuse.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';
        btnFuse.style.pointerEvents = 'none';

        try {
            // Determine the URL based on where the frontend is hosted
            let fetchUrl = '/predict';
            if (window.location.hostname.includes('github.io') || window.location.protocol === 'file:') {
                fetchUrl = 'https://omnihealth-ai-1008998645318.europe-west1.run.app/predict';
            }

            const response = await fetch(fetchUrl, { method: 'POST', body: formData });
            
            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`Server returned ${response.status}: ${errText}`);
            }

            const data = await response.json();

            if (data.error) {
                showToast(data.error, 'error');
                showValidationWarning(data.error);
                resultsPanel.style.display = 'none';
                return;
            }

            // Build input summary
            const summaryParts = [];
            if (hasImage) summaryParts.push('<span class="summary-tag tag-image"><i class="fa-regular fa-image"></i> Image</span>');
            if (hasAudio) summaryParts.push('<span class="summary-tag tag-audio"><i class="fa-solid fa-microphone-lines"></i> Audio</span>');
            if (hasAccel) summaryParts.push('<span class="summary-tag tag-accel"><i class="fa-solid fa-person-running"></i> Accelerometer</span>');
            if (hasWater) summaryParts.push('<span class="summary-tag tag-vitals"><i class="fa-solid fa-droplet"></i> Water: ' + waterRaw + 'L</span>');
            if (hasSleep) summaryParts.push('<span class="summary-tag tag-vitals"><i class="fa-solid fa-bed"></i> Sleep: ' + sleepRaw + 'h</span>');
            document.getElementById('inputSummary').innerHTML = '<span class="summary-label">Inputs used:</span> ' + summaryParts.join(' ');

            // Reveal Results
            resultsPanel.style.display = 'block';

            const styleLabel = (val) => {
                if (!val) return `<span class="val-na">Not Assessed</span>`;
                let lower = val.toLowerCase();
                if (lower.includes("not assessed")) return `<span class="val-na">${val}</span>`;
                if (lower.includes("healthy") || lower.includes("none") || lower.includes("normal") || lower.includes("optimal")) return `<span class="val-pred">${val}</span>`;
                if (lower.includes("borderline") || lower.includes("mild") || lower.includes("suboptimal") || lower.includes("moderate")) return `<span class="val-warn">${val}</span>`;
                return `<span class="val-crit">${val}</span>`;
            };

            document.getElementById('res-anemia').querySelector('.val').innerHTML = styleLabel(data.anemia);
            document.getElementById('res-stress').querySelector('.val').innerHTML = styleLabel(data.stress);
            document.getElementById('res-fatigue').querySelector('.val').innerHTML = styleLabel(data.fatigue);
            document.getElementById('res-water').querySelector('.val').innerHTML = styleLabel(data.dehydration);
            document.getElementById('res-sleep').querySelector('.val').innerHTML = styleLabel(data.sleep_disorder);

            // Add confidence scores (simulated based on prediction certainty)
            const getConfidence = (val) => {
                if (!val || val.toLowerCase().includes('not assessed')) return '';
                if (val.toLowerCase().includes('healthy') || val.toLowerCase().includes('normal')) return 'High confidence';
                if (val.toLowerCase().includes('detected') || val.toLowerCase().includes('critical')) return 'Requires attention';
                return 'Moderate confidence';
            };

            document.getElementById('conf-anemia').textContent = getConfidence(data.anemia);
            document.getElementById('conf-stress').textContent = getConfidence(data.stress);
            document.getElementById('conf-fatigue').textContent = getConfidence(data.fatigue);
            document.getElementById('conf-water').textContent = getConfidence(data.dehydration);
            document.getElementById('conf-sleep').textContent = getConfidence(data.sleep_disorder);

            // Calculate scores for charts safely ignoring Not Assessed values
            const getScore = (val, goodKw, badKw, defaultGood, defaultBad, fallback) => {
                if (!val || val.toLowerCase().includes('not assessed')) return 0; // 0 will hide/skip it in charting if adapted
                if (val.toLowerCase().includes(goodKw)) return defaultGood;
                if (val.toLowerCase().includes(badKw)) return defaultBad;
                return fallback;
            };

            const scores = {
                anemia: getScore(data.anemia, 'healthy', 'detected', 85, 35, 50),
                stress: getScore(data.stress, 'healthy', 'critical', 80, 25, 45),
                fatigue: getScore(data.fatigue, 'normal', 'severe', 85, 30, 50),
                hydration: getScore(data.dehydration, 'optimal', 'dehydrated', 90, 35, 55),
                sleep: getScore(data.sleep_disorder, 'normal', 'apnea', 85, 30, 50)
            };

            // Update charts and save to history
            if (typeof updateCharts === 'function') updateCharts(data);
            if (typeof saveToHistory === 'function') saveToHistory(data, scores);

            window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });

        } catch (error) {
            console.error('Fetch error:', error);
            showToast(error.message || 'Failed to connect to backend engine.', 'error');
        } finally {
            btnFuse.innerHTML = '<i class="fa-solid fa-network-wired"></i> Execute Multimodal Fusion';
            btnFuse.style.pointerEvents = 'auto';
        }
    });

    function showValidationWarning(msg) {
        validationMsg.textContent = msg;
        validationWarning.style.display = 'flex';
    }

    function hideValidationWarning() {
        validationWarning.style.display = 'none';
    }

    // ============================================
    // INITIALIZATION
    // ============================================
    initCharts();
    updateStatsOverview();
    updateHistoryTable();
    updateTrendsChart();

});
