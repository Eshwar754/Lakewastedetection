/* SmartLake AI Dashboard Frontend Controller */

document.addEventListener('DOMContentLoaded', () => {
  initSystemStatus();
  initChart();
  initEventListeners();
  fetchStatistics();
  fetchDetectionsLog();

  // Poll statistics every 5 seconds
  setInterval(fetchStatistics, 5000);
  setInterval(fetchCameraStatus, 1000);
});

let materialChart = null;

// Initialize Chart.js Donut Chart
function initChart() {
  const ctx = document.getElementById('chartMaterialBreakdown').getContext('2d');
  materialChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Plastic', 'Paper', 'Metal', 'Rubber', 'Fabric', 'Wood', 'Fishing Gear', 'Other Waste'],
      datasets: [{
        data: [0, 0, 0, 0, 0, 0, 0, 0],
        backgroundColor: [
          '#00f2fe', '#4facfe', '#7f53ac', '#f59e0b',
          '#10b981', '#647eee', '#ef4444', '#ec4899'
        ],
        borderWidth: 1,
        borderColor: '#0a0f1d'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'right',
          labels: { color: '#f0f4f8', font: { family: 'Outfit', size: 11 } }
        }
      }
    }
  });
}

// System Status Check
async function initSystemStatus() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    document.getElementById('valDeviceName').innerText = data.device || 'CPU';
  } catch (err) {
    document.getElementById('valDeviceName').innerText = 'Offline';
  }
}

// Fetch Aggregated Statistics
async function fetchStatistics() {
  try {
    const res = await fetch('/api/statistics');
    const data = await res.json();

    document.getElementById('valTotalWaste').innerText = data.total_waste || 0;
    document.getElementById('valDominantWaste').innerText = data.most_common_waste || 'None';

    // Update Pollution Level Badge
    const badge = document.getElementById('badgePollutionLevel');
    const pollLevel = data.pollution_level || 'Low';
    badge.innerText = `${pollLevel} Pollution`;
    badge.className = 'badge ' + `badge-${pollLevel.toLowerCase()}`;

    updateMaterialChart(data.category_counts);
  } catch (err) {
    console.error("Error fetching statistics:", err);
  }
}

// Fetch the current frame's live camera statistics.
async function fetchCameraStatus() {
  try {
    const res = await fetch('/api/predict/camera/status');
    if (!res.ok) return;

    const data = await res.json();
    if (!data.is_running || !data.statistics) return;

    const stats = data.statistics;
    document.getElementById('valTotalWaste').innerText = stats.total_waste || 0;
    document.getElementById('valDominantWaste').innerText = stats.most_common_waste || 'None';

    const badge = document.getElementById('badgePollutionLevel');
    const pollLevel = stats.pollution_level || 'Low';
    badge.innerText = `${pollLevel} Pollution`;
    badge.className = 'badge ' + `badge-${pollLevel.toLowerCase()}`;
    updateMaterialChart(stats.category_counts);
  } catch (err) {
    console.error("Error fetching camera status:", err);
  }
}

function updateMaterialChart(counts) {
  if (!materialChart || !counts) return;

  materialChart.data.datasets[0].data = [
    counts['Plastic'] || 0,
    counts['Paper'] || 0,
    counts['Metal'] || 0,
    counts['Rubber'] || 0,
    counts['Fabric'] || 0,
    counts['Wood'] || 0,
    counts['Fishing_Gear'] || 0,
    counts['Other_Waste'] || 0
  ];
  materialChart.update();
}

// Fetch Detections History Log
async function fetchDetectionsLog() {
  try {
    const res = await fetch('/api/detections?limit=15');
    const data = await res.json();
    const tbody = document.getElementById('tblDetectionsBody');

    if (!data || data.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" style="text-align:center; color: var(--text-muted);">No detections logged yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = data.map(item => `
      <tr>
        <td>#${item.id}</td>
        <td><code>${item.session_id}</code></td>
        <td>${item.timestamp.replace('T', ' ').substring(0, 19)}</td>
        <td><span style="text-transform: capitalize;">${item.source_type}</span></td>
        <td><strong>${item.class}</strong></td>
        <td>${Math.round(item.confidence * 100)}%</td>
        <td><code>${item.bbox}</code></td>
      </tr>
    `).join('');
  } catch (err) {
    console.error("Error fetching detection logs:", err);
  }
}

// Initialize Event Listeners
function initEventListeners() {
  const btnStartCam = document.getElementById('btnStartCam');
  const btnStopCam = document.getElementById('btnStopCam');
  const liveImg = document.getElementById('liveStreamImg');
  const placeholder = document.getElementById('placeholderStream');

  // Camera Controls
  btnStartCam.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/predict/camera/start', { method: 'POST' });
      if (!res.ok) {
        let detail = 'Camera start failed';
        try {
          const errorData = await res.json();
          detail = errorData.detail || detail;
        } catch (parseError) {
          // Keep the generic message when the server returns a non-JSON error.
        }
        throw new Error(detail);
      }
      liveImg.src = '/api/predict/camera/stream?' + new Date().getTime();
      liveImg.style.display = 'block';
      placeholder.style.display = 'none';
    } catch (err) {
      const message = err instanceof TypeError
        ? 'The SmartLake backend is not running. Start it with: python backend\\main.py'
        : err.message;
      alert(`Could not start camera stream.\n\n${message}`);
    }
  });

  btnStopCam.addEventListener('click', async () => {
    try {
      await fetch('/api/predict/camera/stop', { method: 'POST' });
      liveImg.style.display = 'none';
      liveImg.src = '';
      placeholder.style.display = 'block';
    } catch (err) {
      console.error(err);
    }
  });

  // Drag & Drop File Upload
  const dropArea = document.getElementById('dropArea');
  const fileInput = document.getElementById('fileInput');

  dropArea.addEventListener('click', () => fileInput.click());

  dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.style.borderColor = 'var(--primary-cyan)';
  });

  dropArea.addEventListener('dragleave', () => {
    dropArea.style.borderColor = 'var(--border-color)';
  });

  dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Refresh Logs Button
  document.getElementById('btnRefreshLogs').addEventListener('click', fetchDetectionsLog);

  // Generate Report Button
  document.getElementById('btnGenerateReport').addEventListener('click', async () => {
    try {
      const res = await fetch('/api/reports/generate');
      const data = await res.json();
      alert(`Report Generated Successfully!\nJSON: ${data.json_report_path}\nMarkdown: ${data.markdown_report_path}`);
    } catch (err) {
      alert("Error generating pollution report.");
    }
  });
}

// Handle Image / Video File Upload
async function handleFileUpload(file) {
  const statusDiv = document.getElementById('uploadStatus');
  const previewImg = document.getElementById('uploadPreviewImg');
  const previewPlaceholder = document.getElementById('uploadPreviewPlaceholder');

  statusDiv.innerText = `Processing ${file.name}... Please wait.`;

  const formData = new FormData();
  formData.append('file', file);

  if (file.type.startsWith('image/')) {
    try {
      const res = await fetch('/api/predict/image', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      statusDiv.innerText = `Done! Found ${data.total_objects} objects. Session ID: ${data.session_id}`;
      
      if (data.annotated_image_base64) {
        previewImg.src = data.annotated_image_base64;
        previewImg.style.display = 'block';
        previewPlaceholder.style.display = 'none';
      }

      fetchStatistics();
      fetchDetectionsLog();
    } catch (err) {
      statusDiv.innerText = `Error processing image: ${err.message}`;
    }
  } else if (file.type.startsWith('video/') || file.name.endsWith('.mp4')) {
    try {
      const res = await fetch('/api/predict/video', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      statusDiv.innerText = `Video processed (${data.frames_processed} frames in ${data.processing_time_sec}s). Total waste: ${data.statistics.total_waste}`;
      fetchStatistics();
      fetchDetectionsLog();
    } catch (err) {
      statusDiv.innerText = `Error processing video: ${err.message}`;
    }
  } else {
    statusDiv.innerText = "Unsupported file type. Please upload JPG, PNG, or MP4.";
  }
}
