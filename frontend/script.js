/* SatQuery AI — Interactive Vision-Language Assistant Client Logic */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  const state = {
    sessionId: null,
    currentImageId: null,
    currentImageUrl: null,
    currentMetadata: null,
    activeLayer: 'original',
    masks: {},
    statistics: { vegetation: 0, water: 0, built_up: 0, barren: 0, roads_linear: 0 },
    isCompareMode: false,
    compareT1Id: null,
    compareT2Id: null,
    chartInstance: null,
    mapInstance: null,
    mapMarker: null,
    mapBoundsPolygon: null
  };

  // DOM Elements
  const dom = {
    // Dropzone & Inputs
    uploadDropzone: document.getElementById('uploadDropzone'),
    imageFileInput: document.getElementById('imageFileInput'),
    btnNewSession: document.getElementById('btnNewSession'),
    presetsList: document.getElementById('presetsList'),
    historyList: document.getElementById('historyList'),
    btnRefreshHistory: document.getElementById('btnRefreshHistory'),
    
    // Viewer Elements
    viewerPlaceholder: document.getElementById('viewerPlaceholder'),
    imageLayersWrapper: document.getElementById('imageLayersWrapper'),
    baseImageElement: document.getElementById('baseImageElement'),
    overlayImageElement: document.getElementById('overlayImageElement'),
    viewerLoader: document.getElementById('viewerLoader'),
    loaderStatusText: document.getElementById('loaderStatusText'),
    layerTabs: document.getElementById('layerTabs'),
    layerOpacitySlider: document.getElementById('layerOpacitySlider'),
    opacityVal: document.getElementById('opacityVal'),
    btnQuickSampleDemo: document.getElementById('btnQuickSampleDemo'),
    compareViewContainer: document.getElementById('compareViewContainer'),
    compareImgT1: document.getElementById('compareImgT1'),
    compareImgT2: document.getElementById('compareImgT2'),
    compareOverlayImg: document.getElementById('compareOverlayImg'),

    // Chat Elements
    chatStream: document.getElementById('chatStream'),
    queryInput: document.getElementById('queryInput'),
    btnSendQuery: document.getElementById('btnSendQuery'),
    btnClearChat: document.getElementById('btnClearChat'),
    quickQueries: document.getElementById('quickQueries'),

    // Right Panel Elements
    metaDimensions: document.getElementById('metaDimensions'),
    metaFileSize: document.getElementById('metaFileSize'),
    metaFormat: document.getElementById('metaFormat'),
    metaChannels: document.getElementById('metaChannels'),
    geoStatusBadge: document.getElementById('geoStatusBadge'),
    mapOverlayStatus: document.getElementById('mapOverlayStatus'),
    confidenceBadge: document.getElementById('confidenceBadge'),
    landCoverChart: document.getElementById('landCoverChart'),
    pctVegetation: document.getElementById('pctVegetation'),
    pctWater: document.getElementById('pctWater'),
    pctBuiltup: document.getElementById('pctBuiltup'),
    pctBarren: document.getElementById('pctBarren'),
    barVegetation: document.getElementById('barVegetation'),
    barWater: document.getElementById('barWater'),
    barBuiltup: document.getElementById('barBuiltup'),
    barBarren: document.getElementById('barBarren'),
    clustersTags: document.getElementById('clustersTags'),

    // Action Buttons & Modals
    btnDownloadReport: document.getElementById('btnDownloadReport'),
    btnToggleCompareMode: document.getElementById('btnToggleCompareMode'),
    compareBtnText: document.getElementById('compareBtnText'),
    compareModal: document.getElementById('compareModal'),
    btnCloseCompareModal: document.getElementById('btnCloseCompareModal'),
    btnCancelCompare: document.getElementById('btnCancelCompare'),
    btnExecuteCompare: document.getElementById('btnExecuteCompare'),
    selectCompareT1: document.getElementById('selectCompareT1'),
    selectCompareT2: document.getElementById('selectCompareT2'),
    btnLoadTemporalBenchmark: document.getElementById('btnLoadTemporalBenchmark'),
    btnHelpModal: document.getElementById('btnHelpModal'),
    infoModal: document.getElementById('infoModal'),
    btnCloseInfoModal: document.getElementById('btnCloseInfoModal'),
    engineStatusBadge: document.getElementById('engineStatusBadge')
  };

  // 1. Initialize UI & Libraries
  function initApp() {
    if (window.lucide) {
      window.lucide.createIcons();
    }
    initLeafletMap();
    initChart();
    fetchSamplePresets();
    fetchHistory();
    setupEventListeners();
  }

  // 2. Leaflet Map Setup
  function initLeafletMap() {
    state.mapInstance = L.map('leafletMap', {
      zoomControl: false,
      attributionControl: false
    }).setView([20.5937, 78.9629], 4); // Center of India

    // High quality satellite tile layer (Esri World Imagery)
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      maxZoom: 18
    }).addTo(state.mapInstance);

    L.control.zoom({ position: 'bottomright' }).addTo(state.mapInstance);
  }

  function updateMapLocation(lat, lon, bbox) {
    if (lat && lon) {
      dom.mapOverlayStatus.classList.add('hidden');
      dom.geoStatusBadge.textContent = `${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E`;
      dom.geoStatusBadge.style.color = '#38bdf8';

      state.mapInstance.setView([lat, lon], 12);

      if (state.mapMarker) {
        state.mapMarker.setLatLng([lat, lon]);
      } else {
        const customIcon = L.divIcon({
          className: 'custom-map-pin',
          html: `<div style="background:#0ea5e9;width:14px;height:14px;border-radius:50%;border:2px solid #fff;box-shadow:0 0 10px #0ea5e9;"></div>`,
          iconSize: [14, 14],
          iconAnchor: [7, 7]
        });
        state.mapMarker = L.marker([lat, lon], { icon: customIcon }).addTo(state.mapInstance);
      }

      // Draw bounding box if present
      if (bbox && bbox.length === 4) {
        if (state.mapBoundsPolygon) {
          state.mapInstance.removeLayer(state.mapBoundsPolygon);
        }
        const bounds = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]];
        state.mapBoundsPolygon = L.rectangle(bounds, {
          color: '#38bdf8',
          weight: 2,
          fillOpacity: 0.2
        }).addTo(state.mapInstance);
        state.mapInstance.fitBounds(bounds);
      }
      setTimeout(() => state.mapInstance.invalidateSize(), 200);
    } else {
      dom.mapOverlayStatus.classList.remove('hidden');
      dom.geoStatusBadge.textContent = 'Optical Only';
      dom.geoStatusBadge.style.color = 'var(--text-muted)';
      if (state.mapMarker) {
        state.mapInstance.removeLayer(state.mapMarker);
        state.mapMarker = null;
      }
      if (state.mapBoundsPolygon) {
        state.mapInstance.removeLayer(state.mapBoundsPolygon);
        state.mapBoundsPolygon = null;
      }
    }
  }

  // 3. Chart.js Donut Chart Setup
  function initChart() {
    const ctx = dom.landCoverChart.getContext('2d');
    state.chartInstance = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Vegetation', 'Water', 'Built-Up', 'Barren / Soil'],
        datasets: [{
          data: [25, 25, 25, 25],
          backgroundColor: ['#10b981', '#0ea5e9', '#f97316', '#a8a29e'],
          borderColor: '#0f172a',
          borderWidth: 2,
          hoverOffset: 4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '70%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 10,
              color: '#94a3b8',
              font: { size: 10, family: 'Inter' }
            }
          }
        }
      }
    });
  }

  function updateChartData(stats) {
    if (!state.chartInstance) return;
    state.chartInstance.data.datasets[0].data = [
      stats.vegetation || 0,
      stats.water || 0,
      stats.built_up || 0,
      stats.barren || 0
    ];
    state.chartInstance.update();

    // Update Numeric Metric Cards
    dom.pctVegetation.textContent = `${stats.vegetation}%`;
    dom.pctWater.textContent = `${stats.water}%`;
    dom.pctBuiltup.textContent = `${stats.built_up}%`;
    dom.pctBarren.textContent = `${stats.barren}%`;

    dom.barVegetation.style.width = `${Math.min(100, stats.vegetation)}%`;
    dom.barWater.style.width = `${Math.min(100, stats.water)}%`;
    dom.barBuiltup.style.width = `${Math.min(100, stats.built_up)}%`;
    dom.barBarren.style.width = `${Math.min(100, stats.barren)}%`;

    if (stats.confidence) {
      dom.confidenceBadge.textContent = `Est. ${Math.round(stats.confidence * 100)}%`;
    }
  }

  // 4. API Requests & Uploads
  async function uploadFile(file) {
    showLoader("Uploading and parsing raster metadata...");
    const formData = new FormData();
    formData.append("file", file);
    if (state.sessionId) {
      formData.append("session_id", state.sessionId);
    }

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }
      const data = await res.json();
      handleImageLoaded(data);
    } catch (e) {
      hideLoader();
      appendSystemMessage(`Upload Error: ${e.message}`, true);
    }
  }

  async function loadSample(filename) {
    showLoader(`Loading calibrated preset: ${filename}...`);
    const formData = new FormData();
    formData.append("sample_filename", filename);
    if (state.sessionId) {
      formData.append("session_id", state.sessionId);
    }

    try {
      const res = await fetch("/api/sample/load", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Failed to load sample");
      }
      const data = await res.json();
      handleImageLoaded(data);
    } catch (e) {
      hideLoader();
      appendSystemMessage(`Sample Load Error: ${e.message}`, true);
    }
  }

  function handleImageLoaded(data) {
    state.sessionId = data.session_id;
    state.currentImageId = data.image_id;
    state.currentImageUrl = data.url;
    state.currentMetadata = data.metadata;

    // Update Metadata Cards
    dom.metaDimensions.textContent = `${data.metadata.width} × ${data.metadata.height} px`;
    dom.metaFileSize.textContent = `${(data.metadata.file_size / 1024).toFixed(1)} KB`;
    dom.metaFormat.textContent = `${data.metadata.format} (${data.metadata.color_mode})`;
    dom.metaChannels.textContent = `${data.metadata.channels}-Band Raster`;

    // Update Geolocation Map
    if (data.metadata.has_georeference) {
      updateMapLocation(data.metadata.latitude, data.metadata.longitude, data.metadata.bounding_box);
    } else {
      updateMapLocation(null, null, null);
    }

    // Display image in Stage
    dom.viewerPlaceholder.classList.add("hidden");
    dom.compareViewContainer.classList.add("hidden");
    dom.imageLayersWrapper.classList.remove("hidden");
    dom.baseImageElement.src = data.url;
    dom.overlayImageElement.classList.add("hidden");
    dom.btnDownloadReport.disabled = false;

    // Trigger full automated analysis
    runAnalysis(data.image_id);
    fetchHistory();
  }

  async function runAnalysis(imageId) {
    showLoader("Extracting spectral indices (VARI, NDWI, Texture)...");
    const formData = new FormData();
    formData.append("image_id", imageId);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Analysis failed");
      }
      const data = await res.json();
      hideLoader();

      state.statistics = data.statistics;
      state.masks = data.masks;

      updateChartData(data.statistics);
      renderDetectedRegions(data.detected_regions);

      // Append introductory AI observation
      appendAiMessage(data.answer, data.intent);
    } catch (e) {
      hideLoader();
      appendSystemMessage(`Analysis Error: ${e.message}`, true);
    }
  }

  // 5. Query Handling
  async function submitQuery(queryText) {
    if (!queryText || !queryText.trim()) return;
    if (!state.currentImageId) {
      appendSystemMessage("Please upload or load a satellite image before submitting queries.", true);
      return;
    }

    appendUserMessage(queryText);
    dom.queryInput.value = "";
    showTypingIndicator();

    try {
      const payload = {
        image_id: state.currentImageId,
        query: queryText,
        session_id: state.sessionId
      };

      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      removeTypingIndicator();

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query processing error");
      }

      const data = await res.json();
      state.statistics = data.statistics;
      if (data.masks) {
        state.masks = data.masks;
      }

      updateChartData(data.statistics);
      if (data.detected_regions) {
        renderDetectedRegions(data.detected_regions);
      }

      appendAiMessage(data.answer, data.intent);

      // Automatically activate corresponding layer overlay for visual feedback!
      activateLayerForIntent(data.intent);

    } catch (e) {
      removeTypingIndicator();
      appendSystemMessage(`Query Error: ${e.message}`, true);
    }
  }

  function activateLayerForIntent(intent) {
    let targetLayer = null;
    if (intent === 'VEGETATION') targetLayer = 'vegetation';
    else if (intent === 'WATER') targetLayer = 'water';
    else if (intent === 'BUILT_UP_AREA') targetLayer = 'builtup';
    else if (intent === 'ROAD') targetLayer = 'roads';
    else if (intent === 'LAND_COVER') targetLayer = 'segmentation';

    if (targetLayer) {
      switchLayer(targetLayer);
    }
  }

  // 6. Layer Switching & Opacity Slider
  function switchLayer(layerName) {
    state.activeLayer = layerName;

    // Update active tab button
    document.querySelectorAll('.layer-tab').forEach(tab => {
      tab.classList.toggle('active', tab.dataset.layer === layerName);
    });

    if (layerName === 'original') {
      dom.overlayImageElement.classList.add('hidden');
      return;
    }

    const maskUrl = state.masks[layerName];
    if (maskUrl) {
      dom.overlayImageElement.src = maskUrl;
      dom.overlayImageElement.classList.remove('hidden');
      applyOpacity();
    } else {
      dom.overlayImageElement.classList.add('hidden');
    }
  }

  function applyOpacity() {
    const val = dom.layerOpacitySlider.value;
    dom.opacityVal.textContent = `${val}%`;
    dom.overlayImageElement.style.opacity = val / 100.0;
  }

  // 7. Dual-Image Change Detection
  async function executeComparison(img1Id, img2Id) {
    showLoader("Aligning rasters & computing temporal change map...");
    closeModal(dom.compareModal);

    try {
      const payload = {
        image_id_1: img1Id,
        image_id_2: img2Id,
        session_id: state.sessionId
      };

      const res = await fetch("/api/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      hideLoader();

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Comparison failed");
      }

      const data = await res.json();
      state.masks['change'] = data.change_mask_url;

      // Enable change tab
      const changeTab = document.querySelector('[data-layer="change"]');
      if (changeTab) changeTab.classList.remove('hidden');

      // Display in compare view container
      dom.viewerPlaceholder.classList.add('hidden');
      dom.imageLayersWrapper.classList.add('hidden');
      dom.compareViewContainer.classList.remove('hidden');

      dom.compareImgT1.src = `/api/image/${img1Id}`;
      dom.compareImgT2.src = `/api/image/${img2Id}`;
      dom.compareOverlayImg.src = data.change_mask_url;
      dom.compareOverlayImg.classList.remove('hidden');

      // Add to chat
      appendAiMessage(data.answer, 'CHANGE_DETECTION');
      dom.btnDownloadReport.disabled = false;

    } catch (e) {
      hideLoader();
      appendSystemMessage(`Compare Error: ${e.message}`, true);
    }
  }

  // 8. Sample Presets and History Fetching
  async function fetchSamplePresets() {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) return;
      const samples = await res.json();

      dom.presetsList.innerHTML = '';
      samples.forEach(s => {
        const card = document.createElement('div');
        card.className = 'preset-card';
        card.innerHTML = `
          <span class="preset-name">${s.name}</span>
          <span class="preset-desc">${s.description}</span>
        `;
        card.addEventListener('click', () => {
          document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
          card.classList.add('active');
          loadSample(s.filename);
        });
        dom.presetsList.appendChild(card);
      });
    } catch (e) {
      console.warn("Could not fetch presets:", e);
    }
  }

  async function fetchHistory() {
    try {
      const res = await fetch('/api/history');
      if (!res.ok) return;
      const history = await res.json();

      dom.historyList.innerHTML = '';
      if (history.length === 0) {
        dom.historyList.innerHTML = '<div class="empty-state-mini">No past sessions recorded</div>';
        return;
      }

      // Populate select elements for comparison modal too
      dom.selectCompareT1.innerHTML = '';
      dom.selectCompareT2.innerHTML = '';

      history.forEach((item, index) => {
        const el = document.createElement('div');
        el.className = 'history-item';
        if (item.session_id === state.sessionId) el.classList.add('active');
        
        const dateStr = item.created_at ? new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';
        el.innerHTML = `
          <div class="history-info">
            <span class="history-title">${item.title}</span>
            <span class="history-time">${dateStr} • ${item.query_count} queries</span>
          </div>
        `;
        el.addEventListener('click', () => loadSessionDetail(item.session_id));
        dom.historyList.appendChild(el);

        if (item.primary_image_id) {
          const opt1 = new Option(item.title, item.primary_image_id);
          const opt2 = new Option(item.title, item.primary_image_id);
          dom.selectCompareT1.add(opt1);
          dom.selectCompareT2.add(opt2);
        }
      });

      if (dom.selectCompareT2.options.length > 1) {
        dom.selectCompareT2.selectedIndex = 1;
      }

    } catch (e) {
      console.warn("Could not fetch history:", e);
    }
  }

  async function loadSessionDetail(sessionId) {
    try {
      const res = await fetch(`/api/session/${sessionId}`);
      if (!res.ok) return;
      const data = await res.json();

      state.sessionId = data.session_id;
      if (data.images && data.images.length > 0) {
        const first = data.images[0];
        handleImageLoaded({
          session_id: data.session_id,
          image_id: first.id,
          url: first.url,
          metadata: {
            width: first.width,
            height: first.height,
            file_size: first.file_size,
            format: 'RGB',
            color_mode: 'RGB',
            channels: 3,
            has_georeference: false
          }
        });
      }
    } catch (e) {
      console.warn("Error loading session:", e);
    }
  }

  // 9. Chat Stream Rendering Helpers
  function appendUserMessage(text) {
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble user-message';
    bubble.innerHTML = `
      <div class="msg-avatar"><i data-lucide="user"></i></div>
      <div class="msg-body">
        <div class="msg-header">
          <strong>User Query</strong>
          <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="msg-text">${escapeHtml(text)}</div>
      </div>
    `;
    dom.chatStream.appendChild(bubble);
    if (window.lucide) window.lucide.createIcons();
    dom.chatStream.scrollTop = dom.chatStream.scrollHeight;
  }

  function appendAiMessage(markdownText, intent) {
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble ai-message';
    const formattedHtml = formatMarkdown(markdownText);

    bubble.innerHTML = `
      <div class="msg-avatar"><i data-lucide="satellite"></i></div>
      <div class="msg-body">
        <div class="msg-header">
          <strong>SatQuery AI (${intent || 'REASONING'})</strong>
          <span class="msg-time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        <div class="msg-text">${formattedHtml}</div>
        <div class="msg-actions">
          <button class="msg-action-btn btn-copy" title="Copy answer to clipboard">
            <i data-lucide="copy"></i> Copy
          </button>
        </div>
      </div>
    `;

    const copyBtn = bubble.querySelector('.btn-copy');
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(markdownText).then(() => {
        copyBtn.innerHTML = '<i data-lucide="check"></i> Copied!';
        setTimeout(() => {
          copyBtn.innerHTML = '<i data-lucide="copy"></i> Copy';
          if (window.lucide) window.lucide.createIcons();
        }, 2000);
      });
    });

    dom.chatStream.appendChild(bubble);
    if (window.lucide) window.lucide.createIcons();
    dom.chatStream.scrollTop = dom.chatStream.scrollHeight;
  }

  function appendSystemMessage(text, isError = false) {
    const bubble = document.createElement('div');
    bubble.className = `message-bubble system-message ${isError ? 'error-bubble' : ''}`;
    bubble.innerHTML = `
      <div class="msg-avatar"><i data-lucide="${isError ? 'alert-triangle' : 'info'}"></i></div>
      <div class="msg-body">
        <div class="msg-text" style="color: ${isError ? '#f87171' : '#94a3b8'}">${escapeHtml(text)}</div>
      </div>
    `;
    dom.chatStream.appendChild(bubble);
    if (window.lucide) window.lucide.createIcons();
    dom.chatStream.scrollTop = dom.chatStream.scrollHeight;
  }

  function showTypingIndicator() {
    const ind = document.createElement('div');
    ind.id = 'typingIndicator';
    ind.className = 'message-bubble ai-message';
    ind.innerHTML = `
      <div class="msg-avatar"><i data-lucide="loader"></i></div>
      <div class="msg-body">
        <div class="typing-indicator">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    `;
    dom.chatStream.appendChild(ind);
    if (window.lucide) window.lucide.createIcons();
    dom.chatStream.scrollTop = dom.chatStream.scrollHeight;
  }

  function removeTypingIndicator() {
    const ind = document.getElementById('typingIndicator');
    if (ind) ind.remove();
  }

  function renderDetectedRegions(regions) {
    dom.clustersTags.innerHTML = '';
    if (!regions || regions.length === 0) {
      dom.clustersTags.innerHTML = '<span class="tag-pill">No isolated hotspots</span>';
      return;
    }
    regions.slice(0, 6).forEach(r => {
      const pill = document.createElement('span');
      pill.className = 'tag-pill';
      pill.textContent = `${r.class_name} (${r.area_percentage}%)`;
      dom.clustersTags.appendChild(pill);
    });
  }

  // 10. Format Markdown text safely to HTML
  function formatMarkdown(text) {
    if (!text) return '';
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/\n\n/g, '</p><p>');
    html = html.replace(/\n/g, '<br/>');
    html = html.replace(/^- (.*)/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    return `<p>${html}</p>`;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // 11. Loader Helper
  function showLoader(text) {
    dom.loaderStatusText.textContent = text || 'Processing...';
    dom.viewerLoader.classList.remove('hidden');
  }
  function hideLoader() {
    dom.viewerLoader.classList.add('hidden');
  }

  // 12. Modal Helpers
  function openModal(modal) { modal.classList.remove('hidden'); }
  function closeModal(modal) { modal.classList.add('hidden'); }

  // 13. Event Listeners Setup
  function setupEventListeners() {
    // Dropzone events
    dom.uploadDropzone.addEventListener('click', () => dom.imageFileInput.click());
    dom.imageFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadFile(e.target.files[0]);
      }
    });

    ['dragenter', 'dragover'].forEach(name => {
      dom.uploadDropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dom.uploadDropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(name => {
      dom.uploadDropzone.addEventListener(name, (e) => {
        e.preventDefault();
        dom.uploadDropzone.classList.remove('dragover');
      });
    });
    dom.uploadDropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });

    // New Session
    dom.btnNewSession.addEventListener('click', () => {
      state.sessionId = null;
      state.currentImageId = null;
      dom.viewerPlaceholder.classList.remove('hidden');
      dom.imageLayersWrapper.classList.add('hidden');
      dom.compareViewContainer.classList.add('hidden');
      dom.chatStream.innerHTML = `
        <div class="message-bubble system-message">
          <div class="msg-avatar"><i data-lucide="satellite"></i></div>
          <div class="msg-body">
            <div class="msg-header"><strong>SatQuery AI Assistant</strong><span class="msg-time">Ready</span></div>
            <div class="msg-text">New session started. Upload a satellite raster or select a demo scene.</div>
          </div>
        </div>
      `;
      if (window.lucide) window.lucide.createIcons();
      updateMapLocation(null, null, null);
      updateChartData({ vegetation: 0, water: 0, built_up: 0, barren: 0 });
      dom.btnDownloadReport.disabled = true;
    });

    // Quick Sample Demo Button in Placeholder
    dom.btnQuickSampleDemo.addEventListener('click', () => {
      loadSample('agriculture_delta.jpg');
    });

    // Layer Switcher Tabs
    dom.layerTabs.addEventListener('click', (e) => {
      const tab = e.target.closest('.layer-tab');
      if (tab && tab.dataset.layer) {
        switchLayer(tab.dataset.layer);
      }
    });

    // Opacity Slider
    dom.layerOpacitySlider.addEventListener('input', applyOpacity);

    // Chat Submit
    dom.btnSendQuery.addEventListener('click', () => submitQuery(dom.queryInput.value));
    dom.queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitQuery(dom.queryInput.value);
      }
    });

    // Clear Chat
    dom.btnClearChat.addEventListener('click', () => {
      dom.chatStream.innerHTML = `
        <div class="message-bubble system-message">
          <div class="msg-avatar"><i data-lucide="trash-2"></i></div>
          <div class="msg-body"><div class="msg-text">Chat history cleared for current session.</div></div>
        </div>
      `;
      if (window.lucide) window.lucide.createIcons();
    });

    // Quick Query Pills
    dom.quickQueries.addEventListener('click', (e) => {
      const pill = e.target.closest('.query-pill');
      if (pill && pill.dataset.query) {
        submitQuery(pill.dataset.query);
      }
    });

    // Download PDF Report
    dom.btnDownloadReport.addEventListener('click', () => {
      if (!state.sessionId) return;
      window.open(`/api/report/${state.sessionId}`, '_blank');
    });

    // Compare Mode Modal
    dom.btnToggleCompareMode.addEventListener('click', () => openModal(dom.compareModal));
    dom.btnCloseCompareModal.addEventListener('click', () => closeModal(dom.compareModal));
    dom.btnCancelCompare.addEventListener('click', () => closeModal(dom.compareModal));
    dom.btnExecuteCompare.addEventListener('click', () => {
      const t1 = parseInt(dom.selectCompareT1.value);
      const t2 = parseInt(dom.selectCompareT2.value);
      if (isNaN(t1) || isNaN(t2)) {
        alert("Please ensure two distinct image passes are selected.");
        return;
      }
      executeComparison(t1, t2);
    });

    // Load Temporal Benchmark
    dom.btnLoadTemporalBenchmark.addEventListener('click', async () => {
      closeModal(dom.compareModal);
      showLoader("Loading pre & post development temporal benchmark...");

      // Load T1
      const formData1 = new FormData();
      formData1.append("sample_filename", "temporal_t1.jpg");
      const res1 = await fetch("/api/sample/load", { method: "POST", body: formData1 });
      const d1 = await res1.json();

      // Load T2
      const formData2 = new FormData();
      formData2.append("sample_filename", "temporal_t2.jpg");
      formData2.append("session_id", d1.session_id);
      const res2 = await fetch("/api/sample/load", { method: "POST", body: formData2 });
      const d2 = await res2.json();

      state.sessionId = d1.session_id;
      executeComparison(d1.image_id, d2.image_id);
    });

    // Info Modal
    dom.btnHelpModal.addEventListener('click', () => openModal(dom.infoModal));
    dom.btnCloseInfoModal.addEventListener('click', () => closeModal(dom.infoModal));

    // Refresh history
    dom.btnRefreshHistory.addEventListener('click', fetchHistory);
  }

  // Run on start
  initApp();
});
