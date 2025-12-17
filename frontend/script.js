document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const statusText = document.getElementById('uploadStatus');
    const logContainer = document.getElementById('logContainer');
    const toggleLogsBtn = document.getElementById('toggleLogs');
    const searchInput = document.getElementById('searchInput');
    const sendBtn = document.getElementById('searchBtn');
    const chatHistory = document.getElementById('chatHistory');

    // --- Theme Toggle ---
    const themeToggleBtn = document.getElementById('themeToggle');
    const themeIcon = themeToggleBtn ? themeToggleBtn.querySelector('i') : null;

    function setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        
        if (themeIcon) {
            if (theme === 'light') {
                themeIcon.classList.replace('fa-sun', 'fa-moon');
            } else {
                themeIcon.classList.replace('fa-moon', 'fa-sun');
            }
        }
    }

    // Initialize theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    setTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            setTheme(newTheme);
        });
    }

    // --- Configuration Management (A/B Testing) ---
    const configurations = {
        'c1': {
            id: 'c1',
            name: 'Configuration 1',
            vision_model: 'qwen2.5-vl',
            speech_model: 'faster-whisper-base',
            embedding_model: 'all-minilm-l6-v2',
            enable_vision: true
        },
        'c2': {
            id: 'c2',
            name: 'Configuration 2',
            vision_model: 'blip-image-captioning-large',
            speech_model: 'faster-whisper-large',
            embedding_model: 'nomic-embed',
            enable_vision: true
        },
        'c3': {
            id: 'c3',
            name: 'Configuration 3',
            vision_model: 'florence-2-large',
            speech_model: 'distil-whisper',
            embedding_model: 'bge-m3',
            enable_vision: false
        }
    };

    let currentConfigId = 'c1';
    const configSelect = document.getElementById('configSelect');
    const configSummary = document.getElementById('configSummary');
    const editConfigBtn = document.getElementById('editConfigBtn');
    const addConfigBtn = document.getElementById('addConfigBtn');

    function initConfigUI() {
        if (!configSelect) return;

        // Populate Dropdown
        configSelect.innerHTML = '';
        Object.values(configurations).forEach(config => {
            const option = document.createElement('option');
            option.value = config.id;
            option.textContent = config.name;
            if (config.id === currentConfigId) option.selected = true;
            configSelect.appendChild(option);
        });

        updateConfigSummary();

        // Event Listeners
        configSelect.addEventListener('change', (e) => {
            currentConfigId = e.target.value;
            updateConfigSummary();
            addLog(`Switched to ${configurations[currentConfigId].name}`);
            
            // Refresh data list to simulate context switch
            fetchIngestedData(); 
        });

        if (editConfigBtn) {
            editConfigBtn.addEventListener('click', () => {
                openSettingsModal();
            });
        }

        if (addConfigBtn) {
            addConfigBtn.addEventListener('click', () => {
                alert("Create New Configuration feature coming soon!");
            });
        }
    }

    function updateConfigSummary() {
        if (!configSummary) return;
        const config = configurations[currentConfigId];
        configSummary.textContent = `${config.vision_model} • ${config.speech_model}`;
    }

    initConfigUI();

    // --- Logging System ---
    function addLog(message) {
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', { hour12: false });
        
        entry.innerHTML = `<span class="timestamp">[${timeString}]</span> ${message}`;
        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    addLog('System initialized.');
    addLog('Ready for video ingestion.');

    // --- Log Toggle ---
    if (toggleLogsBtn) {
        toggleLogsBtn.addEventListener('click', () => {
            logContainer.classList.toggle('minimized');
            const icon = toggleLogsBtn.querySelector('i');
            if (logContainer.classList.contains('minimized')) {
                icon.classList.replace('fa-chevron-down', 'fa-chevron-up');
            } else {
                icon.classList.replace('fa-chevron-up', 'fa-chevron-down');
            }
        });
    }

    // --- Past Data Tabs ---
    const dataTabs = document.querySelectorAll('.data-tab');
    const dataList = document.getElementById('ingestedDataList');

    async function fetchIngestedData(view = 'all') {
        try {
            dataList.innerHTML = '<div class="data-item" style="justify-content:center; color:var(--text-muted);">Loading...</div>';
            
            // Get active embedding model to filter videos
            const config = configurations[currentConfigId];
            const modelParam = config ? `?model=${encodeURIComponent(config.embedding_model)}` : '';

            // Fetch videos filtered by model
            const response = await fetch(`/videos${modelParam}`);
            if (!response.ok) throw new Error('Failed to fetch videos');
            
            let videos = await response.json();
            
            // Client-side filtering for POC
            if (view === 'recent') {
                // Sort by created_at desc (already done by backend) and take top 5
                videos = videos.slice(0, 5);
            } else if (view === 'favorites') {
                // Placeholder: Filter by some favorite flag (not yet in DB)
                videos = videos.filter(v => v.is_favorite); 
            }

            renderDataList(videos);
        } catch (error) {
            console.error('Error fetching data:', error);
            dataList.innerHTML = '<div class="data-item" style="justify-content:center; color:var(--error);">Failed to load data</div>';
        }
    }

    function renderDataList(videos) {
        dataList.innerHTML = '';
        
        if (videos.length === 0) {
            dataList.innerHTML = '<div class="data-item" style="justify-content:center; color:var(--text-muted);">No videos found</div>';
            return;
        }

        videos.forEach(video => {
            const item = document.createElement('div');
            item.className = 'data-item';
            if (selectedVideoId === video.id) {
                item.classList.add('selected');
            }
            
            let statusClass = 'processing';
            let statusText = video.status || 'Unknown';
            
            if (statusText === 'ready') {
                statusClass = 'success';
                statusText = 'Indexed';
            } else if (statusText.includes('error')) {
                statusClass = 'error'; 
            }

            item.innerHTML = `
                <div class="data-info">
                    <i class="fas fa-video"></i>
                    <span class="data-name" title="${video.filename}">${video.filename}</span>
                </div>
                <span class="data-status ${statusClass}">${statusText}</span>
            `;
            
            // Click to select video for filtering
            item.addEventListener('click', () => {
                // Toggle selection
                if (selectedVideoId === video.id) {
                    selectedVideoId = null;
                    item.classList.remove('selected');
                    addLog(`Cleared video filter.`);
                } else {
                    selectedVideoId = video.id;
                    // Remove selected class from others
                    document.querySelectorAll('.data-item').forEach(el => el.classList.remove('selected'));
                    item.classList.add('selected');
                    addLog(`Selected video for search: ${video.filename}`);
                }
            });

            dataList.appendChild(item);
        });
    }

    if (dataTabs) {
        dataTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                dataTabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const view = tab.getAttribute('data-view');
                fetchIngestedData(view);
            });
        });
    }

    // Initial Load
    fetchIngestedData();

    // Poll for updates every 5 seconds
    setInterval(() => {
        // Only poll if we are on the 'all' or 'recent' view to avoid jitter if user is interacting
        // For simplicity, we just refresh. A more robust solution would check if user is dragging/selecting.
        const activeTab = document.querySelector('.data-tab.active');
        const view = activeTab ? activeTab.getAttribute('data-view') : 'all';
        fetchIngestedData(view);
    }, 5000);

    // --- Drag & Drop Upload Handling ---
    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
    }

    async function handleFileUpload(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        // Get Config from State
        const config = configurations[currentConfigId];
        formData.append('config', JSON.stringify(config));

        statusText.textContent = `Uploading ${file.name}...`;
        addLog(`Starting upload: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
        addLog(`Using Pipeline: ${config.name}`);
        addLog(`Configuration: ${JSON.stringify(config)}`);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (response.ok) {
                const result = await response.json();
                statusText.textContent = 'Upload complete!';
                statusText.style.color = 'var(--success)';
                addLog(`Upload successful: ${result.filename}`);
                addLog('Video queued for processing (Ingestion Agent notified).');
                
                // Simulate processing logs
                setTimeout(() => addLog(`Extracting audio from ${result.filename}...`), 2000);
                setTimeout(() => addLog(`Transcribing audio...`), 4500);
                setTimeout(() => addLog(`Generating embeddings...`), 8000);
                setTimeout(() => addLog(`Indexing complete. Video ready for search.`), 12000);
            } else {
                const err = await response.text();
                throw new Error(err);
            }
        } catch (error) {
            console.error('Upload error:', error);
            statusText.textContent = 'Upload failed.';
            statusText.style.color = 'var(--error)';
            addLog(`Error uploading file: ${error.message}`);
        } finally {
            fileInput.value = ''; 
        }
    }

    // --- Chat / Search Handling ---
    let selectedVideoId = null; // Global state for selected video

    function appendMessage(sender, content, isHtml = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isHtml) {
            contentDiv.innerHTML = content;
        } else {
            contentDiv.textContent = content;
        }
        
        msgDiv.appendChild(contentDiv);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function handleSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        appendMessage('user', query);
        searchInput.value = '';
        
        const loadingId = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message system-message';
        loadingDiv.id = loadingId;
        loadingDiv.innerHTML = '<div class="message-content">Searching video archives...</div>';
        chatHistory.appendChild(loadingDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        addLog(`Processing query: "${query}"`);
        if (selectedVideoId) {
            addLog(`Filtering by video ID: ${selectedVideoId}`);
        }

        try {
            let url = `/search?q=${encodeURIComponent(query)}`;
            if (selectedVideoId) {
                url += `&video_id=${selectedVideoId}`;
            }

            // Add active embedding model from configuration
            const config = configurations[currentConfigId];
            if (config && config.embedding_model) {
                url += `&model=${encodeURIComponent(config.embedding_model)}`;
            }
            
            const response = await fetch(url);
            const results = await response.json();
            
            const loadingEl = document.getElementById(loadingId);
            if (loadingEl) loadingEl.remove();

            if (results.length === 0) {
                appendMessage('system', 'No relevant video clips found.');
                addLog('Search completed. No results.');
                return;
            }

            addLog(`Found ${results.length} matches.`);

            let resultHtml = `Found ${results.length} relevant clips:<br>`;
            
            results.forEach(res => {
                // Determine score badge class
                const scorePct = res.score * 100;
                let badgeClass = 'score-med';
                if (scorePct >= 80) badgeClass = 'score-high';

                // Highlight keywords in text (simple implementation)
                const highlightedText = res.text.replace(new RegExp(query, 'gi'), match => `<span class="highlight">${match}</span>`);

                resultHtml += `
                    <div class="video-result">
                        <div class="video-header">
                            <span class="filename">${res.filename}</span>
                            <span class="score-badge ${badgeClass}">${scorePct.toFixed(0)}% Match</span>
                        </div>
                        <div class="video-body">
                            <div class="transcript-text">"${highlightedText}"</div>
                            <div class="video-player-container">
                                <video controls width="100%" src="${res.clip_url}"></video>
                            </div>
                            <div class="video-footer">
                                ${res.start_time.toFixed(1)}s - ${res.end_time.toFixed(1)}s
                            </div>
                        </div>
                    </div>
                `;
            });

            appendMessage('system', resultHtml, true);

        } catch (error) {
            console.error('Search error:', error);
            const loadingEl = document.getElementById(loadingId);
            if (loadingEl) loadingEl.remove();
            
            appendMessage('system', 'Sorry, an error occurred while searching.');
            addLog(`Search error: ${error.message}`);
        }
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', handleSearch);
    }

    if (searchInput) {
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                handleSearch();
            }
        });
    }

    // --- Modal Handling ---
    const modal = document.getElementById("settingsModal");
    const span = document.getElementsByClassName("close-modal")[0];
    const saveBtn = document.getElementById("saveConfigBtn");
    const cancelBtn = document.getElementById("cancelConfigBtn");

    function openSettingsModal() {
        if (!modal) return;
        
        const config = configurations[currentConfigId];
        
        // Set Vision Model
        const visionRadio = document.querySelector(`input[name="visionModel"][value="${config.vision_model}"]`);
        if (visionRadio) visionRadio.checked = true;

        // Set Speech Model
        const speechRadio = document.querySelector(`input[name="speechModel"][value="${config.speech_model}"]`);
        if (speechRadio) speechRadio.checked = true;

        // Set Embedding Model
        const embedRadio = document.querySelector(`input[name="embeddingModel"][value="${config.embedding_model}"]`);
        if (embedRadio) embedRadio.checked = true;

        // Set Enable Vision Checkbox
        const visionCheck = document.getElementById('enableVisionCheck');
        if (visionCheck) visionCheck.checked = config.enable_vision;

        modal.style.display = "flex";
    }

    function saveConfiguration() {
        const config = configurations[currentConfigId];

        // Get Vision Model
        const visionRadio = document.querySelector('input[name="visionModel"]:checked');
        if (visionRadio) config.vision_model = visionRadio.value;

        // Get Speech Model
        const speechRadio = document.querySelector('input[name="speechModel"]:checked');
        if (speechRadio) config.speech_model = speechRadio.value;

        // Get Embedding Model
        const embedRadio = document.querySelector('input[name="embeddingModel"]:checked');
        if (embedRadio) config.embedding_model = embedRadio.value;

        // Get Enable Vision Checkbox
        const visionCheck = document.getElementById('enableVisionCheck');
        if (visionCheck) config.enable_vision = visionCheck.checked;

        updateConfigSummary();
        modal.style.display = "none";
        addLog(`Configuration '${config.name}' updated.`);
    }

    if (span) {
        span.onclick = function() {
            modal.style.display = "none";
        }
    }

    if (cancelBtn) {
        cancelBtn.onclick = function() {
            modal.style.display = "none";
        }
    }

    if (saveBtn) {
        saveBtn.onclick = function() {
            saveConfiguration();
        }
    }

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }

});
