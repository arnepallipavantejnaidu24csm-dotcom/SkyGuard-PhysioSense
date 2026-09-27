/**
 * Main AWS Web Dashboard Application Controller
 * Handles live polling/SSE, station map, health gauges, alerts filtering, and live anomaly injection.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State
    const state = {
        selectedStationId: 'AWS-01-ISRIKA1-Bheemili',
        isPlaying: true,
        pollIntervalMs: 2000,
        timerId: null,
        activeTab: 'live',
        alertFilterSeverity: 'ALL',
        alertSearchQuery: '',
        stations: [],
        map: null,
        markers: {},
    };

    // Initialize Charts
    const chartManager = window.awsChartManager;
    chartManager.initTelemetryChart('chart-temp', 'Temperature', '°C', '#f43f5e', '#fb7185');
    chartManager.initTelemetryChart('chart-pres', 'Pressure', 'hPa', '#06b6d4', '#67e8f9');
    chartManager.initTelemetryChart('chart-rh', 'Humidity', '%', '#3b82f6', '#93c5fd');
    chartManager.initTelemetryChart('chart-wind', 'Wind Speed', 'm/s', '#10b981', '#6ee7b7');
    chartManager.initDualAxisComparisonChart('chart-dual-axis');

    function initMap(stations) {
    if (state.map) return;

    // Create map centered around the station network
    state.map = L.map('map', {
        zoomControl: false
    });

    // Dark map tiles
    L.tileLayer('https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_3ya9_1_d39ba8e9a72b3708ef25cf8d', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(state.map);

    L.control.zoom({
        position: 'bottomright'
    }).addTo(state.map);

    // Add station markers
    const stationBounds = [];

    stations.forEach(st => {
        const markerColor =
            st.status === 'OPERATIONAL'
                ? '#10b981'
                : st.status === 'DEGRADED_SENSORS'
                    ? '#f59e0b'
                    : '#3b82f6';

        const customIcon = L.divIcon({
            className: 'custom-station-pin',
            html: `
                <div style="
                    background-color: ${markerColor};
                    width: 14px;
                    height: 14px;
                    border-radius: 50%;
                    border: 2px solid #ffffff;
                    box-shadow: 0 0 10px ${markerColor};
                "></div>
            `,
            iconSize: [14, 14],
            iconAnchor: [7, 7]
        });

        const marker = L.marker(
            [Number(st.lat), Number(st.lon)],
            { icon: customIcon }
        ).addTo(state.map);

        marker.bindPopup(`
            <div style="
                color: #111827;
                font-family: sans-serif;
                font-size: 12px;
            ">
                <strong>${st.name}</strong><br>
                ID: <code>${st.id}</code><br>
                Elevation: ${st.elevation_m}m<br>
                Status:
                <span style="
                    font-weight:bold;
                    color:${markerColor}
                ">
                    ${st.status}
                </span><br>

                <button
                    onclick="window.switchStation('${st.id}')"
                    style="
                        margin-top:6px;
                        background:#3b82f6;
                        color:#fff;
                        border:none;
                        padding:3px 8px;
                        border-radius:4px;
                        cursor:pointer;
                    "
                >
                    Select Station
                </button>
            </div>
        `);

        state.markers[st.id] = marker;

        stationBounds.push([Number(st.lat), Number(st.lon)]);
    });

    // Automatically zoom the map to all stations
    if (stationBounds.length > 0) {
        state.map.fitBounds(stationBounds, {
            padding: [25, 25],
            maxZoom: 12
        });
    }
}

    // Expose switchStation globally
    window.switchStation = function(stationId) {
        state.selectedStationId = stationId;
        document.getElementById('station-select').value = stationId;
        document.getElementById('current-station-name').innerText = stationId;
        
        // Pan map
        const marker = state.markers[stationId];
        if (marker && state.map) {
            state.map.setView(marker.getLatLng(), 8, { animate: true });
            marker.openPopup();
        }

        fetchLiveTelemetry();
        fetchHealthMetrics();
        fetchAlerts();
        updateSensorIsolationTable();
        updateScenarioMap();
        const activeScenario = regionalScenario.mode;
        if (activeScenario === 'normal') closeScenarioGraphs(); else renderScenarioGraphs(activeScenario);
    };

    // 1. Fetch Stations List
    async function fetchStations() {
        try {
            const res = await fetch('/api/stations');
            const data = await res.json();
            if (data.status === 'success') {
                state.stations = data.stations;
                
                // Populate select dropdown
                const select = document.getElementById('station-select');
                select.innerHTML = '';
                data.stations.forEach(st => {
                    const opt = document.createElement('option');
                    opt.value = st.id;
                    opt.innerText = `${st.id} - ${st.name} (${st.status})`;
                    if (st.id === state.selectedStationId) opt.selected = true;
                    select.appendChild(opt);
                });

                initMap(data.stations);
                updateSensorIsolationTable();
                updateScenarioMap();
            }
        } catch (e) {
            console.error('Error fetching stations:', e);
        }
    }

    // 2. Fetch Live Telemetry
    async function fetchLiveTelemetry() {
        try {
            const res = await fetch(`/api/telemetry/live?station_id=${state.selectedStationId}`);
            const data = await res.json();
            if (data.status === 'success') {
                const cur = data.current;
                const cs = data.chart_series;

                // Update Live Gauge / Value Cards
                updateTelemetryCard('temp', cur.temperature, '°C');
                updateTelemetryCard('pres', cur.pressure, 'hPa');
                updateTelemetryCard('rh', cur.humidity, '%');
                updateTelemetryCard('wind', cur.wind_speed, 'm/s');

                // Anomaly Status Banner
                const anomBanner = document.getElementById('anomaly-status-banner');
                if (cur.anomaly_status.is_anomaly) {
                    const isCrit = cur.anomaly_status.severity === 'CRITICAL';
                    anomBanner.className = `p-3 rounded-lg flex items-center justify-between text-xs font-semibold ${isCrit ? 'bg-red-950/80 text-red-300 border border-red-800 animate-pulse' : 'bg-amber-950/80 text-amber-300 border border-amber-800'}`;
                    anomBanner.innerHTML = `
                        <div class="flex items-center gap-2">
                            <span class="w-2.5 h-2.5 rounded-full ${isCrit ? 'bg-red-500' : 'bg-amber-500'}"></span>
                            <span>[${cur.anomaly_status.severity}] ${cur.anomaly_status.type} (Conf: ${(cur.anomaly_status.confidence * 100).toFixed(0)}%)</span>
                        </div>
                        <span class="underline cursor-pointer" onclick="document.getElementById('tab-alerts').click()">View Diagnostic</span>
                    `;
                } else {
                    anomBanner.className = 'p-3 rounded-lg flex items-center justify-between text-xs font-semibold bg-emerald-950/50 text-emerald-300 border border-emerald-800/50';
                    anomBanner.innerHTML = `
                        <div class="flex items-center gap-2">
                            <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
                            <span>All Instruments Operating Nominally (Zero Active Physics Violations)</span>
                        </div>
                        <span class="text-xs text-gray-400">Timestamp: ${cur.timestamp}</span>
                    `;
                }

                // Update Chart Series
                chartManager.updateStreamChart('chart-temp', cs.timestamps, cs.temperature_corrected, cs.temperature_raw, cs.temperature_ci95_upper, cs.temperature_ci95_lower);
                chartManager.updateStreamChart('chart-pres', cs.timestamps, cs.pressure_corrected, cs.pressure_raw, cs.pressure_ci95_upper, cs.pressure_ci95_lower);
                chartManager.updateStreamChart('chart-rh', cs.timestamps, cs.humidity_corrected, cs.humidity_raw);
                chartManager.updateStreamChart('chart-wind', cs.timestamps, cs.wind_speed_corrected, cs.wind_speed_raw);

                // Dual Axis Chart update
                const dualChart = chartManager.charts['chart-dual-axis'];
                if (dualChart) {
                    dualChart.data.labels = cs.timestamps.map(t => t.split(' ')[1] || t);
                    dualChart.data.datasets[0].data = cs.temperature_corrected;
                    dualChart.data.datasets[1].data = cs.pressure_corrected;
                    dualChart.update('none');
                }
            }
        } catch (e) {
            console.error('Error in live telemetry stream:', e);
        }
    }

    function updateTelemetryCard(idPrefix, valObj, unit) {
        document.getElementById(`val-${idPrefix}-corr`).innerText = valObj.corrected.toFixed(1);
        document.getElementById(`val-${idPrefix}-raw`).innerText = valObj.raw.toFixed(1) + ' ' + unit;
        
        const biasEl = document.getElementById(`val-${idPrefix}-bias`);
        biasEl.innerText = (valObj.bias >= 0 ? '+' : '') + valObj.bias.toFixed(2) + ' ' + unit;
        if (Math.abs(valObj.bias) > 0.5) {
            biasEl.className = 'text-xs text-amber-400 font-mono font-bold';
        } else {
            biasEl.className = 'text-xs text-gray-400 font-mono';
        }

        if (valObj.ci95_lower !== undefined) {
            document.getElementById(`val-${idPrefix}-ci`).innerText = `[${valObj.ci95_lower.toFixed(1)}, ${valObj.ci95_upper.toFixed(1)}]`;
        }
    }

    // 3. Fetch Sensor Health Diagnostics
    async function fetchHealthMetrics() {
        try {
            const res = await fetch(`/api/health?station_id=${state.selectedStationId}`);
            const data = await res.json();
            if (data.status === 'success') {
                const hm = data.health_metrics;
                const container = document.getElementById('health-cards-grid');
                container.innerHTML = '';

                Object.keys(hm).forEach(sKey => {
                    const m = hm[sKey];
                    const card = document.createElement('div');
                    card.className = 'glass-card p-4 rounded-xl flex flex-col justify-between';
                    
                    const score = m.health_score;
                    const gradeColor = score >= 85 ? 'text-emerald-400 border-emerald-500' : score >= 60 ? 'text-amber-400 border-amber-500' : 'text-rose-400 border-rose-500';
                    const progressDash = (score / 100) * 220;

                    card.innerHTML = `
                        <div class="flex items-center justify-between mb-2">
                            <div>
                                <h4 class="font-bold text-gray-200 capitalize text-sm">${m.sensor_name}</h4>
                                <p class="text-xs text-gray-400 truncate max-w-[160px]">${m.instrument_model}</p>
                            </div>
                            <span class="text-xs px-2 py-0.5 rounded font-bold border ${gradeColor}">${m.health_grade}</span>
                        </div>

                        <div class="flex items-center gap-4 my-2">
                            <div class="relative w-16 h-16 flex items-center justify-center">
                                <svg class="w-16 h-16 transform -rotate-90">
                                    <circle cx="32" cy="32" r="28" stroke="rgba(255,255,255,0.1)" stroke-width="5" fill="transparent"/>
                                    <circle cx="32" cy="32" r="28" stroke="${score >= 85 ? '#10b981' : score >= 60 ? '#f59e0b' : '#f43f5e'}" stroke-width="5" fill="transparent"
                                        stroke-dasharray="175" stroke-dashoffset="${175 - (score / 100) * 175}" stroke-linecap="round"/>
                                </svg>
                                <span class="absolute font-bold text-sm text-gray-100">${score.toFixed(0)}%</span>
                            </div>
                            <div class="text-xs space-y-1 text-gray-300">
                                <div>Completeness: <strong class="text-gray-100">${m.completeness_pct}%</strong></div>
                                <div>Drift Rate: <strong class="text-gray-100">${m.drift_rate_per_day >= 0 ? '+' : ''}${m.drift_rate_per_day.toFixed(3)}/day</strong></div>
                                <div>Noise: <strong class="text-gray-100">σ=${m.noise_std.toFixed(2)}</strong></div>
                                <div>RUL: <strong class="text-cyan-400 font-bold">${m.predicted_rul_days.toFixed(0)} Days</strong></div>
                            </div>
                        </div>

                        <div class="mt-2 pt-2 border-t border-gray-800 text-[11px] text-gray-400">
                            <strong>Action:</strong> ${m.recommended_action}
                        </div>
                    `;
                    container.appendChild(card);
                });
            }
        } catch (e) {
            console.error('Error fetching health metrics:', e);
        }
    }

    // 4. Fetch Actionable Alerts Log
    async function fetchAlerts() {
        try {
            const url = `/api/alerts?station_id=${state.selectedStationId}&severity=${state.alertFilterSeverity}&q=${encodeURIComponent(state.alertSearchQuery)}`;
            const res = await fetch(url);
            const data = await res.json();
            if (data.status === 'success') {
                const tbody = document.getElementById('alerts-table-body');
                tbody.innerHTML = '';

                document.getElementById('alerts-total-badge').innerText = data.total_count;

                if (data.alerts.length === 0) {
                    tbody.innerHTML = `<tr><td colspan="6" class="p-6 text-center text-gray-500">No alerts found matching current filter criteria.</td></tr>`;
                    return;
                }

                data.alerts.forEach(a => {
                    const tr = document.createElement('tr');
                    tr.className = 'border-b border-gray-800/80 hover:bg-gray-800/30 text-xs transition';
                    
                    const sevBadgeClass = a.severity === 'CRITICAL' ? 'badge-critical' : a.severity === 'WARNING' ? 'badge-warning' : 'badge-info';

                    tr.innerHTML = `
                        <td class="p-3 font-mono text-gray-400">${a.alert_id}</td>
                        <td class="p-3 font-mono text-gray-300 whitespace-nowrap">${a.timestamp}</td>
                        <td class="p-3"><span class="px-2 py-0.5 rounded font-bold text-[10px] ${sevBadgeClass}">${a.severity}</span></td>
                        <td class="p-3">
                            <strong class="text-gray-200 block">${a.anomaly_type}</strong>
                            <span class="text-gray-400 text-[11px]">Sensors: ${a.affected_parameters.join(', ')}</span>
                        </td>
                        <td class="p-3">
                            <div class="text-[11px] text-gray-300">Raw: <code>${a.raw_reading}</code></div>
                            <div class="text-[11px] text-cyan-300">Corr: <code>${a.corrected_estimate}</code></div>
                            <div class="text-[10px] text-gray-400 mt-0.5 font-sans">${a.historical_context}</div>
                        </td>
                        <td class="p-3 text-gray-300 text-[11px] font-medium max-w-xs">
                            ${a.recommended_action}
                        </td>
                    `;
                    tbody.appendChild(tr);
                });
            }
        } catch (e) {
            console.error('Error fetching alerts:', e);
        }
    }

    // 5. Fetch System Performance Metrics
    async function fetchSystemMetrics() {
        try {
            const res = await fetch('/api/metrics');
            const data = await res.json();
            if (data.status === 'success') {
                document.getElementById('metric-latency').innerText = data.last_latency_ms + ' ms';
                document.getElementById('metric-throughput').innerText = data.throughput_rps + ' rec/s';
                document.getElementById('metric-inferences').innerText = data.total_inferences.toLocaleString();
                document.getElementById('metric-uptime').innerText = (data.uptime_seconds / 60).toFixed(1) + ' min';
            }
        } catch (e) {
            console.error('Error fetching metrics:', e);
        }
    }

    // Event Listeners
    document.getElementById('station-select').addEventListener('change', (e) => {
        window.switchStation(e.target.value);
    });

    // Play / Pause Stream Toggle
    const playBtn = document.getElementById('btn-play-pause');
    playBtn.addEventListener('click', () => {
        state.isPlaying = !state.isPlaying;
        playBtn.innerHTML = state.isPlaying ? '<i class="fas fa-pause"></i> Pause' : '<i class="fas fa-play"></i> Resume';
        playBtn.className = state.isPlaying ? 'px-3 py-1.5 rounded-lg bg-gray-800 text-gray-200 hover:bg-gray-700 text-xs font-semibold' : 'px-3 py-1.5 rounded-lg bg-emerald-600 text-white hover:bg-emerald-500 text-xs font-semibold';
    });

    // Anomaly Injection Buttons
    window.injectLiveAnomaly = async function(type) {
        try {
            const res = await fetch('/api/inject_anomaly', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ station_id: state.selectedStationId, type: type })
            });
            const data = await res.json();
            if (data.status === 'success') {
                showToast(`Injected '${type}' into live stream for ${state.selectedStationId}`);
                fetchLiveTelemetry();
                fetchAlerts();
            }
        } catch (e) {
            console.error('Error injecting anomaly:', e);
        }
    };

    function showToast(msg) {
        const toast = document.createElement('div');
        toast.className = 'fixed bottom-5 right-5 bg-cyan-900 border border-cyan-500 text-cyan-200 px-4 py-2 rounded-lg shadow-xl text-xs z-50 animate-bounce';
        toast.innerText = '⚡ ' + msg;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3500);
    }

    // Alerts Filter Buttons
    document.querySelectorAll('.filter-sev-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-sev-btn').forEach(b => b.classList.remove('bg-blue-600', 'text-white'));
            btn.classList.add('bg-blue-600', 'text-white');
            state.alertFilterSeverity = btn.dataset.severity;
            fetchAlerts();
        });
    });

    // Alerts Search
    document.getElementById('alert-search-input').addEventListener('input', (e) => {
        state.alertSearchQuery = e.target.value;
        fetchAlerts();
    });


    // ================================================================
    // Regional Event Corroboration + Sensor Fault Isolation
    // Integrated from the SIH regional-event feature build.
    // This is a front-end scenario simulator; live telemetry/health
    // APIs remain the source of truth for the production dashboard.
    // ================================================================
    const regionalScenario = {
        mode: 'normal',
        replayTimer: null,
        lastEvent: 'normal',
        graphOne: null,
        graphTwo: null
    };

    function ensureScenarioCharts() {
        const c1 = document.getElementById('scenario-chart-one');
        const c2 = document.getElementById('scenario-chart-two');
        if (!c1 || !c2 || typeof Chart === 'undefined') return;
        const common = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 350 },
            interaction: { mode: 'index', intersect: false },
            plugins: { legend: { labels: { color: '#9ca3af', boxWidth: 12, font: { size: 10 } } } },
            scales: {
                x: { ticks: { color: '#94a3b8', font: { size: 9 } }, grid: { color: 'rgba(148,163,184,.08)' } },
                y: { min: 0, max: 100, ticks: { color: '#94a3b8', callback: v => v + '%', font: { size: 9 } }, grid: { color: 'rgba(148,163,184,.08)' } }
            }
        };
        if (!regionalScenario.graphOne) regionalScenario.graphOne = new Chart(c1, { type:'line', data:{labels:[],datasets:[]}, options:common });
        if (!regionalScenario.graphTwo) regionalScenario.graphTwo = new Chart(c2, { type:'line', data:{labels:[],datasets:[]}, options:common });
    }

    function renderScenarioGraphs(mode) {
        const panel=document.getElementById('scenario-graph-panel');
        if(!panel) return;
        ensureScenarioCharts();
        if(!regionalScenario.graphOne || !regionalScenario.graphTwo) return;
        const labels=['T-4','T-3','T-2','T-1','T0','T+1','T+2','T+3'];
        const configs={
            regional:{
                title:'Regional Event — Anomaly Progression', subtitle:'Scenario visualization of a multi-station weather event.',
                oneTitle:'AI anomaly score / event confidence', twoTitle:'Spatial agreement across stations',
                one:[12,18,27,44,67,82,91,89], two:[8,16,29,48,63,78,86,82],
                l1:'Anomaly score', l2:'Spatial agreement',
                d2:[8,16,31,52,70,83,88,82]
            },
            fault:{
                title:'Sensor Fault — Fault Isolation Analysis', subtitle:'Scenario visualization of a local sensor deviation against neighboring stations.',
                oneTitle:'Fault/anomaly severity', twoTitle:'Sensor health vs spatial consistency',
                one:[8,11,17,34,58,76,88,86], two:[96,94,91,82,68,54,43,42],
                l1:'Fault severity', l2:'Sensor health', d2:[96,95,94,93,91,90,89,88]
            },
            normal:{
                title:'Scenario Analysis', subtitle:'No active scenario selected.',
                oneTitle:'Baseline anomaly score', twoTitle:'Baseline station consistency',
                one:[10,11,12,11,12,11,12,12], two:[96,96,97,96,97,96,97,96],
                l1:'Anomaly score', l2:'Consistency', d2:[96,97,96,97,96,97,96,97]
            }
        }[mode] || null;
        document.getElementById('scenario-graph-title').textContent=configs.title;
        document.getElementById('scenario-graph-subtitle').textContent=configs.subtitle;
        document.getElementById('scenario-chart-one-title').textContent=configs.oneTitle;
        document.getElementById('scenario-chart-two-title').textContent=configs.twoTitle;
        const isDark = document.documentElement.classList.contains('dark');
        const palette = mode === 'regional'
            ? { primary: isDark ? '#22d3ee' : '#0891b2', secondary: isDark ? '#fb7185' : '#e11d48', fill: isDark ? 'rgba(34,211,238,.14)' : 'rgba(8,145,178,.12)' }
            : mode === 'fault'
            ? { primary: isDark ? '#fbbf24' : '#d97706', secondary: isDark ? '#60a5fa' : '#2563eb', fill: isDark ? 'rgba(251,191,36,.14)' : 'rgba(217,119,6,.12)' }
            : { primary: isDark ? '#34d399' : '#059669', secondary: isDark ? '#94a3b8' : '#64748b', fill: isDark ? 'rgba(52,211,153,.12)' : 'rgba(5,150,105,.10)' };
        const applyScenarioChartTheme = (chart) => {
            if (!chart) return;
            chart.options.plugins.legend.labels.color = isDark ? '#cbd5e1' : '#334155';
            chart.options.scales.x.ticks.color = isDark ? '#94a3b8' : '#475569';
            chart.options.scales.y.ticks.color = isDark ? '#94a3b8' : '#475569';
            chart.options.scales.x.grid.color = isDark ? 'rgba(148,163,184,.08)' : 'rgba(71,85,105,.12)';
            chart.options.scales.y.grid.color = isDark ? 'rgba(148,163,184,.08)' : 'rgba(71,85,105,.12)';
        };
        regionalScenario.graphOne.data={labels,datasets:[{label:configs.l1,data:configs.one,tension:.35,fill:true,borderColor:palette.primary,backgroundColor:palette.fill,borderWidth:2,pointRadius:2,pointBackgroundColor:palette.primary}]};
        regionalScenario.graphTwo.data={labels,datasets:[{label:configs.l2,data:configs.two,tension:.35,borderColor:palette.primary,backgroundColor:palette.fill,borderWidth:2,pointRadius:2,pointBackgroundColor:palette.primary},{label:mode==='fault'?'Neighboring stations':'Corroboration',data:configs.d2,tension:.35,borderColor:palette.secondary,backgroundColor:'transparent',borderDash:[5,4],borderWidth:2,pointRadius:2,pointBackgroundColor:palette.secondary}]};
        applyScenarioChartTheme(regionalScenario.graphOne);
        applyScenarioChartTheme(regionalScenario.graphTwo);
        regionalScenario.graphOne.update(); regionalScenario.graphTwo.update();
        panel.classList.remove('hidden');
        requestAnimationFrame(()=>{ regionalScenario.graphOne.resize(); regionalScenario.graphTwo.resize(); panel.scrollIntoView({behavior:'smooth',block:'nearest'}); });
    }

    function closeScenarioGraphs(){
        document.getElementById('scenario-graph-panel')?.classList.add('hidden');
    }

    function renderRegionalScenario() {
        const mode = regionalScenario.mode;
        const values = {
            normal:   {status:'NORMAL', sub:'No regional corroboration detected', score:12, spatial:0, decision:'NORMAL CONDITIONS', confidence:12, wind:'Stable', pressure:'Stable', humidity:'Stable'},
            regional: {status:'REGIONAL EVENT', sub:'3+ nearby stations corroborate anomaly', score:89, spatial:82, decision:'PROBABLE REGIONAL WEATHER EVENT', confidence:89, wind:'Rising', pressure:'Falling', humidity:'Rising'},
            fault:    {status:'SENSOR FAULT', sub:'Local anomaly lacks spatial corroboration', score:86, spatial:8, decision:'PROBABLE SENSOR FAULT', confidence:86, wind:'Abnormal', pressure:'Stable', humidity:'Stable'}
        }[mode];

        const setText = (id, value) => { const el=document.getElementById(id); if(el) el.textContent=value; };
        const setWidth = (id, value) => { const el=document.getElementById(id); if(el) el.style.width=value+'%'; };

        setText('regional-status', values.status);
        setText('regional-status-sub', values.sub);
        setText('regional-score', values.score+'%');
        setWidth('regional-score-bar', values.score);
        setText('regional-spatial', values.spatial+'%');
        setText('regional-decision', values.decision);
        setText('regional-confidence', 'Confidence: '+values.confidence+'%');
        setText('regional-wind-trend', values.wind);
        setText('regional-pressure-trend', values.pressure);
        setText('regional-humidity-trend', values.humidity);

        const statusEl=document.getElementById('regional-status');
        statusEl.className='text-lg font-extrabold mt-1 '+(mode==='normal'?'text-emerald-400':mode==='regional'?'text-rose-400':'text-amber-400');

        const reasons = mode==='normal'
            ? ['✓ Wind pattern within expected range','✓ Pressure trend stable','✓ Nearby stations agree','✓ No multi-parameter anomaly']
            : mode==='regional'
            ? ['✓ Wind increased sharply','✓ Pressure decreased across stations','✓ Humidity increased','✓ 3+ nearby stations show similar pattern','✓ AI ensemble detects abnormal signature']
            : ['✓ Selected sensor deviates strongly','✓ Neighboring stations remain stable','✓ Spatial agreement is very low','✓ Pattern inconsistent with regional event'];

        const reasonsEl=document.getElementById('regional-reasons');
        if(reasonsEl) reasonsEl.innerHTML=reasons.map(x=>`<div>${x}</div>`).join('');

        updateSensorIsolationTable();
        updateScenarioMap();
    }

    function updateScenarioMap() {
        if (!state.stations || !state.markers) return;
        Object.keys(state.markers).forEach(id => {
            const marker=state.markers[id];
            const regionalAffected=regionalScenario.mode==='regional';
            const faultAffected=regionalScenario.mode==='fault' && id===state.selectedStationId;
            const abnormal=regionalAffected || faultAffected;
            if (marker && marker.setStyle) {
                marker.setStyle({
                    color: abnormal ? '#f43f5e' : '#10b981',
                    fillColor: abnormal ? '#f43f5e' : '#10b981'
                });
            }
        });
    }

    function updateSensorIsolationTable() {
        const body=document.getElementById('sensor-health-isolation-body');
        if(!body || !state.stations) return;
        const mode=regionalScenario.mode;
        body.innerHTML='';
        state.stations.forEach(st=>{
            const tr=document.createElement('tr');
            const isFault=mode==='fault' && st.id===state.selectedStationId;
            const isRegional=mode==='regional';
            const health=isFault?'42%':isRegional?'91%':'96%';
            const quality=isFault?'Poor':'Good';
            const spatial=isFault?'Inconsistent':isRegional?'Matched':'Consistent';
            const diagnosis=isFault?'Sensor fault suspected':isRegional?'Regional event corroborated':'Healthy';
            const cls=isFault?'text-rose-400':isRegional?'text-amber-400':'text-emerald-400';
            tr.innerHTML=`
                <td class="p-3 text-xs font-semibold text-gray-200">${st.name || st.id}</td>
                <td class="p-3 text-xs font-mono ${cls}">${health}</td>
                <td class="p-3 text-xs ${isFault?'text-rose-400':'text-gray-300'}">${quality}</td>
                <td class="p-3 text-xs ${isFault?'text-rose-400':isRegional?'text-amber-400':'text-gray-300'}">${spatial}</td>
                <td class="p-3 text-xs font-semibold ${cls}">${diagnosis}</td>`;
            body.appendChild(tr);
        });
    }

    function showRegionalTimeline(type) {
        const box=document.getElementById('regional-timeline');
        if(!box) return;
        const labels=['10:00','10:30','10:50','11:10','11:30'];
        const items=type==='regional'
            ? ['Normal','Wind rise','Pressure drop','Multi-station match','Regional event']
            : type==='fault'
            ? ['Normal','Local spike','Neighbor check','No match','Sensor fault']
            : ['Normal','Normal','Normal','Normal','Normal'];
        box.innerHTML=items.map((x,i)=>`
            <div class="min-w-[120px] px-2.5 py-2 rounded-lg border border-gray-800 bg-gray-950/30">
                <div class="text-[10px] font-bold text-gray-300">${labels[i]}</div>
                <div class="text-[10px] text-gray-500 mt-1">${x}</div>
            </div>`).join('');
        document.getElementById('regional-replay-text').textContent='Scenario ready. Press Replay Last Event to visualize the anomaly evolution.';
        document.getElementById('regional-replay-line').style.width='0%';
        document.getElementById('regional-replay-dot').style.left='0%';
    }

    function setRegionalMode(mode) {
        clearInterval(regionalScenario.replayTimer);
        regionalScenario.mode=mode;
        regionalScenario.lastEvent=mode;
        renderRegionalScenario();
        showRegionalTimeline(mode);
        if (mode === 'normal') {
            closeScenarioGraphs();
        } else {
            renderScenarioGraphs(mode);
        }
    }

    function replayRegionalEvent() {
        clearInterval(regionalScenario.replayTimer);
        const mode=regionalScenario.lastEvent || 'normal';
        const seq=mode==='regional'
            ? ['10:00 Normal','10:30 Wind increasing','10:50 Pressure falling','11:10 3 stations agree','11:30 Regional event corroborated']
            : mode==='fault'
            ? ['10:00 Normal','10:30 Local sensor spike','10:50 Neighbor check','11:10 No spatial correlation','11:30 Sensor fault suspected']
            : ['10:00 Normal','10:30 Stable','11:00 Stable','11:30 Stable'];
        let i=0;
        const textEl=document.getElementById('regional-replay-text');
        const lineEl=document.getElementById('regional-replay-line');
        const dotEl=document.getElementById('regional-replay-dot');
        const tick=()=>{
            if(i>=seq.length){clearInterval(regionalScenario.replayTimer);return;}
            if(textEl) textEl.textContent=seq[i];
            const pct=(i/(seq.length-1))*100;
            if(lineEl) lineEl.style.width=pct+'%';
            if(dotEl) dotEl.style.left=pct+'%';
            i++;
        };
        tick();
        regionalScenario.replayTimer=setInterval(tick,700);
    }

    document.getElementById('regional-event-btn')?.addEventListener('click',()=>setRegionalMode('regional'));
    document.getElementById('sensor-fault-btn')?.addEventListener('click',()=>setRegionalMode('fault'));
    document.getElementById('regional-reset-btn')?.addEventListener('click',()=>setRegionalMode('normal'));
    document.getElementById('regional-replay-btn')?.addEventListener('click',replayRegionalEvent);
    document.getElementById('scenario-graph-close')?.addEventListener('click', closeScenarioGraphs);

    setRegionalMode('normal');

    // Start Polling Loops
    fetchStations().then(() => {
        fetchLiveTelemetry();
        fetchHealthMetrics();
        fetchAlerts();
        fetchSystemMetrics();
    });

    // Stream Loop
    setInterval(() => {
        if (state.isPlaying) {
            fetchLiveTelemetry();
            fetchSystemMetrics();
        }
    }, state.pollIntervalMs);

    // Slower Loop for Alerts & Health
    setInterval(() => {
        fetchHealthMetrics();
        fetchAlerts();
    }, 6000);
});
