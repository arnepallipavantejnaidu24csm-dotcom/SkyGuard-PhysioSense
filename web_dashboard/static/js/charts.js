/**
 * AWS Dashboard Chart.js Integration
 * Implements real-time dual-axis charts, raw vs. corrected lines, and confidence bands.
 */

class AWSChartManager {
    constructor() {
        this.charts = {};
    }

    initTelemetryChart(canvasId, label, unit, colorPrimary, colorRaw) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    // Corrected Value (Thick Solid Line)
                    {
                        label: `Corrected (${unit})`,
                        data: [],
                        borderColor: colorPrimary,
                        backgroundColor: colorPrimary + '22',
                        borderWidth: 2.5,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        tension: 0.3,
                        fill: false,
                        yAxisID: 'y',
                    },
                    // Raw Sensor Reading (Dotted / Dashed Line)
                    {
                        label: `Raw Obs (${unit})`,
                        data: [],
                        borderColor: colorRaw || '#9ca3af',
                        borderWidth: 1.5,
                        borderDash: [4, 4],
                        pointRadius: 0,
                        pointHoverRadius: 3,
                        tension: 0.2,
                        fill: false,
                        yAxisID: 'y',
                    },
                    // 95% Upper CI
                    {
                        label: '95% CI Upper',
                        data: [],
                        borderColor: 'transparent',
                        backgroundColor: colorPrimary + '18',
                        fill: '+1', // Fill down to next dataset (Lower CI)
                        pointRadius: 0,
                        tension: 0.3,
                        yAxisID: 'y',
                    },
                    // 95% Lower CI
                    {
                        label: '95% CI Lower',
                        data: [],
                        borderColor: 'transparent',
                        backgroundColor: 'transparent',
                        fill: false,
                        pointRadius: 0,
                        tension: 0.3,
                        yAxisID: 'y',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 300 },
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            color: '#9ca3af',
                            font: { size: 11 },
                            filter: (item) => !item.text.includes('CI') // Hide CI from legend clutter
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(17, 24, 39, 0.95)',
                        borderColor: 'rgba(255, 255, 255, 0.1)',
                        borderWidth: 1,
                        titleColor: '#f3f4f6',
                        bodyColor: '#e5e7eb',
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: {
                            color: '#6b7280',
                            font: { size: 10 },
                            maxTicksLimit: 6,
                        }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.06)' },
                        ticks: {
                            color: '#9ca3af',
                            font: { size: 10 },
                        }
                    }
                }
            }
        });

        this.charts[canvasId] = chart;
        return chart;
    }

    initDualAxisComparisonChart(canvasId) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return null;

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Temperature (°C)',
                        data: [],
                        borderColor: '#f43f5e',
                        backgroundColor: 'rgba(244, 63, 94, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0,
                        yAxisID: 'yTemp',
                    },
                    {
                        label: 'Pressure (hPa)',
                        data: [],
                        borderColor: '#06b6d4',
                        backgroundColor: 'rgba(6, 182, 212, 0.1)',
                        borderWidth: 2,
                        tension: 0.3,
                        pointRadius: 0,
                        yAxisID: 'yPres',
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#9ca3af', font: { size: 11 } } },
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.04)' },
                        ticks: { color: '#6b7280', maxTicksLimit: 8 }
                    },
                    yTemp: {
                        type: 'linear',
                        position: 'left',
                        grid: { color: 'rgba(244, 63, 94, 0.08)' },
                        ticks: { color: '#f43f5e' },
                        title: { display: true, text: 'Temperature (°C)', color: '#f43f5e', font: { size: 11 } }
                    },
                    yPres: {
                        type: 'linear',
                        position: 'right',
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#06b6d4' },
                        title: { display: true, text: 'Pressure (hPa)', color: '#06b6d4', font: { size: 11 } }
                    }
                }
            }
        });

        this.charts[canvasId] = chart;
        return chart;
    }

    updateStreamChart(canvasId, timestamps, correctedVals, rawVals, ciUpper, ciLower) {
        const chart = this.charts[canvasId];
        if (!chart) return;

        // Truncate timestamps for clean display
        const shortLabels = timestamps.map(t => {
            if (typeof t === 'string' && t.includes(' ')) {
                return t.split(' ')[1].substring(0, 5);
            }
            return t;
        });

        chart.data.labels = shortLabels;
        chart.data.datasets[0].data = correctedVals;
        chart.data.datasets[1].data = rawVals;
        if (ciUpper && ciLower) {
            chart.data.datasets[2].data = ciUpper;
            chart.data.datasets[3].data = ciLower;
        }
        chart.update('none');
    }
    refreshTheme() {
        const isDark = document.documentElement.classList.contains('dark');

        const textColor = isDark ? '#94a3b8' : '#475569';
        const gridColor = isDark
            ? 'rgba(148, 163, 184, 0.12)'
            : 'rgba(71, 85, 105, 0.15)';

        Object.values(this.charts).forEach(chart => {
            if (!chart) return;

            if (chart.options.scales) {
                Object.values(chart.options.scales).forEach(scale => {
                    if (scale.ticks) {
                        scale.ticks.color = textColor;
                    }

                    if (scale.grid) {
                        scale.grid.color = gridColor;
                    }
                });
            }

            chart.update('none');
        });
    }
}

window.awsChartManager = new AWSChartManager();
