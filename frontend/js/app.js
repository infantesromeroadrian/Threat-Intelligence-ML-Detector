/**
 * Main Application Logic
 * Threat Intelligence Aggregator Frontend
 */

const app = {
    currentSection: 'dashboard',
    currentPage: { cves: 0, iocs: 0, threats: 0, alerts: 0 },
    pageSize: 20,

    // =========================================================================
    // Initialization
    // =========================================================================

    async init() {
        console.log('🚀 Initializing Threat Intelligence Aggregator...');
        
        this.setupEventListeners();
        await this.checkAPIStatus();
        await this.loadDashboard();
    },

    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.navigate(e.target.dataset.section);
            });
        });

        // Search inputs (debounced)
        const searchInputs = ['cve-search', 'ioc-search', 'threat-search'];
        searchInputs.forEach(id => {
            const input = document.getElementById(id);
            if (input) {
                let timeout;
                input.addEventListener('input', () => {
                    clearTimeout(timeout);
                    timeout = setTimeout(() => this.handleSearch(id), 500);
                });
            }
        });

        // Filter selects
        document.getElementById('cve-severity-filter')?.addEventListener('change', () => this.loadCVEs());
        document.getElementById('ioc-type-filter')?.addEventListener('change', () => this.loadIOCs());
        document.getElementById('threat-type-filter')?.addEventListener('change', () => this.loadThreats());
        document.getElementById('alert-status-filter')?.addEventListener('change', () => this.loadAlerts());
        document.getElementById('alert-severity-filter')?.addEventListener('change', () => this.loadAlerts());
    },

    // =========================================================================
    // Navigation
    // =========================================================================

    navigate(section) {
        this.currentSection = section;

        // Update nav buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.section === section);
        });

        // Show/hide sections
        document.querySelectorAll('.section').forEach(sec => {
            sec.classList.toggle('active', sec.id === `${section}-section`);
        });

        // Load data for section
        this.loadSectionData(section);
    },

    async loadSectionData(section) {
        switch (section) {
            case 'dashboard':
                await this.loadDashboard();
                break;
            case 'cves':
                await this.loadCVEs();
                break;
            case 'iocs':
                await this.loadIOCs();
                break;
            case 'threats':
                await this.loadThreats();
                break;
            case 'topics':
                await this.loadTopics();
                break;
            case 'alerts':
                await this.loadAlerts();
                break;
        }
    },

    // =========================================================================
    // API Status
    // =========================================================================

    async checkAPIStatus() {
        const indicator = document.getElementById('apiStatus');
        const statusText = document.getElementById('apiStatusText');

        try {
            await api.checkHealth();
            indicator.classList.add('online');
            indicator.classList.remove('offline');
            statusText.textContent = 'API Online';
        } catch (error) {
            indicator.classList.add('offline');
            indicator.classList.remove('online');
            statusText.textContent = 'API Offline';
            console.error('API health check failed:', error);
        }
    },

    // =========================================================================
    // Dashboard
    // =========================================================================

    async loadDashboard() {
        try {
            // Load stats
            const [cveStats, iocStats, threatStats, alertStats] = await Promise.all([
                api.getCVEStats().catch(() => ({ total_cves: 0, critical_count: 0 })),
                api.getIOCStats().catch(() => ({ total_iocs: 0, active_count: 0 })),
                api.getThreatStats().catch(() => ({ total_documents: 0, by_severity: {} })),
                api.getAlertStats().catch(() => ({ active_count: 0, critical_count: 0 }))
            ]);

            // Update stat cards
            document.getElementById('stat-total-cves').textContent = cveStats.total_cves || 0;
            document.getElementById('stat-critical-cves').textContent = `${cveStats.critical_count || 0} Critical`;
            
            document.getElementById('stat-total-iocs').textContent = iocStats.total_iocs || 0;
            document.getElementById('stat-active-iocs').textContent = `${iocStats.active_count || 0} Active`;
            
            document.getElementById('stat-total-threats').textContent = threatStats.total_documents || 0;
            const highThreats = (threatStats.by_severity?.HIGH || 0) + (threatStats.by_severity?.CRITICAL || 0);
            document.getElementById('stat-high-threats').textContent = `${highThreats} High Severity`;
            
            document.getElementById('stat-active-alerts').textContent = alertStats.active_count || 0;
            document.getElementById('stat-critical-alerts').textContent = `${alertStats.critical_count || 0} Critical`;

            // Load recent CVEs
            const recentCVEs = await api.getCriticalCVEs().catch(() => []);
            this.renderRecentCVEs(recentCVEs.slice(0, 5));

            // Load active alerts
            const activeAlerts = await api.getActiveAlerts().catch(() => []);
            this.renderActiveAlerts(activeAlerts.slice(0, 5));

        } catch (error) {
            console.error('Error loading dashboard:', error);
        }
    },

    renderRecentCVEs(cves) {
        const container = document.getElementById('recent-cves');
        
        if (!cves || cves.length === 0) {
            container.innerHTML = '<p class="empty-state">No critical CVEs found</p>';
            return;
        }

        container.innerHTML = cves.map(cve => `
            <div class="item-card">
                <div class="item-header">
                    <h4 class="item-title">${cve.cve_id}</h4>
                    <span class="item-badge badge-${cve.severity.toLowerCase()}">${cve.severity}</span>
                </div>
                <p class="item-description">${this.truncate(cve.description, 150)}</p>
                <div class="item-meta">
                    <span>📅 ${this.formatDate(cve.published_date)}</span>
                    ${cve.cvss ? `<span>🎯 CVSS: ${cve.cvss.base_score}</span>` : ''}
                </div>
            </div>
        `).join('');
    },

    renderActiveAlerts(alerts) {
        const container = document.getElementById('active-alerts');
        
        if (!alerts || alerts.length === 0) {
            container.innerHTML = '<p class="empty-state">No active alerts</p>';
            return;
        }

        container.innerHTML = alerts.map(alert => `
            <div class="item-card">
                <div class="item-header">
                    <h4 class="item-title">${alert.title}</h4>
                    <span class="item-badge badge-${alert.severity.toLowerCase()}">${alert.severity}</span>
                </div>
                <p class="item-description">${this.truncate(alert.description, 120)}</p>
                <div class="item-meta">
                    <span>🕐 ${this.formatAge(alert.age_hours)}</span>
                    <span>📊 ${alert.status}</span>
                </div>
                <div class="item-actions">
                    <button class="btn btn-sm btn-success" onclick="app.acknowledgeAlert('${alert.alert_id}')">
                        ✅ Acknowledge
                    </button>
                    <button class="btn btn-sm btn-warning" onclick="app.showAlertDetails('${alert.alert_id}')">
                        👁️ Details
                    </button>
                </div>
            </div>
        `).join('');
    },

    // =========================================================================
    // CVEs
    // =========================================================================

    async loadCVEs() {
        const container = document.getElementById('cves-list');
        container.innerHTML = '<div class="loading">Loading CVEs...</div>';

        try {
            const params = {
                skip: this.currentPage.cves * this.pageSize,
                limit: this.pageSize
            };

            const severity = document.getElementById('cve-severity-filter')?.value;
            if (severity) params.severity = severity;

            const keyword = document.getElementById('cve-search')?.value;
            if (keyword) params.keyword = keyword;

            const data = await api.getCVEs(params);
            this.renderCVEs(data.items || []);
            this.renderPagination('cves', data.total || 0);

        } catch (error) {
            container.innerHTML = `<div class="error">Error loading CVEs: ${error.message}</div>`;
        }
    },

    renderCVEs(cves) {
        const container = document.getElementById('cves-list');
        
        if (cves.length === 0) {
            container.innerHTML = '<p class="empty-state">No CVEs found</p>';
            return;
        }

        container.innerHTML = cves.map(cve => `
            <div class="item-card">
                <div class="item-header">
                    <h4 class="item-title">${cve.cve_id}</h4>
                    <span class="item-badge badge-${cve.severity.toLowerCase()}">${cve.severity}</span>
                </div>
                <p class="item-description">${this.truncate(cve.description, 200)}</p>
                <div class="item-meta">
                    <span>📅 ${this.formatDate(cve.published_date)}</span>
                    ${cve.cvss ? `<span>🎯 CVSS: ${cve.cvss.base_score}</span>` : ''}
                    ${cve.cwe_ids?.length ? `<span>🔍 CWE: ${cve.cwe_ids[0]}</span>` : ''}
                    <span>📦 Source: ${cve.source}</span>
                </div>
            </div>
        `).join('');
    },

    // =========================================================================
    // IOCs
    // =========================================================================

    async loadIOCs() {
        const container = document.getElementById('iocs-list');
        container.innerHTML = '<div class="loading">Loading IOCs...</div>';

        try {
            const params = {
                skip: this.currentPage.iocs * this.pageSize,
                limit: this.pageSize
            };

            const type = document.getElementById('ioc-type-filter')?.value;
            if (type) params.ioc_type = type;

            const search = document.getElementById('ioc-search')?.value;
            if (search) params.search = search;

            const data = await api.getIOCs(params);
            this.renderIOCs(data.items || []);
            this.renderPagination('iocs', data.total || 0);

        } catch (error) {
            container.innerHTML = `<div class="error">Error loading IOCs: ${error.message}</div>`;
        }
    },

    renderIOCs(iocs) {
        const container = document.getElementById('iocs-list');
        
        if (iocs.length === 0) {
            container.innerHTML = '<p class="empty-state">No IOCs found</p>';
            return;
        }

        container.innerHTML = iocs.map(ioc => `
            <div class="item-card">
                <div class="item-header">
                    <h4 class="item-title"><code>${ioc.value}</code></h4>
                    <span class="item-badge badge-${ioc.threat_level.toLowerCase()}">${ioc.threat_level}</span>
                </div>
                <div class="item-meta">
                    <span>🏷️ Type: ${ioc.ioc_type}</span>
                    <span>📊 Confidence: ${(ioc.confidence_score * 100).toFixed(0)}%</span>
                    <span>📅 ${this.formatDate(ioc.first_seen)}</span>
                </div>
                ${ioc.context ? `<p class="item-description">${this.truncate(ioc.context, 150)}</p>` : ''}
            </div>
        `).join('');
    },

    // =========================================================================
    // Threats
    // =========================================================================

    async loadThreats() {
        const container = document.getElementById('threats-list');
        container.innerHTML = '<div class="loading">Loading threats...</div>';

        try {
            const params = {
                skip: this.currentPage.threats * this.pageSize,
                limit: this.pageSize
            };

            const type = document.getElementById('threat-type-filter')?.value;
            if (type) params.threat_type = type;

            const keyword = document.getElementById('threat-search')?.value;
            if (keyword) params.keyword = keyword;

            const data = await api.getThreats(params);
            this.renderThreats(data.items || []);
            this.renderPagination('threats', data.total || 0);

        } catch (error) {
            container.innerHTML = `<div class="error">Error loading threats: ${error.message}</div>`;
        }
    },

    renderThreats(threats) {
        const container = document.getElementById('threats-list');
        
        if (threats.length === 0) {
            container.innerHTML = '<p class="empty-state">No threats found</p>';
            return;
        }

        container.innerHTML = threats.map(threat => `
            <div class="item-card">
                <div class="item-header">
                    <h4 class="item-title">${threat.title}</h4>
                    <span class="item-badge badge-${threat.severity.toLowerCase()}">${threat.severity}</span>
                </div>
                <p class="item-description">${this.truncate(threat.content, 200)}</p>
                <div class="item-meta">
                    <span>🏷️ ${threat.threat_type}</span>
                    <span>📅 ${this.formatDate(threat.published_date)}</span>
                    <span>📦 ${threat.source}</span>
                    <span>🎯 IOCs: ${threat.iocs_count}</span>
                </div>
            </div>
        `).join('');
    },

    // =========================================================================
    // Topics
    // =========================================================================

    async loadTopics() {
        const container = document.getElementById('topics-list');
        container.innerHTML = '<div class="loading">Loading topics...</div>';

        try {
            const data = await api.getTopics({ skip: 0, limit: 100 });
            this.renderTopics(data.items || []);
        } catch (error) {
            container.innerHTML = `<div class="error">Error loading topics: ${error.message}</div>`;
        }
    },

    renderTopics(topics) {
        const container = document.getElementById('topics-list');
        
        if (topics.length === 0) {
            container.innerHTML = '<p class="empty-state">No topics discovered yet</p>';
            return;
        }

        container.innerHTML = topics.map(topic => {
            const keywords = topic.keywords.slice(0, 10).map(kw => kw.word).join(', ');
            
            return `
                <div class="item-card">
                    <div class="item-header">
                        <h4 class="item-title">${topic.label || `Topic ${topic.topic_number}`}</h4>
                        ${topic.is_significant ? '<span class="item-badge badge-high">⭐ Significant</span>' : ''}
                    </div>
                    <p class="item-description"><strong>Keywords:</strong> ${keywords}</p>
                    <div class="item-meta">
                        <span>📄 Documents: ${topic.document_count}</span>
                        ${topic.coherence_score ? `<span>📊 Coherence: ${topic.coherence_score.toFixed(3)}</span>` : ''}
                        <span>📅 ${this.formatDate(topic.discovery_date)}</span>
                    </div>
                </div>
            `;
        }).join('');
    },

    // =========================================================================
    // Alerts
    // =========================================================================

    async loadAlerts() {
        const container = document.getElementById('alerts-list');
        container.innerHTML = '<div class="loading">Loading alerts...</div>';

        try {
            const params = {
                skip: this.currentPage.alerts * this.pageSize,
                limit: this.pageSize
            };

            const status = document.getElementById('alert-status-filter')?.value;
            if (status) params.status = status;

            const severity = document.getElementById('alert-severity-filter')?.value;
            if (severity) params.severity = severity;

            const data = await api.getAlerts(params);
            this.renderAlerts(data.items || []);
            this.renderPagination('alerts', data.total || 0);

        } catch (error) {
            container.innerHTML = `<div class="error">Error loading alerts: ${error.message}</div>`;
        }
    },

    renderAlerts(alerts) {
        const container = document.getElementById('alerts-list');
        
        if (alerts.length === 0) {
            container.innerHTML = '<p class="empty-state">No alerts found</p>';
            return;
        }

        container.innerHTML = alerts.map(alert => `
            <div class="item-card">
                <div class="item-header">
                    <h4 class="item-title">${alert.title}</h4>
                    <span class="item-badge badge-${alert.severity.toLowerCase()}">${alert.severity}</span>
                </div>
                <p class="item-description">${this.truncate(alert.description, 150)}</p>
                <div class="item-meta">
                    <span>📊 Status: ${alert.status}</span>
                    <span>🕐 ${this.formatAge(alert.age_hours)}</span>
                    <span>🎯 Confidence: ${(alert.confidence_score * 100).toFixed(0)}%</span>
                </div>
                ${alert.is_active ? `
                    <div class="item-actions">
                        ${alert.status === 'NEW' ? `
                            <button class="btn btn-sm btn-success" onclick="app.acknowledgeAlert('${alert.alert_id}')">
                                ✅ Acknowledge
                            </button>
                        ` : ''}
                        <button class="btn btn-sm btn-primary" onclick="app.resolveAlert('${alert.alert_id}')">
                            ✔️ Resolve
                        </button>
                        <button class="btn btn-sm btn-warning" onclick="app.showAlertDetails('${alert.alert_id}')">
                            👁️ Details
                        </button>
                    </div>
                ` : ''}
            </div>
        `).join('');
    },

    // =========================================================================
    // Alert Actions
    // =========================================================================

    async acknowledgeAlert(alertId) {
        try {
            await api.acknowledgeAlert(alertId, 'Web Dashboard User');
            alert('Alert acknowledged successfully!');
            await this.loadAlerts();
            await this.loadDashboard();
        } catch (error) {
            alert(`Error acknowledging alert: ${error.message}`);
        }
    },

    async resolveAlert(alertId) {
        const notes = prompt('Resolution notes (optional):');
        if (notes === null) return; // Cancelled

        try {
            await api.resolveAlert(alertId, 'Web Dashboard User', notes || '');
            alert('Alert resolved successfully!');
            await this.loadAlerts();
            await this.loadDashboard();
        } catch (error) {
            alert(`Error resolving alert: ${error.message}`);
        }
    },

    async showAlertDetails(alertId) {
        try {
            const alert = await api.getAlertById(alertId);
            const modalBody = document.getElementById('alertModalBody');
            
            modalBody.innerHTML = `
                <h2>${alert.title}</h2>
                <p><strong>Severity:</strong> <span class="item-badge badge-${alert.severity.toLowerCase()}">${alert.severity}</span></p>
                <p><strong>Status:</strong> ${alert.status}</p>
                <p><strong>Type:</strong> ${alert.alert_type}</p>
                <p><strong>Description:</strong></p>
                <p>${alert.description}</p>
                <p><strong>Created:</strong> ${this.formatDate(alert.created_at)}</p>
                <p><strong>Age:</strong> ${this.formatAge(alert.age_hours)}</p>
                ${alert.actionable_items?.length ? `
                    <p><strong>Recommended Actions:</strong></p>
                    <ul>
                        ${alert.actionable_items.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                ` : ''}
            `;

            document.getElementById('alertModal').classList.add('active');
        } catch (error) {
            alert(`Error loading alert details: ${error.message}`);
        }
    },

    closeModal() {
        document.getElementById('alertModal').classList.remove('active');
    },

    // =========================================================================
    // Pagination
    // =========================================================================

    renderPagination(section, total) {
        const container = document.getElementById(`${section}-pagination`);
        if (!container) return;

        const totalPages = Math.ceil(total / this.pageSize);
        const currentPage = this.currentPage[section];

        container.innerHTML = `
            <button ${currentPage === 0 ? 'disabled' : ''} 
                    onclick="app.changePage('${section}', ${currentPage - 1})">
                ← Previous
            </button>
            <span>Page ${currentPage + 1} of ${totalPages || 1}</span>
            <button ${currentPage >= totalPages - 1 ? 'disabled' : ''} 
                    onclick="app.changePage('${section}', ${currentPage + 1})">
                Next →
            </button>
        `;
    },

    changePage(section, page) {
        this.currentPage[section] = page;
        this.loadSectionData(section);
    },

    // =========================================================================
    // Utility Functions
    // =========================================================================

    truncate(text, length) {
        if (!text) return '';
        return text.length > length ? text.substring(0, length) + '...' : text;
    },

    formatDate(dateString) {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
    },

    formatAge(hours) {
        if (hours < 1) return `${Math.round(hours * 60)} minutes ago`;
        if (hours < 24) return `${Math.round(hours)} hours ago`;
        return `${Math.round(hours / 24)} days ago`;
    },

    handleSearch(inputId) {
        const section = inputId.split('-')[0]; // 'cve', 'ioc', 'threat'
        this.currentPage[section + 's'] = 0; // Reset to first page
        this.loadSectionData(section + 's');
    },

    showDiscoverTopics() {
        alert('Topic discovery feature coming soon! This will run LDA topic modeling on recent threat intelligence documents.');
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    app.init();

    // Refresh dashboard every 60 seconds
    setInterval(() => {
        if (app.currentSection === 'dashboard') {
            app.loadDashboard();
        }
    }, 60000);
});
