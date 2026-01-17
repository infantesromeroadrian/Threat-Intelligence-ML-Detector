/**
 * API Client for Threat Intelligence Aggregator
 * 
 * Handles all HTTP requests to the backend REST API.
 */

const API_BASE_URL = 'http://localhost:8000';

class ThreatIntelAPI {
    constructor(baseURL = API_BASE_URL) {
        this.baseURL = baseURL;
    }

    /**
     * Generic request handler
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ 
                    error: `HTTP ${response.status}` 
                }));
                throw new Error(error.error || error.detail || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error (${endpoint}):`, error);
            throw error;
        }
    }

    // =========================================================================
    // Health & Status
    // =========================================================================

    async checkHealth() {
        return this.request('/health');
    }

    // =========================================================================
    // CVEs
    // =========================================================================

    async getCVEs(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/api/cves?${query}`);
    }

    async getCVEStats() {
        return this.request('/api/cves/stats');
    }

    async getRecentCVEs(limit = 10) {
        return this.request(`/api/cves/recent?limit=${limit}`);
    }

    async getCriticalCVEs() {
        return this.request('/api/cves/critical');
    }

    async getCVEById(cveId) {
        return this.request(`/api/cves/${cveId}`);
    }

    async createCVE(data) {
        return this.request('/api/cves', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async deleteCVE(cveId) {
        return this.request(`/api/cves/${cveId}`, {
            method: 'DELETE'
        });
    }

    // =========================================================================
    // IOCs
    // =========================================================================

    async getIOCs(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/api/iocs?${query}`);
    }

    async getIOCStats() {
        return this.request('/api/iocs/stats');
    }

    async getRecentIOCs(limit = 10) {
        return this.request(`/api/iocs/recent?limit=${limit}`);
    }

    async getActiveIOCs() {
        return this.request('/api/iocs/active');
    }

    async getIOCsByType(type) {
        return this.request(`/api/iocs/type/${type}`);
    }

    async getIOCByValue(value) {
        return this.request(`/api/iocs/${value}`);
    }

    async createIOC(data) {
        return this.request('/api/iocs', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    // =========================================================================
    // Threat Intelligence
    // =========================================================================

    async getThreats(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/api/threats?${query}`);
    }

    async getThreatStats() {
        return this.request('/api/threats/stats');
    }

    async getRecentThreats(limit = 10) {
        return this.request(`/api/threats/recent?limit=${limit}`);
    }

    async getHighSeverityThreats() {
        return this.request('/api/threats/high-severity');
    }

    async getThreatById(documentId) {
        return this.request(`/api/threats/${documentId}`);
    }

    // =========================================================================
    // Topics
    // =========================================================================

    async getTopics(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/api/topics?${query}`);
    }

    async getTopicStats() {
        return this.request('/api/topics/stats');
    }

    async getSignificantTopics() {
        return this.request('/api/topics/significant');
    }

    async getTopicById(topicId) {
        return this.request(`/api/topics/${topicId}`);
    }

    async updateTopicLabel(topicId, label) {
        return this.request(`/api/topics/${topicId}/label`, {
            method: 'PUT',
            body: JSON.stringify({ label })
        });
    }

    // =========================================================================
    // Alerts
    // =========================================================================

    async getAlerts(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/api/alerts?${query}`);
    }

    async getAlertStats() {
        return this.request('/api/alerts/stats');
    }

    async getActiveAlerts() {
        return this.request('/api/alerts/active');
    }

    async getCriticalAlerts() {
        return this.request('/api/alerts/critical');
    }

    async getAlertById(alertId) {
        return this.request(`/api/alerts/${alertId}`);
    }

    async acknowledgeAlert(alertId, acknowledgedBy) {
        return this.request(`/api/alerts/${alertId}/acknowledge`, {
            method: 'POST',
            body: JSON.stringify({ acknowledged_by: acknowledgedBy })
        });
    }

    async resolveAlert(alertId, resolvedBy, notes = '') {
        return this.request(`/api/alerts/${alertId}/resolve`, {
            method: 'POST',
            body: JSON.stringify({ 
                resolved_by: resolvedBy, 
                resolution_notes: notes 
            })
        });
    }

    async markFalsePositive(alertId, markedBy, notes = '') {
        return this.request(`/api/alerts/${alertId}/false-positive`, {
            method: 'POST',
            body: JSON.stringify({ 
                marked_by: markedBy, 
                notes: notes 
            })
        });
    }
}

// Create global API instance
const api = new ThreatIntelAPI();
