# 🎨 Frontend Architecture Plan

**Framework**: Vanilla HTML + JavaScript + CSS  
**Server**: Nginx (static files)  
**API Communication**: Fetch API (REST calls to FastAPI backend)

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    USER BROWSER                             │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │            HTML + CSS + JavaScript                    │ │
│  │  (Single Page Application - Vanilla JS)               │ │
│  └────────────────────┬──────────────────────────────────┘ │
└───────────────────────┼──────────────────────────────────────┘
                        │
                        │ HTTP/REST (Fetch API)
                        │
┌───────────────────────▼──────────────────────────────────────┐
│                  NGINX (Static Server)                       │
│              Port 80 - Serves HTML/JS/CSS                    │
└────────────────────────────────────────────────────────────┬─┘
                        │
                        │ Proxy /api → http://api:8000
                        │
┌───────────────────────▼──────────────────────────────────────┐
│                  FastAPI Backend                             │
│              Port 8000 - REST API + JSON                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Frontend Directory Structure

```
frontend/
├── index.html                  # Main entry point
├── css/
│   ├── main.css               # Global styles
│   ├── components.css         # Component-specific styles
│   └── themes.css             # Dark/Light theme
├── js/
│   ├── app.js                 # Main application logic
│   ├── api.js                 # API client (fetch wrapper)
│   ├── router.js              # Client-side routing
│   ├── components/
│   │   ├── cve-card.js       # CVE display component
│   │   ├── ioc-table.js      # IOC table component
│   │   ├── topic-chart.js    # LDA topic visualization
│   │   ├── alert-badge.js    # Alert notification
│   │   └── search-bar.js     # Search component
│   ├── pages/
│   │   ├── dashboard.js      # Main dashboard
│   │   ├── cves.js           # CVE explorer
│   │   ├── iocs.js           # IOC viewer
│   │   ├── topics.js         # Topic analysis
│   │   └── alerts.js         # Alerts management
│   └── utils/
│       ├── date.js           # Date formatting
│       ├── format.js         # Data formatters
│       └── validation.js     # Input validation
├── assets/
│   ├── icons/                # SVG icons
│   └── images/               # Images
├── lib/                       # Third-party libraries (optional)
│   ├── chart.js              # Charting (if needed)
│   └── d3.min.js             # D3.js for complex viz
└── nginx.conf                # Nginx configuration
```

---

## 🎨 UI Components (Vanilla JS)

### 1. CVE Card Component
```html
<div class="cve-card" data-severity="critical">
  <div class="cve-header">
    <span class="cve-id">CVE-2024-1234</span>
    <span class="severity-badge critical">CRITICAL</span>
  </div>
  <p class="cve-description">Buffer overflow in...</p>
  <div class="cve-footer">
    <span class="cvss-score">9.8</span>
    <span class="date">2024-01-15</span>
  </div>
</div>
```

**JavaScript:**
```javascript
// js/components/cve-card.js
export function createCVECard(cve) {
  return `
    <div class="cve-card" data-severity="${cve.severity}">
      <div class="cve-header">
        <span class="cve-id">${cve.cve_id}</span>
        <span class="severity-badge ${cve.severity.toLowerCase()}">
          ${cve.severity}
        </span>
      </div>
      <p class="cve-description">${cve.description}</p>
      <div class="cve-footer">
        <span class="cvss-score">${cve.cvss_score || 'N/A'}</span>
        <span class="date">${formatDate(cve.published_date)}</span>
      </div>
    </div>
  `;
}
```

### 2. IOC Table Component
```javascript
// js/components/ioc-table.js
export function createIOCTable(iocs) {
  const rows = iocs.map(ioc => `
    <tr class="ioc-row" data-confidence="${ioc.confidence}">
      <td><code>${ioc.value}</code></td>
      <td><span class="ioc-type">${ioc.ioc_type}</span></td>
      <td><span class="confidence ${ioc.confidence.toLowerCase()}">
        ${ioc.confidence}
      </span></td>
      <td>${formatDate(ioc.extracted_at)}</td>
      <td><button onclick="copyIOC('${ioc.value}')">Copy</button></td>
    </tr>
  `).join('');
  
  return `
    <table class="ioc-table">
      <thead>
        <tr>
          <th>Value</th>
          <th>Type</th>
          <th>Confidence</th>
          <th>Extracted</th>
          <th>Actions</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}
```

### 3. Topic Visualization (D3.js)
```javascript
// js/components/topic-chart.js
import * as d3 from '../lib/d3.min.js';

export function renderTopicChart(topics, containerId) {
  const svg = d3.select(`#${containerId}`)
    .append('svg')
    .attr('width', 800)
    .attr('height', 600);
    
  // Create force-directed graph of topics
  const nodes = topics.map(topic => ({
    id: topic.topic_id,
    label: topic.label || `Topic ${topic.topic_number}`,
    size: topic.document_count
  }));
  
  // ... D3 visualization logic
}
```

---

## 🔌 API Client (Fetch Wrapper)

```javascript
// js/api.js
const API_BASE_URL = '/api';  // Proxied by Nginx to http://api:8000

class APIClient {
  async request(endpoint, options = {}) {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    };
    
    try {
      const response = await fetch(url, config);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
  
  // CVE endpoints
  async getCVEs(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/cves?${query}`);
  }
  
  async getCVEById(cveId) {
    return this.request(`/cves/${cveId}`);
  }
  
  // IOC endpoints
  async getIOCs(params = {}) {
    const query = new URLSearchParams(params).toString();
    return this.request(`/iocs?${query}`);
  }
  
  // Topic endpoints
  async getTopics() {
    return this.request('/topics');
  }
  
  // Alert endpoints
  async getAlerts(status = 'active') {
    return this.request(`/alerts?status=${status}`);
  }
  
  async acknowledgeAlert(alertId) {
    return this.request(`/alerts/${alertId}/acknowledge`, {
      method: 'POST'
    });
  }
}

export const api = new APIClient();
```

---

## 🧭 Client-Side Routing

```javascript
// js/router.js
class Router {
  constructor(routes) {
    this.routes = routes;
    this.currentPage = null;
    
    window.addEventListener('hashchange', () => this.route());
    this.route();
  }
  
  route() {
    const hash = window.location.hash.slice(1) || '/';
    const route = this.routes[hash] || this.routes['/'];
    
    if (this.currentPage !== hash) {
      this.currentPage = hash;
      route();
    }
  }
  
  navigate(path) {
    window.location.hash = path;
  }
}

// Usage in app.js
const routes = {
  '/': renderDashboard,
  '/cves': renderCVEsPage,
  '/iocs': renderIOCsPage,
  '/topics': renderTopicsPage,
  '/alerts': renderAlertsPage
};

const router = new Router(routes);
```

---

## 🎨 Styling Strategy

### CSS Variables for Theming
```css
/* css/themes.css */
:root {
  /* Colors */
  --color-primary: #3b82f6;
  --color-danger: #ef4444;
  --color-warning: #f59e0b;
  --color-success: #10b981;
  
  /* Severity levels */
  --severity-critical: #dc2626;
  --severity-high: #ea580c;
  --severity-medium: #f59e0b;
  --severity-low: #84cc16;
  
  /* Backgrounds */
  --bg-primary: #ffffff;
  --bg-secondary: #f3f4f6;
  --bg-card: #ffffff;
  
  /* Text */
  --text-primary: #111827;
  --text-secondary: #6b7280;
}

[data-theme="dark"] {
  --bg-primary: #111827;
  --bg-secondary: #1f2937;
  --bg-card: #1f2937;
  --text-primary: #f9fafb;
  --text-secondary: #d1d5db;
}
```

### Component Styles
```css
/* css/components.css */
.cve-card {
  background: var(--bg-card);
  border-left: 4px solid var(--severity-color);
  padding: 1rem;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.severity-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.severity-badge.critical {
  background: var(--severity-critical);
  color: white;
}
```

---

## 📦 Docker Setup (Nginx)

### nginx.conf
```nginx
server {
    listen 80;
    server_name localhost;
    
    root /usr/share/nginx/html;
    index index.html;
    
    # Frontend static files
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # Proxy API requests to backend
    location /api {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API docs (optional)
    location /docs {
        proxy_pass http://api:8000/docs;
    }
    
    location /redoc {
        proxy_pass http://api:8000/redoc;
    }
}
```

### docker-compose.yml (Frontend service)
```yaml
frontend:
  image: nginx:alpine
  container_name: threat-intel-frontend
  restart: unless-stopped
  ports:
    - "80:80"
  volumes:
    - ./frontend:/usr/share/nginx/html:ro
    - ./frontend/nginx.conf:/etc/nginx/conf.d/default.conf:ro
  networks:
    - threat-intel-network
  depends_on:
    - api
```

---

## 📊 Libraries (Optional)

### Minimal Dependencies (CDN)
```html
<!-- Chart.js for simple charts -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>

<!-- D3.js for complex visualizations -->
<script src="https://d3js.org/d3.v7.min.js"></script>

<!-- Optional: Tailwind CSS for quick styling -->
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@3.4.0/dist/tailwind.min.css" rel="stylesheet">
```

---

## 🚀 Implementation Plan (Session 5)

### Phase 1: Basic Setup (2 hours)
1. ✅ Create frontend directory structure
2. ✅ Setup index.html with navigation
3. ✅ Implement API client (api.js)
4. ✅ Create router (router.js)
5. ✅ Basic CSS theme

### Phase 2: Core Components (3 hours)
1. ✅ CVE card component
2. ✅ IOC table component
3. ✅ Search & filter components
4. ✅ Alert notifications

### Phase 3: Pages (4 hours)
1. ✅ Dashboard page (overview stats)
2. ✅ CVE explorer page
3. ✅ IOC viewer page
4. ✅ Topics page (LDA viz with D3)
5. ✅ Alerts management page

### Phase 4: Polish (2 hours)
1. ✅ Dark mode toggle
2. ✅ Responsive design
3. ✅ Loading states
4. ✅ Error handling
5. ✅ Empty states

### Phase 5: Integration (1 hour)
1. ✅ Docker setup with Nginx
2. ✅ Test end-to-end
3. ✅ Documentation

**Total Estimated Time**: ~12 hours

---

## 🎯 Features

### Dashboard
- 📊 Real-time stats (total CVEs, IOCs, alerts)
- 📈 Recent CVE trend chart
- 🚨 Active alerts summary
- 📋 Top threats (topics)

### CVE Explorer
- 🔍 Search by CVE ID, keyword
- 🎚️ Filter by severity, date range
- 📄 Pagination
- 📱 CVE detail modal
- 📥 Export to CSV/JSON

### IOC Viewer
- 📊 Table with sorting/filtering
- 🔎 Search by value, type
- 📋 Copy to clipboard
- 🏷️ Group by type
- 📥 Export

### Topics Page
- 🌐 Interactive topic network (D3.js force graph)
- 📊 Topic keywords word cloud
- 📈 Document distribution chart
- 🔗 Related CVEs/IOCs per topic

### Alerts Page
- 🚨 Active alerts list
- ✅ Acknowledge/resolve actions
- 🔔 Severity filtering
- 📅 Timeline view
- 🔕 Mute/unmute notifications

---

## 🔒 Security Considerations

- ✅ **No sensitive data in frontend** - All auth/secrets in backend
- ✅ **CORS properly configured** - FastAPI CORS middleware
- ✅ **Input validation** - Client-side + server-side
- ✅ **XSS prevention** - Escape user input
- ✅ **HTTPS ready** - Nginx SSL config (production)

---

## 📝 Notes

- **No build step required** - Pure HTML/JS/CSS
- **Fast development** - Edit and refresh
- **Small bundle size** - No framework overhead
- **Easy to understand** - Vanilla JS is readable
- **Production ready** - Nginx serves static files efficiently

---

**Status**: 📅 Planned for Session 5  
**Technology**: HTML + JavaScript + CSS + Nginx  
**Estimated Time**: ~12 hours  
**Dependencies**: Chart.js (CDN), D3.js (CDN), Tailwind CSS (optional, CDN)
