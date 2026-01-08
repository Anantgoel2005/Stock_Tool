// DOM Elements
const logArea = document.getElementById("log-area");
const resultArea = document.getElementById("result-area");
const articlesList = document.getElementById("articles-list");
const predictBtn = document.getElementById("predict-btn");
const clearLogsBtn = document.getElementById("clear-logs-btn");
const articlesCount = document.getElementById("articles-count");
const toggleAdvanced = document.getElementById("toggle-advanced");
const advancedConfig = document.getElementById("advanced-config");
const expandAllBtn = document.getElementById("expand-all-articles");
const collapseAllBtn = document.getElementById("collapse-all-articles");

const symbolInput = document.getElementById("symbol");
const horizonInput = document.getElementById("horizon");
const newsDaysInput = document.getElementById("news-days");
const newsQueryInput = document.getElementById("news-query");

// State
let isProcessing = false;

// Utility Functions
function formatTime(date = new Date()) {
    return date.toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit'
    });
}

function appendLog(message, tone = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry log-${tone}`;
    
    const time = document.createElement("span");
    time.className = "log-time";
    time.textContent = formatTime();
    
    const msg = document.createElement("span");
    msg.className = "log-message";
    msg.textContent = message;
    
    entry.appendChild(time);
    entry.appendChild(msg);
    logArea.appendChild(entry);
    logArea.scrollTop = logArea.scrollHeight;
}

function clearLogs() {
    logArea.innerHTML = '';
    appendLog("Logs cleared.", "info");
}

function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.classList.add("loading");
        button.disabled = true;
    } else {
        button.classList.remove("loading");
        button.disabled = false;
    }
}

function setButtonsLoading(isLoading) {
    setButtonLoading(predictBtn, isLoading);
    isProcessing = isLoading;
}

function validateInputs() {
    const symbol = symbolInput.value.trim();
    if (!symbol) {
        appendLog("❌ Please enter a stock ticker symbol.", "error");
        symbolInput.focus();
        return false;
    }
    
    const newsDays = parseInt(newsDaysInput.value);
    if (isNaN(newsDays) || newsDays < 1 || newsDays > 30) {
        appendLog("❌ News lookback period must be between 1 and 30 days.", "error");
        newsDaysInput.focus();
        return false;
    }
    
    return true;
}

function parsePredictionResult(content) {
    if (!content || content.trim() === '') {
        return null;
    }
    
    const lines = content.split('\n').filter(line => line.trim());
    const result = {
        symbol: '',
        horizon: '',
        signal: '',
        confidence: '',
        reasoning: []
    };
    
    // Extract symbol and horizon
    const symbolMatch = content.match(/Prediction for ([A-Z0-9.]+)/i);
    if (symbolMatch) {
        result.symbol = symbolMatch[1];
    }
    
    const horizonMatch = content.match(/PREDICTION FOR: ([^-]+)/i);
    if (horizonMatch) {
        result.horizon = horizonMatch[1].trim();
    }
    
    // Extract signal
    const signalMatch = content.match(/SIGNAL:\s+(.+)/i);
    if (signalMatch) {
        result.signal = signalMatch[1].trim();
    }
    
    // Extract confidence
    const confidenceMatch = content.match(/CONFIDENCE:\s+(.+)/i);
    if (confidenceMatch) {
        result.confidence = confidenceMatch[1].trim();
    }
    
    // Extract reasoning items
    const reasoningStart = content.indexOf('REASONING:');
    if (reasoningStart !== -1) {
        const reasoningText = content.substring(reasoningStart);
        const reasoningLines = reasoningText.split('\n');

        reasoningLines.forEach(rawLine => {
            let trimmed = rawLine.trim();
            if (!trimmed || trimmed.startsWith('REASONING:')) {
                return;
            }
            // Expect lines like "  - SHORT-TERM STRENGTH: ..."
            if (trimmed.startsWith('-')) {
                trimmed = trimmed.replace(/^[-•]\s*/, '');
            } else if (trimmed.startsWith('•')) {
                trimmed = trimmed.slice(1).trim();
            } else if (!trimmed.startsWith('- ') && !trimmed.startsWith('• ')) {
                // Skip stray lines that aren't formatted as bullet points
                return;
            }

            if (trimmed && trimmed.length > 5) {
                let type = 'neutral';
                let label = '';
                
                if (trimmed.includes('POSITIVE')) {
                    type = 'positive';
                    label = 'Positive';
                } else if (trimmed.includes('NEGATIVE')) {
                    type = 'negative';
                    label = 'Negative';
                } else if (trimmed.includes('WARNING') || trimmed.includes('ALERT')) {
                    type = 'warning';
                    label = 'Warning';
                } else if (trimmed.includes('NEUTRAL')) {
                    type = 'neutral';
                    label = 'Neutral';
                }
                
                result.reasoning.push({
                    text: trimmed,
                    type: type,
                    label: label
                });
            }
        });
    }
    
    return result;
}

function showResult(content, isError = false) {
    resultArea.innerHTML = '';
    
    if (isError) {
        resultArea.innerHTML = `<div class="empty-state">
            <span class="empty-icon">⚠️</span>
            <p style="color: var(--danger);">${escapeHtml(content)}</p>
        </div>`;
        return;
    }
    
    if (!content || content.trim() === '') {
        resultArea.innerHTML = `<div class="empty-state">
            <span class="empty-icon">💡</span>
            <p>Click "Predict" to see AI-powered stock analysis</p>
        </div>`;
        return;
    }
    
    const parsed = parsePredictionResult(content);
    
    if (!parsed || !parsed.signal) {
        // Fallback to formatted text if parsing fails
        const formatted = escapeHtml(content)
            .replace(/\n/g, '<br>');
        resultArea.innerHTML = `<div style="padding: 1rem; font-family: monospace; white-space: pre-wrap;">${formatted}</div>`;
        return;
    }
    
    const isBuy = parsed.signal.toUpperCase().includes('BUY');
    const signalClass = isBuy ? 'buy' : 'sell';
    const signalIcon = isBuy ? '📈' : '📉';
    
    let html = `
        <div class="prediction-header">
            <div>
                <div class="prediction-symbol">${escapeHtml(parsed.symbol || 'N/A')}</div>
                <div class="prediction-horizon">${escapeHtml(parsed.horizon || '')}</div>
            </div>
            <div class="prediction-signal">
                <div class="signal-badge ${signalClass}">
                    ${signalIcon} ${escapeHtml(parsed.signal)}
                </div>
                <div class="confidence-badge">
                    Confidence: <span class="confidence-value">${escapeHtml(parsed.confidence)}</span>
                </div>
            </div>
        </div>
    `;
    
    if (parsed.reasoning.length > 0) {
        // Group reasoning into tidy sections
        const sections = {};
        const addToSection = (name, item) => {
            if (!sections[name]) sections[name] = [];
            sections[name].push(item);
        };

        parsed.reasoning.forEach(item => {
            const text = (item.text || "").toUpperCase();
            if (text.includes("TREND") || text.includes("SMA")) {
                addToSection("Trend & Direction", item);
            } else if (text.includes("RSI") || text.includes("MOMENTUM") || text.includes("MACD")) {
                addToSection("Momentum", item);
            } else if (text.includes("RETURN") || text.includes("SHORT-TERM")) {
                addToSection("Short-term Performance", item);
            } else if (text.includes("SENTIMENT") || text.includes("NEWS")) {
                addToSection("News & Sentiment", item);
            } else {
                addToSection("Other Factors", item);
            }
        });

        const orderedNames = [
            "Trend & Direction",
            "Momentum",
            "Short-term Performance",
            "News & Sentiment",
            "Other Factors",
        ].filter(name => sections[name] && sections[name].length > 0);

        const sectionIcon = (name) => {
            if (name === "Trend & Direction") return "📈";
            if (name === "Momentum") return "⚡";
            if (name === "Short-term Performance") return "⏱️";
            if (name === "News & Sentiment") return "📰";
            return "ℹ️";
        };

        html += `
            <div class="reasoning-section">
                <div class="reasoning-title">
                    <span>🔍</span>
                    <span>Detailed Breakdown</span>
                    <span class="reasoning-legend">
                        <span class="dot positive"></span> Positive
                        <span class="dot warning"></span> Watch
                        <span class="dot neutral"></span> Neutral
                        <span class="dot negative"></span> Risk
                    </span>
                </div>
                <div class="reasoning-grid">
        `;

        orderedNames.forEach(name => {
            const items = sections[name] || [];
            if (items.length === 0) {
                return;
            }

            // Use only the first (most important) line per section to reduce clutter.
            const primary = items[0];
            const extraCount = items.length - 1;
            const t = primary.type || "neutral";
            
            html += `
                <div class="reasoning-block">
                    <div class="reasoning-block-header">
                        <span class="reasoning-block-icon">${sectionIcon(name)}</span>
                        <span class="reasoning-block-title">${escapeHtml(name)}</span>
                        <span class="reasoning-block-count">${items.length}</span>
                    </div>
                    <ul class="reasoning-block-list">
                        <li class="${t}">${escapeHtml(primary.text)}</li>
                    </ul>
                    ${extraCount > 0
                        ? `<div class="reasoning-extra-note">+ ${extraCount} more factor${extraCount > 1 ? "s" : ""} considered</div>`
                        : ""}
                </div>
            `;
        });
        
        html += `
                </div>
            </div>
        `;
    }
    
    resultArea.innerHTML = html;
}

function toggleArticle(card) {
    card.classList.toggle('expanded');
}

function expandAllArticles() {
    const cards = articlesList.querySelectorAll('.article-card');
    cards.forEach(card => card.classList.add('expanded'));
}

function collapseAllArticles() {
    const cards = articlesList.querySelectorAll('.article-card');
    cards.forEach(card => card.classList.remove('expanded'));
}

function updateArticles(articles) {
    articlesList.innerHTML = "";
    
    if (!articles || articles.length === 0) {
        articlesList.innerHTML = `
            <div class="empty-state-item">
                <span class="empty-icon">📄</span>
                <p>No articles fetched for this prediction</p>
            </div>
        `;
        articlesCount.textContent = "0";
        return;
    }
    
    articlesCount.textContent = articles.length.toString();
    
    articles.forEach((article, idx) => {
        const card = document.createElement("div");
        card.className = "article-card";
        
        const preview = article.length > 120 ? article.substring(0, 120) + '...' : article;
        
        card.innerHTML = `
            <div class="article-header" onclick="toggleArticle(this.parentElement)">
                <div class="article-number">${idx + 1}</div>
                <div class="article-preview">${escapeHtml(preview)}</div>
                <div class="article-expand">▼</div>
            </div>
            <div class="article-content">
                <div class="article-full-text">${escapeHtml(article)}</div>
            </div>
        `;
        
        articlesList.appendChild(card);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// API Functions
async function postJSON(url, payload) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minute timeout
    
    try {
        const response = await fetch(url, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            const message = error.detail || `Request failed with status ${response.status}`;
            throw new Error(message);
        }
        
        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        if (error.name === 'AbortError') {
            throw new Error("Request timed out. Please try again.");
        }
        throw error;
    }
}

function getPayload() {
    return {
        symbol: symbolInput.value.trim().toUpperCase(),
        horizon: horizonInput.value,
        news_query: newsQueryInput.value.trim() || null,
        days_back: Math.max(1, Math.min(30, parseInt(newsDaysInput.value) || 1)),
    };
}

// Event Handlers
predictBtn.addEventListener("click", async () => {
    if (isProcessing) {
        appendLog("⚠️ Another operation is in progress. Please wait.", "warning");
        return;
    }
    
    if (!validateInputs()) {
        return;
    }
    
    const payload = getPayload();
    
    appendLog(`🔮 Running prediction for ${payload.symbol} (${payload.horizon.replace('_', '-')})...`, "info");
    setButtonsLoading(true);
    showResult("⏳ Fetching data and analyzing...", false);
    updateArticles([]);
    
    try {
        const startTime = Date.now();
        const res = await postJSON("/api/predict", payload);
        const duration = ((Date.now() - startTime) / 1000).toFixed(1);
        
        showResult(res.result || "No result returned.", false);
        updateArticles(res.articles || []);
        appendLog(`✅ Prediction completed successfully (${duration}s)`, "success");
    } catch (err) {
        const errorMsg = err.message || "An unknown error occurred";
        showResult(`Error: ${errorMsg}`, true);
        updateArticles([]);
        appendLog(`❌ Prediction failed: ${errorMsg}`, "error");
    } finally {
        setButtonsLoading(false);
    }
});

clearLogsBtn.addEventListener("click", clearLogs);

// Advanced options toggle
toggleAdvanced.addEventListener("click", () => {
    advancedConfig.classList.toggle("expanded");
});

// Article controls
expandAllBtn.addEventListener("click", expandAllArticles);
collapseAllBtn.addEventListener("click", collapseAllArticles);

// Make toggleArticle available globally for onclick handlers
window.toggleArticle = toggleArticle;

// Form validation
symbolInput.addEventListener("input", function() {
    this.value = this.value.toUpperCase();
});

newsDaysInput.addEventListener("input", function() {
    let value = parseInt(this.value);
    if (isNaN(value) || value < 1) value = 1;
    if (value > 30) value = 30;
    this.value = value;
});

// Prevent form submission on Enter
document.getElementById("config-form").addEventListener("submit", (e) => {
    e.preventDefault();
});

// Initialize
appendLog("🚀 Stock Prediction Tool initialized.", "info");
appendLog("💡 Enter a ticker symbol and click 'Get Prediction' to start.", "info");
