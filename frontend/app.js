/* ===================================
   NeuroTidy Frontend JavaScript
   =================================== */

let currentMode = 'explain';

// ─── Setup & Persistence ──────────────────────────────────
(function init() {
  // Restore saved endpoint
  const saved = localStorage.getItem('neurotidy_endpoint') || '';
  if (saved) document.getElementById('apiEndpoint').value = saved;
})();

function saveEndpoint() {
  const ep = document.getElementById('apiEndpoint').value.trim().replace(/\/$/, '');
  localStorage.setItem('neurotidy_endpoint', ep);
  showNote('✅ Endpoint saved!', 'success');
}

function getEndpoint() {
  return (document.getElementById('apiEndpoint').value.trim().replace(/\/$/, ''));
}

function setMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
  // Show/hide mode-specific options
  document.getElementById('opts-explain').classList.toggle('hidden', mode !== 'explain');
  document.getElementById('opts-debug').classList.toggle('hidden', mode !== 'debug');
  // Update button label
  const labels = { explain: '📖 Explain Code', analyze: '🔍 Analyze Code', optimize: '⚡ Optimize Code', debug: '🐛 Debug Error', review: '🔍 Review PR' };
  document.getElementById('runBtnText').textContent = labels[mode] || '🚀 Run';
  // Clear results
  document.getElementById('resultPanel').classList.add('hidden');
}

// ─── API Call ─────────────────────────────────────────────
async function runAnalysis() {
  const endpoint = getEndpoint();
  if (!endpoint) {
    showNote('❌ Please enter your API endpoint above.', 'error');
    document.getElementById('apiEndpoint').focus();
    return;
  }

  const code = document.getElementById('codeInput').value.trim();
  const errorMsg = document.getElementById('errorMsg').value.trim();

  if (!code && currentMode !== 'debug') {
    showNote('⚠️ Please enter some code first.', 'warn');
    document.getElementById('codeInput').focus();
    return;
  }
  if (currentMode === 'debug' && !errorMsg && !code) {
    showNote('⚠️ Enter an error message or some code for the debug mode.', 'warn');
    return;
  }

  // Build payload
  let payload = {};
  if (currentMode === 'explain') {
    const mode = document.querySelector('input[name="explainMode"]:checked')?.value || 'intermediate';
    payload = { code, mode };
  } else if (currentMode === 'debug') {
    payload = { code, error: errorMsg };
  } else {
    payload = { code, use_ai: true };
  }

  setLoading(true);
  showNote('Sending to AWS Bedrock…');

  try {
    const url = `${endpoint}/${currentMode}`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showNote(`❌ API Error: ${data.error || res.statusText}`, 'error');
      setLoading(false);
      return;
    }

    renderResult(data);
    showNote('✅ Analysis complete!', 'success');
  } catch (err) {
    showNote(`❌ ${err.message}`, 'error');
  } finally {
    setLoading(false);
  }
}

// ─── Render Results ───────────────────────────────────────
function renderResult(data) {
  const panel = document.getElementById('resultPanel');
  const body = document.getElementById('resultBody');
  const title = document.getElementById('resultTitle');
  const idEl = document.getElementById('resultId');

  panel.classList.remove('hidden');
  idEl.textContent = `ID: ${data.analysis_id || ''}`;

  const titles = { explain: '📖 Explanation', analyze: '🔍 Static Analysis', optimize: '⚡ DL Optimization', debug: '🐛 Bug Report', review: '🔍 PR Review Report' };
  title.textContent = titles[currentMode] || 'Result';

  if (currentMode === 'explain') body.innerHTML = renderExplain(data);
  if (currentMode === 'analyze') body.innerHTML = renderAnalyze(data);
  if (currentMode === 'optimize') body.innerHTML = renderOptimize(data);
  if (currentMode === 'debug') body.innerHTML = renderDebug(data);
  if (currentMode === 'review') body.innerHTML = renderReview(data);

  panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderExplain(data) {
  const exp = data.explanation;
  const text = (typeof exp === 'string') ? exp : JSON.stringify(exp, null, 2);
  return `<div class="explanation-text">${escHtml(text)}</div>`;
}

function renderAnalyze(data) {
  let html = `<div class="summary-text">${escHtml(data.summary || '')}</div>`;
  const violations = data.violations || [];
  if (violations.length === 0) {
    html += `<div style="color:var(--accent3);font-size:15px;font-weight:600;">✅ No issues found! Clean code.</div>`;
  } else {
    violations.forEach(v => {
      html += makeViolationCard(v, v.suggested_fix || '');
    });
  }
  // Metrics
  const m = data.metrics || {};
  if (m.total_lines) {
    html += `<div class="metrics-row">
      <div class="metric-chip">Lines: <span>${m.code_lines || m.total_lines}</span></div>
      <div class="metric-chip">Functions: <span>${m.function_count ?? '?'}</span></div>
      <div class="metric-chip">Comment ratio: <span>${m.comment_ratio ?? '?'}%</span></div>
    </div>`;
  }
  html += renderAI(data.ai_insights);
  return html;
}

function renderOptimize(data) {
  let html = `<div class="summary-text">${escHtml(data.summary || '')}</div>`;
  const violations = data.violations || [];
  if (violations.length === 0) {
    html += `<div style="color:var(--accent3);font-size:15px;font-weight:600;">✅ No DL optimization issues found!</div>`;
  } else {
    violations.forEach(v => {
      html += makeViolationCard(v, v.suggested_fix || '');
    });
  }
  html += renderAI(data.ai_insights);
  return html;
}

function renderDebug(data) {
  let html = `
    <div style="margin-bottom:16px">
      <div style="font-size:13px;color:var(--text-muted);margin-bottom:4px">Error Type</div>
      <div style="font-size:17px;font-weight:700;color:var(--danger)">${escHtml(data.error_type || 'Unknown')}</div>
    </div>
    <div style="margin-bottom:20px">
      <div style="font-size:13px;color:var(--text-muted);margin-bottom:4px">Root Cause</div>
      <div style="font-size:15px">${escHtml(data.root_cause || '')}</div>
    </div>`;

  const tips = data.learning_tips || [];
  if (tips.length) {
    html += `<div style="margin-bottom:16px">
      <div style="font-size:13px;font-weight:600;color:var(--accent2);margin-bottom:8px">🎓 Learning Tips</div>
      <ul style="padding-left:20px;color:var(--text-muted);font-size:14px;line-height:2">
        ${tips.map(t => `<li>${escHtml(t)}</li>`).join('')}
      </ul>
    </div>`;
  }

  const fixes = data.suggested_fixes || [];
  if (fixes.length) {
    html += `<div style="margin-bottom:16px">
      <div style="font-size:13px;font-weight:600;color:var(--accent3);margin-bottom:8px">🔧 Suggested Fixes</div>
      <ul style="padding-left:20px;font-size:14px;line-height:2">
        ${fixes.map(f => `<li>${escHtml(f)}</li>`).join('')}
      </ul>
    </div>`;
  }

  const exp = data.explanation;
  if (exp && typeof exp === 'object' && exp.simple_explanation) {
    html += `<div class="ai-insights-box">
      <h4>🤖 AI Explanation</h4>
      <p style="font-size:14px;color:var(--text);margin-bottom:10px">${escHtml(exp.simple_explanation)}</p>
      ${(exp.step_by_step_fix || []).map((s, i) => `<div style="font-size:13px;color:var(--text-muted);margin-bottom:4px">${i + 1}. ${escHtml(s)}</div>`).join('')}
    </div>`;
  }
  return html;
}

function renderReview(data) {
  const status = data.status || 'unknown';
  const statusColor = status === 'reviewed' ? 'var(--accent3)' : status === 'error' ? 'var(--danger)' : 'var(--text-muted)';
  let html = `
    <div style="margin-bottom:16px">
      <div style="font-size:13px;color:var(--text-muted);margin-bottom:4px">Status</div>
      <div style="font-size:17px;font-weight:700;color:${statusColor}">${escHtml(status)}</div>
    </div>`;
  if (data.pr_number) {
    html += `<div style="margin-bottom:12px">
      <div style="font-size:13px;color:var(--text-muted);margin-bottom:4px">Pull Request</div>
      <div style="font-size:15px">PR #${data.pr_number} in <code>${escHtml(data.repo || '')}</code></div>
    </div>`;
  }
  if (data.files_reviewed !== undefined) {
    html += `<div class="metrics-row">
      <div class="metric-chip">Files reviewed: <span>${data.files_reviewed}</span></div>
      <div class="metric-chip">Comments posted: <span>${data.comments_posted}</span></div>
    </div>`;
  }
  if (data.reason) {
    html += `<div style="color:var(--text-muted);font-size:14px;margin-top:12px">ℹ️ ${escHtml(data.reason)}</div>`;
  }
  html += `<div style="margin-top:20px;padding:12px;background:var(--card-bg);border-radius:8px;font-size:13px;color:var(--text-muted)">`;
  html += `<strong>💡 PR Review Bot Setup:</strong> Configure <code>GITHUB_TOKEN</code> and <code>GITHUB_WEBHOOK_SECRET</code> in <code>config.env</code>, then point your GitHub repo webhook to <code>/review</code>.</div>`;
  return html;
}

function makeViolationCard(v, fix) {
  return `<div class="violation-card ${v.severity || ''}">
    <div>
      <div class="v-badge">${v.severity || 'INFO'}</div>
    </div>
    <div class="v-content">
      <div class="v-rule">[${v.rule_id || ''}] ${v.line_number ? `Line ${v.line_number}` : ''}</div>
      <div class="v-desc">${escHtml(v.description || '')}</div>
      ${fix ? `<div class="v-fix">→ ${escHtml(fix)}</div>` : ''}
    </div>
  </div>`;
}

function renderAI(ai) {
  if (!ai || ai.error || Object.keys(ai).length === 0) return '';
  let html = '<div class="ai-insights-box"><h4>🤖 AI Insights</h4>';
  if (ai.readability_score != null) html += `<div style="font-size:14px;margin-bottom:8px">Readability: <strong>${ai.readability_score}/100</strong> &nbsp; Maintainability: <strong>${ai.maintainability_score || '?'}/100</strong></div>`;
  if (ai.performance_score != null) html += `<div style="font-size:14px;margin-bottom:8px">Performance score: <strong>${ai.performance_score}/100</strong></div>`;
  if (ai.top_recommendation) html += `<div style="font-size:14px;margin-bottom:8px;color:var(--accent3)">💡 ${escHtml(ai.top_recommendation)}</div>`;
  if (ai.estimated_speedup) html += `<div style="font-size:14px;color:var(--accent2)">⚡ ${escHtml(ai.estimated_speedup)}</div>`;
  html += '</div>';
  return html;
}

// ─── Utilities ────────────────────────────────────────────
function setLoading(on) {
  const btn = document.getElementById('runBtn');
  btn.disabled = on;
  document.getElementById('runBtnText').style.display = on ? 'none' : 'inline';
  document.getElementById('spinner').classList.toggle('hidden', !on);
}

function showNote(msg, type = 'info') {
  const el = document.getElementById('runNote');
  el.textContent = msg;
  el.style.color = type === 'error' ? 'var(--danger)' : type === 'success' ? 'var(--accent3)' : type === 'warn' ? 'var(--warn)' : 'var(--text-muted)';
}

function clearCode() {
  document.getElementById('codeInput').value = '';
  document.getElementById('resultPanel').classList.add('hidden');
}

function copyResult() {
  const text = document.getElementById('resultBody').innerText;
  navigator.clipboard.writeText(text).then(() => showNote('📋 Copied!', 'success'));
}

function loadSample() {
  const samples = {
    explain: `import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)`,

    analyze: `def process(data=[]):
    result = []
    for i in range(len(data)):
        result.append(data[i] * 2)
    total = None
    if total == None:
        total = sum(result)
    return result, total`,

    optimize: `import torch
from torch.utils.data import DataLoader

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
loader = DataLoader(dataset, batch_size=32)

for epoch in range(100):
    for batch in loader:
        data, labels = batch
        data = data.to('cuda')
        labels = labels.to('cuda')
        output = model(data)
        loss = criterion(torch.sigmoid(output), labels)
        loss.backward()
        optimizer.step()`,

    debug: `model = SimpleNet(784, 256, 10)
optimizer = torch.optim.Adam(model.parameters())`,

    review: `diff --git a/train.py b/train.py
--- a/train.py
+++ b/train.py
@@ -0,0 +1,10 @@
+import torch
+from torch.utils.data import DataLoader
+
+def train(model, dataset):
+    loader = DataLoader(dataset, batch_size=32)
+    optimizer = torch.optim.Adam(model.parameters())
+    for epoch in range(100):
+        for data, labels in loader:
+            output = model(data)
+            loss = criterion(output, labels)
+            loss.backward()
+            optimizer.step()`
  };

  document.getElementById('codeInput').value = samples[currentMode] || samples.explain;

  if (currentMode === 'debug') {
    document.getElementById('errorMsg').value = 'NameError: name \'SimpleNet\' is not defined';
  }
}

function setApiExample(mode) {
  setMode(mode);
  loadSample();
  document.getElementById('analyzer').scrollIntoView({ behavior: 'smooth' });
}

function escHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Allow Ctrl+Enter to run
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runAnalysis();
});
