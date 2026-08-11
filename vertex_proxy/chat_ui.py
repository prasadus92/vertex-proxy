"""Mini chat web UI served at GET /chat."""

CHAT_HTML = r"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>vertex-proxy chat</title>
<style>
  :root {
    --bg: #1a1a2e;
    --bg2: #16213e;
    --bg3: #0f3460;
    --accent: #e94560;
    --text: #eee;
    --text2: #aaa;
    --user-bg: #0f3460;
    --assistant-bg: #1a1a2e;
    --input-bg: #16213e;
    --border: #333;
    --code-bg: #0d1117;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    flex-direction: column;
  }
  .header {
    padding: 12px 20px;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .header h1 {
    font-size: 16px;
    color: var(--accent);
    white-space: nowrap;
  }
  .header select, .header input[type=range] {
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 13px;
  }
  .header select { min-width: 180px; }
  .header label { font-size: 12px; color: var(--text2); white-space: nowrap; }
  .temp-group { display: flex; align-items: center; gap: 6px; }
  .temp-group input[type=range] { width: 80px; accent-color: var(--accent); }
  .temp-val { font-size: 12px; color: var(--text2); min-width: 28px; }
  .sys-toggle {
    font-size: 12px;
    color: var(--accent);
    cursor: pointer;
    background: none;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 4px 10px;
    margin-left: auto;
  }
  .sys-toggle:hover { background: var(--accent); color: #fff; }
  .system-prompt {
    display: none;
    padding: 8px 20px;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
  }
  .system-prompt.open { display: block; }
  .system-prompt textarea {
    width: 100%;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px;
    font-size: 13px;
    resize: vertical;
    min-height: 50px;
    font-family: inherit;
  }
  .messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg {
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 12px;
    font-size: 14px;
    line-height: 1.6;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }
  .msg.user {
    align-self: flex-end;
    background: var(--user-bg);
    border-bottom-right-radius: 4px;
  }
  .msg.assistant {
    align-self: flex-start;
    background: var(--assistant-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .msg.assistant pre {
    background: var(--code-bg);
    padding: 10px;
    border-radius: 6px;
    overflow-x: auto;
    margin: 8px 0;
  }
  .msg.assistant code {
    font-family: 'Fira Code', 'Cascadia Code', monospace;
    font-size: 13px;
  }
  .msg.assistant p { margin: 6px 0; }
  .msg.assistant ul, .msg.assistant ol { margin: 6px 0 6px 20px; }
  .msg.assistant h1, .msg.assistant h2, .msg.assistant h3 {
    margin: 10px 0 4px 0;
    color: var(--accent);
  }
  .msg.assistant h1 { font-size: 18px; }
  .msg.assistant h2 { font-size: 16px; }
  .msg.assistant h3 { font-size: 14px; }
  .msg.assistant blockquote {
    border-left: 3px solid var(--accent);
    padding-left: 12px;
    margin: 6px 0;
    color: var(--text2);
  }
  .msg.assistant a { color: var(--accent); }
  .msg.assistant hr { border: none; border-top: 1px solid var(--border); margin: 10px 0; }
  .msg .role-label {
    font-size: 11px;
    color: var(--text2);
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .input-area {
    padding: 12px 20px;
    background: var(--bg2);
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
  }
  .input-area textarea {
    flex: 1;
    background: var(--input-bg);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 14px;
    font-family: inherit;
    resize: none;
    min-height: 44px;
    max-height: 200px;
    line-height: 1.5;
  }
  .input-area textarea:focus { outline: none; border-color: var(--accent); }
  .input-area button {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-size: 14px;
    cursor: pointer;
    white-space: nowrap;
    align-self: flex-end;
  }
  .input-area button:hover { opacity: 0.9; }
  .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
  .typing-indicator { color: var(--text2); font-style: italic; font-size: 13px; }
  .clear-btn {
    font-size: 12px;
    color: var(--text2);
    cursor: pointer;
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
  }
  .clear-btn:hover { border-color: var(--accent); color: var(--accent); }
</style>
</head>
<body>
  <div class="header">
    <h1>&#x1f680; vertex-proxy</h1>
    <select id="modelSelect"><option>loading models...</option></select>
    <div class="temp-group">
      <label>Temp</label>
      <input type="range" id="tempSlider" min="0" max="2" step="0.1" value="0.7">
      <span class="temp-val" id="tempVal">0.7</span>
    </div>
    <button class="clear-btn" onclick="clearChat()">Clear</button>
    <button class="sys-toggle" id="sysToggle" onclick="toggleSystem()">System</button>
  </div>
  <div class="system-prompt" id="systemPrompt">
    <textarea id="sysInput" placeholder="Enter system prompt..."></textarea>
  </div>
  <div class="messages" id="messages"></div>
  <div class="input-area">
    <textarea id="userInput" placeholder="Type a message..." rows="1"
      onkeydown="handleKey(event)"></textarea>
    <button id="sendBtn" onclick="sendMessage()">Send</button>
  </div>

<script>
const messagesDiv = document.getElementById('messages');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const modelSelect = document.getElementById('modelSelect');
const tempSlider = document.getElementById('tempSlider');
const tempVal = document.getElementById('tempVal');
const sysInput = document.getElementById('sysInput');
let chatHistory = [];
let isStreaming = false;

tempSlider.addEventListener('input', () => { tempVal.textContent = tempSlider.value; });

// Auto-resize textarea
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto';
  userInput.style.height = Math.min(userInput.scrollHeight, 200) + 'px';
});

async function loadModels() {
  try {
    const r = await fetch('/v1/models');
    const d = await r.json();
    modelSelect.innerHTML = '';
    (d.data || []).forEach(m => {
      const o = document.createElement('option');
      o.value = m.id;
      o.textContent = m.id + (m.provider ? ' (' + m.provider + ')' : '');
      modelSelect.appendChild(o);
    });
  } catch(e) {
    modelSelect.innerHTML = '<option>error loading models</option>';
  }
}

function toggleSystem() {
  document.getElementById('systemPrompt').classList.toggle('open');
}

function clearChat() {
  chatHistory = [];
  messagesDiv.innerHTML = '';
}

function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = role;
  div.appendChild(label);
  const body = document.createElement('div');
  body.className = 'msg-body';
  if (role === 'assistant') {
    body.innerHTML = renderMarkdown(content);
  } else {
    body.textContent = content;
  }
  div.appendChild(body);
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return body;
}

function renderMarkdown(text) {
  // Escape HTML
  let s = text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // Code blocks
  s = s.replace(/```(\w*)\n([\s\S]*?)```/g, function(m, lang, code) {
    return '<pre><code>' + code.trim() + '</code></pre>';
  });
  // Inline code
  s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Headers
  s = s.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  s = s.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  s = s.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  // Bold + italic
  s = s.replace(/\*\*\*(.+?)\*\*\*/g, '<b><i>$1</i></b>');
  s = s.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>');
  s = s.replace(/\*(.+?)\*/g, '<i>$1</i>');
  // Strikethrough
  s = s.replace(/~~(.+?)~~/g, '<s>$1</s>');
  // Blockquotes
  s = s.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');
  // Horizontal rule
  s = s.replace(/^---$/gm, '<hr>');
  // Unordered lists
  s = s.replace(/^[*-] (.+)$/gm, '<li>$1</li>');
  s = s.replace(/(<li>.*<\/li>)/gs, '<ul>$1</ul>');
  // Links
  s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  // Paragraphs (double newline)
  s = s.replace(/\n\n/g, '</p><p>');
  // Single newline -> <br>
  s = s.replace(/\n/g, '<br>');
  return '<p>' + s + '</p>';
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text || isStreaming) return;
  isStreaming = true;
  sendBtn.disabled = true;
  sendBtn.textContent = 'Stop';
  sendBtn.onclick = stopStreaming;

  userInput.value = '';
  userInput.style.height = 'auto';
  addMessage('user', text);
  chatHistory.push({ role: 'user', content: text });

  const messages = [];
  const sys = sysInput.value.trim();
  if (sys) messages.push({ role: 'system', content: sys });
  messages.push(...chatHistory);

  const body = {
    model: modelSelect.value,
    messages: messages,
    temperature: parseFloat(tempSlider.value),
    stream: true,
  };

  const assistantBody = addMessage('assistant', '');
  let fullText = '';

  try {
    const apiKey = new URLSearchParams(window.location.search).get('key') || '';
    const headers = { 'Content-Type': 'application/json' };
    if (apiKey) headers['Authorization'] = 'Bearer ' + apiKey;

    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: headers,
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.text();
      assistantBody.innerHTML = '<span style="color:#f44">Error: ' + resp.status + ' ' + err.substring(0,500) + '</span>';
      isStreaming = false;
      resetSendBtn();
      return;
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    window._stopStream = false;
    while (true) {
      if (window._stopStream) { reader.cancel(); break; }
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') continue;
        try {
          const j = JSON.parse(data);
          const delta = j.choices?.[0]?.delta?.content;
          if (delta) {
            fullText += delta;
            assistantBody.innerHTML = renderMarkdown(fullText);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
          }
        } catch(e) {}
      }
    }
  } catch(e) {
    if (!window._stopStream) {
      assistantBody.innerHTML += '<br><span style="color:#f44">Stream error: ' + e.message + '</span>';
    }
  }

  if (fullText) {
    chatHistory.push({ role: 'assistant', content: fullText });
  }
  isStreaming = false;
  resetSendBtn();
}

function stopStreaming() {
  window._stopStream = true;
}

function resetSendBtn() {
  sendBtn.disabled = false;
  sendBtn.textContent = 'Send';
  sendBtn.onclick = sendMessage;
}

loadModels();
userInput.focus();
</script>
</body>
</html>
"""
