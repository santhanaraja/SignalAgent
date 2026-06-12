/**
 * Market Pulse AI Chat Widget
 * Floating chat bubble that expands into a chat window.
 * Powered by Claude API with live dashboard data context.
 */
(function () {
  const SESSION_ID = 'mp-' + Math.random().toString(36).slice(2, 10);
  let isOpen = false;
  let isLoading = false;

  // --- Inject CSS ---
  const style = document.createElement('style');
  style.textContent = `
    #mp-chat-fab {
      position: fixed; bottom: 24px; right: 24px; z-index: 9999;
      width: 56px; height: 56px; border-radius: 50%;
      background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 100%);
      border: none; cursor: pointer; box-shadow: 0 4px 20px rgba(88,166,255,.4);
      display: flex; align-items: center; justify-content: center;
      transition: transform .2s, box-shadow .2s;
    }
    #mp-chat-fab:hover { transform: scale(1.08); box-shadow: 0 6px 28px rgba(88,166,255,.5); }
    #mp-chat-fab svg { width: 26px; height: 26px; fill: #fff; }
    #mp-chat-fab .close-icon { display: none; }
    #mp-chat-fab.open .chat-icon { display: none; }
    #mp-chat-fab.open .close-icon { display: block; }

    #mp-chat-panel {
      position: fixed; bottom: 92px; right: 24px; z-index: 9998;
      width: 400px; max-height: 560px; border-radius: 14px;
      background: #0d1117; border: 1px solid #30363d;
      box-shadow: 0 12px 48px rgba(0,0,0,.6);
      display: none; flex-direction: column; overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    #mp-chat-panel.open { display: flex; }

    .mp-chat-header {
      padding: 14px 16px; background: #161b22; border-bottom: 1px solid #30363d;
      display: flex; align-items: center; gap: 10px;
    }
    .mp-chat-header-dot { width: 8px; height: 8px; border-radius: 50%; background: #3fb950; }
    .mp-chat-header-title { font-size: 14px; font-weight: 700; color: #e6edf3; flex: 1; }
    .mp-chat-header-clear {
      background: none; border: 1px solid #30363d; color: #8b949e; font-size: 10px;
      padding: 3px 10px; border-radius: 4px; cursor: pointer; transition: .15s;
    }
    .mp-chat-header-clear:hover { color: #e6edf3; border-color: #58a6ff; }

    .mp-chat-messages {
      flex: 1; overflow-y: auto; padding: 14px 16px; min-height: 300px; max-height: 400px;
      display: flex; flex-direction: column; gap: 10px;
    }
    .mp-chat-messages::-webkit-scrollbar { width: 4px; }
    .mp-chat-messages::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }

    .mp-msg {
      max-width: 88%; padding: 10px 14px; border-radius: 12px;
      font-size: 13px; line-height: 1.55; color: #e6edf3; word-wrap: break-word;
    }
    .mp-msg.user {
      align-self: flex-end; background: #1f3a5f; border-bottom-right-radius: 4px;
    }
    .mp-msg.assistant {
      align-self: flex-start; background: #161b22; border: 1px solid #30363d;
      border-bottom-left-radius: 4px;
    }
    .mp-msg.assistant strong { color: #58a6ff; }
    .mp-msg.system {
      align-self: center; background: none; color: #8b949e; font-size: 11px;
      text-align: center; padding: 4px;
    }
    .mp-msg.assistant p { margin: 6px 0; }
    .mp-msg.assistant ul, .mp-msg.assistant ol { margin: 4px 0; padding-left: 18px; }
    .mp-msg.assistant li { margin: 2px 0; }
    .mp-msg.assistant code {
      background: #21262d; padding: 1px 5px; border-radius: 3px; font-size: 12px;
      font-family: 'SF Mono', 'Fira Code', monospace;
    }

    .mp-typing {
      align-self: flex-start; padding: 10px 14px; display: flex; gap: 4px;
    }
    .mp-typing span {
      width: 6px; height: 6px; border-radius: 50%; background: #8b949e;
      animation: mp-bounce .6s infinite alternate;
    }
    .mp-typing span:nth-child(2) { animation-delay: .2s; }
    .mp-typing span:nth-child(3) { animation-delay: .4s; }
    @keyframes mp-bounce { to { opacity: .3; transform: translateY(-4px); } }

    .mp-chat-input-area {
      padding: 12px; border-top: 1px solid #30363d; background: #161b22;
      display: flex; gap: 8px;
    }
    .mp-chat-input {
      flex: 1; background: #0d1117; border: 1px solid #30363d; color: #e6edf3;
      padding: 10px 14px; border-radius: 8px; font-size: 13px;
      font-family: inherit; resize: none; outline: none; transition: border-color .15s;
    }
    .mp-chat-input:focus { border-color: #58a6ff; }
    .mp-chat-input::placeholder { color: #484f58; }
    .mp-chat-send {
      background: #58a6ff; border: none; color: #fff; width: 38px; height: 38px;
      border-radius: 8px; cursor: pointer; display: flex; align-items: center;
      justify-content: center; transition: .15s; flex-shrink: 0;
    }
    .mp-chat-send:hover { filter: brightness(1.15); }
    .mp-chat-send:disabled { opacity: .4; cursor: not-allowed; }
    .mp-chat-send svg { width: 18px; height: 18px; fill: #fff; }

    .mp-chat-disclaimer {
      padding: 6px 16px 8px; font-size: 9px; color: #484f58; text-align: center;
      background: #161b22;
    }

    @media (max-width: 500px) {
      #mp-chat-panel { width: calc(100vw - 16px); right: 8px; bottom: 84px; max-height: 70vh; }
      #mp-chat-fab { bottom: 16px; right: 16px; }
    }
  `;
  document.head.appendChild(style);

  // --- Build DOM ---
  // FAB button
  const fab = document.createElement('button');
  fab.id = 'mp-chat-fab';
  fab.title = 'Ask Market Pulse AI';
  fab.innerHTML = `
    <svg class="chat-icon" viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H5.2L4 17.2V4h16v12z"/><path d="M7 9h10v2H7zm0-3h10v2H7z"/></svg>
    <svg class="close-icon" viewBox="0 0 24 24"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
  `;
  document.body.appendChild(fab);

  // Chat panel
  const panel = document.createElement('div');
  panel.id = 'mp-chat-panel';
  panel.innerHTML = `
    <div class="mp-chat-header">
      <div class="mp-chat-header-dot"></div>
      <div class="mp-chat-header-title">Market Pulse AI</div>
      <button class="mp-chat-header-clear" id="mp-clear-btn">Clear</button>
    </div>
    <div class="mp-chat-messages" id="mp-messages">
      <div class="mp-msg system">Ask me anything about the dashboard data — tickers, signals, groups, or thesis breakers.</div>
    </div>
    <div class="mp-chat-input-area">
      <input type="text" class="mp-chat-input" id="mp-input" placeholder="e.g. What are the top performing groups?" maxlength="2000" autocomplete="off">
      <button class="mp-chat-send" id="mp-send-btn" title="Send">
        <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    </div>
    <div class="mp-chat-disclaimer">AI-generated analysis based on algorithmic signals. Not financial advice.</div>
  `;
  document.body.appendChild(panel);

  // --- References ---
  const messagesEl = document.getElementById('mp-messages');
  const inputEl = document.getElementById('mp-input');
  const sendBtn = document.getElementById('mp-send-btn');
  const clearBtn = document.getElementById('mp-clear-btn');

  // --- Toggle ---
  fab.addEventListener('click', () => {
    isOpen = !isOpen;
    fab.classList.toggle('open', isOpen);
    panel.classList.toggle('open', isOpen);
    if (isOpen) inputEl.focus();
  });

  // --- Send message ---
  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isLoading) return;

    // Add user message
    appendMsg('user', text);
    inputEl.value = '';
    isLoading = true;
    sendBtn.disabled = true;

    // Show typing indicator
    const typing = document.createElement('div');
    typing.className = 'mp-typing';
    typing.innerHTML = '<span></span><span></span><span></span>';
    messagesEl.appendChild(typing);
    scrollBottom();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: SESSION_ID }),
      });
      const data = await res.json();

      typing.remove();

      if (data.status === 'success') {
        appendMsg('assistant', formatMarkdown(data.message));
      } else {
        appendMsg('system', 'Error: ' + (data.error || 'Something went wrong'));
      }
    } catch (e) {
      typing.remove();
      appendMsg('system', 'Connection error — please try again.');
    } finally {
      isLoading = false;
      sendBtn.disabled = false;
      inputEl.focus();
    }
  }

  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // --- Clear ---
  clearBtn.addEventListener('click', async () => {
    messagesEl.innerHTML = '<div class="mp-msg system">Chat cleared. Ask me anything about the dashboard data.</div>';
    try {
      await fetch('/api/chat/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: SESSION_ID }),
      });
    } catch (e) { /* ignore */ }
  });

  // --- Helpers ---
  function appendMsg(role, html) {
    const div = document.createElement('div');
    div.className = `mp-msg ${role}`;
    if (role === 'user') {
      div.textContent = html; // plain text for user
    } else {
      div.innerHTML = html;
    }
    messagesEl.appendChild(div);
    scrollBottom();
  }

  function scrollBottom() {
    requestAnimationFrame(() => {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  }

  function formatMarkdown(text) {
    // Basic markdown to HTML: bold, code, lists, paragraphs
    let html = text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      // Bold: **text**
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // Inline code: `code`
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      // Bullet lists
      .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
      // Numbered lists
      .replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Wrap consecutive <li> in <ul>
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

    // Paragraphs: double newline
    html = html.replace(/\n{2,}/g, '</p><p>');
    // Single newlines within paragraphs
    html = html.replace(/\n/g, '<br>');

    // Wrap in paragraph
    html = '<p>' + html + '</p>';
    // Clean empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, '');

    return html;
  }
})();
