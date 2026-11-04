// Keeps a conversation list up to date and handles typed replies.
(() => {
  const list = document.querySelector('[data-live-messages]');
  if (!list) return;
  const sessionId = list.dataset.sessionId;
  let lastSignature = null;

  const SOURCE_LABEL = { sign: 'Signed', speech: 'Spoken', text: 'Typed', emergency: 'Emergency' };

  function render(messages) {
    const ordered = 'newestFirst' in list.dataset ? [...messages].reverse().slice(0, 10) : messages.slice(-40);
    list.replaceChildren(...ordered.map(m => {
      const fromStaff = m.sender_role !== 'patient';
      const item = document.createElement('li');
      item.className = 'message ' + (fromStaff ? 'message--staff' : 'message--patient');
      const dot = document.createElement('span');
      dot.className = 'message__dot' + (fromStaff ? ' message__dot--doctor' : '');
      const body = document.createElement('div');
      body.className = 'message__body';
      const tag = document.createElement('span');
      tag.className = 'message-tag';
      tag.textContent = `${m.sender} · ${SOURCE_LABEL[m.source] || m.source}`;
      const text = document.createElement('p');
      text.textContent = m.body;
      const meta = document.createElement('small');
      meta.textContent = m.created_at;
      body.append(tag, text, meta);
      item.append(dot, body);
      return item;
    }));
    if (!messages.length) {
      const empty = document.createElement('li');
      empty.className = 'empty';
      empty.textContent = list.dataset.empty;
      list.append(empty);
    }
  }

  async function refresh() {
    try {
      const res = await fetch(`/api/messages/${sessionId}`);
      if (!res.ok) return;
      const { messages } = await res.json();
      const signature = messages.map(m => m.id).join(',');
      if (signature !== lastSignature) {
        lastSignature = signature;
        render(messages);
      }
    } catch (e) { /* try again on the next tick */ }
  }

  const form = document.getElementById('replyForm');
  form?.addEventListener('submit', async event => {
    event.preventDefault();
    const box = document.getElementById('replyText');
    const text = box.value.trim();
    if (!text) return;
    const sent = await SignBridge.sendMessage(text, 'text', null, Number(form.dataset.sessionId));
    if (sent) {
      box.value = '';
      refresh();
    }
  });

  refresh();
  setInterval(refresh, 2500);
})();
