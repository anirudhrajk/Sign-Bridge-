document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash__x').forEach(btn => {
    btn.addEventListener('click', () => btn.closest('.flash').remove());
  });

  const netStatus = document.getElementById('netStatus');
  function updateNet() {
    if (!netStatus) return;
    // Only shown when there is a problem; "Online" is the normal state.
    netStatus.hidden = navigator.onLine;
  }
  window.addEventListener('online', updateNet);
  window.addEventListener('offline', updateNet);
  updateNet();

  document.querySelectorAll('[data-quick-phrase], [data-emergency]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const text = btn.dataset.quickPhrase || btn.dataset.emergency;
      btn.disabled = true;
      const sent = await sendMessage(text, 'emergency');
      btn.disabled = false;
      if (sent) toast(`Sent to doctor: ${text}`);
    });
  });

  document.querySelectorAll('[data-a11y]').forEach(box => {
    const key = box.dataset.a11y;
    box.checked = document.documentElement.classList.contains(`a11y-${key}`);
    box.addEventListener('change', () => {
      document.documentElement.classList.toggle(`a11y-${key}`, box.checked);
      try { localStorage.setItem(`sb-${key}`, box.checked ? '1' : '0'); } catch (e) {}
    });
  });
});

async function sendMessage(body, source = 'text', confidence = null, sessionId = null) {
  try {
    const res = await fetch('/api/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ body, source, confidence, session_id: sessionId })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      toast(err.error || 'Could not send message');
      return null;
    }
    return await res.json();
  } catch (e) {
    toast('Network error — message not sent');
    return null;
  }
}

function toast(message) {
  const stack = document.getElementById('toastStack');
  if (!stack) return;
  const el = document.createElement('div');
  el.className = 'toast';
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => el.remove(), 3200);
}

window.SignBridge = { sendMessage, toast };