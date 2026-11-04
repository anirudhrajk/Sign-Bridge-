document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash__x').forEach(btn => {
    btn.addEventListener('click', () => btn.closest('.flash').remove());
  });

  const netStatus = document.getElementById('netStatus');
  const netText = netStatus?.querySelector('[data-net-text]');
  function updateNet() {
    if (!netStatus) return;
    const online = navigator.onLine;
    netText.textContent = online ? 'Online' : 'Offline';
    netStatus.querySelector('.dot').style.background = online ? 'var(--success)' : 'var(--danger)';
  }
  window.addEventListener('online', updateNet);
  window.addEventListener('offline', updateNet);
  updateNet();
});

async function sendMessage(body, source = 'text', confidence = null) {
  try {
    const res = await fetch('/api/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ body, source, confidence })
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