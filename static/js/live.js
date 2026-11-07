// Keeps a conversation up to date. On the patient's screen it also signs new doctor messages,
// asks "did you understand?", and shows the full-screen "it's your turn" alert.
(() => {
  const list = document.querySelector('[data-live-messages]');
  if (!list) return;
  const sessionId = list.dataset.sessionId;
  const isPatient = list.dataset.viewer === 'patient';
  const avatar = isPatient ? document.getElementById('patientAvatar') : null;
  let lastSignature = null, lastSeenId = null, lastCall = null;

  const SOURCE_LABEL = { sign: 'Signed', speech: 'Spoken', text: 'Typed', emergency: 'Emergency', pain: 'Pain report', symptoms: 'Symptoms', request: 'Request' };
  const ANSWER_LABEL = { understood: '✓ Understood', repeat: 'Asked to repeat', simpler: 'Asked for simpler words' };
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

  async function answer(m, value) {
    const res = await fetch(`/api/messages/${m.id}/feedback`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ answer: value }),
    });
    if (res.ok) { SignBridge.toast(value === 'understood' ? 'Thanks, your doctor can see you understood.' : 'Your doctor has been asked.'); refresh(); }
    else SignBridge.toast('Could not send your answer.');
    if (value === 'repeat' && avatar) avatar.say(m.body);
  }

  function render(messages) {
    const ordered = 'newestFirst' in list.dataset ? [...messages].reverse().slice(0, 10) : messages.slice(-40);
    list.replaceChildren(...ordered.map(m => {
      const fromStaff = m.sender_role !== 'patient';
      const item = el('li', 'message ' + (fromStaff ? 'message--staff' : 'message--patient') +
        (['emergency', 'pain', 'symptoms'].includes(m.source) ? ' message--urgent' : ''));
      const body = el('div', 'message__body');
      body.append(el('span', 'message-tag', `${m.sender} · ${SOURCE_LABEL[m.source] || m.source}`),
                  el('p', null, m.body), el('small', null, m.created_at));
      if (fromStaff && isPatient) {
        const row = el('div', 'msg-actions');
        if (avatar) {
          const watch = el('button', 'btn btn--quiet btn--sm', 'Watch in sign');
          watch.type = 'button';
          watch.addEventListener('click', () => avatar.say(m.body));
          row.append(watch);
        }
        if (m.understood) row.append(el('span', 'chip chip--soft', ANSWER_LABEL[m.understood]));
        else for (const [v, label] of [['understood', 'I understand'], ['repeat', 'Please repeat'], ['simpler', 'Simpler please']]) {
          const b = el('button', 'btn btn--ghost btn--sm' + (v === 'understood' ? ' btn--ok' : ''), label);
          b.type = 'button';
          b.addEventListener('click', () => answer(m, v));
          row.append(b);
        }
        body.append(row);
      } else if (fromStaff && !isPatient) {
        body.append(el('span', 'msg-status' + (m.understood ? ' msg-status--' + m.understood : ''),
          m.understood ? ANSWER_LABEL[m.understood] : 'Waiting for the patient to confirm'));
      }
      item.append(el('span', 'message__dot' + (fromStaff ? ' message__dot--doctor' : '')), body);
      return item;
    }));
    if (!messages.length) list.append(el('li', 'empty', list.dataset.empty));
  }

  const showCall = call => SignBridge.showCall(call);

  async function refresh() {
    try {
      const res = await fetch(`/api/messages/${sessionId}`);
      if (!res.ok) return;
      const { messages, call } = await res.json();
      const signature = messages.map(m => `${m.id}:${m.understood || ''}`).join(',');
      if (signature !== lastSignature) {
        lastSignature = signature;
        render(messages);
        const newest = messages.filter(m => m.sender_role !== 'patient').at(-1);
        if (avatar && newest && lastSeenId !== null && newest.id > lastSeenId) avatar.say(newest.body);
        if (newest) lastSeenId = Math.max(lastSeenId ?? 0, newest.id);
        else if (lastSeenId === null) lastSeenId = 0;
      }
      if (isPatient && call) {
        // Show only calls made after this page opened, or within the last two minutes.
        const fresh = Date.now() - Date.parse(call.at) < 120000;
        if (call.at !== lastCall && (lastCall !== null || fresh)) showCall(call);
        lastCall = call.at;
      } else if (lastCall === null) lastCall = '';
    } catch (e) { /* try again on the next tick */ }
  }

  const form = document.getElementById('replyForm');
  form?.addEventListener('submit', async event => {
    event.preventDefault();
    const box = document.getElementById('replyText');
    const text = box.value.trim();
    if (!text) return;
    const sent = await SignBridge.sendMessage(text, 'text', null, Number(form.dataset.sessionId));
    if (sent) { box.value = ''; refresh(); }
  });

  document.getElementById('callInBtn')?.addEventListener('click', async () => {
    const room = (document.getElementById('callRoom')?.value || '').trim();
    const res = await fetch(`/api/sessions/${sessionId}/call`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ room }),
    });
    SignBridge.toast(res.ok ? "The patient's screen is flashing and vibrating." : 'Could not call the patient.');
  });

  refresh();
  setInterval(refresh, 2500);
})();
