// Face-to-face mode: everything stays on this device. The patient's signed messages and the
// doctor's replies go into a local log instead of the server, so it works fully offline.
(() => {
  const KEY = 'sb-together-log';
  const logEl = document.getElementById('log');
  const said = document.getElementById('patientSaid');
  const doctorText = document.getElementById('doctorText');
  const avatar = document.getElementById('togetherAvatar');
  let log = [];
  try { log = JSON.parse(localStorage.getItem(KEY) || '[]'); } catch (e) { log = []; }

  const save = () => { try { localStorage.setItem(KEY, JSON.stringify(log.slice(-200))); } catch (e) {} };
  const time = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  function render() {
    logEl.replaceChildren(...log.map(m => {
      const li = document.createElement('li');
      li.className = 'message ' + (m.from === 'doctor' ? 'message--staff' : 'message--patient');
      const body = document.createElement('div');
      body.className = 'message__body';
      const tag = document.createElement('span');
      tag.className = 'message-tag';
      tag.textContent = `${m.from === 'doctor' ? 'Doctor' : 'Patient'} · ${m.at}`;
      const p = document.createElement('p');
      p.textContent = m.text;
      body.append(tag, p);
      if (m.from === 'doctor') {
        const b = document.createElement('button');
        b.type = 'button'; b.className = 'btn btn--quiet btn--sm'; b.textContent = 'Sign again';
        b.addEventListener('click', () => avatar.say(m.text));
        body.append(b);
      }
      li.append(body);
      return li;
    }));
    if (!log.length) logEl.replaceChildren(Object.assign(document.createElement('li'), { className: 'empty', textContent: 'No messages yet.' }));
    const last = [...log].reverse().find(m => m.from === 'patient');
    const p = said.querySelector('p');
    p.textContent = last ? last.text : 'Nothing yet.';
    p.className = last ? '' : 'empty';
  }

  // The sign panel calls SignBridge.sendMessage; keep it on this device instead.
  SignBridge.sendMessage = async text => {
    log.push({ from: 'patient', text, at: time() });
    save(); render();
    said.classList.remove('said--flash'); void said.offsetWidth; said.classList.add('said--flash');
    return { local: true };
  };

  function reply(text) {
    text = (text || '').trim();
    if (!text) { SignBridge.toast('Type a reply first.'); return; }
    log.push({ from: 'doctor', text, at: time() });
    save(); render();
    avatar.say(text);
    doctorText.value = '';
  }

  document.getElementById('signItBtn').addEventListener('click', () => reply(doctorText.value));
  doctorText.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); reply(doctorText.value); } });
  document.querySelectorAll('[data-quick]').forEach(b => b.addEventListener('click', () => reply(b.dataset.quick)));
  document.getElementById('clearLog').addEventListener('click', () => {
    if (!confirm('Clear this conversation from the device?')) return;
    log = []; save(); render();
  });

  // Speech needs an internet connection in most browsers; typing always works.
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const mic = document.getElementById('micBtn');
  if (!Rec) mic.hidden = true;
  else mic.addEventListener('click', () => {
    if (!navigator.onLine) { SignBridge.toast('Speaking needs internet. Please type instead.'); return; }
    const r = new Rec();
    r.lang = 'en-IN'; r.interimResults = true;
    mic.disabled = true; mic.lastChild.textContent = ' Listening…';
    r.onresult = e => { doctorText.value = Array.from(e.results).map(x => x[0].transcript).join(' '); };
    r.onend = () => { mic.disabled = false; mic.lastChild.textContent = ' Speak'; };
    r.onerror = () => SignBridge.toast('Could not hear that. Please type instead.');
    r.start();
  });

  render();
})();
