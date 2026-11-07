// Face-to-face over video. The patient calls; a doctor answers from their waiting list.
// Video and sound go straight between the two browsers (WebRTC). The server only passes the
// connection details between them (/api/calls) and keeps what is sent, as in any visit (/api/messages).
// Patient: the camera picture that is sent is also read for signs (camera.js); their message goes to the
// doctor in English, and the doctor's replies are captioned and signed by the avatar.
// Doctor: sees the patient's signs as English captions, and types or speaks a reply.
(() => {
  const app = document.getElementById('vcApp');
  if (!app) return;
  const $ = id => document.getElementById(id);
  const role = app.dataset.role;                  // 'patient' | 'doctor'
  const isPatient = role === 'patient';
  let iceServers;
  try { iceServers = JSON.parse(app.dataset.ice); } catch (e) { iceServers = [{ urls: 'stun:stun.l.google.com:19302' }]; }

  const stage = app.querySelector('.vc-stage');
  const slot = $('vcSlot'), screen = $('vcScreen'), callSlot = $('vcCallSlot');
  const remote = $('vcRemote'), backdrop = $('vcBackdrop');
  const ring = $('vcRing'), ringTitle = $('vcRingTitle'), ringText = $('vcRingText');
  const caption = $('vcCaption'), captionWho = $('vcCaptionWho'), captionText = $('vcCaptionText');
  const who = $('vcWho'), timerEl = $('vcTimer'), dot = $('vcDot');
  const log = $('vcLog'), drawer = $('vcDrawer'), unread = $('vcUnread');
  const ended = $('vcEnded');
  const avatar = $('vcAvatar');
  const toast = msg => window.SignBridge?.toast(msg);
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  document.body.append(screen);   // the full-screen call sits above the app shell
  // Phones: captions sit just above the message box, however tall it grows.
  if ('ResizeObserver' in window) {
    const bottom = screen.querySelector('.vc-bottom');
    new ResizeObserver(() => screen.style.setProperty('--vc-bottom-space', `${bottom.offsetHeight + 20}px`)).observe(bottom);
  }

  async function api(url, body) {
    const opts = body === undefined
      ? { headers: { Accept: 'application/json' } }
      : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
    const res = await fetch(url, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw Object.assign(new Error(data.error || 'Something went wrong. Please try again.'), { status: res.status });
    return data;
  }

  // ---------- state ----------
  let call = null;          // the call, as /api/calls returns it
  let pc = null, dc = null; // the peer connection and its message channel
  let local = null;         // what this side sends
  let mic = null;           // patient: the microphone (the camera belongs to the sign reader)
  let lastSignal = 0, pollToken = 0, live = false, ending = false, ringSince = 0;
  let pendingIce = [];
  const seen = new Set();   // message ids already shown
  let msgTimer = 0, clock = 0, unreadCount = 0;
  let startQueue = () => {};   // doctor: watch the waiting list again (set below)

  // ---------- messages ----------
  const LABEL = { sign: 'signed', speech: 'spoken', text: 'typed' };
  function setCaption(whoText, text) {
    captionWho.textContent = whoText;
    captionText.textContent = text;
    caption.hidden = false;
    caption.classList.remove('is-new');
    void caption.offsetWidth;
    caption.classList.add('is-new');
  }

  function showMessage(m) {
    if (m.id != null) {
      if (seen.has(m.id)) return;
      seen.add(m.id);
    }
    const mine = m.from === role;
    const li = document.createElement('li');
    li.className = 'vc-log__item ' + (mine ? 'is-mine' : 'is-theirs');
    const tag = document.createElement('small');
    tag.textContent = `${mine ? 'You' : m.from === 'doctor' ? 'Doctor' : 'Patient'} · ${LABEL[m.source] || 'sent'}`;
    const p = document.createElement('p');
    p.textContent = m.text;
    li.append(tag, p);
    if (isPatient && !mine && avatar) {
      const again = document.createElement('button');
      again.type = 'button';
      again.className = 'btn btn--quiet btn--sm';
      again.textContent = 'Watch in sign';
      again.addEventListener('click', () => avatar.say(m.text));
      li.append(again);
    }
    log.querySelector('.empty')?.remove();
    log.append(li);
    log.scrollTop = log.scrollHeight;
    if (mine) return;
    if (isPatient) {
      setCaption('Doctor', m.text);
      avatar?.say(m.text);
    } else {
      setCaption(`Patient ${LABEL[m.source] || 'sent'}`, m.text);
    }
    if (drawer.hidden) {
      unreadCount++;
      unread.textContent = String(unreadCount);
      unread.hidden = false;
    }
  }

  // The server copy is the record; the data channel only makes delivery instant. This catches anything
  // the channel missed. Messages from before the call are marked as seen, not replayed.
  async function syncMessages(initial = false) {
    if (!call) return;
    try {
      const { messages } = await api(`/api/messages/${call.session_id}`);
      for (const m of messages) {
        if (initial) { seen.add(m.id); continue; }
        showMessage({ id: m.id, from: m.direction === 'patient_to_doctor' ? 'patient' : 'doctor', text: m.body, source: m.source });
      }
    } catch (e) { /* try again on the next tick */ }
  }

  async function send(text, source, confidence = null) {
    if (!call || !live) { toast(isPatient ? 'Call a doctor first, then send your message.' : 'Answer a call first.'); return null; }
    try {
      const res = await api('/api/messages', { body: text, source, confidence, session_id: call.session_id });
      const m = { id: res.id, from: role, text, source };
      showMessage(m);
      if (dc?.readyState === 'open') dc.send(JSON.stringify({ type: 'msg', ...m }));
      return res;
    } catch (e) {
      toast(e.message);
      return null;
    }
  }
  // The sign reader (camera.js) sends through SignBridge.sendMessage: route it into the call.
  if (isPatient && window.SignBridge) window.SignBridge.sendMessage = (text, source, confidence) => send(text, source || 'sign', confidence);

  // ---------- connection ----------
  async function signal(kind, payload) {
    if (!call) return;
    try { await api(`/api/calls/${call.id}/signal`, { kind, payload }); } catch (e) { /* the call ended */ }
  }

  function wireChannel(channel) {
    dc = channel;
    dc.onmessage = event => {
      let data;
      try { data = JSON.parse(event.data); } catch (e) { return; }
      if (data.type === 'msg') showMessage(data);
      if (data.type === 'bye') finish(isPatient ? 'The doctor ended the call.' : 'The patient ended the call.');
    };
  }

  function makePeer() {
    pc = new RTCPeerConnection({ iceServers });
    pc.onicecandidate = event => { if (event.candidate) signal('ice', event.candidate.toJSON()); };
    pc.ontrack = event => {
      const stream = event.streams[0] || new MediaStream([event.track]);
      if (remote.srcObject !== stream) {
        remote.srcObject = stream;
        backdrop.srcObject = stream;
      }
      remote.play().catch(() => {});
      backdrop.play().catch(() => {});
      goLive();
    };
    pc.onconnectionstatechange = () => {
      const state = pc?.connectionState;
      if (state === 'connected') { setStatus('live'); goLive(); }
      else if (state === 'disconnected') setStatus('Reconnecting…');
      else if (state === 'failed') hangUp('The connection was lost. Please call again.');
    };
    pc.ondatachannel = event => wireChannel(event.channel);
  }

  function addTracks() {
    local?.getTracks().forEach(track => {
      // Signing is all movement: keep the frame rate up and let sharpness drop first on a weak network.
      if (track.kind === 'video' && 'contentHint' in track) track.contentHint = 'motion';
      const sender = pc.addTrack(track, local);
      if (track.kind === 'video') {
        try {
          const params = sender.getParameters();
          params.degradationPreference = 'maintain-framerate';
          sender.setParameters(params).catch(() => {});
        } catch (e) { /* not supported here */ }
      }
    });
  }

  function flushIce() {
    pendingIce.splice(0).forEach(c => pc.addIceCandidate(c).catch(() => {}));
  }

  async function onSignal(s) {
    if (s.kind === 'offer' && isPatient) {
      if (!pc) { makePeer(); addTracks(); }
      await pc.setRemoteDescription(s.payload);
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      await signal('answer', { type: answer.type, sdp: answer.sdp });
      flushIce();
    } else if (s.kind === 'answer' && !isPatient && pc) {
      await pc.setRemoteDescription(s.payload);
      flushIce();
    } else if (s.kind === 'ice') {
      if (pc?.remoteDescription) await pc.addIceCandidate(s.payload).catch(() => {});
      else pendingIce.push(s.payload);
    }
  }

  const RING_LIMIT = 10 * 60 * 1000;   // same as the server's waiting list
  async function poll() {
    const token = ++pollToken;
    while (call && token === pollToken) {
      try {
        const data = await api(`/api/calls/${call.id}?after=${lastSignal}`);
        if (token !== pollToken) return;
        Object.assign(call, { status: data.status, doctor: data.doctor, patient: data.patient, answered_at: data.answered_at });
        for (const s of data.signals) {
          lastSignal = Math.max(lastSignal, s.id);
          try { await onSignal(s); } catch (e) { console.error(e); }
        }
        if (data.status === 'ended') {
          finish(live ? (isPatient ? 'The doctor ended the call.' : 'The patient ended the call.') : 'The call ended.');
          return;
        }
        if (isPatient && data.status === 'active' && !live) {
          ringTitle.textContent = `${data.doctor} is joining…`;
          ringText.textContent = 'Connecting the video.';
        }
        if (isPatient && data.status === 'ringing' && Date.now() - ringSince > RING_LIMIT) {
          hangUp('No doctor could answer just now. Please try again, or ask staff at the desk.');
          return;
        }
      } catch (e) {
        if (e.status === 403 || e.status === 404) { finish('The call ended.'); return; }
      }
      await sleep(live ? 2000 : 700);
    }
  }

  // ---------- look and feel ----------
  function setStatus(text) {
    dot.classList.toggle('is-live', text === 'live');
    dot.classList.toggle('is-wait', text !== 'live');
    if (text !== 'live') timerEl.textContent = text;
  }

  function startClock() {
    const from = call?.answered_at ? Date.parse(call.answered_at) : Date.now();
    const tick = () => {
      const s = Math.max(0, Math.round((Date.now() - from) / 1000));
      timerEl.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    };
    tick();
    clearInterval(clock);
    clock = setInterval(tick, 1000);
  }

  // Zoom between the stage's place on the page and the full-screen call, centre to centre.
  function zoom(move) {
    const first = stage.getBoundingClientRect();
    move();
    const last = stage.getBoundingClientRect();
    if (reduce || !first.width || !last.width) return;
    const s = first.width / last.width;
    const dx = first.left + first.width / 2 - (last.left + last.width / 2);
    const dy = first.top + first.height / 2 - (last.top + last.height / 2);
    stage.animate([
      { transform: `translate(${dx}px, ${dy}px) scale(${s})` },
      { transform: 'none' }
    ], { duration: 700, easing: 'cubic-bezier(.2, .85, .2, 1)' });
  }

  function showRing(on, title, text) {
    ring.hidden = !on;
    if (title) ringTitle.textContent = title;
    if (text) ringText.textContent = text;
  }

  function goLive() {
    if (live || !call) return;
    live = true;
    showRing(false);
    ended.hidden = true;
    screen.hidden = false;
    document.documentElement.classList.add('vc-live');
    zoom(() => {
      callSlot.append(stage);
      stage.classList.add('is-live');
    });
    if (!reduce) screen.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 450, easing: 'ease-out' });
    who.textContent = isPatient ? (call.doctor || 'Your doctor') : (call.patient || 'Patient');
    setStatus('live');
    startClock();
    syncMessages();
    clearInterval(msgTimer);
    msgTimer = setInterval(syncMessages, 3000);
    (isPatient ? $('messageEdit') : $('vcReply'))?.focus({ preventScroll: true });
  }

  function resetControls() {
    ['vcMute', 'vcCam', 'vcChat'].forEach(id => $(id).setAttribute('aria-pressed', 'false'));
    drawer.hidden = true;
    unreadCount = 0;
    unread.hidden = true;
    caption.hidden = true;
    log.replaceChildren(Object.assign(document.createElement('li'), { className: 'empty', textContent: 'Nothing said yet.' }));
    seen.clear();
  }

  function finish(message) {
    if (ending || !call) return;
    ending = true;
    const done = call;
    const wasLive = live;
    pollToken++;
    clearInterval(msgTimer);
    clearInterval(clock);
    try { dc?.close(); } catch (e) {}
    try { pc?.close(); } catch (e) {}
    pc = dc = null;
    pendingIce = [];
    lastSignal = 0;
    remote.srcObject = null;
    backdrop.srcObject = null;
    mic?.getTracks().forEach(t => t.stop());
    mic = null;
    if (isPatient) {
      local?.getVideoTracks().forEach(t => { t.enabled = true; });   // the sign reader keeps the camera
    } else {
      local?.getTracks().forEach(t => t.stop());
      $('vcLocal').srcObject = null;
      $('vcLocalEmpty').hidden = false;
    }
    local = null;
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    showRing(false);
    if (wasLive) {
      zoom(() => {
        slot.append(stage);
        stage.classList.remove('is-live');
      });
      const hide = () => {
        if (live) return;   // a new call started meanwhile
        screen.hidden = true;
        document.documentElement.classList.remove('vc-live');
      };
      if (reduce) hide();
      else {
        screen.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 350, easing: 'ease-in' }).finished.then(hide, hide);
        setTimeout(hide, 450);   // background tabs pause animations; close the call screen anyway
      }
      const mins = done.answered_at ? Math.max(1, Math.round((Date.now() - Date.parse(done.answered_at)) / 60000)) : 0;
      $('vcEndedTitle').textContent = mins ? `Call ended · ${mins} min` : 'Call ended';
      $('vcSummary').href = `/visit/${done.session_id}/summary`;
      ended.hidden = false;
    }
    resetControls();
    who.textContent = isPatient ? 'Your camera' : 'Waiting room';
    timerEl.textContent = '';
    dot.className = 'vc-dot';
    call = null;
    live = false;
    ending = false;
    if (isPatient) $('vcCall').disabled = false;
    else startQueue();
    if (message) toast(message);
  }

  async function hangUp(message = 'Call ended.') {
    if (!call) return;
    const id = call.id;
    try { if (dc?.readyState === 'open') dc.send(JSON.stringify({ type: 'bye' })); } catch (e) {}
    api(`/api/calls/${id}/end`, {}).catch(() => {});
    finish(message);
  }
  // Closing the tab ends the call for the other side too.
  window.addEventListener('pagehide', () => {
    if (call) navigator.sendBeacon?.(`/api/calls/${call.id}/end`);
  });

  // ---------- call controls ----------
  function toggleTracks(button, kind, offMessage) {
    const tracks = kind === 'audio' ? local?.getAudioTracks() : local?.getVideoTracks();
    if (!tracks?.length) { toast(kind === 'audio' ? 'No microphone is in use.' : 'No camera is in use.'); return; }
    const off = button.getAttribute('aria-pressed') !== 'true';
    tracks.forEach(t => { t.enabled = !off; });
    button.setAttribute('aria-pressed', String(off));
    if (off && offMessage) toast(offMessage);
  }
  $('vcMute').addEventListener('click', e => toggleTracks(e.currentTarget, 'audio'));
  $('vcCam').addEventListener('click', e => toggleTracks(e.currentTarget, 'video',
    isPatient ? 'Camera off. Your doctor cannot see your signs until you turn it back on.' : ''));
  const setDrawer = open => {
    drawer.hidden = !open;
    $('vcChat').setAttribute('aria-pressed', String(open));
    if (open) { unreadCount = 0; unread.hidden = true; log.scrollTop = log.scrollHeight; }
  };
  $('vcChat').addEventListener('click', () => setDrawer(drawer.hidden));
  $('vcDrawerClose').addEventListener('click', () => setDrawer(false));
  const full = $('vcFull');
  if (!screen.requestFullscreen) full.hidden = true;
  full.addEventListener('click', () => {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else screen.requestFullscreen().catch(() => toast('Full screen is not available here.'));
  });
  document.addEventListener('fullscreenchange', () => full.classList.toggle('is-on', !!document.fullscreenElement));
  $('vcHang').addEventListener('click', () => hangUp());
  $('vcCancel').addEventListener('click', () => hangUp('Call cancelled.'));

  // Grow the message boxes with their text, up to four lines.
  const grow = box => { box.style.height = 'auto'; box.style.height = Math.min(box.scrollHeight, 120) + 'px'; };

  // ---------- patient ----------
  if (isPatient) {
    const callBtn = $('vcCall');
    const feed = $('cameraFeed');
    const edit = $('messageEdit');
    edit.addEventListener('input', () => grow(edit));
    edit.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('confirmSendBtn').click(); }
    });

    const cameraReady = () => new Promise(resolve => {
      if (feed.srcObject) { resolve(feed.srcObject); return; }
      const onCam = e => { window.removeEventListener('sb:camera', onCam); resolve(e.detail.stream); };
      window.addEventListener('sb:camera', onCam);
      $('startCameraBtn').click();
    });

    callBtn.addEventListener('click', async () => {
      if (call) return;
      callBtn.disabled = true;
      ended.hidden = true;
      const cam = await cameraReady();
      if (!cam) {
        toast('Turn on your camera to call, so the doctor can see your signs.');
        callBtn.disabled = false;
        return;
      }
      try {
        mic = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      } catch (e) { mic = null; }   // a call works without sound
      local = new MediaStream([...cam.getVideoTracks(), ...(mic ? mic.getAudioTracks() : [])]);
      try {
        call = await api('/api/calls', {});
      } catch (e) {
        toast(e.message);
        mic?.getTracks().forEach(t => t.stop());
        mic = local = null;
        callBtn.disabled = false;
        return;
      }
      await syncMessages(true);
      ringSince = Date.now();
      showRing(true, 'Calling a doctor…', 'A doctor will answer soon. Keep this page open.');
      setStatus('Calling…');
      poll();
    });
    // If the camera is switched off before a call, the lobby button turns it back on.
    window.addEventListener('sb:camera', e => {
      if (!e.detail.stream && call && !live) hangUp('The camera turned off, so the call was cancelled.');
    });
  }

  // ---------- doctor ----------
  if (!isPatient) {
    const queue = $('vcQueue'), count = $('vcWaitCount');
    let queueTimer = 0;
    const baseTitle = document.title;
    const ago = iso => {
      const m = Math.round((Date.now() - Date.parse(iso)) / 60000);
      return m < 1 ? 'just now' : `${m} min`;
    };

    async function refreshQueue() {
      if (call) return;
      let calls = [];
      try { ({ calls } = await api('/api/calls/waiting')); } catch (e) { return; }
      if (call) return;
      count.textContent = calls.length ? `${calls.length} waiting` : 'No one waiting';
      count.classList.toggle('chip--accent', calls.length > 0);
      document.title = calls.length ? `(${calls.length}) ${baseTitle}` : baseTitle;
      if (!calls.length) {
        queue.replaceChildren(Object.assign(document.createElement('li'), {
          className: 'empty', textContent: 'No patients are waiting. New calls appear here on their own.'
        }));
        return;
      }
      queue.replaceChildren(...calls.map(c => {
        const li = document.createElement('li');
        li.className = 'vc-queue__item';
        const meta = document.createElement('div');
        const name = document.createElement('strong');
        name.textContent = c.patient;
        const since = document.createElement('small');
        since.textContent = `Waiting ${ago(c.since)}`;
        meta.append(name, since);
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn--primary btn--sm';
        btn.textContent = 'Answer';
        btn.addEventListener('click', () => answer(c, btn));
        li.append(meta, btn);
        return li;
      }));
    }
    startQueue = () => {
      clearInterval(queueTimer);
      refreshQueue();
      queueTimer = setInterval(refreshQueue, 3000);
    };

    async function answer(c, btn) {
      if (call) return;
      btn.disabled = true;
      ended.hidden = true;
      clearInterval(queueTimer);
      try {
        local = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: 'user' },
          audio: { echoCancellation: true, noiseSuppression: true }
        });
      } catch (e) {
        try { local = await navigator.mediaDevices.getUserMedia({ audio: true }); }
        catch (e2) { local = new MediaStream(); }
        toast('Your camera is not available. The patient will not see you, but you can still read and reply.');
      }
      $('vcLocal').srcObject = local;
      $('vcLocalEmpty').hidden = local.getVideoTracks().length > 0;
      try {
        call = await api(`/api/calls/${c.id}/accept`, {});
      } catch (e) {
        toast(e.message);
        local.getTracks().forEach(t => t.stop());
        local = null;
        $('vcLocal').srcObject = null;
        $('vcLocalEmpty').hidden = false;
        startQueue();
        return;
      }
      document.title = baseTitle;
      await syncMessages(true);
      showRing(true, `Connecting to ${call.patient}…`, 'The video starts in a moment.');
      setStatus('Connecting…');
      makePeer();
      addTracks();
      // Always receive the patient's picture and sound, even with no camera or microphone here.
      if (!local.getVideoTracks().length) pc.addTransceiver('video', { direction: 'recvonly' });
      if (!local.getAudioTracks().length) pc.addTransceiver('audio', { direction: 'recvonly' });
      wireChannel(pc.createDataChannel('chat'));
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await signal('offer', { type: offer.type, sdp: offer.sdp });
      poll();
    }

    // Reply: type, or speak (the words fill the box so they can be checked before sending).
    const form = $('vcReplyForm'), reply = $('vcReply'), micBtn = $('vcMic');
    let spoken = false;
    reply.addEventListener('input', () => { grow(reply); spoken = false; });
    reply.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
    });
    form.addEventListener('submit', async e => {
      e.preventDefault();
      const text = reply.value.trim();
      if (!text) return;
      const sent = await send(text, spoken ? 'speech' : 'text');
      if (sent) { reply.value = ''; spoken = false; grow(reply); }
    });
    form.querySelectorAll('[data-quick]').forEach(b => b.addEventListener('click', () => send(b.dataset.quick, 'text')));

    const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
    let rec = null;
    if (!Rec) micBtn.hidden = true;
    else micBtn.addEventListener('click', () => {
      if (rec) { rec.stop(); return; }
      rec = new Rec();
      rec.lang = 'en-IN';
      rec.interimResults = true;
      rec.continuous = true;
      const before = reply.value.trim();
      micBtn.setAttribute('aria-pressed', 'true');
      micBtn.querySelector('span').textContent = 'Listening…';
      rec.onresult = event => {
        const said = Array.from(event.results).map(r => r[0].transcript).join(' ').trim();
        reply.value = [before, said].filter(Boolean).join(' ');
        spoken = true;
        grow(reply);
      };
      rec.onerror = event => toast(event.error === 'not-allowed' ? 'Microphone access was blocked. Type your reply instead.'
        : event.error === 'network' ? 'Speaking needs internet. Type your reply instead.' : 'Could not hear that. Try again or type.');
      rec.onend = () => {
        rec = null;
        micBtn.setAttribute('aria-pressed', 'false');
        micBtn.querySelector('span').textContent = 'Speak';
        reply.focus();
      };
      rec.start();
    });

    startQueue();
  }
})();
