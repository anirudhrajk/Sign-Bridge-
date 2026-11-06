// Sign translator: camera -> MediaPipe Holistic -> sign segmentation -> on-device model.
(() => {
  const $ = id => document.getElementById(id);
  const video = $('cameraFeed'), canvas = $('landmarkCanvas'), ctx = canvas.getContext('2d');
  const status = $('cameraStatus'), empty = $('cameraEmpty'), meter = $('signMeter');
  const startBtn = $('startCameraBtn'), stopBtn = $('stopCameraBtn'), nowBtn = $('recognizeBtn');
  const resultBox = $('resultBox'), alts = $('alts'), altsList = $('altsList'), altsLabel = $('altsLabel');
  const message = $('messageEdit'), sendBtn = $('confirmSendBtn'), undoBtn = $('undoBtn'), clearBtn = $('clearBtn');
  const modelStatus = $('modelStatus');

  // Tuning
  const SURE = 0.7;           // add the word automatically at or above this (99.8% right on test data)
  const MAYBE = 0.2;          // below this, ask the patient to sign again
  const MIN_FRAMES = 8;       // shortest sign we try to read
  const MAX_FRAMES = 150;     // read anyway if a sign runs this long
  // Timing is in milliseconds, so it behaves the same on fast and slow devices.
  const END_GAP_MS = 300;     // hands out of view this long ends a sign
  const STILL_MS = 700;       // hands held still this long (after moving) ends a sign
  const MIN_SIGN_MS = 350;    // ignore twitches shorter than this
  const STILL_SPEED = 0.15;   // wrist speed (shoulder widths per second) that counts as still

  let model = null, holistic = null, stream = null, running = false, busy = false;
  let recent = [];            // last ~3 s of frames, for "Read sign now"
  let segment = [], gapSince = 0, stillSince = 0, segStart = 0, moved = 0, lastWrists = null, lastT = 0;
  let lastWord = null;        // {text, start} of the word we last added, so it can be replaced
  const confidences = [];
  let teach = null;           // {label, done} while recording examples of a new sign
  let practice = null;        // {name, label} after "show me how"
  const TEACH_COUNT = 5;
  let fpsT = performance.now(), fpsN = 0, cameraFps = 0;
  let tracker = 1, slowSeconds = 0;   // 1 full tracker, 0 lite
  const SLOW_FPS = 10, SLOW_FOR = 4;  // this slow for this many seconds: drop to the lite tracker
  let lastStatus = '';
  // Only touch the page when the message changes (this runs for every camera frame).
  const say = text => { if (text !== lastStatus) { status.textContent = text; lastStatus = text; } };

  // ---------- model ----------
  const lang = document.getElementById('translator').dataset.lang || 'isl';
  const picker = $('langPicker');
  picker?.addEventListener('change', async () => {
    picker.disabled = true;
    const res = await fetch('/api/preferences/sign-language', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: picker.value }),
    });
    if (res.ok) location.reload();
    else { picker.disabled = false; picker.value = lang; SignBridge.toast('Could not switch sign language.'); }
  });

  SignModel.load(`/static/models/${lang}`).then(m => {
    model = m;
    modelStatus.textContent = `${lang.toUpperCase()} · ${m.classes.length} signs ready`;
    modelStatus.classList.add('chip--soft');
    buildIndex(m);
    loadTaught();
    if (running) nowBtn.disabled = false;
  }).catch(() => {
    modelStatus.textContent = 'Signs unavailable';
    SignBridge.toast('Could not load the sign model. Typing still works.');
  });

  // ---------- camera ----------
  async function start() {
    startBtn.disabled = true;
    say('Starting camera…');
    try {
      // 1280x720 at up to 60 fps; the browser uses the closest the camera offers.
      // Frame size hardly changes tracking time (25 ms at 640, 28 ms at 1920).
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 }, facingMode: 'user' }, audio: false });
      cameraFps = Math.round(stream.getVideoTracks()[0]?.getSettings?.().frameRate || 0);
    } catch (e) {
      say('Camera blocked');
      SignBridge.toast('Allow camera access in your browser to sign.');
      startBtn.disabled = false;
      return;
    }
    video.srcObject = stream;
    await video.play();
    if (!holistic) {
      say('Loading hand tracking (first time takes a few seconds)…');
      holistic = new Holistic({ locateFile: f => `/static/vendor/holistic/${f}` });
      holistic.setOptions({ modelComplexity: tracker, smoothLandmarks: true, refineFaceLandmarks: false,
                            minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
      holistic.onResults(onResults);
    }
    running = true;
    empty.hidden = true;
    stopBtn.disabled = false;
    nowBtn.disabled = !model;
    loop();
  }

  // Track each new camera frame once. requestVideoFrameCallback fires per camera frame, so a
  // 60 Hz screen does not send the same picture twice (which only heats a phone and slows it down).
  // A 50 ms timer takes over if the browser stops reporting frames, so tracking never stalls.
  const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
  function perFrame(cb) {
    if (!hasRVFC) { requestAnimationFrame(cb); return; }
    let done = false, id = 0;
    const go = () => { if (done) return; done = true; clearTimeout(timer); video.cancelVideoFrameCallback?.(id); cb(); };
    const timer = setTimeout(go, 50);
    id = video.requestVideoFrameCallback(go);
  }
  async function loop() {
    if (!running) return;
    if (!busy && video.readyState >= 2) {
      busy = true;
      try { await holistic.send({ image: video }); } catch (e) { console.error(e); }
      busy = false;
    }
    perFrame(loop);
  }

  function stop() {
    running = false;
    stream?.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    say('Camera off');
    empty.hidden = false;
    meter.hidden = true;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    nowBtn.disabled = true;
    segment = []; recent = [];
  }

  // ---------- per frame ----------
  function onResults(res) {
    if (!running) return;
    draw(res);
    if (!model) { say('Loading signs…'); return; }
    const frame = model.toPoints(res);
    updateCheck(res, frame);
    recent.push(frame);
    if (recent.length > 90) recent.shift();
    if (!model) { say('Loading signs…'); return; }

    const hands = model.hasHand(frame);
    if (!res.poseLandmarks) say('Move back so your shoulders are in view');

    const now = (res.time ?? performance.now());
    const dt = lastT ? Math.min(0.25, (now - lastT) / 1000) : 0;
    lastT = now;
    if (hands) {
      const motion = wristMotion(frame);
      if (!segment.length) { segStart = now; stillSince = now; moved = 0; }
      segment.push(frame);
      gapSince = 0;
      if (dt > 0 && motion / dt > STILL_SPEED) { stillSince = now; moved += motion; }
      if (res.poseLandmarks) say('Reading your sign…');
      meter.hidden = false;
      meter.firstElementChild.style.width = `${Math.min(100, ((now - segStart) / 1800) * 100)}%`;
      // A held pause after real movement ends the sign.
      if (now - stillSince >= STILL_MS && moved > 0.15 && now - segStart >= MIN_SIGN_MS && segment.length >= MIN_FRAMES) finish();
      else if (segment.length >= MAX_FRAMES) finish();
    } else if (segment.length) {
      if (!gapSince) gapSince = now;
      if (now - gapSince >= END_GAP_MS) finish();
    } else if (res.poseLandmarks) {
      say('Ready: raise your hands to sign');
      meter.hidden = true;
    }
  }

  function wristMotion(frame) {
    const ls = frame[1], rs = frame[2];
    const scale = ls && rs ? Math.hypot(ls[0] - rs[0], ls[1] - rs[1]) || 0.25 : 0.25;
    const [wl, wr] = model.wrists || [7, 28];
    const w = [frame[wl], frame[wr]];
    let m = 0;
    if (lastWrists) w.forEach((p, i) => { if (p && lastWrists[i]) m = Math.max(m, Math.hypot(p[0] - lastWrists[i][0], p[1] - lastWrists[i][1]) / scale); });
    lastWrists = w;
    return m;
  }

  function finish() {
    const frames = segment;
    segment = []; gapSince = 0; moved = 0; lastWrists = null;
    meter.hidden = true;
    if (frames.length < MIN_FRAMES) return;
    if (teach) { saveExample(frames); return; }
    classify(frames);
  }

  async function classify(frames) {
    const t0 = performance.now();
    const top = await model.predict(frames, 5);
    const ms = Math.round(performance.now() - t0);
    showResult(top, ms);
  }

  // ---------- results ----------
  function showResult(top, ms) {
    const best = top[0];
    if (practice) showPractice(top);
    const sure = best.p >= SURE;
    resultBox.replaceChildren();
    const word = document.createElement('p');
    const meta = document.createElement('small');
    meta.className = 'muted';
    if (best.p < MAYBE) {
      word.className = 'empty';
      word.textContent = 'SignBridge doesn’t know this sign yet. Try again slowly, pick a word below, type it, or teach it under “Teach a new sign”.';
      resultBox.append(word);
      renderAlts(top.slice(0, 4), 'Or pick the word you meant:', false);
      return;
    }
    word.textContent = best.label;
    meta.textContent = `${Math.round(best.p * 100)}% sure · read in ${ms} ms`;
    resultBox.append(word, meta);
    if (sure) {
      addWord(best.label, best.p);
      renderAlts(top.slice(1, 4), 'Not this word? Tap the right one:', true);
    } else {
      renderAlts(top.slice(0, 4), 'Not sure. Tap the word you meant:', false);
    }
  }

  function renderAlts(options, label, replace) {
    altsLabel.textContent = label;
    altsList.replaceChildren(...options.map(o => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip alt';
      b.textContent = o.label;
      b.title = `${Math.round(o.p * 100)}%`;
      b.addEventListener('click', () => {
        if (replace && lastWord) replaceLastWord(o.label); else addWord(o.label, o.p);
        alts.hidden = true;
        resultBox.replaceChildren(Object.assign(document.createElement('p'), { textContent: o.label }));
      });
      return b;
    }));
    alts.hidden = options.length === 0;
  }

  function addWord(text, p) {
    const cur = message.value.replace(/\s+$/, '');
    const start = cur.length ? cur.length + 1 : 0;
    message.value = cur.length ? `${cur} ${text}` : capitalise(text);
    lastWord = { start, text: message.value.slice(start) };
    confidences.push(p);
    sync();
  }

  function replaceLastWord(text) {
    const v = message.value;
    if (lastWord && v.slice(lastWord.start) === lastWord.text) {
      const t = lastWord.start === 0 ? capitalise(text) : text;
      message.value = v.slice(0, lastWord.start) + t;
      lastWord.text = t;
    } else {
      addWord(text, 0.5);
    }
    sync();
  }

  const capitalise = s => (s === 'I' ? s : s.charAt(0).toUpperCase() + s.slice(1));

  function sync() {
    sendBtn.disabled = !message.value.trim();
    undoBtn.disabled = !message.value.trim();
  }

  undoBtn.addEventListener('click', () => {
    const words = message.value.trim().split(/\s+/);
    words.pop();
    message.value = words.join(' ');
    lastWord = null;
    confidences.pop();
    alts.hidden = true;
    sync();
  });
  clearBtn.addEventListener('click', () => {
    message.value = ''; lastWord = null; confidences.length = 0; alts.hidden = true; sync();
  });
  message.addEventListener('input', () => { lastWord = null; sync(); });

  sendBtn.addEventListener('click', async () => {
    const text = message.value.trim();
    if (!text) return;
    sendBtn.disabled = true;
    const avg = confidences.length ? confidences.reduce((a, b) => a + b, 0) / confidences.length : null;
    const sent = await SignBridge.sendMessage(text, 'sign', avg ? Math.round(avg * 100) / 100 : null);
    if (sent) {
      SignBridge.toast('Sent to your doctor.');
      message.value = ''; lastWord = null; confidences.length = 0; alts.hidden = true;
      resultBox.replaceChildren(Object.assign(document.createElement('p'), { className: 'empty', textContent: 'Sent. Sign your next message any time.' }));
    }
    sync();
  });

  nowBtn.addEventListener('click', () => {
    const frames = recent.slice(-60);
    if (!model || frames.filter(f => model.hasHand(f)).length < MIN_FRAMES) {
      SignBridge.toast('No sign seen yet. Make sure your hands are in view.');
      return;
    }
    segment = [];
    classify(frames);
  });
  // ---------- camera check ----------
  const check = $('camCheck');
  function updateCheck(res, frame) {
    check.hidden = false;
    const set = (k, ok) => check.querySelector(`[data-check="${k}"]`).classList.toggle('is-ok', !!ok);
    set('body', res.poseLandmarks); const [wl, wr] = model.wrists || [7, 28];
    set('left', frame[wl]); set('right', frame[wr]);
    fpsN++;
    const now = performance.now();
    if (now - fpsT > 1000) {
      const fps = Math.round(fpsN * 1000 / (now - fpsT));
      check.querySelector('[data-fps]').textContent = `${fps} fps tracked${cameraFps ? ` · camera ${cameraFps} fps` : ''}${fps < 10 ? ' · slow device: sign a little slower' : ''}`;
      fpsT = now; fpsN = 0;
      // A phone that cannot keep up gets the lighter tracker, so the camera never lags behind the signer.
      slowSeconds = fps < SLOW_FPS ? slowSeconds + 1 : 0;
      if (tracker === 1 && slowSeconds >= SLOW_FOR) {
        tracker = 0; slowSeconds = 0;
        holistic.setOptions({ modelComplexity: 0 });
      }
    }
    const ls = res.poseLandmarks?.[11], rs = res.poseLandmarks?.[12];
    if (ls && rs && Math.abs(ls.x - rs.x) > 0.6) say('Move back a little so your hands stay in view');
  }

  // ---------- taught signs ----------
  const CACHE_KEY = `sb-taught-${lang}`;
  let taught = [];   // [{id, label, frames, by, mine}]
  async function loadTaught() {
    try {
      const res = await fetch(`/api/custom-signs?lang=${lang}`, { headers: { Accept: 'application/json' } });
      if (!res.ok || !(res.headers.get('content-type') || '').includes('json')) throw new Error('offline or signed out');
      taught = (await res.json()).signs;
      try { localStorage.setItem(CACHE_KEY, JSON.stringify(taught)); } catch (e) {}
    } catch (e) {
      try { taught = JSON.parse(localStorage.getItem(CACHE_KEY) || '[]'); } catch (e2) { taught = []; }
    }
    const usable = taught.filter(t => t.frames?.[0]?.length === (model.P || 49));   // examples recorded for this model
    const embs = [];
    for (const t of usable) embs.push({ label: t.label, emb: await model.embed(t.frames) });
    model.setCustom(embs);
    renderTaught();
  }

  function renderTaught() {
    const list = $('taughtList');
    const groups = {};
    taught.forEach(t => (groups[t.label] ||= []).push(t));
    const labels = Object.keys(groups).sort();
    $('taughtCount').textContent = `${labels.length} taught`;
    list.replaceChildren(...labels.map(label => {
      const li = document.createElement('li');
      const div = document.createElement('div');
      div.className = 'grow';
      const ex = groups[label];
      const p = document.createElement('p'); p.textContent = label;
      const small = document.createElement('small');
      small.textContent = `${ex.length} example${ex.length === 1 ? '' : 's'} · by ${[...new Set(ex.map(e => e.by))].join(', ')}`;
      div.append(p, small);
      const more = document.createElement('button');
      more.type = 'button'; more.className = 'btn btn--quiet btn--sm'; more.textContent = 'Add examples';
      more.addEventListener('click', () => startTeach(label));
      li.append(div, more);
      if (ex.some(e => e.mine)) {
        const del = document.createElement('button');
        del.type = 'button'; del.className = 'btn btn--quiet btn--icon'; del.setAttribute('aria-label', `Delete ${label}`);
        del.innerHTML = '<svg class="icon"><use href="#i-trash"/></svg>';
        del.addEventListener('click', async () => {
          if (!confirm(`Delete your examples of "${label}"?`)) return;
          for (const e of ex.filter(e => e.mine)) await fetch(`/api/custom-signs/${e.id}`, { method: 'DELETE' });
          loadTaught();
        });
        li.append(del);
      }
      return li;
    }));
  }

  function startTeach(label) {
    label = (label || $('teachWord').value).trim().toLowerCase();
    if (!label) { SignBridge.toast('Type the word you want to teach first.'); $('teachWord').focus(); return; }
    if (!running) { SignBridge.toast('Start the camera first.'); return; }
    teach = { label, done: 0 };
    $('teachWord').value = label;
    $('teachProgress').hidden = false;
    updateTeach(`Sign "${label}" now, then lower your hands or pause.`);
  }

  function updateTeach(text) {
    $('teachDots').replaceChildren(...Array.from({ length: TEACH_COUNT }, (_, i) =>
      Object.assign(document.createElement('span'), { className: i < teach.done ? 'is-done' : '' })));
    $('teachText').textContent = text;
  }

  async function saveExample(frames) {
    const label = teach.label;
    const res = await fetch('/api/custom-signs', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, lang, frames: frames.slice(-150) }),
    }).catch(() => null);
    if (!res || !res.ok) {
      const err = res ? (await res.json().catch(() => ({}))).error : 'You are offline. Teaching needs a connection to the clinic server.';
      SignBridge.toast(err || 'Could not save that example.');
      return;
    }
    if (!teach) return;
    teach.done++;
    if (teach.done >= TEACH_COUNT) {
      teach = null;
      $('teachProgress').hidden = true;
      SignBridge.toast(`Learned "${label}". Try signing it now.`);
      loadTaught();
    } else {
      updateTeach(`Got ${teach.done} of ${TEACH_COUNT}. Sign "${label}" again.`);
    }
  }

  $('teachStart').addEventListener('click', () => startTeach());
  $('teachWord').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); startTeach(); } });
  $('teachCancel').addEventListener('click', () => { teach = null; $('teachProgress').hidden = true; loadTaught(); });

  // ---------- show me how + practice ----------
  const learnCard = $('learnCard'), learnAvatar = $('learnAvatar');
  function showHow(name, label) {
    learnCard.hidden = false;
    $('learnWord').textContent = `“${label}”`;
    practice = { name, label };
    const note = $('practiceNote');
    note.textContent = 'Now sign it to the camera. SignBridge will tell you if it recognised it.';
    note.className = 'practice';
    const play = () => learnAvatar.playSign(name, label);
    if (learnAvatar.ready) play(); else learnAvatar.addEventListener('avatar:ready', play, { once: true });
    learnCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
  $('learnClose').addEventListener('click', () => { learnCard.hidden = true; practice = null; });

  function showPractice(top) {
    const note = $('practiceNote');
    const rank = top.findIndex(t => t.name === practice.name);
    if (rank === 0) { note.textContent = `✓ Recognised "${practice.label}". Well done.`; note.className = 'practice practice--ok'; }
    else if (rank > 0) { note.textContent = `Close: "${practice.label}" was guess #${rank + 1}. It looked more like "${top[0].label}". Watch again and try once more.`; note.className = 'practice practice--near'; }
    else { note.textContent = `Not recognised yet. It looked like "${top[0].label}". Watch the avatar again and copy the hand shape and movement.`; note.className = 'practice practice--no'; }
  }

  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);
  window.addEventListener('pagehide', stop);

  // ---------- drawing ----------
  const HAND_LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  const ARM_LINKS = [[11,12],[11,13],[13,15],[12,14],[14,16]];

  function draw(res) {
    const img = res.image;
    const iw = img?.width || video.videoWidth, ih = img?.height || video.videoHeight;
    if (canvas.width !== iw || canvas.height !== ih) { canvas.width = iw; canvas.height = ih; }
    const W = canvas.width, H = canvas.height;
    if (img) {
      // Draw the exact frame the landmarks came from (over the live video, which stays the
      // source), so the lines never trail behind the hands.
      ctx.drawImage(img, 0, 0, W, H);
    } else ctx.clearRect(0, 0, W, H);
    ctx.lineCap = 'round';
    if (res.poseLandmarks) {
      ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 3;
      for (const [a, b] of ARM_LINKS) line(res.poseLandmarks[a], res.poseLandmarks[b], W, H);
    }
    for (const [hand, color] of [[res.leftHandLandmarks, '#3db5a2'], [res.rightHandLandmarks, '#f08a62']]) {
      if (!hand) continue;
      ctx.strokeStyle = color; ctx.lineWidth = 3;
      for (const [a, b] of HAND_LINKS) line(hand[a], hand[b], W, H);
      ctx.fillStyle = '#fff';
      for (const p of hand) { ctx.beginPath(); ctx.arc(p.x * W, p.y * H, 3, 0, Math.PI * 2); ctx.fill(); }
    }
  }
  function line(a, b, W, H) { if (!a || !b) return; ctx.beginPath(); ctx.moveTo(a.x * W, a.y * H); ctx.lineTo(b.x * W, b.y * H); ctx.stroke(); }

  // ---------- index of known signs ----------
  function buildIndex(m) {
    const host = $('signIndex'), search = $('signSearch');
    $('signCount').textContent = `· ${m.classes.length} words`;
    const groups = {};
    m.classes.forEach((c, i) => { (groups[m.categories[i]] ||= []).push([m.labels[i], c]); });
    const render = q => {
      host.replaceChildren(...Object.keys(groups).sort().map(cat => {
        const words = groups[cat].filter(([w]) => w.toLowerCase().includes(q)).sort((a, b) => a[0].localeCompare(b[0]));
        if (!words.length) return document.createComment('');
        const sec = document.createElement('div');
        sec.className = 'sign-index__group';
        const h = document.createElement('p');
        h.className = 'section-title';
        h.textContent = cat.replace(/_/g, ' ');
        const list = document.createElement('div');
        list.className = 'alts__list';
        list.append(...words.map(([w, name]) => {
          const b = Object.assign(document.createElement('button'), { type: 'button', className: 'chip alt', textContent: w });
          b.addEventListener('click', () => showHow(name, w));
          return b;
        }));
        sec.append(h, list);
        return sec;
      }));
    };
    render('');
    search.addEventListener('input', () => render(search.value.trim().toLowerCase()));
  }
})();
