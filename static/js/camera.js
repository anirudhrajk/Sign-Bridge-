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
  const SURE = 0.55;          // add the word automatically at or above this confidence
  const MAYBE = 0.2;          // below this, ask the patient to sign again
  const MIN_FRAMES = 8;       // shortest sign we try to read
  const MAX_FRAMES = 150;     // read anyway if a sign runs this long
  const END_GAP = 6;          // frames without hands that end a sign
  const STILL_FRAMES = 12;    // frames of stillness that end a sign
  const STILL_MOTION = 0.012; // wrist movement (in shoulder widths) that counts as still

  let model = null, holistic = null, stream = null, running = false, busy = false;
  let recent = [];            // last ~3 s of frames, for "Read sign now"
  let segment = [], gap = 0, still = 0, moved = 0, lastWrists = null;
  let lastWord = null;        // {text, start} of the word we last added, so it can be replaced
  const confidences = [];

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
    if (running) nowBtn.disabled = false;
  }).catch(() => {
    modelStatus.textContent = 'Signs unavailable';
    SignBridge.toast('Could not load the sign model. Typing still works.');
  });

  // ---------- camera ----------
  async function start() {
    startBtn.disabled = true;
    status.textContent = 'Starting camera…';
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720, facingMode: 'user' }, audio: false });
    } catch (e) {
      status.textContent = 'Camera blocked';
      SignBridge.toast('Allow camera access in your browser to sign.');
      startBtn.disabled = false;
      return;
    }
    video.srcObject = stream;
    await video.play();
    if (!holistic) {
      status.textContent = 'Loading hand tracking (first time takes a few seconds)…';
      holistic = new Holistic({ locateFile: f => `/static/vendor/holistic/${f}` });
      holistic.setOptions({ modelComplexity: 0, smoothLandmarks: true, refineFaceLandmarks: false,
                            minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
      holistic.onResults(onResults);
    }
    running = true;
    empty.hidden = true;
    stopBtn.disabled = false;
    nowBtn.disabled = !model;
    loop();
  }

  async function loop() {
    if (!running) return;
    if (!busy && video.readyState >= 2) {
      busy = true;
      try { await holistic.send({ image: video }); } catch (e) { console.error(e); }
      busy = false;
    }
    requestAnimationFrame(loop);
  }

  function stop() {
    running = false;
    stream?.getTracks().forEach(t => t.stop());
    stream = null;
    video.srcObject = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    status.textContent = 'Camera off';
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
    const frame = SignModel.holisticToPoints(res);
    recent.push(frame);
    if (recent.length > 90) recent.shift();
    if (!model) { status.textContent = 'Loading signs…'; return; }

    const hands = SignModel.hasHand(frame);
    if (!res.poseLandmarks) status.textContent = 'Move back so your shoulders are in view';

    if (hands) {
      const motion = wristMotion(frame);
      segment.push(frame);
      gap = 0;
      if (motion < STILL_MOTION) still++; else { still = 0; moved += motion; }
      if (res.poseLandmarks) status.textContent = 'Reading your sign…';
      meter.hidden = false;
      meter.firstElementChild.style.width = `${Math.min(100, (segment.length / 45) * 100)}%`;
      // A held pause after real movement ends the sign.
      if (still >= STILL_FRAMES && moved > 0.15 && segment.length >= MIN_FRAMES) finish();
      else if (segment.length >= MAX_FRAMES) finish();
    } else if (segment.length) {
      gap++;
      if (gap >= END_GAP) finish();
    } else if (res.poseLandmarks) {
      status.textContent = 'Ready: raise your hands to sign';
      meter.hidden = true;
    }
  }

  function wristMotion(frame) {
    const ls = frame[1], rs = frame[2];
    const scale = ls && rs ? Math.hypot(ls[0] - rs[0], ls[1] - rs[1]) || 0.25 : 0.25;
    const w = [frame[7], frame[28]];
    let m = 0;
    if (lastWrists) w.forEach((p, i) => { if (p && lastWrists[i]) m = Math.max(m, Math.hypot(p[0] - lastWrists[i][0], p[1] - lastWrists[i][1]) / scale); });
    lastWrists = w;
    return m;
  }

  function finish() {
    const frames = segment;
    segment = []; gap = 0; still = 0; moved = 0; lastWrists = null;
    meter.hidden = true;
    if (frames.length < MIN_FRAMES) return;
    classify(frames);
  }

  function classify(frames) {
    const t0 = performance.now();
    const top = model.predict(frames, 5);
    const ms = Math.round(performance.now() - t0);
    showResult(top, ms);
  }

  // ---------- results ----------
  function showResult(top, ms) {
    const best = top[0];
    const sure = best.p >= SURE;
    resultBox.replaceChildren();
    const word = document.createElement('p');
    const meta = document.createElement('small');
    meta.className = 'muted';
    if (best.p < MAYBE) {
      word.className = 'empty';
      word.textContent = "Couldn't read that sign. Try again a little slower.";
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
    if (!model || frames.filter(SignModel.hasHand).length < MIN_FRAMES) {
      SignBridge.toast('No sign seen yet. Make sure your hands are in view.');
      return;
    }
    segment = [];
    classify(frames);
  });
  startBtn.addEventListener('click', start);
  stopBtn.addEventListener('click', stop);
  window.addEventListener('pagehide', stop);

  // ---------- drawing ----------
  const HAND_LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
  const ARM_LINKS = [[11,12],[11,13],[13,15],[12,14],[14,16]];

  function draw(res) {
    if (canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
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
    m.classes.forEach((c, i) => { (groups[m.categories[i]] ||= []).push(m.labels[i]); });
    const render = q => {
      host.replaceChildren(...Object.keys(groups).sort().map(cat => {
        const words = groups[cat].filter(w => w.toLowerCase().includes(q)).sort();
        if (!words.length) return document.createComment('');
        const sec = document.createElement('div');
        sec.className = 'sign-index__group';
        const h = document.createElement('p');
        h.className = 'section-title';
        h.textContent = cat.replace(/_/g, ' ');
        const list = document.createElement('div');
        list.className = 'alts__list';
        list.append(...words.map(w => Object.assign(document.createElement('span'), { className: 'chip', textContent: w })));
        sec.append(h, list);
        return sec;
      }));
    };
    render('');
    search.addEventListener('input', () => render(search.value.trim().toLowerCase()));
  }
})();
