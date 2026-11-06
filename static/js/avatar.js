// SignBridge avatar: the 3D signer from the landing page, driven by real recorded sign movements.
// Clips come from ml/build_sign_clips.py (points in shoulder widths from the shoulder centre).
// A device without WebGL gets the drawn 2D signer instead.
(() => {
  const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
  const PALM = [0, 1, 2, 5, 9, 13, 17];
  const TONES = {
    warm:   { skin: '#c98d62', shade: '#a86f48', lip: '#9c5a4a', hair: '#1f1a17' },
    light:  { skin: '#eac3a0', shade: '#c99b78', lip: '#c07a70', hair: '#3b2a20' },
    deep:   { skin: '#8a5a3c', shade: '#6b432b', lip: '#6e3a30', hair: '#141110' },
  };

  class SignAvatar {
    static async load(base) {
      const [m, buf] = await Promise.all([
        fetch(`${base}/clips.json`).then(r => { if (!r.ok) throw new Error('clips.json'); return r.json(); }),
        fetch(`${base}/clips.bin`).then(r => { if (!r.ok) throw new Error('clips.bin'); return r.arrayBuffer(); }),
      ]);
      return { manifest: m, data: new Int16Array(buf) };
    }

    /** The 3D model, fetched once however many signers a page has. */
    static model() {
      return (SignAvatar.glb ||= fetch('/static/landing/avatar/signer.glb').then(r => {
        if (!r.ok) throw new Error('signer.glb');
        return r.arrayBuffer();
      }));
    }

    static async create(canvas, pack, opts = {}) {
      let rig = null;
      try {
        const glb = await SignAvatar.model();
        rig = await window.SB3D.createRig(canvas, glb.slice(0), pack.manifest, pack.data);
      } catch (e) {
        // No WebGL: a canvas that failed to give a 3D context cannot be used for 2D either.
        console.warn('3D signer unavailable, using the drawn signer', e);
        const fresh = canvas.cloneNode();
        canvas.replaceWith(fresh);
        canvas = fresh;
      }
      return new SignAvatar(canvas, pack, opts, rig);
    }

    constructor(canvas, pack, opts = {}, rig = null) {
      this.canvas = canvas;
      this.rig = rig;
      this.ctx = rig ? null : canvas.getContext('2d');
      this.m = pack.manifest;
      this.data = pack.data;
      this.N = this.m.frame_size;
      this.byName = new Map(this.m.clips.map((c, i) => [c.name, i]));
      this.tone = TONES[opts.tone] || TONES.warm;
      this.shirt = opts.shirt || '#0e6b5e';
      this.speed = opts.speed || 0.85;
      this.onWord = opts.onWord || (() => {});
      this.onDone = opts.onDone || (() => {});
      this.idle = Float32Array.from(this.m.idle, v => v / this.m.scale);
      this.pose = this.idle.slice();
      this.queue = [];
      this.seg = null;
      this.blinkAt = performance.now() + 2000;
      this.t0 = performance.now();
      this.lastDraw = 0;
      this.onScreen = true;
      new IntersectionObserver(([e]) => { this.onScreen = e.isIntersecting; }).observe(canvas);
      if (!rig) {
        this.resize();
        new ResizeObserver(() => this.resize()).observe(canvas);
      }
      this.loop = this.loop.bind(this);
      requestAnimationFrame(this.loop);
    }

    resize() {
      const r = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.canvas.width = Math.max(1, Math.round(r.width * dpr));
      this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    }

    frame(ci, f) {
      const c = this.m.clips[ci];
      const i = Math.max(0, Math.min(c.frames - 1, f));
      const o = c.offset + i * this.N, s = this.m.scale;
      const out = new Float32Array(this.N);
      for (let k = 0; k < this.N; k++) out[k] = this.data[o + k] / s;
      return out;
    }

    /** items: [{sign: clipName|null, word: text}] — words without a sign are only captioned. */
    play(items) {
      this.queue = items.map((it, i) => ({ ...it, i }));
      this.seg = null;
      this.next();
    }

    stop() { this.queue = []; this.seg = { kind: 'ease', from: this.pose.slice(), to: this.idle, dur: 350, start: performance.now() }; }

    setTone(name) { this.tone = TONES[name] || this.tone; }

    get playing() { return !!this.seg && this.seg.kind !== 'idle'; }

    next() {
      const now = performance.now();
      const it = this.queue.shift();
      if (!it) {
        this.seg = { kind: 'ease', from: this.pose.slice(), to: this.idle, dur: 450, start: now, done: true };
        return;
      }
      this.onWord(it.i, it);
      const ci = it.sign != null ? this.byName.get(it.sign) : undefined;
      if (ci === undefined) { // caption only: hold briefly so the word can be read
        this.seg = { kind: 'hold', dur: 700 / this.speed, start: now };
        return;
      }
      const first = this.frame(ci, 0);
      this.seg = { kind: 'ease', from: this.pose.slice(), to: first, dur: 260 / this.speed, start: now,
                   then: { kind: 'clip', ci, start: 0 } };
    }

    update(now) {
      const s = this.seg;
      if (!s) return;
      if (s.kind === 'ease') {
        const k = Math.min(1, (now - s.start) / s.dur);
        const e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
        for (let j = 0; j < this.N; j++) this.pose[j] = s.from[j] + (s.to[j] - s.from[j]) * e;
        if (k >= 1) {
          if (s.then) this.seg = { ...s.then, start: now };
          else if (s.done) { this.seg = null; this.onWord(-1); this.onDone(); }
          else this.next();
        }
      } else if (s.kind === 'hold') {
        if (now - s.start >= s.dur) this.next();
      } else if (s.kind === 'clip') {
        const c = this.m.clips[s.ci];
        const pos = ((now - s.start) / 1000) * this.m.fps * this.speed;
        if (pos >= c.frames - 1) { this.pose = this.frame(s.ci, c.frames - 1); this.next(); return; }
        const a = this.frame(s.ci, Math.floor(pos)), b = this.frame(s.ci, Math.floor(pos) + 1), w = pos - Math.floor(pos);
        for (let j = 0; j < this.N; j++) this.pose[j] = a[j] + (b[j] - a[j]) * w;
      }
    }

    loop(now) {
      requestAnimationFrame(this.loop);
      this.update(now);
      if (!this.onScreen) return;                         // keep the camera tracker's time for itself
      if (!this.rig) { this.draw(now); return; }
      if (!this.seg && now - this.lastDraw < 100) return; // waiting for a sign: 10 frames a second is enough
      this.lastDraw = now;
      this.rig.apply(this.pose, now);
      this.rig.render();
    }

    // ---------- drawing ----------
    draw(now) {
      const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
      const L = this.m.layout, p = this.pose;
      // Frame like a video call: head and upper body, hands at rest fall just out of view.
      const u = Math.min(W * 0.34, H * 0.34);           // one shoulder width in pixels
      const breathe = this.seg ? 0 : Math.sin((now - this.t0) / 650) * 0.012;
      const ox = W / 2, oy = H * 0.44;
      const X = x => ox + x * u, Y = y => oy + (y + breathe) * u;
      const pt = (arr, i, d) => [X(arr[i * d]), Y(arr[i * d + 1])];

      const pose = p.subarray(0, L.pose);
      const lh = p.subarray(L.pose, L.pose + L.lh);
      const rh = p.subarray(L.pose + L.lh, L.pose + L.lh + L.rh);
      const face = p.subarray(L.pose + L.lh + L.rh);
      const P = i => pt(pose, i, 2);      // 0 L shoulder,1 R shoulder,2 L elbow,3 R elbow,4 L wrist,5 R wrist,6 L hip,7 R hip
      const F = i => pt(face, i, 2);
      const g = this.m.face_groups;
      const group = name => { const [a, b] = g[name]; const out = []; for (let i = a; i < b; i++) out.push(F(i)); return out; };

      ctx.clearRect(0, 0, W, H);
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      const t = this.tone;

      // torso: rounded shoulders tapering to the waist
      const A = P(0), B = P(1);                       // the two shoulders (either may be on the left)
      const [lft, rgt] = A[0] < B[0] ? [A, B] : [B, A];
      const sw = rgt[0] - lft[0];
      const mx = (lft[0] + rgt[0]) / 2, sy = (lft[1] + rgt[1]) / 2;
      const waistY = sy + 1.55 * u, bottom = Math.max(H + 10, waistY + 0.4 * u);
      const ow = 0.16 * u;                            // shoulder roundness past the joint
      ctx.fillStyle = this.shirt;
      ctx.beginPath();
      ctx.moveTo(mx, sy - 0.1 * u);
      ctx.bezierCurveTo(mx + sw * 0.3, sy - 0.12 * u, rgt[0] - 0.05 * u, rgt[1] - 0.12 * u, rgt[0] + ow * 0.5, rgt[1] - 0.02 * u);
      ctx.quadraticCurveTo(rgt[0] + ow, rgt[1] + 0.12 * u, rgt[0] + ow * 0.6, rgt[1] + 0.45 * u);
      ctx.quadraticCurveTo(mx + sw * 0.4, waistY - 0.3 * u, mx + sw * 0.38, waistY);
      ctx.lineTo(mx + sw * 0.4, bottom); ctx.lineTo(mx - sw * 0.4, bottom); ctx.lineTo(mx - sw * 0.38, waistY);
      ctx.quadraticCurveTo(mx - sw * 0.4, waistY - 0.3 * u, lft[0] - ow * 0.6, lft[1] + 0.45 * u);
      ctx.quadraticCurveTo(lft[0] - ow, lft[1] + 0.12 * u, lft[0] - ow * 0.5, lft[1] - 0.02 * u);
      ctx.bezierCurveTo(lft[0] + 0.05 * u, lft[1] - 0.12 * u, mx - sw * 0.3, sy - 0.12 * u, mx, sy - 0.1 * u);
      ctx.fill();

      // head geometry from the face outline
      const oval = group('oval');
      const xs = oval.map(q => q[0]), ys = oval.map(q => q[1]);
      const fw = Math.max(...xs) - Math.min(...xs), fh = Math.max(...ys) - Math.min(...ys);
      const cx = xs.reduce((a, b) => a + b, 0) / xs.length, cy = ys.reduce((a, b) => a + b, 0) / ys.length;
      const chin = oval[18];

      // neck and collar
      ctx.fillStyle = t.shade;
      ctx.beginPath();
      ctx.moveTo(cx - fw * 0.28, chin[1] - fh * 0.25); ctx.lineTo(cx + fw * 0.28, chin[1] - fh * 0.25);
      ctx.lineTo(mx + fw * 0.34, sy - 0.04 * u); ctx.lineTo(mx - fw * 0.34, sy - 0.04 * u); ctx.closePath(); ctx.fill();
      ctx.fillStyle = shade(this.shirt, -22);
      ctx.beginPath();
      ctx.moveTo(mx - fw * 0.45, sy - 0.08 * u); ctx.quadraticCurveTo(mx, sy + 0.16 * u, mx + fw * 0.45, sy - 0.08 * u);
      ctx.quadraticCurveTo(mx, sy + 0.04 * u, mx - fw * 0.45, sy - 0.08 * u); ctx.fill();

      // back of the head (hair), ears, face
      ctx.fillStyle = t.hair;
      ctx.beginPath(); ctx.ellipse(cx, cy - fh * 0.14, fw * 0.62, fh * 0.64, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = t.shade;
      for (const side of [8, 28]) {
        const e = oval[side];
        ctx.beginPath(); ctx.ellipse(e[0] + (e[0] - cx) * 0.06, e[1] - fh * 0.04, fw * 0.08, fh * 0.13, 0, 0, Math.PI * 2); ctx.fill();
      }
      const faceShape = oval.map(([x, y]) => [cx + (x - cx) * 1.04, cy + (y - cy) * 1.02]);
      ctx.fillStyle = t.skin; poly(ctx, faceShape); ctx.fill();
      const browTop = Math.min(...group('brow_l').concat(group('brow_r')).map(q => q[1]));
      ctx.save(); poly(ctx, faceShape); ctx.clip();
      ctx.fillStyle = t.hair;
      ctx.beginPath();
      ctx.moveTo(cx - fw, browTop - fh * 0.1);
      ctx.bezierCurveTo(cx - fw * 0.3, browTop - fh * 0.26, cx + fw * 0.2, browTop - fh * 0.12, cx + fw, browTop - fh * 0.2);
      ctx.lineTo(cx + fw, browTop - fh); ctx.lineTo(cx - fw, browTop - fh); ctx.fill();
      ctx.restore();

      // face features
      ctx.strokeStyle = t.hair; ctx.lineWidth = 0.035 * u;
      for (const b of ['brow_l', 'brow_r']) { line(ctx, group(b)); ctx.stroke(); }
      if (now > this.blinkAt + 140) this.blinkAt = now + 2200 + Math.random() * 2600;
      const blink = now > this.blinkAt ? 0.12 : 1;
      for (const e of ['eye_l', 'eye_r']) {
        const pts = group(e);
        const ex = pts.reduce((a, q) => a + q[0], 0) / pts.length, ey = pts.reduce((a, q) => a + q[1], 0) / pts.length;
        ctx.fillStyle = '#fbf7f2';
        poly(ctx, pts.map(([x, y]) => [x, ey + (y - ey) * blink * 1.25])); ctx.fill();
        ctx.fillStyle = '#2a211c';
        ctx.beginPath(); ctx.ellipse(ex, ey, 0.032 * u, 0.032 * u * Math.max(0.15, blink), 0, 0, Math.PI * 2); ctx.fill();
      }
      const lo = group('lips_outer'), li = group('lips_inner');
      const eyeMid = [(F(g.eye_l[0])[0] + F(g.eye_r[0])[0]) / 2, (F(g.eye_l[0])[1] + F(g.eye_r[0])[1]) / 2];
      const mouthMid = [lo.reduce((a, q) => a + q[0], 0) / lo.length, lo.reduce((a, q) => a + q[1], 0) / lo.length];
      const nose = [eyeMid[0] + (mouthMid[0] - eyeMid[0]) * 0.62, eyeMid[1] + (mouthMid[1] - eyeMid[1]) * 0.62];
      ctx.strokeStyle = t.shade; ctx.lineWidth = 0.022 * u;
      ctx.beginPath(); ctx.arc(nose[0], nose[1] - 0.02 * u, 0.045 * u, 0.15 * Math.PI, 0.85 * Math.PI); ctx.stroke();
      ctx.fillStyle = t.lip; poly(ctx, lo); ctx.fill();
      ctx.fillStyle = '#4a1f1a'; poly(ctx, li); ctx.fill();

      // arms (sleeves) then hands, nearer hand last
      const arm = (s, e, w, hand) => {
        const wr = [X(hand[0]), Y(hand[1])];
        const S = P(s), E = P(e), cuff = lerp(E, wr, 0.84);
        ctx.strokeStyle = shade(this.shirt, -10); ctx.lineWidth = 0.21 * u;
        ctx.beginPath(); ctx.moveTo(...S); ctx.lineTo(...E); ctx.stroke();
        ctx.lineWidth = 0.18 * u;
        ctx.beginPath(); ctx.moveTo(...E); ctx.lineTo(...cuff); ctx.stroke();
        ctx.strokeStyle = t.skin; ctx.lineWidth = 0.12 * u;
        ctx.beginPath(); ctx.moveTo(...lerp(E, wr, 0.8)); ctx.lineTo(...wr); ctx.stroke();
      };
      arm(0, 2, 4, lh); arm(1, 3, 5, rh);
      this.hand(lh, X, Y, u); this.hand(rh, X, Y, u);
    }

    hand(h, X, Y, u) {
      const ctx = this.ctx, t = this.tone;
      const q = i => [X(h[i * 3]), Y(h[i * 3 + 1])];
      const z = i => h[i * 3 + 2];
      const size = Math.hypot(h[27] - h[0], h[28] - h[1]) || 0.6;
      const k = Math.max(0.6, Math.min(1.4, size / 0.62));
      const order = FINGERS.map((f, i) => ({ f, i, d: (z(f[1]) + z(f[3])) / 2 })).sort((a, b) => b.d - a.d);
      const width = i => (i === 0 ? 0.1 : i === 4 ? 0.075 : 0.088) * u * k;
      ctx.fillStyle = t.skin; ctx.strokeStyle = t.shade; ctx.lineWidth = 0.022 * u * k;
      poly(ctx, PALM.map(q)); ctx.fill(); ctx.stroke();
      for (const { f, i } of order) {
        const pts = [i === 0 ? q(1) : lerp(q(0), q(f[0]), 0.85), ...f.map(q)];
        ctx.strokeStyle = t.shade; ctx.lineWidth = width(i) + 0.024 * u * k; line(ctx, pts); ctx.stroke();
        ctx.strokeStyle = t.skin; ctx.lineWidth = width(i); line(ctx, pts); ctx.stroke();
      }
      ctx.fillStyle = t.skin; poly(ctx, PALM.map(q)); ctx.fill();
    }
  }

  function poly(ctx, pts) { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); }
  function line(ctx, pts) { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); }
  function lerp(a, b, w) { return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w]; }
  function shade(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    const c = v => Math.max(0, Math.min(255, v + amt));
    return `rgb(${c(n >> 16)}, ${c((n >> 8) & 255)}, ${c(n & 255)})`;
  }

  window.SignAvatar = SignAvatar;
  window.SignAvatar.TONES = Object.keys(TONES);
})();
