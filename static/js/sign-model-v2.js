// v2 sign models (127 points, CNN + transformer) run with ONNX Runtime Web, offline.
// The network computes its own features from raw points + a presence mask (ml/train_v2.py),
// so this file only gathers points and resamples time exactly like ml/prepare_v2.py.
(() => {
  const FRAMES_STORED = 48;

  function evenly(n, k) {
    const out = new Array(k);
    for (let i = 0; i < k; i++) out[i] = Math.floor((k === 1 ? 0 : (i * (n - 1)) / (k - 1)) + 0.5);
    return out;
  }

  class SignModelV2 {
    static async create(base, manifest) {
      ort.env.wasm.wasmPaths = '/static/vendor/onnxruntime/';
      ort.env.wasm.numThreads = 1;   // works without cross-origin isolation, on every browser
      const session = await ort.InferenceSession.create(`${base}/model.onnx`, { executionProviders: ['wasm'] });
      return new SignModelV2(manifest, session);
    }

    constructor(m, session) {
      this.m = m; this.session = session;
      this.classes = m.classes; this.categories = m.categories; this.labels = m.labels || m.classes;
      this.T = m.frames; this.P = m.points;
      this.face = [...m.lips, ...m.eye_r, ...m.eye_l, ...m.nose];
      this.custom = [];
      this.format = 2;
      this.wrists = [9, 30];
    }

    /** One MediaPipe Holistic result -> 127 [x, y] points (null = missing). */
    toPoints(res) {
      const pts = new Array(this.P).fill(null);
      const pose = res.poseLandmarks;
      if (pose) this.m.pose.forEach((id, i) => { if (pose[id]) pts[i] = [pose[id].x, pose[id].y]; });
      // Give each hand to the arm it is attached to (labels can be swapped between cameras).
      let hands = [res.leftHandLandmarks, res.rightHandLandmarks];
      const lw = pose?.[15], rw = pose?.[16];
      if (lw && rw) {
        const d = (h, w) => (h ? Math.hypot(h[0].x - w.x, h[0].y - w.y) : Infinity);
        const [a, b] = hands;
        if (a && b) { if (d(a, rw) + d(b, lw) < d(a, lw) + d(b, rw)) hands = [b, a]; }
        else if (a || b) { const h = a || b; hands = d(h, lw) <= d(h, rw) ? [h, null] : [null, h]; }
      }
      if (hands[0]) hands[0].forEach((p, i) => { pts[9 + i] = [p.x, p.y]; });
      if (hands[1]) hands[1].forEach((p, i) => { pts[30 + i] = [p.x, p.y]; });
      const face = res.faceLandmarks;
      if (face) this.face.forEach((id, i) => { if (face[id]) pts[51 + i] = [face[id].x, face[id].y]; });
      return pts;
    }

    hasHand(f) { return !!(f[9] || f[30]); }

    async run(frames) {
      const { T, P } = this;
      let a = 0, b = frames.length - 1;
      while (a <= b && !this.hasHand(frames[a])) a++;
      while (b >= a && !this.hasHand(frames[b])) b--;
      const seq = a > b ? frames : frames.slice(a, b + 1);
      const i48 = evenly(seq.length, FRAMES_STORED), i32 = evenly(FRAMES_STORED, T);
      const x = new Float32Array(T * P * 2), m = new Float32Array(T * P);
      for (let t = 0; t < T; t++) {
        const f = seq[i48[i32[t]]];
        for (let p = 0; p < P; p++) {
          const q = f[p];
          if (!q) continue;
          x[(t * P + p) * 2] = q[0]; x[(t * P + p) * 2 + 1] = q[1]; m[t * P + p] = 1;
        }
      }
      const out = await this.session.run({
        points: new ort.Tensor('float32', x, [1, T, P, 2]),
        mask: new ort.Tensor('float32', m, [1, T, P]),
      });
      const logits = out.logits.data, emb = Float32Array.from(out.embedding.data);
      let hi = -Infinity;
      for (const v of logits) if (v > hi) hi = v;
      let z = 0;
      for (const v of logits) z += Math.exp(v - hi);
      const lz = hi + Math.log(z);
      return { logp: Array.from(logits, v => v - lz), emb };
    }

    async embed(frames) { return (await this.run(frames)).emb; }
    setCustom(examples) { this.custom = examples || []; }

    async predict(frames, top = 5) {
      const { logp, emb } = await this.run(frames);
      const scores = logp.map((s, n) => ({ index: n, s, name: this.classes[n], label: this.labels[n] }));
      const best = new Map();
      for (const ex of this.custom) {
        let sim = 0;
        for (let j = 0; j < emb.length; j++) sim += emb[j] * ex.emb[j];
        if (!best.has(ex.label) || sim > best.get(ex.label)) best.set(ex.label, sim);
      }
      for (const [label, sim] of best) scores.push({ index: -1, s: 20 * sim - 19, name: 'taught:' + label, label, custom: true, sim });
      const hi = Math.max(...scores.map(x => x.s));
      let z = 0;
      for (const x of scores) z += Math.exp(x.s - hi);
      for (const x of scores) x.p = Math.exp(x.s - hi) / z;
      return scores.sort((p, q) => q.p - p.p).slice(0, top).map(({ s, ...r }) => r);
    }
  }

  window.SignModelV2 = SignModelV2;
})();
