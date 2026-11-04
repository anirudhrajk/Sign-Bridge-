// Runs the ISL sign classifier in the browser: no server round trip, works offline.
// Feature extraction mirrors ml/features.py exactly. Change both together.
(() => {
  const POSE_IDS = [0, 11, 12, 13, 14, 15, 16];

  /**
   * Turn one MediaPipe Holistic result into 49 [x, y] points (null when not detected).
   * Each hand is assigned to the arm whose wrist it is attached to, not to the tracker's
   * left/right label: a swapped label drops accuracy from ~98% to ~12%, and labels can differ
   * between cameras and browsers. The body tracker's own left and right arms are consistent.
   */
  function holisticToPoints(res) {
    const pts = new Array(49).fill(null);
    const pose = res.poseLandmarks;
    if (pose) POSE_IDS.forEach((id, i) => { if (pose[id]) pts[i] = [pose[id].x, pose[id].y]; });
    let hands = [res.leftHandLandmarks, res.rightHandLandmarks];
    const lw = pose?.[15], rw = pose?.[16];          // the pose's left and right wrists
    if (lw && rw) {
      const d = (h, w) => (h ? Math.hypot(h[0].x - w.x, h[0].y - w.y) : Infinity);
      const [a, b] = hands;
      if (a && b) {
        if (d(a, rw) + d(b, lw) < d(a, lw) + d(b, rw)) hands = [b, a];
      } else if (a || b) {                            // one hand: put it on the nearer wrist
        const h = a || b;
        hands = d(h, lw) <= d(h, rw) ? [h, null] : [null, h];
      }
    }
    const put = (hand, offset) => { if (hand) hand.forEach((p, i) => { pts[offset + i] = [p.x, p.y]; }); };
    put(hands[0], 7);
    put(hands[1], 28);
    SignModel.lastSwap = hands[0] !== res.leftHandLandmarks && !!(res.leftHandLandmarks || res.rightHandLandmarks);
    return pts;
  }

  const hasHand = f => f[7] !== null || f[28] !== null;

  function frameFeatures(p, out, o) {
    const ls = p[1], rs = p[2];
    let cx = 0.5, cy = 0.5, scale = 0.25;
    if (ls && rs) {
      cx = (ls[0] + rs[0]) / 2; cy = (ls[1] + rs[1]) / 2;
      scale = Math.max(Math.hypot(ls[0] - rs[0], ls[1] - rs[1]), 1e-3);
    }
    for (let i = 0; i < 49; i++) {
      const q = p[i];
      out[o + 2 * i] = q ? (q[0] - cx) / scale : 0;
      out[o + 2 * i + 1] = q ? (q[1] - cy) / scale : 0;
    }
    let k = o + 98;
    const flags = [];
    for (const start of [7, 28]) {
      let present = true;
      for (let i = 0; i < 21; i++) if (!p[start + i]) { present = false; break; }
      if (!present) { for (let i = 0; i < 42; i++) out[k++] = 0; flags.push(0); continue; }
      const w = p[start], m = p[start + 9];
      const size = Math.max(Math.hypot(m[0] - w[0], m[1] - w[1]), 1e-3);
      for (let i = 0; i < 21; i++) {
        out[k++] = (p[start + i][0] - w[0]) / size;
        out[k++] = (p[start + i][1] - w[1]) / size;
      }
      flags.push(1);
    }
    out[k++] = flags[0];
    out[k] = flags[1];
  }

  function trim(frames) {
    let a = 0, b = frames.length - 1;
    while (a <= b && !hasHand(frames[a])) a++;
    while (b >= a && !hasHand(frames[b])) b--;
    return a > b ? frames : frames.slice(a, b + 1);
  }

  class SignModel {
    /** Loads whichever model the folder holds: v2 (model.onnx) or v1 (weights.bin). */
    static async load(base) {
      const manifest = await fetch(`${base}/model.json`).then(r => { if (!r.ok) throw new Error('model.json'); return r.json(); });
      if (manifest.format === 2) return window.SignModelV2.create(base, manifest);
      const buf = await fetch(`${base}/weights.bin`).then(r => { if (!r.ok) throw new Error('weights.bin'); return r.arrayBuffer(); });
      return new SignModel(manifest, new Float32Array(buf));
    }

    get wrists() { return [7, 28]; }
    get P() { return 49; }
    toPoints(res) { return holisticToPoints(res); }
    hasHand(f) { return hasHand(f); }

    constructor(m, data) {
      this.m = m;
      this.T = m.seq_len; this.F = m.feat_dim;
      this.classes = m.classes;
      this.categories = m.categories;
      this.labels = m.labels || m.classes;
      this.t = {};
      for (const t of m.tensors) {
        const size = t.shape.reduce((a, b) => a * b, 1);
        this.t[t.name] = { data: data.subarray(t.offset, t.offset + size), shape: t.shape };
      }
      this.C = this.t['inp.w'].shape[0];
    }

    /** Run the network. Returns {logp: log-probabilities of known signs, emb: unit-length fingerprint}. */
    run(frames) {
      const { T, F, C } = this;
      const seq = trim(frames);
      const x = new Float32Array(T * F);
      for (let t = 0; t < T; t++) {
        const src = seq.length === 1 ? 0 : Math.round((t * (seq.length - 1)) / (T - 1));
        frameFeatures(seq[src], x, t * F);
      }
      // Optional motion features (ml/features.py add_velocity): change of the 49 positions since
      // the previous frame, zero for a hand that is missing in either frame.
      if (this.m.velocity) {
        for (let t = 1; t < T; t++) {
          const o = t * F, p = (t - 1) * F;
          const lOk = x[o + 182] > 0 && x[p + 182] > 0, rOk = x[o + 183] > 0 && x[p + 183] > 0;
          for (let j = 0; j < 98; j++) {
            const ok = j < 14 || (j < 56 ? lOk : rOk);
            x[o + 184 + j] = ok ? x[o + j] - x[p + j] : 0;
          }
        }
      }
      const mean = this.t.mean.data, std = this.t.std.data;
      for (let t = 0; t < T; t++) for (let f = 0; f < F; f++) x[t * F + f] = (x[t * F + f] - mean[f]) / std[f];

      // Input projection + ReLU -> h stored as (C, T)
      let h = new Float32Array(C * T);
      const W = this.t['inp.w'].data, B = this.t['inp.b'].data;
      for (let c = 0; c < C; c++) {
        for (let t = 0; t < T; t++) {
          let s = B[c];
          const wo = c * F, xo = t * F;
          for (let f = 0; f < F; f++) s += W[wo + f] * x[xo + f];
          h[c * T + t] = s > 0 ? s : 0;
        }
      }
      // Residual dilated conv blocks (BatchNorm already folded into the weights)
      this.m.blocks.forEach((blk, i) => {
        const w = this.t[`b${i}.w`].data, b = this.t[`b${i}.b`].data;
        const k = blk.kernel, d = blk.dilation, pad = d * (k >> 1);
        const out = new Float32Array(C * T);
        for (let co = 0; co < C; co++) {
          for (let t = 0; t < T; t++) {
            let s = b[co];
            for (let ci = 0; ci < C; ci++) {
              const wo = (co * C + ci) * k, ho = ci * T;
              for (let j = 0; j < k; j++) {
                const tt = t + j * d - pad;
                if (tt >= 0 && tt < T) s += w[wo + j] * h[ho + tt];
              }
            }
            out[co * T + t] = h[co * T + t] + (s > 0 ? s : 0);
          }
        }
        h = out;
      });
      // Mean + max pooling over time, then the classifier
      const pooled = new Float32Array(2 * C);
      for (let c = 0; c < C; c++) {
        let sum = 0, max = -Infinity;
        for (let t = 0; t < T; t++) { const v = h[c * T + t]; sum += v; if (v > max) max = v; }
        pooled[c] = sum / T; pooled[C + c] = max;
      }
      const HW = this.t['head.w'].data, HB = this.t['head.b'].data, N = HB.length;
      const logits = new Float32Array(N);
      let hi = -Infinity;
      for (let n = 0; n < N; n++) {
        let s = HB[n];
        for (let j = 0; j < 2 * C; j++) s += HW[n * 2 * C + j] * pooled[j];
        logits[n] = s; if (s > hi) hi = s;
      }
      let z = 0;
      for (let n = 0; n < N; n++) z += Math.exp(logits[n] - hi);
      const lz = hi + Math.log(z);
      const logp = logits.map(v => v - lz);
      let norm = 0;
      for (let j = 0; j < 2 * C; j++) norm += pooled[j] * pooled[j];
      norm = Math.sqrt(norm) || 1;
      return { logp, emb: pooled.map(v => v / norm) };
    }

    /** Fingerprint only (for taught signs). */
    embed(frames) { return this.run(frames).emb; }

    /**
     * Taught signs: [{label, emb}] (several examples per label). They compete with the known
     * signs on one scale: score = 20 x best similarity - 19, calibrated on 30 held-out signs
     * so known words keep ~95.5% accuracy (see docs/teach-a-sign.md).
     */
    setCustom(examples) { this.custom = examples || []; }

    /** frames: array of 49-point frames. Returns [{index, name, label, p, custom?}] sorted by p. */
    predict(frames, top = 5) {
      const { logp, emb } = this.run(frames);
      const scores = Array.from(logp, (v, n) => ({ index: n, s: v, name: this.m.classes[n], label: this.labels[n] }));
      const best = new Map();
      for (const ex of this.custom || []) {
        let sim = 0;
        for (let j = 0; j < emb.length; j++) sim += emb[j] * ex.emb[j];
        if (!best.has(ex.label) || sim > best.get(ex.label)) best.set(ex.label, sim);
      }
      for (const [label, sim] of best) scores.push({ index: -1, s: 20 * sim - 19, name: 'taught:' + label, label, custom: true, sim });
      const hi = Math.max(...scores.map(x => x.s));
      let z = 0;
      for (const x of scores) z += Math.exp(x.s - hi);
      for (const x of scores) x.p = Math.exp(x.s - hi) / z;
      return scores.sort((a, b) => b.p - a.p).slice(0, top).map(({ s, ...r }) => r);
    }
  }

  window.SignModel = SignModel;
  window.SignModel.holisticToPoints = holisticToPoints;
  window.SignModel.hasHand = hasHand;
})();
