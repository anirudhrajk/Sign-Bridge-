(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const hasGsap = !!(window.gsap && window.ScrollTrigger);

  /* ---------- Team: real entries come from team.js ---------- */
  function buildTeam() {
    const team = window.SB_TEAM || [];
    if (!team.length) return;

    const list = $('#people');
    team.forEach((person, i) => {
      const item = document.createElement('li');
      item.className = 'person';
      const id = `person-tip-${i}`;
      const button = document.createElement('button');
      button.className = 'person__photo';
      button.type = 'button';
      button.setAttribute('aria-expanded', 'false');
      button.setAttribute('aria-controls', id);
      button.setAttribute('aria-label', window.SBProfile ? `Open the profile card of ${person.name}` : `Show the role of ${person.name}`);
      if (window.SBProfile) button.setAttribute('aria-haspopup', 'dialog');
      button.dataset.index = i;
      const img = document.createElement('img');
      img.src = person.photo;
      img.alt = person.name;
      button.append(img);
      const tip = document.createElement('div');
      tip.className = 'person__tip';
      tip.id = id;
      tip.setAttribute('role', 'status');
      const name = document.createElement('strong');
      name.textContent = person.name;
      const role = document.createElement('span');
      role.textContent = person.role;
      tip.append(name, role);
      item.append(tip, button);
      list.append(item);
    });

    const close = except => $$('.person.is-open').forEach(p => {
      if (p === except) return;
      p.classList.remove('is-open');
      $('.person__photo', p).setAttribute('aria-expanded', 'false');
    });
    list.addEventListener('click', event => {
      const button = event.target.closest('.person__photo');
      if (!button) return;
      // Team card pop-up (spotlight-card.js); the small role note below stays as the fallback.
      if (window.SBProfile) { window.SBProfile.open(team[Number(button.dataset.index)], button); return; }
      const person = button.closest('.person');
      close(person);
      const open = person.classList.toggle('is-open');
      button.setAttribute('aria-expanded', String(open));
    });
    document.addEventListener('click', event => { if (!event.target.closest('.person')) close(); });
    document.addEventListener('keydown', event => { if (event.key === 'Escape') close(); });

    $('#team').hidden = false;
    $('[data-team-link]').hidden = false;
  }

  /* ---------- Hero signer: real recorded movements from static/signs/isl ---------- */
  const SENTENCE = [['hello', 'hello'], ['doctor', 'doctor'], ['i', 'I'], ['sick', 'sick']];
  const FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]];
  const TRAIL = 12;
  const COLORS = { hand: '#4fe6e0', body: '#5d716c', joint: '#eef6f4' };

  class Signer {
    constructor(canvas, manifest, data, rig) {
      this.canvas = canvas;
      this.rig = rig;
      this.ctx = rig ? null : canvas.getContext('2d');
      this.m = manifest;
      this.data = data;
      this.N = manifest.frame_size;
      this.clips = new Map(manifest.clips.map(c => [c.name, c]));
      this.idle = Float32Array.from(manifest.idle, v => v / manifest.scale);
      this.pose = this.idle.slice();
      this.trails = [[], []];
      this.dpr = 1;
      this.pausedAt = null;
      this.running = true;
      this.visible = true;
      this.step = -1;
      this.state = { kind: 'wait', until: performance.now() + 500 };
      if (!rig) {
        this.resize();
        new ResizeObserver(() => this.resize()).observe(canvas);
      }
      new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; }).observe(canvas);
      this.tick = this.tick.bind(this);
      requestAnimationFrame(this.tick);
    }

    resize() {
      const r = this.canvas.getBoundingClientRect();
      const dpr = Math.min(devicePixelRatio || 1, 2);
      this.canvas.width = Math.max(1, Math.round(r.width * dpr));
      this.canvas.height = Math.max(1, Math.round(r.height * dpr));
      this.dpr = dpr;
      if (!this.running) this.draw();
    }

    frame(clip, index) {
      const f = Math.max(0, Math.min(clip.frames - 1, index));
      const offset = clip.offset + f * this.N;
      const out = new Float32Array(this.N);
      for (let k = 0; k < this.N; k++) out[k] = this.data[offset + k] / this.m.scale;
      return out;
    }

    mix(a, b, w) {
      const out = new Float32Array(this.N);
      for (let k = 0; k < this.N; k++) out[k] = a[k] + (b[k] - a[k]) * w;
      return out;
    }

    reset() {
      $$('#gloss li').forEach(li => { li.className = ''; });
      $('#saidPatient').classList.remove('is-on');
      $('#saidDoctor').classList.remove('is-on');
    }

    advance(now) {
      this.step += 1;
      const items = $$('#gloss li');
      items.forEach((li, i) => {
        li.className = i < this.step ? 'is-done' : i === this.step ? 'is-on' : '';
      });
      if (this.step >= SENTENCE.length) {
        this.state = { kind: 'ease', from: this.pose.slice(), to: this.idle, start: now, dur: 520, then: { kind: 'finish' } };
        return;
      }
      const clip = this.clips.get(SENTENCE[this.step][0]);
      if (!clip) { this.advance(now); return; }
      this.state = { kind: 'ease', from: this.pose.slice(), to: this.frame(clip, 0), start: now, dur: 300, then: { kind: 'clip', clip } };
    }

    update(now) {
      const s = this.state;
      if (s.kind === 'wait') {
        if (now >= s.until) {
          if (s.then) s.then();
          else { this.reset(); this.step = -1; this.advance(now); }
        }
      } else if (s.kind === 'ease') {
        const t = Math.min(1, Math.max(0, (now - s.start) / s.dur));
        const e = t * t * (3 - 2 * t);
        this.pose = this.mix(s.from, s.to, e);
        if (t >= 1) {
          if (s.then.kind === 'clip') this.state = { kind: 'clip', clip: s.then.clip, start: now };
          else this.finish(now);
        }
      } else if (s.kind === 'clip') {
        const position = (now - s.start) / 1000 * this.m.fps * .85;
        if (position >= s.clip.frames - 1) { this.pose = this.frame(s.clip, s.clip.frames - 1); this.advance(now); return; }
        const i = Math.floor(position);
        this.pose = this.mix(this.frame(s.clip, i), this.frame(s.clip, i + 1), position - i);
      }
    }

    finish(now) {
      $('#saidPatient').classList.add('is-on');
      this.state = {
        kind: 'wait', until: now + 1300,
        then: () => {
          $('#saidDoctor').classList.add('is-on');
          this.state = { kind: 'wait', until: performance.now() + 3600 };
        },
      };
    }

    tick(now) {
      requestAnimationFrame(this.tick);
      if (!this.running || !this.visible) {
        if (this.pausedAt === null) this.pausedAt = now;
        return;
      }
      if (this.pausedAt !== null) {
        // Slide the timeline forward by exactly the time spent paused.
        const gap = now - this.pausedAt;
        if (this.state.start !== undefined) this.state.start += gap;
        if (this.state.until !== undefined) this.state.until += gap;
        this.pausedAt = null;
      }
      this.update(now);
      this.draw(now);
    }

    draw(now = performance.now()) {
      if (this.rig) {
        this.rig.apply(this.pose, now);
        this.rig.render();
        return;
      }
      const { ctx, canvas } = this;
      const W = canvas.width, H = canvas.height, L = this.m.layout, p = this.pose;
      const u = Math.min(W * .3, H * .34), ox = W / 2, oy = H * .44;
      const X = v => ox + v * u, Y = v => oy + v * u;
      const body = p.subarray(0, L.pose);
      const left = p.subarray(L.pose, L.pose + L.lh);
      const right = p.subarray(L.pose + L.lh, L.pose + L.lh + L.rh);
      const face = p.subarray(L.pose + L.lh + L.rh);

      ctx.clearRect(0, 0, W, H);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      const path = (points, close) => {
        ctx.beginPath();
        points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        if (close) ctx.closePath();
        ctx.stroke();
      };
      const B = i => [X(body[i * 2]), Y(body[i * 2 + 1])];

      // Body and face stay quiet so the hands carry the meaning.
      ctx.strokeStyle = COLORS.body;
      ctx.lineWidth = 1.5 * this.dpr;
      path([B(2), B(0), B(1), B(3)]);
      path([B(0), B(6)]);
      path([B(1), B(7)]);
      path([B(2), [X(left[0]), Y(left[1])]]);
      path([B(3), [X(right[0]), Y(right[1])]]);
      const group = (name, close) => {
        const [a, b] = this.m.face_groups[name];
        const pts = [];
        for (let i = a; i < b; i++) pts.push([X(face[i * 2]), Y(face[i * 2 + 1])]);
        path(pts, close);
      };
      const [ovalStart] = this.m.face_groups.oval;
      const chin = [X(face[(ovalStart + 18) * 2]), Y(face[(ovalStart + 18) * 2 + 1])];
      const a = B(0), b = B(1);
      path([chin, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]]);
      group('oval', true);
      ['eye_l', 'eye_r', 'lips_outer'].forEach(g => group(g, true));
      ['brow_l', 'brow_r'].forEach(g => group(g, false));

      [left, right].forEach((hand, h) => {
        const q = i => [X(hand[i * 3]), Y(hand[i * 3 + 1])];
        const trail = this.trails[h];
        trail.push(q(8));
        if (trail.length > TRAIL) trail.shift();
        trail.forEach((pt, i) => {
          if (!i) return;
          ctx.globalAlpha = (i / trail.length) * .5;
          ctx.strokeStyle = COLORS.hand;
          ctx.lineWidth = 2 * this.dpr;
          path([trail[i - 1], pt]);
        });
        ctx.globalAlpha = 1;
        ctx.strokeStyle = COLORS.hand;
        ctx.lineWidth = 2.25 * this.dpr;
        FINGERS.forEach(f => path(f.map(q)));
        ctx.fillStyle = COLORS.joint;
        for (let i = 0; i < 21; i++) {
          const [x, y] = q(i);
          ctx.beginPath();
          ctx.arc(x, y, 2.2 * this.dpr, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }
  }

  async function loadAvatar(onProgress) {
    const canvas = $('#signer');
    if (!window.SB3D) return null;
    if (window.SB_AVATAR) {
      onProgress(1);
      return Uint8Array.from(atob(window.SB_AVATAR), c => c.charCodeAt(0)).buffer;
    }
    const res = await fetch(canvas.dataset.avatar);
    if (!res.ok) throw new Error('avatar');
    const total = Number(res.headers.get('content-length')) || 0;
    if (!res.body || !total) return res.arrayBuffer();
    const reader = res.body.getReader();
    const bytes = new Uint8Array(total);
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes.set(value, got);
      got += value.length;
      onProgress(got / total);
    }
    return bytes.buffer;
  }

  async function loadClips(onProgress) {
    const base = $('#signer').dataset.clips;
    const manifestRes = await fetch(`${base}/clips.json`);
    if (!manifestRes.ok) throw new Error('clips.json');
    const manifest = await manifestRes.json();
    const res = await fetch(`${base}/clips.bin`);
    if (!res.ok) throw new Error('clips.bin');
    const total = Number(res.headers.get('content-length')) || 0;
    if (!res.body || !total) return { manifest, data: new Int16Array(await res.arrayBuffer()) };
    const reader = res.body.getReader();
    const bytes = new Uint8Array(total);
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes.set(value, got);
      got += value.length;
      onProgress(got / total);
    }
    return { manifest, data: new Int16Array(bytes.buffer) };
  }

  async function startSigner(pack, avatar) {
    if (!pack) { $('.signer').hidden = true; return; }
    let rig = null;
    if (avatar) {
      try {
        rig = await window.SB3D.createRig($('#signer'), avatar, pack.manifest, pack.data);
      } catch (e) {
        // No WebGL here: swap in a fresh canvas and fall back to the drawn figure.
        rig = null;
        const fresh = $('#signer').cloneNode();
        $('#signer').replaceWith(fresh);
      }
    }
    const gloss = $('#gloss');
    SENTENCE.forEach(([, word]) => {
      const li = document.createElement('li');
      li.textContent = word;
      gloss.append(li);
    });
    const signer = new Signer($('#signer'), pack.manifest, pack.data, rig);
    const toggle = $('#signerToggle');
    toggle.hidden = false;
    const set = running => {
      signer.running = running;
      toggle.textContent = running ? 'Pause signing' : 'Play signing';
    };
    toggle.addEventListener('click', () => set(!signer.running));
    if (reduce) {
      // Show one still moment of a real sign, with the words already filled in.
      set(false);
      const clip = signer.clips.get('doctor');
      if (clip) signer.pose = signer.frame(clip, Math.floor(clip.frames / 2));
      $$('#gloss li').forEach(li => { li.className = 'is-done'; });
      $('#saidPatient').classList.add('is-on');
      $('#saidDoctor').classList.add('is-on');
      signer.draw();
    }
  }

  /* ---------- Questions: smooth open and close ---------- */
  function buildFaq() {
    $$('#faq details').forEach(item => {
      const panel = $('div', item);
      $('summary', item).addEventListener('click', event => {
        event.preventDefault();
        const opening = !item.open;
        $$('#faq details[open]').forEach(other => {
          if (other === item) return;
          if (hasGsap && !reduce) gsap.to($('div', other), { height: 0, duration: .3, onComplete: () => { other.open = false; gsap.set($('div', other), { clearProps: 'height' }); } });
          else other.open = false;
        });
        if (!hasGsap || reduce) { item.open = opening; window.ScrollTrigger?.refresh(); return; }
        if (opening) {
          item.open = true;
          gsap.from(panel, { height: 0, duration: .4, ease: 'power3.out', clearProps: 'height', onComplete: () => ScrollTrigger.refresh() });
        } else {
          gsap.to(panel, { height: 0, duration: .3, ease: 'power2.in', onComplete: () => { item.open = false; gsap.set(panel, { clearProps: 'height' }); ScrollTrigger.refresh(); } });
        }
      });
    });
  }

  /* ---------- Scroll motion ---------- */
  function splitWords(el) {
    const words = el.textContent.trim().split(/\s+/);
    el.setAttribute('aria-label', el.textContent.trim());
    el.textContent = '';
    words.forEach((w, i) => {
      const s = document.createElement('span');
      s.className = 'word';
      s.setAttribute('aria-hidden', 'true');
      s.textContent = w;
      el.append(s, i < words.length - 1 ? ' ' : '');
    });
    return $$('.word', el);
  }

  function setStep(index) {
    $$('#talkSteps li').forEach((li, i) => li.classList.toggle('is-on', i === index));
    if (!hasGsap) return;
    $$('#talkChat [data-step]').forEach(el => {
      const on = Number(el.dataset.step) <= index;
      gsap.to(el, { autoAlpha: on ? 1 : 0, y: on ? 0 : 16, duration: .4, ease: 'power2.out', overwrite: true });
    });
  }

  function setupMotion() {
    gsap.registerPlugin(ScrollTrigger);
    const words = splitWords($('#statement'));
    gsap.to(words, {
      opacity: 1, stagger: .05, ease: 'none',
      scrollTrigger: { trigger: '.statement', start: 'top 72%', end: 'bottom 62%', scrub: .4 },
    });

    const mm = gsap.matchMedia();
    mm.add('(min-width: 901px)', () => {
      gsap.set('#talkChat [data-step]', { autoAlpha: 0, y: 16 });
      setStep(0);
      ScrollTrigger.create({
        trigger: '.talk__pin', start: 'top top', end: '+=180%', pin: true,
        onUpdate: self => {
          const index = Math.min(2, Math.floor(self.progress * 3));
          if (index !== setupMotion.step) { setupMotion.step = index; setStep(index); }
        },
      });

      const track = $('#deckTrack');
      const tiles = $$('.tile', track);
      const distance = () => Math.max(0, track.scrollWidth - innerWidth);
      const now = $('#deckNow');
      const tilt = self => {
        const shift = gsap.getProperty(track, 'x');
        tiles.forEach(tile => {
          const offset = (tile.offsetLeft + tile.offsetWidth / 2 + shift - innerWidth / 2) / innerWidth;
          gsap.set(tile, { rotateY: gsap.utils.clamp(-20, 20, offset * -30), rotateZ: offset * 1.6, z: -Math.abs(offset) * 90 });
        });
        now.textContent = Math.round((self?.progress || 0) * (tiles.length - 1)) + 1;
      };
      gsap.to(track, {
        x: () => -distance(), ease: 'none',
        scrollTrigger: {
          trigger: '.deck__pin', start: 'top top', end: () => `+=${distance()}`,
          pin: true, scrub: .6, invalidateOnRefresh: true, onUpdate: tilt,
        },
      });
      tilt();
    });

    const links = $$('.dock a');
    links.forEach(link => {
      const section = $(link.getAttribute('href'));
      if (!section || section.hidden) return;
      ScrollTrigger.create({
        trigger: section, start: 'top 55%', end: 'bottom 55%',
        onToggle: self => {
          if (!self.isActive) return;
          links.forEach(l => l.classList.toggle('is-on', l === link));
        },
      });
    });
  }

  function enter() {
    const loader = $('#loader');
    if (!hasGsap || reduce) { loader.remove(); return; }
    const logo = $('#loaderLogo');
    // The metal logo swells a little and fades, then the black loader dissolves into the page.
    gsap.timeline()
      .to(logo, { scale: 1.12, autoAlpha: 0, duration: .75, ease: 'power2.in' })
      .to(loader, { autoAlpha: 0, duration: .8, ease: 'power2.inOut', onComplete: () => { logo?._metallicPaint?.destroy(); loader.remove(); } }, '-=.35')
      .from('.mask > span', { yPercent: 108, duration: 1.05, ease: 'expo.out', stagger: .08 }, '-=.42')
      .from('[data-rise]', { autoAlpha: 0, y: 22, duration: .8, ease: 'power3.out', stagger: .1 }, '-=.75');
  }

  async function start() {
    buildTeam();
    buildFaq();
    if (reduce) $$('#talkChat [data-step]').forEach(el => { el.style.opacity = 1; });

    // The loader shows the SignBridge logo in liquid metal while the signer loads, for at least LOGO_MS.
    const LOGO_MS = 3000;
    const shownAt = performance.now();
    const progress = () => {};

    let pack = null, avatar = null;
    const parts = [0, 0];
    const part = i => p => { parts[i] = p; progress((parts[0] + parts[1]) / 2); };
    try {
      [pack, avatar] = await Promise.race([
        Promise.all([loadClips(part(0)), loadAvatar(part(1)).catch(() => null)]),
        new Promise((_, no) => setTimeout(() => no(new Error('timeout')), 15000)),
      ]);
    } catch (e) { pack = null; }

    if (document.fonts?.ready) await document.fonts.ready;
    if (hasGsap && !reduce) {
      const left = LOGO_MS - (performance.now() - shownAt);
      if (left > 0) await new Promise(done => setTimeout(done, left));
    }
    await startSigner(pack, avatar);
    if (hasGsap && !reduce) setupMotion();
    enter();
    if (hasGsap && !reduce) ScrollTrigger.refresh();
  }

  start();
})();
