/* SpotlightCard from React Bits (reactbits.dev/components/spotlight-card), JS-CSS variant.
   Themes, the light falloff and the whole pointer effect are copied unchanged.
   mountSpotlightCard(root, props) takes the component's props and replaces its React shell.
   SignBridge glue at the end: the team pop-up on the landing page (window.SBProfile). */
(() => {
'use strict';

const THEMES = {
  dark: {
    surface: '#111111',
    border: 'rgba(255, 255, 255, 0.08)',
    shadow: '0 24px 48px -24px rgba(0, 0, 0, 0.6)',
    light: '#ffffff',
    fill: 1,
    edge: 1
  },
  light: {
    surface: '#ffffff',
    border: 'rgba(24, 24, 27, 0.1)',
    shadow: '0 1px 2px rgba(24, 24, 27, 0.04), 0 18px 40px -20px rgba(24, 24, 27, 0.16)',
    light: '#18181b',
    fill: 0.28,
    edge: 0.7
  }
};

const SHAPES = ['circle', 'beam'];

const NOISE =
  "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")";

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const falloff = (color, softness) => {
  const soft = clamp(softness, 0, 1);
  const core = (1 - soft) ** 1.5 * 0.85;
  const stop = (alpha, at) =>
    `color-mix(in srgb, ${color} ${(alpha * 100).toFixed(2)}%, transparent) ${(at * 100).toFixed(2)}%`;
  const list = [stop(1, 0)];
  for (let i = 0; i <= 14; i++) {
    const t = core + ((1 - core) * i) / 14;
    const g = (t - core) / (1 - core);
    const smooth = 1 - g * g * (3 - 2 * g);
    const glow = Math.exp(-4.6 * g * g) * (1 - g ** 6);
    list.push(stop(smooth + (glow - smooth) * soft, t));
  }
  return list.join(', ');
};

function mountSpotlightCard(root, props = {}) {
  const {
    spotlightColor,
    intensity = 0.15,
    spotlightSize = 240,
    softness = 0.7,
    shape = 'circle',
    borderGlow = 0.6,
    proximity = 80,
    smoothing = 0.3,
    ambient = false,
    flare = true,
    grain = 0,
    theme = 'dark'
  } = props;

  const palette = THEMES[theme] ?? THEMES.dark;
  const color = spotlightColor ?? palette.light;
  const form = SHAPES.includes(shape) ? shape : 'circle';
  const settings = {
    shape: form,
    fill: falloff(color, softness),
    edge: falloff(color, 1),
    size: Math.max(20, spotlightSize),
    intensity: clamp(intensity, 0, 1) * palette.fill,
    borderGlow: clamp(borderGlow, 0, 1) * palette.edge,
    proximity: Math.max(0, proximity),
    smoothing: clamp(smoothing, 0, 1),
    ambient,
    flare,
    grain: clamp(grain, 0, 1)
  };
  const settingsRef = { current: settings };
  const wakeRef = { current: null };

  root.classList.add('spotlight-card');
  root.style.setProperty('--spotlight-card-surface', palette.surface);
  root.style.setProperty('--spotlight-card-border', palette.border);
  root.style.setProperty('--spotlight-card-shadow', palette.shadow);
  const span = cls => {
    const el = document.createElement('span');
    el.className = cls;
    el.setAttribute('aria-hidden', 'true');
    return el;
  };
  const lightEl = span('spotlight-card__light');
  const grainEl = span('spotlight-card__grain');
  grainEl.style.backgroundImage = NOISE;
  const borderEl = span('spotlight-card__border');
  const edgeEl = span('spotlight-card__edge');
  root.prepend(lightEl, grainEl);
  root.append(borderEl, edgeEl);
  const rootRef = { current: root };
  const lightRef = { current: lightEl };
  const grainRef = { current: grainEl };
  const edgeRef = { current: edgeEl };

  const cleanup = (() => {
    const root = rootRef.current;
    const light = lightRef.current;
    const noise = grainRef.current;
    const edge = edgeRef.current;
    if (!root || !light || !noise || !edge) return undefined;

    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;
    const state = { x: 0, y: 0, presence: 0, flare: 0, time: Math.random() * 60, pressed: false, visible: true };
    const pointer = { x: 0, y: 0, active: false };
    const drawn = { light: '', edge: '', noise: '' };
    let focus = null;
    let raf = 0;
    let last = 0;
    let alive = true;

    const gradient = (s, size, x, y, stops) => {
      if (s.shape === 'beam') {
        const reach = Math.max(size * 0.6, y + size * 0.55);
        return `radial-gradient(${(size * 0.55).toFixed(1)}px ${reach.toFixed(1)}px at ${x.toFixed(1)}px 0px, ${stops})`;
      }
      return `radial-gradient(circle ${size.toFixed(1)}px at ${x.toFixed(1)}px ${y.toFixed(1)}px, ${stops})`;
    };

    const paint = s => {
      const size = s.size * (1 + 0.2 * state.flare);
      const boost = 1 + 0.7 * state.flare;
      const fill = gradient(s, size, state.x, state.y, s.fill);
      const rim = gradient(s, size * 0.6, state.x, s.shape === 'beam' ? 0 : state.y, s.edge);
      if (fill !== drawn.light) {
        drawn.light = fill;
        light.style.background = fill;
      }
      if (rim !== drawn.edge) {
        drawn.edge = rim;
        edge.style.background = rim;
      }
      light.style.opacity = String(Math.min(1, state.presence * s.intensity * boost));
      edge.style.opacity = String(Math.min(1, state.presence * s.borderGlow * boost));
      const grainy = s.grain > 0 ? gradient(s, size, state.x, state.y, '#000 0%, transparent 100%') : 'none';
      if (grainy !== drawn.noise) {
        drawn.noise = grainy;
        noise.style.webkitMaskImage = grainy;
        noise.style.maskImage = grainy;
      }
      noise.style.opacity = String(state.presence * s.grain);
    };

    const tick = now => {
      raf = 0;
      if (!alive) return;
      const s = settingsRef.current;
      const dt = Math.min(0.05, Math.max(0.001, (now - last) / 1000));
      last = now;
      const rect = root.getBoundingClientRect();
      let tx = state.x;
      let ty = state.y;
      let target = 0;
      if (focus && root.contains(focus)) {
        const box = focus.getBoundingClientRect();
        tx = box.left + box.width / 2 - rect.left;
        ty = box.top + box.height / 2 - rect.top;
        target = 1;
      } else if (pointer.active) {
        tx = pointer.x - rect.left;
        ty = pointer.y - rect.top;
        const gap = Math.hypot(Math.max(0, -tx, tx - rect.width), Math.max(0, -ty, ty - rect.height));
        if (gap === 0) target = 1;
        else if (s.proximity > 0) target = Math.max(0, 1 - gap / s.proximity) ** 2;
      }
      if (target < 0.6 && s.ambient && state.visible) {
        if (!reduce) state.time += dt;
        const drift = 0.6 - target;
        tx += (rect.width * (0.5 + 0.34 * Math.sin(state.time * 0.43)) - tx) * (drift / 0.6);
        ty += (rect.height * (0.5 + 0.3 * Math.sin(state.time * 0.61 + 1.3)) - ty) * (drift / 0.6);
        target = 0.6;
      }
      if (state.presence < 0.02 && target > 0 && !s.ambient) {
        state.x = tx;
        state.y = ty;
      }
      const follow = reduce || s.smoothing === 0 ? 1 : 1 - Math.exp(-dt / (0.015 + s.smoothing * 0.22));
      state.x += (tx - state.x) * follow;
      state.y += (ty - state.y) * follow;
      const fade = reduce ? 1 : 1 - Math.exp(-dt / (target > state.presence ? 0.1 : 0.32));
      state.presence += (target - state.presence) * fade;
      if (Math.abs(target - state.presence) < 0.002) state.presence = target;
      const press = state.pressed && s.flare && !reduce ? 1 : 0;
      state.flare += (press - state.flare) * (1 - Math.exp(-dt / (press ? 0.05 : 0.28)));
      if (Math.abs(press - state.flare) < 0.002) state.flare = press;
      paint(s);
      const moving = Math.abs(tx - state.x) > 0.1 || Math.abs(ty - state.y) > 0.1;
      const settled = !moving && state.presence === target && state.flare === press;
      if (!settled || (s.ambient && state.visible && !reduce)) raf = requestAnimationFrame(tick);
    };

    const wake = () => {
      if (raf || !alive) return;
      last = performance.now();
      raf = requestAnimationFrame(tick);
    };
    wakeRef.current = wake;

    const onMove = event => {
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      pointer.active = true;
      if (state.presence > 0 || settingsRef.current.proximity > 0 || root.contains(event.target)) wake();
    };
    const onOut = event => {
      if (event.relatedTarget) return;
      pointer.active = false;
      wake();
    };
    const onDown = event => {
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      pointer.active = true;
      state.pressed = true;
      wake();
    };
    const onUp = event => {
      if (event.pointerType !== 'mouse') pointer.active = false;
      if (!state.pressed && event.pointerType === 'mouse') return;
      state.pressed = false;
      wake();
    };
    const onFocusIn = event => {
      if (!(event.target instanceof Element) || !event.target.matches(':focus-visible')) return;
      focus = event.target;
      wake();
    };
    const onFocusOut = event => {
      if (root.contains(event.relatedTarget)) return;
      focus = null;
      wake();
    };
    const onScroll = () => {
      if (pointer.active || state.presence > 0) wake();
    };

    const observer = new IntersectionObserver(entries => {
      state.visible = entries.some(entry => entry.isIntersecting);
      if (state.visible) wake();
    });
    observer.observe(root);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
    window.addEventListener('scroll', onScroll, { capture: true, passive: true });
    document.addEventListener('pointerout', onOut);
    root.addEventListener('pointerdown', onDown);
    root.addEventListener('focusin', onFocusIn);
    root.addEventListener('focusout', onFocusOut);
    wake();

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      observer.disconnect();
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      window.removeEventListener('scroll', onScroll, { capture: true });
      document.removeEventListener('pointerout', onOut);
      root.removeEventListener('pointerdown', onDown);
      root.removeEventListener('focusin', onFocusIn);
      root.removeEventListener('focusout', onFocusOut);
      wakeRef.current = null;
    };
  })();

  return {
    wake: () => wakeRef.current?.(),
    destroy() {
      cleanup?.();
      lightEl.remove(); grainEl.remove(); borderEl.remove(); edgeEl.remove();
    }
  };
}

window.SpotlightCard = { mount: mountSpotlightCard };

// ---------- SignBridge: the team pop-up on the landing page ----------
const SVG = (size, body, fill = 'none') => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="${fill}" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
const SHARE_ICON = SVG(20, '<path d="M12 15V3m0 0-4 4m4-4 4 4"/><path d="M8 10H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-7a2 2 0 0 0-2-2h-2"/>');
const SAVE_ICON = SVG(20, '<path d="M6 4h12v17l-6-4-6 4z"/>');
const STAR_ICON = SVG(15, '<path d="m12 3 2.8 5.7 6.2.9-4.5 4.4 1.1 6.2L12 17.3 6.4 20.2l1.1-6.2L3 9.6l6.2-.9z"/>', 'currentColor');
const flash = (button, text) => {
  const tip = document.createElement('span');
  tip.className = 'sb-spot__tip';
  tip.textContent = text;
  button.append(tip);
  setTimeout(() => tip.remove(), 1600);
};
let current = null;

function close() {
  if (!current) return;
  const { overlay, card, opener, onKey } = current;
  current = null;
  document.removeEventListener('keydown', onKey);
  overlay.classList.remove('is-open');
  const done = () => { card.destroy(); overlay.remove(); };
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) done();
  else setTimeout(done, 320);
  opener?.focus({ preventScroll: true });
}

function open(person, opener) {
  close();
  const overlay = document.createElement('div');
  overlay.className = 'sb-spot';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', person.name);

  const shut = document.createElement('button');
  shut.type = 'button';
  shut.className = 'sb-spot__close';
  shut.setAttribute('aria-label', 'Close');
  shut.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg>';
  shut.addEventListener('click', close);

  // The card's content, laid out like the React Bits demo: small photo and share at the top,
  // name, role, two tags, a stats row, then "Get in touch" and a save button.
  const root = document.createElement('div');
  root.className = 'sb-spot__card';
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  };
  const top = el('div', 'sb-spot__top');
  const img = el('img', 'sb-spot__avatar');
  img.src = person.portrait || person.photo;
  img.alt = '';
  img.decoding = 'async';
  const share = el('button', 'sb-spot__icon');
  share.type = 'button';
  share.setAttribute('aria-label', `Share ${person.name}'s card`);
  share.innerHTML = SHARE_ICON;
  share.addEventListener('click', async () => {
    const url = `${location.origin}/#team`;
    try {
      if (navigator.share) await navigator.share({ title: `${person.name} · SignBridge`, text: person.role, url });
      else { await navigator.clipboard.writeText(url); flash(share, 'Link copied'); }
    } catch (e) { /* closed the share sheet */ }
  });
  top.append(img, share);

  const name = el('h3', null, person.name);
  const role = el('p', 'sb-spot__role', person.role);
  const tags = el('div', 'sb-spot__tags');
  (person.tags || []).slice(0, 3).forEach(t => tags.append(el('span', 'sb-spot__tag', t)));

  const stats = el('div', 'sb-spot__stats');
  (person.stats || [{ value: '5.0', label: 'Rating', stars: 5 }]).forEach(s => {
    const cell = el('div', 'sb-spot__stat');
    const value = el('strong');
    if (s.stars) {
      const row = el('span', 'sb-spot__stars');
      row.innerHTML = STAR_ICON.repeat(s.stars);
      value.append(row);
    }
    value.append(s.value);
    cell.append(value, el('small', null, s.label));
    stats.append(cell);
  });

  const actions = el('div', 'sb-spot__actions');
  // Every card has "Get in touch"; it opens an email only once that person's address is filled in.
  const cta = el('a', 'sb-spot__cta', 'Get in touch');
  if (person.email) cta.href = `mailto:${person.email}`;
  else {
    cta.setAttribute('role', 'button');
    cta.tabIndex = 0;
    cta.setAttribute('aria-disabled', 'true');
    cta.title = 'Contact details coming soon';
  }
  const save = el('button', 'sb-spot__icon sb-spot__save');
  save.type = 'button';
  save.innerHTML = SAVE_ICON;
  const KEY = `sb-team-saved-${person.name}`;
  let saved = false;
  try { saved = localStorage.getItem(KEY) === '1'; } catch (e) { /* private window */ }
  const showSaved = () => {
    save.setAttribute('aria-pressed', String(saved));
    save.setAttribute('aria-label', saved ? `Saved ${person.name}` : `Save ${person.name}`);
  };
  showSaved();
  save.addEventListener('click', () => {
    saved = !saved;
    try { localStorage.setItem(KEY, saved ? '1' : '0'); } catch (e) { /* not kept */ }
    showSaved();
  });
  actions.append(cta, save);
  root.append(top, name, role);
  if (tags.childElementCount) root.append(tags);
  root.append(stats, actions);

  overlay.append(shut, root);
  overlay.addEventListener('click', event => { if (event.target === overlay) close(); });
  const onKey = event => {
    if (event.key === 'Escape') close();
    if (event.key === 'Tab') {
      const stops = [...overlay.querySelectorAll('button, a[href], [tabindex="0"]')];
      const first = stops[0], last = stops[stops.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  };
  document.addEventListener('keydown', onKey);
  document.body.append(overlay);
  const card = mountSpotlightCard(root, {
    theme: 'dark',
    spotlightColor: '#ffffff',
    intensity: 0.15,
    spotlightSize: 240,
    softness: 0.7,
    shape: 'circle',
    borderGlow: 0.6,
    proximity: 80,
    smoothing: 0.3,
    grain: 0,
    ambient: false,
    flare: true
  });
  current = { overlay, card, opener, onKey };
  requestAnimationFrame(() => overlay.classList.add('is-open'));
  setTimeout(() => overlay.classList.add('is-open'), 50);
  shut.focus({ preventScroll: true });
}

window.SBProfile = { open, close };
})();
