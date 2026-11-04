// Wires every [data-avatar] block: loads the signer, captions each word, and exposes el.say(text).
(() => {
  const packs = {};
  const getPack = lang => (packs[lang] ||= SignAvatar.load(`/static/signs/${lang}`));
  const pref = (k, d) => { try { return localStorage.getItem(k) || d; } catch (e) { return d; } };
  const save = (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} };

  document.querySelectorAll('[data-avatar]').forEach(el => {
    const canvas = el.querySelector('canvas');
    const caption = el.querySelector('[data-avatar-caption]');
    const loading = el.querySelector('[data-avatar-loading]');
    const replay = el.querySelector('[data-avatar-replay]');
    const speed = el.querySelector('[data-avatar-speed]');
    const tone = el.querySelector('[data-avatar-tone]');
    let avatar = null, planner = null, lastText = '', pending = null, chips = [];

    speed.value = pref('sb-avatar-speed', '0.85');
    tone.value = pref('sb-avatar-tone', 'warm');

    function renderCaption(plan) {
      chips = plan.map(p => {
        const s = document.createElement('span');
        s.className = 'gloss' + (p.sign ? '' : ' gloss--text');
        s.textContent = p.sign ? planner.label.get(p.sign) : p.word;
        if (!p.sign) s.title = 'No sign for this word yet; shown as text';
        return s;
      });
      caption.replaceChildren(...chips);
      if (!plan.length) caption.replaceChildren(Object.assign(document.createElement('span'), { className: 'muted small', textContent: 'Nothing to sign.' }));
    }

    el.say = text => {
      lastText = text;
      if (!avatar) { pending = text; return; }
      const plan = planner.plan(text);
      renderCaption(plan);
      replay.disabled = !plan.length;
      avatar.play(plan);
      return plan;
    };
    el.plan = text => (planner ? planner.plan(text) : []);
    // Play one sign by its name (used by "show me how").
    el.playSign = (name, label) => {
      if (!avatar) { pending = label; return; }
      const plan = [{ word: label, sign: name }];
      lastText = label;
      renderCaption(plan);
      replay.disabled = false;
      avatar.play(plan);
    };

    getPack(el.dataset.lang).then(pack => {
      avatar = new SignAvatar(canvas, pack, {
        tone: tone.value, speed: Number(speed.value),
        onWord: i => chips.forEach((c, j) => c.classList.toggle('is-current', j === i)),
      });
      planner = new SignPlanner(pack.manifest.clips);
      loading.hidden = true;
      el.ready = true;
      el.dispatchEvent(new CustomEvent('avatar:ready'));
      if (pending) { el.say(pending); pending = null; }
    }).catch(() => { loading.textContent = 'Signer unavailable. Messages still appear as text.'; });

    replay.addEventListener('click', () => lastText && el.say(lastText));
    speed.addEventListener('change', () => { save('sb-avatar-speed', speed.value); if (avatar) avatar.speed = Number(speed.value); });
    tone.addEventListener('change', () => { save('sb-avatar-tone', tone.value); avatar?.setTone(tone.value); });
  });
})();
