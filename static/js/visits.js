// Visit summary as a folder. Tapping the dashboard's "Visit summary" tile opens a folder of visits
// (React Bits Folder: book, glass, #7aa7ff). Tap the folder to open it, tap a page to read that visit's
// chat up close, and tap the same page again to open the full chat. The tile's link still works without JS.
(() => {
  const tile = document.querySelector('[data-visits-open]');
  const source = document.getElementById('visitPapers');
  if (!tile || !source || !window.Folder) return;

  let current = null;

  function close() {
    if (!current) return;
    const { overlay, folder, onKey, onDown } = current;
    current = null;
    document.removeEventListener('keydown', onKey, true);
    document.removeEventListener('pointerdown', onDown, true);
    overlay.classList.remove('is-open');
    const done = () => { folder.destroy(); overlay.remove(); };
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) done(); else setTimeout(done, 380);
    tile.focus({ preventScroll: true });
  }

  function open() {
    const papers = [...source.content.querySelectorAll('.sb-paper')].map(node => node.cloneNode(true));
    const items = papers.map(paper => ({ content: paper, href: paper.dataset.href, title: paper.querySelector('.sb-paper__title')?.textContent }));

    const overlay = document.createElement('div');
    overlay.className = 'sb-visits';
    overlay.setAttribute('role', 'dialog');
    overlay.setAttribute('aria-modal', 'true');
    overlay.setAttribute('aria-labelledby', 'sbVisitsTitle');
    overlay.innerHTML = `
      <button type="button" class="sb-visits__close" aria-label="Close"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M18 6 6 18M6 6l12 12"/></svg></button>
      <header class="sb-visits__head">
        <h2 id="sbVisitsTitle">Your visits</h2>
        <p>Tap the folder to open it. Tap a page to read it.</p>
      </header>
      <div class="sb-visits__desk"></div>`;
    document.body.append(overlay);
    const desk = overlay.querySelector('.sb-visits__desk');

    // Fit the open book (about 450 x 380 units) to the screen.
    const size = Math.max(0.7, Math.min(1.7, (innerWidth - 32) / 450, (innerHeight - 220) / 380));
    let zoomed = -1;
    const folder = Folder.mount(desk, {
      mode: 'book',
      variant: 'glass',
      color: '#7aa7ff',
      label: 'Visits',
      theme: 'dark',
      items: items.length ? items : [{ title: 'No visits yet' }],
      size,
      onItemClick: (item, index) => {
        if (zoomed === index && item.href) { location.href = item.href; return; }
        zoomed = zoomed === index ? -1 : index;
      },
      onOpenChange: value => { if (!value) zoomed = -1; }
    });

    overlay.querySelector('.sb-visits__close').addEventListener('click', close);
    // A tap on the empty backdrop (not the folder or its pages) closes the overlay.
    const onDown = event => {
      if (!event.target.closest('.folder-item')) zoomed = -1;
      if (event.target === overlay || event.target === desk) {
        if (!folder.isOpen()) close();
      }
    };
    const onKey = event => {
      if (event.key === 'Escape') {
        // The folder handles Escape first (closes the page, then itself); close the overlay when it is shut.
        if (zoomed < 0 && !folder.isOpen()) { event.stopPropagation(); close(); }
        zoomed = -1;
      }
      if (event.key === 'Tab') {
        const stops = [...overlay.querySelectorAll('button, [tabindex="0"]')];
        const first = stops[0], last = stops[stops.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    };
    document.addEventListener('keydown', onKey, true);
    document.addEventListener('pointerdown', onDown, true);
    current = { overlay, folder, onKey, onDown };
    requestAnimationFrame(() => overlay.classList.add('is-open'));
    setTimeout(() => overlay.classList.add('is-open'), 50);
    overlay.querySelector('.folder-stage')?.focus({ preventScroll: true });
  }

  tile.addEventListener('click', event => {
    if (event.metaKey || event.ctrlKey || event.shiftKey) return;   // new tab still opens the page
    event.preventDefault();
    open();
  });
})();
