/* SignBridge visual refresh: pointer spotlight, scroll reveal and a gentle magnetic pull on
   main buttons. Adds classes and CSS variables only; never touches app behaviour.
   The landing page's own scroll animations (statement, steps, feature deck) are left alone. */
(() => {
  const root = document.documentElement;
  const reduceMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches || root.classList.contains('a11y-motion');
  const finePointer = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
  const inApp = !!document.querySelector('.shell, .public');

  // Spotlight and lift on app cards.
  if (inApp && finePointer) {
    document.querySelectorAll(':is(.content, .public) :is(.card:not(.note), .dict-card, .deaf-card)').forEach((el) => {
      el.classList.add('sb-spotlight', 'sb-hover');
      el.addEventListener('pointermove', (event) => {
        const rect = el.getBoundingClientRect();
        el.style.setProperty('--sb-x', `${event.clientX - rect.left}px`);
        el.style.setProperty('--sb-y', `${event.clientY - rect.top}px`);
      }, { passive: true });
    });
  }

  // Reveal on scroll. Only cards below the first screen wait; everything on screen shows at once.
  const targets = inApp
    ? document.querySelectorAll(':is(.content, .public) :is(.card, .dict-card)')
    : document.querySelectorAll('.team, .faq, .close');
  if (!reduceMotion() && 'IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) { entry.target.classList.add('sb-visible'); observer.unobserve(entry.target); }
      });
    }, { threshold: 0.08, rootMargin: '0px 0px -30px 0px' });
    targets.forEach((el) => {
      if (el.getBoundingClientRect().top < window.innerHeight) return;
      el.classList.add('sb-reveal');
      observer.observe(el);
    });
  }

  // Magnetic pull on main buttons, mouse only.
  if (finePointer) {
    document.querySelectorAll(':is(.shell, .public) .btn--primary, .hero .btn--solid, .close .btn--solid').forEach((button) => {
      button.addEventListener('pointermove', (event) => {
        if (reduceMotion() || button.disabled) return;
        const rect = button.getBoundingClientRect();
        const dx = (event.clientX - rect.left - rect.width / 2) * 0.045;
        const dy = (event.clientY - rect.top - rect.height / 2) * 0.045;
        button.style.translate = `${dx}px ${dy}px`;
      });
      button.addEventListener('pointerleave', () => { button.style.translate = ''; });
    });
  }
})();
