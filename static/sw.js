// SignBridge service worker: makes the app open and work without internet.
// Served at /sw.js by Flask, which fills in VERSION and PRECACHE (see views.py).
const VERSION = '__VERSION__';
const PRECACHE = __PRECACHE__;
const CACHE = `signbridge-${VERSION}`;
const OFFLINE_URL = '/offline';

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    // Fetch one by one so a slow phone isn't swamped by 38 MB of parallel requests.
    for (const url of PRECACHE) {
      try {
        const res = await fetch(url, { cache: 'reload' });
        if (res.ok) await cache.put(url, res);
      } catch (e) { /* cached on first use instead */ }
    }
    self.skipWaiting();
  })());
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key.startsWith('signbridge-') && key !== CACHE) await caches.delete(key);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', event => {
  const req = event.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin) return;

  // Live data always comes from the server.
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(fetch(req).catch(() => new Response(JSON.stringify({ error: 'You are offline' }),
      { status: 503, headers: { 'Content-Type': 'application/json' } })));
    return;
  }

  // App files, models and the sign library: cache first, then the network.
  if (url.pathname.startsWith('/static/')) {
    event.respondWith((async () => {
      const cache = await caches.open(CACHE);
      const hit = await cache.match(req, { ignoreSearch: true });
      if (hit) return hit;
      const res = await fetch(req);
      if (res.ok) cache.put(req, res.clone());
      return res;
    })());
    return;
  }

  // Pages: always try the network. Only public pages are kept for offline use, so a shared
  // device never shows someone else's conversation.
  if (req.mode === 'navigate') {
    event.respondWith((async () => {
      try {
        return await fetch(req);
      } catch (e) {
        const cache = await caches.open(CACHE);
        return (await cache.match(url.pathname)) || (await cache.match(OFFLINE_URL)) || Response.error();
      }
    })());
  }
});
