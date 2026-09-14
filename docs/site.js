const progress = document.querySelector('.read-progress span');
const nav = document.querySelector('.site-nav');
const menuButton = document.querySelector('.menu-toggle');
const siteHeader = document.querySelector('.site-header');
const immersiveHero = document.querySelector('[data-immersive-hero]');
const darkOpening = document.querySelector('#project-film') || immersiveHero;
const navLinks = [...document.querySelectorAll('.site-nav a[href^="#"]')];
const sections = navLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);

function updateProgress() {
  const available = document.documentElement.scrollHeight - window.innerHeight;
  const ratio = available > 0 ? window.scrollY / available : 0;
  progress.style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
}

function updateCurrentSection() {
  const marker = window.scrollY + 140;
  let active = sections[0];
  for (const section of sections) {
    if (section.offsetTop <= marker) active = section;
  }
  for (const link of navLinks) {
    link.toggleAttribute(
      'aria-current',
      active && link.getAttribute('href') === `#${active.id}`,
    );
  }
}

function updateHeaderTone() {
  if (!siteHeader || !immersiveHero) return;
  const threshold = Math.max(
    80,
    darkOpening.offsetTop + darkOpening.offsetHeight - siteHeader.offsetHeight - 24,
  );
  siteHeader.classList.toggle('is-scrolled', window.scrollY >= threshold);
}

function closeMenu() {
  nav.classList.remove('open');
  menuButton.setAttribute('aria-expanded', 'false');
  menuButton.setAttribute('aria-label', 'Open navigation');
}

menuButton.addEventListener('click', () => {
  const open = nav.classList.toggle('open');
  menuButton.setAttribute('aria-expanded', String(open));
  menuButton.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
});

for (const link of navLinks) link.addEventListener('click', closeMenu);

const revealObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    }
  },
  { rootMargin: '0px 0px -8% 0px', threshold: 0.08 },
);

for (const element of document.querySelectorAll('.reveal')) {
  revealObserver.observe(element);
}

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function startVideo(video) {
  // The HTML muted= attribute is not always reflected onto the .muted property,
  // which makes browsers treat autoplay as unmuted and block it. Force it here.
  video.muted = true;
  video.playsInline = true;
  if (video.readyState === HTMLMediaElement.HAVE_NOTHING) video.load();
  video.play().catch(() => {
    // Autoplay was blocked; retry on the first user gesture (allowed by policy).
    const onGesture = () => { video.muted = true; video.play().catch(() => {}); };
    window.addEventListener('pointerdown', onGesture, { once: true });
    window.addEventListener('keydown', onGesture, { once: true });
    window.addEventListener('touchstart', onGesture, { once: true });
  });
}

function initializeProjectFilm() {
  const video = document.querySelector('[data-autoplay-film]');
  const fallback = document.querySelector('[data-film-play]');
  if (!video) return;

  // Keep the project film on a dedicated autoplay path. Some browsers drop an
  // off-screen autoplay request and do not resume it when the element enters
  // the viewport, even though the declarative attributes are present.
  video.defaultMuted = true;
  video.muted = true;
  video.autoplay = true;
  video.loop = true;
  video.playsInline = true;
  video.setAttribute('muted', '');
  video.setAttribute('playsinline', '');

  let visible = false;
  let retryTimer = 0;

  async function ensurePlayback() {
    window.clearTimeout(retryTimer);
    video.muted = true;
    try {
      await video.play();
      if (fallback) fallback.hidden = true;
    } catch (error) {
      if (fallback) fallback.hidden = false;
      retryTimer = window.setTimeout(() => {
        if (visible && !document.hidden) ensurePlayback();
      }, 1200);
    }
  }

  fallback?.addEventListener('click', ensurePlayback);
  video.addEventListener('loadedmetadata', ensurePlayback);
  video.addEventListener('canplay', ensurePlayback);
  video.addEventListener('playing', () => {
    if (fallback) fallback.hidden = true;
  });
  window.addEventListener('pageshow', ensurePlayback);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) ensurePlayback();
  });

  new IntersectionObserver((entries) => {
    visible = entries.some((entry) => entry.isIntersecting);
    if (visible) ensurePlayback();
  }, { rootMargin: '240px 0px', threshold: 0.01 }).observe(video);

  if (video.readyState === HTMLMediaElement.HAVE_NOTHING) video.load();
  ensurePlayback();
}

function pearson(points) {
  if (points.length < 3) return Number.NaN;
  const meanX = points.reduce((sum, point) => sum + point.cka, 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point.sr, 0) / points.length;
  let covariance = 0;
  let varianceX = 0;
  let varianceY = 0;
  for (const point of points) {
    const dx = point.cka - meanX;
    const dy = point.sr - meanY;
    covariance += dx * dy;
    varianceX += dx * dx;
    varianceY += dy * dy;
  }
  return covariance / Math.sqrt(varianceX * varianceY);
}

async function initializeCorrelationViz() {
  const root = document.querySelector('[data-correlation-viz]');
  if (!root) return;

  const response = await fetch('assets/goal-intact-alignment.json?v=20260729-4');
  if (!response.ok) throw new Error(`Alignment data request failed: ${response.status}`);
  const { points } = await response.json();
  const pointLayer = root.querySelector('[data-correlation-points]');
  const fit = root.querySelector('[data-correlation-fit]');
  const buttons = [...root.querySelectorAll('[data-correlation-epoch]')];
  const current = root.querySelector('[data-correlation-current]');
  const srValue = root.querySelector('[data-correlation-sr]');
  const ckaValue = root.querySelector('[data-correlation-cka]');
  const rValue = root.querySelector('[data-correlation-r]');
  const note = root.querySelector('[data-correlation-note]');
  const colors = ['#5637d5', '#198f79', '#e4a03a', '#e53935', '#2196f3'];
  const xMap = (value) => 90 + Math.max(0, Math.min(1, (value - 0.40) / 0.16)) * 620;
  const yMap = (value) => 420 - Math.max(0, Math.min(1, (value - 25) / 75)) * 360;
  let manuallySelected = false;

  function setEpoch(epoch, manual = false) {
    if (manual) manuallySelected = true;
    const visible = points.filter((point) => point.epoch <= epoch);
    const epochPoints = points.filter((point) => point.epoch === epoch);
    const meanSr = epochPoints.reduce((sum, point) => sum + point.sr, 0) / epochPoints.length;
    const meanCka = epochPoints.reduce((sum, point) => sum + point.cka, 0) / epochPoints.length;
    const r = pearson(visible);

    pointLayer.replaceChildren();
    for (const point of visible) {
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.classList.add('correlation-point');
      circle.setAttribute('cx', String(xMap(point.cka)));
      circle.setAttribute('cy', String(yMap(point.sr)));
      circle.setAttribute('r', point.epoch === epoch ? '10' : '7');
      circle.setAttribute('fill', colors[point.epoch - 1]);
      circle.setAttribute('opacity', point.epoch === epoch ? '1' : '0.62');
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      label.textContent = `E${point.epoch}, seed ${point.seed}: CKA ${point.cka.toFixed(3)}, SR ${point.sr.toFixed(2)}%`;
      circle.append(label);
      pointLayer.append(circle);
    }

    const xs = visible.map((point) => point.cka);
    const ys = visible.map((point) => point.sr);
    const meanX = xs.reduce((sum, value) => sum + value, 0) / xs.length;
    const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length;
    const denominator = xs.reduce((sum, value) => sum + (value - meanX) ** 2, 0);
    const slope = denominator > 0
      ? visible.reduce((sum, point) => sum + (point.cka - meanX) * (point.sr - meanY), 0) / denominator
      : 0;
    const intercept = meanY - slope * meanX;
    const lowX = Math.max(0.40, Math.min(...xs) - 0.004);
    const highX = Math.min(0.56, Math.max(...xs) + 0.004);
    fit.setAttribute('x1', String(xMap(lowX)));
    fit.setAttribute('y1', String(yMap(slope * lowX + intercept)));
    fit.setAttribute('x2', String(xMap(highX)));
    fit.setAttribute('y2', String(yMap(slope * highX + intercept)));

    current.textContent = String(epoch);
    srValue.textContent = `${meanSr.toFixed(2)}%`;
    ckaValue.textContent = meanCka.toFixed(3);
    rValue.textContent = `${r >= 0 ? '+' : ''}${r.toFixed(3)}`;
    note.textContent = epoch === 1
      ? 'Provisional · cumulative n = 3'
      : `Cumulative n = ${visible.length} checkpoints`;
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(Number(button.dataset.correlationEpoch) === epoch));
    }
  }

  for (const button of buttons) {
    button.addEventListener('click', () => setEpoch(Number(button.dataset.correlationEpoch), true));
  }

  setEpoch(5);
  if (reduceMotion) return;
  const observer = new IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    observer.disconnect();
    setEpoch(1);
    for (let epoch = 2; epoch <= 5; epoch += 1) {
      window.setTimeout(() => {
        if (!manuallySelected) setEpoch(epoch);
      }, (epoch - 1) * 1250);
    }
  }, { threshold: 0.35 });
  observer.observe(root);
}

initializeCorrelationViz().catch((error) => {
  console.error('Could not initialize CKA/SR visualization.', error);
});

initializeProjectFilm();

const mediaObserver = new IntersectionObserver(
  (entries) => {
    for (const entry of entries) {
      const video = entry.target;
      if (entry.isIntersecting && !reduceMotion) {
        startVideo(video);
      } else {
        video.pause();
      }
    }
  },
  { rootMargin: '180px 0px', threshold: 0.08 },
);

for (const video of document.querySelectorAll('[data-autoplay-video]')) {
  video.muted = true;
  if (reduceMotion) video.pause();
  mediaObserver.observe(video);
}

window.addEventListener('scroll', () => {
  updateProgress();
  updateCurrentSection();
  updateHeaderTone();
}, { passive: true });

window.addEventListener('resize', () => {
  updateProgress();
  updateHeaderTone();
  if (window.innerWidth > 760) closeMenu();
});

updateProgress();
updateCurrentSection();
updateHeaderTone();
