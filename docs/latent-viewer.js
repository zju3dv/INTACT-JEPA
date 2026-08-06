import * as THREE from './vendor/three/three.module.min.js';
import { OrbitControls } from './vendor/three/OrbitControls.js';

const root = document.querySelector('[data-latent-viewer]');

if (root) {
  // Bail gracefully if WebGL is unavailable; otherwise the loading pill hangs forever.
  try {
    const probeCanvas = document.createElement('canvas');
    const gl = probeCanvas.getContext('webgl2') || probeCanvas.getContext('webgl');
    if (!gl) throw new Error('WebGL unavailable');
  } catch (viewerError) {
    const viewerLoading = root.querySelector('[data-viewer-loading]');
    if (viewerLoading) {
      viewerLoading.textContent = 'The interactive geometry viewer needs WebGL, which is unavailable in this browser. The rest of the page is unaffected.';
      viewerLoading.classList.add('viewer-error');
    }
    root.classList.add('viewer-unavailable');
    throw viewerError;
  }

  const stage = root.querySelector('[data-geometry-stage]');
  const loading = root.querySelector('[data-viewer-loading]');
  const variantControls = root.querySelector('[data-variant-controls]');
  const dimensionControls = root.querySelector('[data-dimension-controls]');
  const timeline = root.querySelector('[data-timeline]');
  const playButton = root.querySelector('[data-play]');
  const resetButton = root.querySelector('[data-reset-view]');
  const epochValue = root.querySelector('[data-epoch]');
  const stepValue = root.querySelector('[data-step]');
  const frameValue = root.querySelector('[data-frame]');
  const rankValue = root.querySelector('[data-rank]');
  const cosineValue = root.querySelector('[data-cosine]');
  const variantTitle = root.querySelector('[data-variant-title]');
  const variantDescription = root.querySelector('[data-variant-description]');
  const taskLegend = root.querySelector('[data-task-legend]');
  const endpointLabel = root.querySelector('[data-endpoint-label]');
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const state = {
    manifest: null,
    variant: 'complete',
    mode: '3d',
    frame: 0,
    playing: false,
    metadata: new Map(),
    coordinates: new Map(),
    loadToken: 0,
    playStartedAt: 0,
    playStartedFrame: 0,
  };

  function dimensionsForMode() {
    if (state.mode === 'split') return [2, 3];
    return [Number(state.mode[0])];
  }

  function hexToRgb(hex) {
    const value = Number.parseInt(hex.slice(1), 16);
    return [(value >> 16) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
  }

  function pointMaterial() {
    return new THREE.ShaderMaterial({
      transparent: true,
      depthWrite: false,
      vertexColors: true,
      uniforms: {
        pointScale: { value: 14.0 },
        opacity: { value: 0.80 },
      },
      vertexShader: `
        uniform float pointScale;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * viewPosition;
          gl_PointSize = pointScale * (2.6 / max(1.0, -viewPosition.z));
        }
      `,
      fragmentShader: `
        uniform float opacity;
        varying vec3 vColor;
        void main() {
          float radius = distance(gl_PointCoord, vec2(0.5));
          float edge = 1.0 - smoothstep(0.40, 0.50, radius);
          if (edge <= 0.0) discard;
          gl_FragColor = vec4(vColor, opacity * edge);
        }
      `,
    });
  }

  function createView(dimensions) {
    const container = root.querySelector(`[data-geometry-view="${dimensions}"]`);
    const canvas = container.querySelector('canvas');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7f9fb);

    const camera = dimensions === 3
      ? new THREE.PerspectiveCamera(34, 1, 0.01, 100)
      : new THREE.OrthographicCamera(-1.55, 1.55, 1.55, -1.55, 0.01, 100);
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.zoomToCursor = true;
    controls.enableRotate = dimensions === 3;
    controls.enablePan = dimensions === 2;
    controls.minDistance = 2.15;
    controls.maxDistance = 7;
    controls.minZoom = 0.7;
    controls.maxZoom = 4;

    const geometry = new THREE.BufferGeometry();
    const points = new THREE.Points(geometry, pointMaterial());
    points.frustumCulled = false;
    scene.add(points);

    if (dimensions === 3) {
      const corner = -1.42;
      const end = 1.48;
      const axisGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(corner, corner, corner), new THREE.Vector3(end, corner, corner),
        new THREE.Vector3(corner, corner, corner), new THREE.Vector3(corner, end, corner),
        new THREE.Vector3(corner, corner, corner), new THREE.Vector3(corner, corner, end),
      ]);
      scene.add(new THREE.LineSegments(
        axisGeometry,
        new THREE.LineBasicMaterial({ color: 0x161616, transparent: true, opacity: 0.76 }),
      ));
    }

    function reset() {
      if (dimensions === 3) {
        camera.position.set(2.28, 1.66, 3.42);
      } else {
        camera.position.set(0, 0, 4);
        camera.zoom = 1;
        camera.updateProjectionMatrix();
      }
      controls.target.set(0, 0, 0);
      controls.update();
    }

    function resize() {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width < 2 || height < 2) return;
      renderer.setSize(width, height, false);
      const aspect = width / height;
      if (dimensions === 3) {
        camera.aspect = aspect;
      } else {
        camera.left = -1.52 * aspect;
        camera.right = 1.52 * aspect;
        camera.top = 1.52;
        camera.bottom = -1.52;
      }
      camera.updateProjectionMatrix();
    }

    function setTaskColors(metadata) {
      const colors = new Float32Array(metadata.points * 3);
      let pointIndex = 0;
      for (const task of state.manifest.tasks) {
        const rgb = hexToRgb(task.color);
        for (let index = 0; index < metadata.points_per_task; index += 1) {
          colors.set(rgb, pointIndex * 3);
          pointIndex += 1;
        }
      }
      geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }

    function prepare(metadata) {
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(new Float32Array(metadata.points * 3), 3),
      );
      setTaskColors(metadata);
    }

    function setFrame(metadata, coordinates, frame) {
      const positions = geometry.getAttribute('position').array;
      const frameOffset = frame * metadata.points * dimensions;
      const radii = new Float32Array(metadata.points);
      for (let point = 0; point < metadata.points; point += 1) {
        const source = frameOffset + point * dimensions;
        const x = coordinates[source];
        const y = coordinates[source + 1];
        const z = dimensions === 3 ? coordinates[source + 2] : 0;
        radii[point] = Math.hypot(x, y, z);
      }
      radii.sort();
      const robustRadius = Math.max(radii[Math.floor((metadata.points - 1) * 0.98)], 1);
      const scale = 1.16 / robustRadius;
      for (let point = 0; point < metadata.points; point += 1) {
        const source = frameOffset + point * dimensions;
        const target = point * 3;
        positions[target] = coordinates[source] * scale;
        positions[target + 1] = coordinates[source + 1] * scale;
        positions[target + 2] = dimensions === 3 ? coordinates[source + 2] * scale : 0;
      }
      geometry.getAttribute('position').needsUpdate = true;
    }

    function render() {
      controls.update();
      renderer.render(scene, camera);
    }

    new ResizeObserver(resize).observe(container);
    reset();
    resize();
    return { dimensions, container, prepare, reset, resize, setFrame, render };
  }

  const views = new Map([[2, createView(2)], [3, createView(3)]]);

  function setLoading(message) {
    loading.textContent = message;
    loading.hidden = !message;
    root.toggleAttribute('aria-busy', Boolean(message));
  }

  function activeVariant() {
    return state.manifest.variants.find((variant) => variant.key === state.variant);
  }

  function coordinateKey(dimensions) {
    return `${state.variant}:${dimensions}d`;
  }

  async function metadataFor(variant) {
    if (!state.metadata.has(variant.key)) {
      const response = await fetch(`assets/latent-geometry/${variant.metadata}`);
      if (!response.ok) throw new Error(`Metadata request failed: ${response.status}`);
      state.metadata.set(variant.key, await response.json());
    }
    return state.metadata.get(variant.key);
  }

  async function coordinatesFor(variant, metadata, dimensions) {
    const key = coordinateKey(dimensions);
    if (!state.coordinates.has(key)) {
      const projection = metadata.files[`${dimensions}d`];
      const response = await fetch(`assets/latent-geometry/${projection.file}`);
      if (!response.ok) throw new Error(`Coordinate request failed: ${response.status}`);
      const coordinates = new Int16Array(await response.arrayBuffer());
      const expected = metadata.frames * metadata.points * dimensions;
      if (coordinates.length !== expected) {
        throw new Error(`Coordinate length ${coordinates.length} does not match ${expected}`);
      }
      state.coordinates.set(key, coordinates);
    }
    return state.coordinates.get(key);
  }

  function updateMetrics(metadata) {
    const frame = state.frame;
    epochValue.textContent = metadata.epochs[frame].toFixed(2);
    stepValue.textContent = metadata.steps[frame].toLocaleString('en-US');
    frameValue.textContent = `${frame + 1} / ${metadata.frames.toLocaleString('en-US')}`;
    rankValue.textContent = metadata.effective_rank[frame].toFixed(1);
    cosineValue.textContent = metadata.mean_pairwise_cosine[frame].toFixed(3);
    timeline.value = String(frame);
  }

  function updateFrame(metadata) {
    state.frame = Math.max(0, Math.min(metadata.frames - 1, Math.trunc(state.frame || 0)));
    for (const dimensions of dimensionsForMode()) {
      const coordinates = state.coordinates.get(coordinateKey(dimensions));
      if (coordinates) views.get(dimensions).setFrame(metadata, coordinates, state.frame);
    }
    updateMetrics(metadata);
  }

  function setPlaying(playing) {
    state.playing = playing;
    playButton.textContent = playing ? '❚❚' : '▶';
    playButton.setAttribute('aria-label', playing ? 'Pause training evolution' : 'Play training evolution');
    playButton.title = playing ? 'Pause' : 'Play';
    if (playing) {
      state.playStartedAt = performance.now();
      state.playStartedFrame = state.frame;
    }
  }

  function applyMode() {
    stage.classList.remove('mode-2d', 'mode-3d', 'mode-split');
    stage.classList.add(`mode-${state.mode}`);
    requestAnimationFrame(() => {
      for (const dimensions of dimensionsForMode()) views.get(dimensions).resize();
    });
  }

  async function selectData({ preserveProgress = true } = {}) {
    const token = ++state.loadToken;
    setPlaying(false);
    setLoading('Loading measured checkpoints…');
    try {
      const variant = activeVariant();
      const metadata = await metadataFor(variant);
      const previousMax = Number(timeline.max) || metadata.frames - 1;
      const previousProgress = previousMax > 0 ? state.frame / previousMax : 0;
      await Promise.all(
        dimensionsForMode().map((dimensions) => coordinatesFor(variant, metadata, dimensions)),
      );
      if (token !== state.loadToken) return;

      timeline.max = String(metadata.frames - 1);
      state.frame = preserveProgress
        ? Math.round(previousProgress * (metadata.frames - 1))
        : 0;
      for (const dimensions of dimensionsForMode()) {
        const view = views.get(dimensions);
        view.prepare(metadata);
        view.reset();
      }
      variantTitle.textContent = variant.label;
      variantDescription.textContent = variant.description;
      endpointLabel.textContent = `Measured E0–E${metadata.final_epoch.toFixed(0)} · ${metadata.frames.toLocaleString('en-US')} saved states`;
      applyMode();
      updateFrame(metadata);
      setLoading('');
      if (!reduceMotion) setPlaying(true);
    } catch (error) {
      console.error(error);
      setLoading('The geometry viewer could not load. Serve the site over HTTP and retry.');
    }
  }

  function renderVariantButtons() {
    variantControls.replaceChildren();
    for (const variant of state.manifest.variants) {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.variant = variant.key;
      button.className = `variant-option variant-option-${variant.key}`;
      button.innerHTML = `<span>${variant.label}</span><strong>rank ${variant.final_effective_rank.toFixed(1)}</strong>`;
      button.setAttribute('aria-pressed', String(variant.key === state.variant));
      button.addEventListener('click', () => {
        if (variant.key === state.variant) return;
        state.variant = variant.key;
        for (const candidate of variantControls.querySelectorAll('button')) {
          candidate.setAttribute('aria-pressed', String(candidate.dataset.variant === state.variant));
        }
        selectData({ preserveProgress: true });
      });
      variantControls.append(button);
    }
  }

  function renderTaskLegend() {
    taskLegend.replaceChildren();
    for (const task of state.manifest.tasks) {
      const item = document.createElement('span');
      item.innerHTML = `<i style="--task-color:${task.color}"></i>${task.label}`;
      taskLegend.append(item);
    }
  }

  function animate(now) {
    const variant = state.manifest && activeVariant();
    const metadata = variant && state.metadata.get(variant.key);
    if (state.playing && metadata) {
      const duration = 90000;
      const elapsed = Math.max(0, now - state.playStartedAt);
      const advanced = Math.floor(elapsed / duration * metadata.frames);
      let nextFrame = state.playStartedFrame + advanced;
      if (nextFrame >= metadata.frames) {
        state.playStartedAt = now;
        state.playStartedFrame = 0;
        nextFrame = 0;
      }
      if (nextFrame !== state.frame) {
        state.frame = nextFrame;
        updateFrame(metadata);
      }
    }
    for (const dimensions of dimensionsForMode()) views.get(dimensions).render();
    requestAnimationFrame(animate);
  }

  variantControls.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    const buttons = [...variantControls.querySelectorAll('button')];
    const current = buttons.findIndex((button) => button.dataset.variant === state.variant);
    const direction = event.key === 'ArrowRight' ? 1 : -1;
    buttons[(current + direction + buttons.length) % buttons.length].click();
    event.preventDefault();
  });

  dimensionControls.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-dimensions]');
    if (!button) return;
    const mode = button.dataset.dimensions === 'split'
      ? 'split'
      : `${button.dataset.dimensions}d`;
    if (mode === state.mode) return;
    state.mode = mode;
    for (const candidate of dimensionControls.querySelectorAll('button')) {
      const candidateMode = candidate.dataset.dimensions === 'split'
        ? 'split'
        : `${candidate.dataset.dimensions}d`;
      candidate.setAttribute('aria-pressed', String(candidateMode === mode));
    }
    applyMode();
    selectData({ preserveProgress: true });
  });

  timeline.addEventListener('input', async () => {
    const metadata = await metadataFor(activeVariant());
    state.frame = Number(timeline.value);
    setPlaying(false);
    updateFrame(metadata);
  });

  playButton.addEventListener('click', async () => {
    const metadata = await metadataFor(activeVariant());
    if (!state.playing && state.frame >= metadata.frames - 1) {
      state.frame = 0;
      updateFrame(metadata);
    }
    setPlaying(!state.playing);
  });

  resetButton.addEventListener('click', () => {
    for (const dimensions of dimensionsForMode()) views.get(dimensions).reset();
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) setPlaying(false);
  });

  fetch('assets/latent-geometry/manifest.json')
    .then((response) => {
      if (!response.ok) throw new Error(`Manifest request failed: ${response.status}`);
      return response.json();
    })
    .then((manifest) => {
      state.manifest = manifest;
      renderVariantButtons();
      renderTaskLegend();
      applyMode();
      selectData({ preserveProgress: false });
      requestAnimationFrame(animate);
    })
    .catch((error) => {
      console.error(error);
      setLoading('The geometry manifest is unavailable.');
    });
}
