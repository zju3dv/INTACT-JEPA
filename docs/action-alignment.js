import * as THREE from './vendor/three/three.module.min.js';
import { OrbitControls } from './vendor/three/OrbitControls.js';

const root = document.querySelector('[data-action-alignment]');

if (root) {
  const canvas = root.querySelector('canvas');
  const stage = root.querySelector('[data-action-stage]');
  const controlsRoot = root.querySelector('[data-action-controls]');
  const legend = root.querySelector('[data-action-legend]');
  const modeLabel = root.querySelector('[data-action-mode]');
  const r2Value = root.querySelector('[data-action-r2]');
  const ckaValue = root.querySelector('[data-action-cka]');
  const knnValue = root.querySelector('[data-action-knn]');
  const resetButton = root.querySelector('[data-action-reset]');
  const loading = root.querySelector('[data-action-loading]');

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf7f9fb);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.075;
  orbit.zoomToCursor = true;
  orbit.minDistance = 2.15;
  orbit.maxDistance = 7;

  const content = new THREE.Group();
  scene.add(content);

  const dotCanvas = document.createElement('canvas');
  dotCanvas.width = 64;
  dotCanvas.height = 64;
  const dotContext = dotCanvas.getContext('2d');
  dotContext.fillStyle = '#ffffff';
  dotContext.beginPath();
  dotContext.arc(32, 32, 29, 0, Math.PI * 2);
  dotContext.fill();
  const dotTexture = new THREE.CanvasTexture(dotCanvas);

  const corner = -1.35;
  const endpoint = 1.42;
  const axes = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(corner, corner, corner), new THREE.Vector3(endpoint, corner, corner),
      new THREE.Vector3(corner, corner, corner), new THREE.Vector3(corner, endpoint, corner),
      new THREE.Vector3(corner, corner, corner), new THREE.Vector3(corner, corner, endpoint),
    ]),
    new THREE.LineBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.72 }),
  );
  scene.add(axes);

  let data;
  let mode = 'pusht';

  function resetCamera() {
    camera.position.set(2.32, 1.58, 3.45);
    orbit.target.set(0, 0, 0);
    orbit.update();
  }

  function geometryFrom(values) {
    return new THREE.BufferGeometry().setAttribute(
      'position',
      new THREE.Float32BufferAttribute(values.flat(), 3),
    );
  }

  function addPoints(values, color, size = 0.045, opacity = 0.86) {
    const object = new THREE.Points(
      geometryFrom(values),
      new THREE.PointsMaterial({
        color,
        map: dotTexture,
        size,
        sizeAttenuation: true,
        transparent: true,
        opacity,
        alphaTest: 0.12,
        depthWrite: false,
      }),
    );
    content.add(object);
  }

  function addPairs(expert, predicted) {
    const vertices = [];
    for (let index = 0; index < expert.length; index += 1) {
      vertices.push(...expert[index], ...predicted[index]);
    }
    content.add(new THREE.LineSegments(
      geometryFrom(Array.from({ length: vertices.length / 3 }, (_, index) => vertices.slice(index * 3, index * 3 + 3))),
      new THREE.LineBasicMaterial({ color: 0x718095, transparent: true, opacity: 0.18 }),
    ));
  }

  function setMetric(element, value) {
    element.textContent = Number.isFinite(value) ? value.toFixed(3) : '—';
  }

  function clearContent() {
    for (const object of [...content.children]) {
      object.geometry?.dispose();
      const materials = Array.isArray(object.material) ? object.material : [object.material];
      for (const material of materials) material?.dispose();
    }
    content.clear();
  }

  function renderMode(nextMode) {
    mode = nextMode;
    clearContent();
    for (const button of controlsRoot.querySelectorAll('button')) {
      button.setAttribute('aria-pressed', String(button.dataset.actionView === mode));
    }

    if (mode === 'intent') {
      for (const task of data.tasks) addPoints(task.intent, task.color, 0.043, 0.82);
      modeLabel.textContent = 'Shared actor-intent geometry';
      r2Value.textContent = '—';
      ckaValue.textContent = '—';
      knnValue.textContent = '—';
      legend.innerHTML = data.tasks
        .map((task) => `<span><i style="--legend-color:${task.color}"></i>${task.label}</span>`)
        .join('');
    } else {
      const task = data.tasks.find((candidate) => candidate.key === mode);
      addPairs(task.expert, task.predicted);
      addPoints(task.expert, '#8793A3', 0.052, 0.58);
      addPoints(task.predicted, task.color, 0.064, 0.92);
      modeLabel.textContent = `${task.label} action correspondence`;
      setMetric(r2Value, task.metrics.action_r2);
      setMetric(ckaValue, task.metrics.linear_cka);
      setMetric(knnValue, task.metrics.knn_overlap);
      legend.innerHTML = `<span><i style="--legend-color:#8793A3"></i>Expert</span><span><i style="--legend-color:${task.color}"></i>Predicted</span><span><b></b>Paired sample</span>`;
    }
    resetCamera();
  }

  function resize() {
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (width < 2 || height < 2) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }

  function animate() {
    orbit.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }

  controlsRoot.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-action-view]');
    if (button && button.dataset.actionView !== mode) renderMode(button.dataset.actionView);
  });
  resetButton.addEventListener('click', resetCamera);
  new ResizeObserver(resize).observe(stage);
  resetCamera();
  resize();
  requestAnimationFrame(animate);

  fetch('assets/action-alignment-e5.json?v=20260729-1')
    .then((response) => {
      if (!response.ok) throw new Error(`Action alignment request failed: ${response.status}`);
      return response.json();
    })
    .then((payload) => {
      data = payload;
      loading.hidden = true;
      renderMode(mode);
    })
    .catch((error) => {
      console.error(error);
      loading.textContent = 'Action correspondence data could not be loaded.';
    });
}
