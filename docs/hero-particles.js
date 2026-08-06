import * as THREE from './vendor/three/three.module.min.js';

const root = document.querySelector('[data-immersive-hero]');
const canvas = root?.querySelector('[data-hero-particles]');

if (root && canvas) {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 30);
  camera.position.set(0, 0, 4.05);

  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setClearColor(0x050914, 1);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  const count = window.innerWidth < 700 ? 920 : 1720;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const phases = new Float32Array(count);
  const blue = new THREE.Color('#279ef0');
  const coral = new THREE.Color('#f0525d');
  const pale = new THREE.Color('#d9e8f7');
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  for (let index = 0; index < count; index += 1) {
    const ratio = (index + 0.5) / count;
    const y = 1 - 2 * ratio;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const angle = index * goldenAngle;
    const shell = 1.02 + 0.16 * Math.sin(index * 12.9898) + 0.05 * Math.sin(index * 0.31);
    const offset = index * 3;
    positions[offset] = Math.cos(angle) * radius * shell;
    positions[offset + 1] = y * shell;
    positions[offset + 2] = Math.sin(angle) * radius * shell;
    phases[index] = (index * 0.61803398875) % 1;

    const mix = 0.5 + 0.5 * Math.sin(angle * 0.72 + y * 3.4);
    const color = index % 31 === 0 ? pale : blue.clone().lerp(coral, mix);
    colors[offset] = color.r;
    colors[offset + 1] = color.g;
    colors[offset + 2] = color.b;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('aPhase', new THREE.BufferAttribute(phases, 1));

  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    vertexColors: true,
    uniforms: {
      uTime: { value: 0 },
      uPointer: { value: new THREE.Vector2(4, 4) },
      uPixelRatio: { value: renderer.getPixelRatio() },
    },
    vertexShader: `
      attribute float aPhase;
      varying vec3 vColor;
      varying float vAlpha;
      uniform float uTime;
      uniform vec2 uPointer;
      uniform float uPixelRatio;

      void main() {
        vec3 p = position;
        float pulse = sin(uTime * 0.55 + aPhase * 18.8496);
        p *= 1.0 + pulse * 0.018;
        vec4 viewPosition = modelViewMatrix * vec4(p, 1.0);
        vec4 clipPosition = projectionMatrix * viewPosition;
        vec2 ndc = clipPosition.xy / clipPosition.w;
        vec2 delta = ndc - uPointer;
        float distanceToPointer = length(delta);
        float influence = exp(-distanceToPointer * distanceToPointer * 10.5);
        vec2 direction = delta / max(distanceToPointer, 0.025);
        clipPosition.xy += direction * influence * 0.092 * clipPosition.w;
        gl_Position = clipPosition;
        gl_PointSize = (2.8 + 4.4 * influence + 1.15 * (pulse + 1.0)) * uPixelRatio;
        vColor = color;
        vAlpha = 0.43 + influence * 0.48 + (pulse + 1.0) * 0.08;
      }
    `,
    fragmentShader: `
      varying vec3 vColor;
      varying float vAlpha;

      void main() {
        vec2 centered = gl_PointCoord - vec2(0.5);
        float radius = length(centered);
        if (radius > 0.5) discard;
        float edge = smoothstep(0.5, 0.22, radius);
        gl_FragColor = vec4(vColor, vAlpha * edge);
      }
    `,
  });

  const particles = new THREE.Points(geometry, material);
  particles.rotation.set(-0.18, -0.42, 0.05);
  scene.add(particles);

  const pointer = new THREE.Vector2(4, 4);
  const targetTilt = new THREE.Vector2();
  const idleSpin = reduceMotion ? 0 : 0.072;
  let spinVelocity = idleSpin;
  let previousPointerX = null;
  let previousPointerTime = null;
  let active = true;
  let previous = performance.now();

  function resize() {
    const width = root.clientWidth;
    const height = root.clientHeight;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
    material.uniforms.uPixelRatio.value = renderer.getPixelRatio();
  }

  function updatePointer(event) {
    const rect = root.getBoundingClientRect();
    if (event.clientY < rect.top || event.clientY > rect.bottom) return;
    const now = performance.now();
    if (previousPointerX !== null && previousPointerTime !== null) {
      const elapsed = Math.max(12, now - previousPointerTime);
      const horizontalSpeed = (event.clientX - previousPointerX) / elapsed;
      spinVelocity += THREE.MathUtils.clamp(horizontalSpeed * 0.065, -0.15, 0.15);
      spinVelocity = THREE.MathUtils.clamp(spinVelocity, -0.11, 0.27);
    }
    previousPointerX = event.clientX;
    previousPointerTime = now;
    pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -(((event.clientY - rect.top) / rect.height) * 2 - 1),
    );
    targetTilt.set(pointer.y * 0.13, pointer.x * 0.18);
  }

  function animate(now) {
    const delta = Math.min(0.04, (now - previous) / 1000);
    previous = now;
    if (active) {
      material.uniforms.uTime.value += delta;
      material.uniforms.uPointer.value.lerp(pointer, 0.08);
      spinVelocity += (idleSpin - spinVelocity) * Math.min(1, delta * 1.35);
      particles.rotation.y += delta * spinVelocity;
      particles.rotation.x += (targetTilt.x - particles.rotation.x + 0.18) * 0.018;
      particles.rotation.z += (targetTilt.y - particles.rotation.z) * 0.012;
      renderer.render(scene, camera);
    }
    requestAnimationFrame(animate);
  }

  window.addEventListener('pointermove', updatePointer, { passive: true });
  root.addEventListener('pointerleave', () => {
    pointer.set(4, 4);
    targetTilt.set(0, 0);
    previousPointerX = null;
    previousPointerTime = null;
  });
  new ResizeObserver(resize).observe(root);
  new IntersectionObserver((entries) => {
    active = entries.some((entry) => entry.isIntersecting);
  }, { threshold: 0.02 }).observe(root);
  document.addEventListener('visibilitychange', () => {
    active = !document.hidden && root.getBoundingClientRect().bottom > 0;
  });

  resize();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
