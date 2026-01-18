import { mat4, vec3 } from 'wgpu-matrix';
import { Pool } from './pool.js';
import { Sphere } from './sphere.js';
import { Vector, Raytracer } from './lightgl.js';

async function init() {
  const gpu = navigator.gpu;
  if (!gpu) {
    document.getElementById('loading').innerHTML = 'WebGPU not supported.';
    return;
  }

  const adapter = await gpu.requestAdapter();
  if (!adapter) {
    document.getElementById('loading').innerHTML = 'No WebGPU adapter found.';
    return;
  }
  const device = await adapter.requestDevice();

  const canvas = document.querySelector('canvas');
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format,
    alphaMode: 'premultiplied',
  });

  const help = document.getElementById('help');
  const ratio = window.devicePixelRatio || 1;

  // Load Texture
  async function loadTexture(url) {
    const res = await fetch(url);
    const blob = await res.blob();
    const source = await createImageBitmap(blob);
    
    const texture = device.createTexture({
      label: url,
      size: [source.width, source.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    device.queue.copyExternalImageToTexture(
      { source, flipY: true },
      { texture },
      { width: source.width, height: source.height }
    );

    return texture;
  }

  const tileTexture = await loadTexture('/tiles.jpg');

  const tileSampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'repeat',
    addressModeV: 'repeat',
  });

  // Camera state
  let angleX = -25;
  let angleY = -200.5;

  function getMatrices() {
    const aspect = canvas.width / canvas.height;
    const projectionMatrix = mat4.perspective(45 * Math.PI / 180, aspect, 0.01, 100);
    
    const viewMatrix = mat4.identity();
    mat4.translate(viewMatrix, [0, 0, -4], viewMatrix);
    mat4.rotateX(viewMatrix, -angleX * Math.PI / 180, viewMatrix);
    mat4.rotateY(viewMatrix, -angleY * Math.PI / 180, viewMatrix);
    mat4.translate(viewMatrix, [0, 0.5, 0], viewMatrix);
    
    return { projectionMatrix, viewMatrix };
  }

  // Uniform Buffer (Matrices) - Shared ViewProjection
  const uniformBufferSize = 4 * 16; // 4x4 matrix
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Lighting State
  let lightDir = new Vector(2.0, 2.0, -1.0).unit();
  const lightUniformBuffer = device.createBuffer({
    size: 16, // vec3 + padding
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  
  function updateLight() {
      device.queue.writeBuffer(lightUniformBuffer, 0, new Float32Array([...lightDir.toArray(), 0]));
  }
  updateLight();

  // Sphere State (Shared)
  const sphereUniformBuffer = device.createBuffer({
    size: 16, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create Pool
  const pool = new Pool(device, format, uniformBuffer, tileTexture, tileSampler, lightUniformBuffer, sphereUniformBuffer);

  // Create Sphere
  const sphere = new Sphere(device, format, uniformBuffer, lightUniformBuffer, sphereUniformBuffer);
  
  // Initial Sphere Physics State
  let center = new Vector(-0.4, -0.75, 0.2);
  let radius = 0.25;
  sphere.update(center.toArray(), radius);

  // Keyboard state
  const keys = {};
  window.addEventListener('keydown', (e) => { keys[e.key.toUpperCase()] = true; });
  window.addEventListener('keyup', (e) => { keys[e.key.toUpperCase()] = false; });

  // Interaction State
  let mode = -1;
  const MODE_ORBIT_CAMERA = 1;
  const MODE_MOVE_SPHERE = 2;
  let oldX, oldY;
  let prevHit;
  let planeNormal;

  function startDrag(x, y) {
    oldX = x;
    oldY = y;
    const { projectionMatrix, viewMatrix } = getMatrices();
    const viewport = [0, 0, canvas.width, canvas.height];
    const tracer = new Raytracer(viewMatrix, projectionMatrix, viewport);
    const ray = tracer.getRayForPixel(x * ratio, y * ratio);
    const sphereHit = Raytracer.hitTestSphere(tracer.eye, ray, center, radius);
    
    if (sphereHit) {
      mode = MODE_MOVE_SPHERE;
      prevHit = sphereHit.hit;
      // Plane perpendicular to camera look vector
      planeNormal = tracer.getRayForPixel(canvas.width / 2, canvas.height / 2).negative();
    } else {
      mode = MODE_ORBIT_CAMERA;
    }
  }

  function duringDrag(x, y) {
    if (mode === MODE_ORBIT_CAMERA) {
      angleY -= x - oldX;
      angleX -= y - oldY;
      angleX = Math.max(-89.999, Math.min(89.999, angleX));
    } else if (mode === MODE_MOVE_SPHERE) {
      const { projectionMatrix, viewMatrix } = getMatrices();
      const viewport = [0, 0, canvas.width, canvas.height];
      const tracer = new Raytracer(viewMatrix, projectionMatrix, viewport);
      const ray = tracer.getRayForPixel(x * ratio, y * ratio);
      
      // Intersect ray with plane defined by prevHit and planeNormal
      // t = dot(prevHit - eye, planeNormal) / dot(ray, planeNormal)
      // or t = -dot(eye - prevHit, planeNormal) / ...
      const t = -planeNormal.dot(tracer.eye.subtract(prevHit)) / planeNormal.dot(ray);
      const nextHit = tracer.eye.add(ray.multiply(t));
      
      center = center.add(nextHit.subtract(prevHit));
      
      // Clamp to pool bounds
      center.x = Math.max(radius - 1, Math.min(1 - radius, center.x));
      center.y = Math.max(radius - 1, Math.min(10, center.y));
      center.z = Math.max(radius - 1, Math.min(1 - radius, center.z));
      
      sphere.update(center.toArray(), radius);
      prevHit = nextHit;
    }
    oldX = x;
    oldY = y;
  }

  function stopDrag() {
    mode = -1;
  }

  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startDrag(e.offsetX, e.offsetY);
  });

  window.addEventListener('mousemove', (e) => {
    if (mode !== -1) {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        duringDrag(x, y);
    }
  });

  window.addEventListener('mouseup', () => {
    stopDrag();
  });


  // Depth Texture
  let depthTexture;

  function onResize() {
    const width = window.innerWidth - help.clientWidth - 20;
    const height = window.innerHeight;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';

    if (depthTexture) depthTexture.destroy();
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: 'depth24plus',
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    render();
  }

  window.addEventListener('resize', onResize);
  document.getElementById('loading').innerHTML = '';
  onResize();

  function updateUniforms() {
    const { projectionMatrix, viewMatrix } = getMatrices();
    const viewProjectionMatrix = mat4.multiply(projectionMatrix, viewMatrix);
    device.queue.writeBuffer(uniformBuffer, 0, viewProjectionMatrix);
  }

  function render() {
    // Logic from @webgl/main.js for L key
    if (keys['L']) {
        lightDir = Vector.fromAngles((90 - angleY) * Math.PI / 180, -angleX * Math.PI / 180);
        updateLight();
    }

    updateUniforms();

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      }
    });

    pool.render(passEncoder);
    sphere.render(passEncoder);
    
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
  }

  function animate() {
    requestAnimationFrame(animate);
    render();
  }
  requestAnimationFrame(animate);
}

init();