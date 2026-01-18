import { mat4, vec3 } from 'wgpu-matrix';
import { Pool } from './pool.js';
import { Sphere } from './sphere.js';

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

  // Uniform Buffer (Matrices) - Shared ViewProjection
  const uniformBufferSize = 4 * 16; // 4x4 matrix
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create Pool
  const pool = new Pool(device, format, uniformBuffer, tileTexture, tileSampler);

  // Create Sphere
  const sphere = new Sphere(device, format, uniformBuffer);
  
  // Initial Sphere Physics State
  let center = [-0.4, -0.75, 0.2];
  let radius = 0.25;
  sphere.update(center, radius);

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
    const aspect = canvas.width / canvas.height;
    const projectionMatrix = mat4.perspective(45 * Math.PI / 180, aspect, 0.01, 100);
    
    const viewMatrix = mat4.identity();
    mat4.translate(viewMatrix, [0, 0, -4], viewMatrix);
    mat4.rotateX(viewMatrix, -angleX * Math.PI / 180, viewMatrix);
    mat4.rotateY(viewMatrix, -angleY * Math.PI / 180, viewMatrix);
    mat4.translate(viewMatrix, [0, 0.5, 0], viewMatrix);

    const viewProjectionMatrix = mat4.multiply(projectionMatrix, viewMatrix);
    
    device.queue.writeBuffer(uniformBuffer, 0, viewProjectionMatrix);
  }

  function render() {
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
