import { mat4, vec3 } from 'wgpu-matrix';

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
  
  // Create Pool Geometry (Cube with -y face removed)
  function pickOctant(i) {
    return [
      (i & 1) * 2 - 1,
      (i & 2) - 1,
      (i & 4) / 2 - 1
    ];
  }

  const cubeData = [
    [0, 4, 2, 6, -1, 0, 0], // -x
    [1, 3, 5, 7, +1, 0, 0], // +x
    // [0, 1, 4, 5, 0, -1, 0], // -y (REMOVED)
    [2, 6, 3, 7, 0, +1, 0], // +y
    [0, 2, 1, 3, 0, 0, -1], // -z
    [4, 5, 6, 7, 0, 0, +1]  // +z
  ];

  const positions = [];
  const indices = [];
  let vertexCount = 0;

  for (const data of cubeData) {
    const vOffset = vertexCount;
    for (let j = 0; j < 4; j++) {
      const d = data[j];
      const pos = pickOctant(d);
      positions.push(...pos);
      vertexCount++;
    }
    // 2 triangles per face
    indices.push(vOffset + 0, vOffset + 1, vOffset + 2);
    indices.push(vOffset + 2, vOffset + 1, vOffset + 3);
  }

  const positionBuffer = device.createBuffer({
    label: 'Pool Vertex Buffer',
    size: positions.length * 4,
    usage: GPUBufferUsage.VERTEX,
    mappedAtCreation: true,
  });
  new Float32Array(positionBuffer.getMappedRange()).set(positions);
  positionBuffer.unmap();

  const indexBuffer = device.createBuffer({
    label: 'Pool Index Buffer',
    size: indices.length * 4,
    usage: GPUBufferUsage.INDEX,
    mappedAtCreation: true,
  });
  new Uint32Array(indexBuffer.getMappedRange()).set(indices);
  indexBuffer.unmap();

  // Uniform Buffer (Matrices)
  const uniformBufferSize = 4 * 16; // 4x4 matrix
  const uniformBuffer = device.createBuffer({
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Pipeline
  const shaderModule = device.createShaderModule({
    label: 'Pool Shader',
    code: `
      struct Uniforms {
        modelViewProjectionMatrix : mat4x4f,
      }
      @binding(0) @group(0) var<uniform> uniforms : Uniforms;
      @binding(1) @group(0) var tileSampler : sampler;
      @binding(2) @group(0) var tileTexture : texture_2d<f32>;

      struct VertexOutput {
        @builtin(position) position : vec4f,
        @location(0) localPos : vec3f,
      }

      @vertex
      fn vs_main(@location(0) position : vec3f) -> VertexOutput {
        var output : VertexOutput;
        
        var transformedPos = position;
        transformedPos.y = ((1.0 - position.y) * (7.0 / 12.0) - 1.0);

        output.position = uniforms.modelViewProjectionMatrix * vec4f(transformedPos, 1.0);
        output.localPos = transformedPos;
        return output;
      }

      @fragment
      fn fs_main(@location(0) localPos : vec3f) -> @location(0) vec4f {
        var wallColor : vec3f;
        let point = localPos;
        
        // Planar mapping logic from WebGL demo
        if (abs(point.x) > 0.999) {
          wallColor = textureSampleLevel(tileTexture, tileSampler, point.yz * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
        } else if (abs(point.z) > 0.999) {
          wallColor = textureSampleLevel(tileTexture, tileSampler, point.yx * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
        } else {
          wallColor = textureSampleLevel(tileTexture, tileSampler, point.xz * 0.5 + 0.5, 0.0).rgb;
        }

        return vec4f(wallColor, 1.0);
      }
    `
  });

  const pipeline = device.createRenderPipeline({
    label: 'Pool Pipeline',
    layout: 'auto',
    vertex: {
      module: shaderModule,
      entryPoint: 'vs_main',
      buffers: [{
        arrayStride: 3 * 4,
        attributes: [{
          shaderLocation: 0,
          offset: 0,
          format: 'float32x3'
        }]
      }]
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fs_main',
      targets: [{ format }]
    },
    primitive: {
      topology: 'triangle-list',
      cullMode: 'back', // Ensure winding order and culling matches
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    }
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: tileSampler },
      { binding: 2, resource: tileTexture.createView() }
    ]
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
    const aspect = canvas.width / canvas.height;
    const projectionMatrix = mat4.perspective(45 * Math.PI / 180, aspect, 0.01, 100);
    
    const viewMatrix = mat4.identity();
    mat4.translate(viewMatrix, [0, 0, -4], viewMatrix);
    mat4.rotateX(viewMatrix, -angleX * Math.PI / 180, viewMatrix);
    mat4.rotateY(viewMatrix, -angleY * Math.PI / 180, viewMatrix);
    mat4.translate(viewMatrix, [0, 0.5, 0], viewMatrix);

    const mvpMatrix = mat4.multiply(projectionMatrix, viewMatrix);
    
    device.queue.writeBuffer(uniformBuffer, 0, mvpMatrix);
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

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setVertexBuffer(0, positionBuffer);
    passEncoder.setIndexBuffer(indexBuffer, 'uint32');
    passEncoder.drawIndexed(indices.length);
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
