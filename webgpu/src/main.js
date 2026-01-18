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

  // Camera state
  let angleX = -25;
  let angleY = -200.5;
  
  // Create Pool Geometry (Cube with -y face removed)
  // Original Cube: -1 to 1
  // Faces: -x, +x, -y, +y, -z, +z
  // We remove -y (bottom in original coords, top in transformed coords)
  // Indices in standard cube generation usually: 
  // 0: -x, 1: +x, 2: -y, 3: +y, 4: -z, 5: +z
  
  // Vertices (Position x, y, z)
  // 4 vertices per face * 6 faces = 24 vertices
  // We keep 5 faces = 20 vertices.
  
  // Helper to pick octant
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
    // 0, 1, 2
    // 2, 1, 3
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
    size: indices.length * 4, // Uint32 is safer, though Uint16 is enough for < 65k
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

      struct VertexOutput {
        @builtin(position) position : vec4f,
        @location(0) localPos : vec3f,
      }

      @vertex
      fn vs_main(@location(0) position : vec3f) -> VertexOutput {
        var output : VertexOutput;
        
        // Transform Y coordinate as per WebGL demo
        // position.y = ((1.0 - position.y) * (7.0 / 12.0) - 1.0) * poolHeight (1.0);
        var transformedPos = position;
        transformedPos.y = ((1.0 - position.y) * (7.0 / 12.0) - 1.0);

        output.position = uniforms.modelViewProjectionMatrix * vec4f(transformedPos, 1.0);
        output.localPos = transformedPos; // Pass to fragment for simple shading
        return output;
      }

      @fragment
      fn fs_main(@location(0) localPos : vec3f) -> @location(0) vec4f {
        // Simple visualization: Color based on position or constant
        return vec4f(0.5, 0.5, 0.5, 1.0); 
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
      cullMode: 'back', // Cull back faces? WebGL uses culling.
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: 'less',
      format: 'depth24plus',
    }
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{
      binding: 0,
      resource: { buffer: uniformBuffer }
    }]
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
    // Projection
    const aspect = canvas.width / canvas.height;
    const projectionMatrix = mat4.perspective(45 * Math.PI / 180, aspect, 0.01, 100);

    // ModelView
    // gl.translate(0, 0, -4);
    // gl.rotate(-angleX, 1, 0, 0);
    // gl.rotate(-angleY, 0, 1, 0);
    // gl.translate(0, 0.5, 0);
    
    // In wgpu-matrix (and gl-matrix), multiplications are usually applied such that:
    // result = A * B implies B happens first (if we think of column vectors).
    // But `mat4.translate(M, v)` does M * T.
    
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