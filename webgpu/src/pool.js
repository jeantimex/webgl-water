export class Pool {
  constructor(device, format, uniformBuffer, tileTexture, tileSampler) {
    this.device = device;
    this.format = format;
    
    this.createGeometry();
    this.createPipeline(uniformBuffer, tileTexture, tileSampler);
  }

  createGeometry() {
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
      indices.push(vOffset + 0, vOffset + 1, vOffset + 2);
      indices.push(vOffset + 2, vOffset + 1, vOffset + 3);
    }

    this.vertexCount = indices.length;

    this.positionBuffer = this.device.createBuffer({
      label: 'Pool Vertex Buffer',
      size: positions.length * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Float32Array(this.positionBuffer.getMappedRange()).set(positions);
    this.positionBuffer.unmap();

    this.indexBuffer = this.device.createBuffer({
      label: 'Pool Index Buffer',
      size: indices.length * 4,
      usage: GPUBufferUsage.INDEX,
      mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();
  }

  createPipeline(uniformBuffer, tileTexture, tileSampler) {
    const shaderModule = this.device.createShaderModule({
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
          // Using textureSampleLevel to avoid non-uniform control flow error
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

    this.pipeline = this.device.createRenderPipeline({
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
        targets: [{ format: this.format }]
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'back',
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth24plus',
      }
    });

    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: tileSampler },
        { binding: 2, resource: tileTexture.createView() }
      ]
    });
  }

  render(passEncoder) {
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.setVertexBuffer(0, this.positionBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
    passEncoder.drawIndexed(this.vertexCount);
  }
}
