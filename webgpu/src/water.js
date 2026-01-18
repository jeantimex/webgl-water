export class Water {
  constructor(device, width, height, uniformBuffer, lightUniformBuffer, sphereUniformBuffer, tileTexture, tileSampler) {
    this.device = device;
    this.width = width;
    this.height = height;
    
    // External resources for surface rendering
    this.commonUniformBuffer = uniformBuffer;
    this.lightUniformBuffer = lightUniformBuffer;
    this.sphereUniformBuffer = sphereUniformBuffer;
    this.tileTexture = tileTexture;
    this.tileSampler = tileSampler;

    // Physics state
    // Texture data: R=Height, G=Velocity, B=NormalX, A=NormalZ
    this.textureA = this.createTexture();
    this.textureB = this.createTexture();
    
    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.createPipelines();
    this.createSurfaceMesh();
    this.createSurfacePipeline();
  }

  createTexture() {
    return this.device.createTexture({
      size: [this.width, this.height],
      format: 'rgba32float', // High precision for physics
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  createPipelines() {
    const fullscreenQuadVS = `
      struct VertexOutput {
        @builtin(position) position : vec4f,
        @location(0) uv : vec2f,
      }

      @vertex
      fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
        var pos = array<vec2f, 6>(
          vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
          vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0)
        );
        var output : VertexOutput;
        output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
        output.uv = vec2f((pos[vertexIndex].x + 1.0) * 0.5, (1.0 - pos[vertexIndex].y) * 0.5);
        return output;
      }
    `;

    // Drop Shader
    this.dropPipeline = this.createPipeline('Drop', fullscreenQuadVS, `
      @group(0) @binding(0) var waterTexture : texture_2d<f32>;
      @group(0) @binding(1) var waterSampler : sampler;
      
      struct DropUniforms {
        center : vec2f,
        radius : f32,
        strength : f32,
      }
      @group(0) @binding(2) var<uniform> u : DropUniforms;

      @fragment
      fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
        var info = textureSample(waterTexture, waterSampler, uv);
        
        let drop = max(0.0, 1.0 - length(u.center * 0.5 + 0.5 - uv) / u.radius);
        let dropVal = 0.5 - cos(drop * 3.14159265) * 0.5;
        
        info.r += dropVal * u.strength;
        
        return info;
      }
    `, 32); 

    // Update Shader
    this.updatePipeline = this.createPipeline('Update', fullscreenQuadVS, `
      @group(0) @binding(0) var waterTexture : texture_2d<f32>;
      @group(0) @binding(1) var waterSampler : sampler;
      
      struct UpdateUniforms {
        delta : vec2f, 
      }
      @group(0) @binding(2) var<uniform> u : UpdateUniforms;

      @fragment
      fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
        var info = textureSample(waterTexture, waterSampler, uv);
        
        let dx = vec2f(u.delta.x, 0.0);
        let dy = vec2f(0.0, u.delta.y);
        
        let average = (
          textureSample(waterTexture, waterSampler, uv - dx).r +
          textureSample(waterTexture, waterSampler, uv - dy).r +
          textureSample(waterTexture, waterSampler, uv + dx).r +
          textureSample(waterTexture, waterSampler, uv + dy).r
        ) * 0.25;
        
        info.g += (average - info.r) * 2.0;
        info.g *= 0.995;
        info.r += info.g;
        
        return info;
      }
    `, 16); 

    // Normal Shader
    this.normalPipeline = this.createPipeline('Normal', fullscreenQuadVS, `
      @group(0) @binding(0) var waterTexture : texture_2d<f32>;
      @group(0) @binding(1) var waterSampler : sampler;
      
      struct NormalUniforms {
        delta : vec2f,
      }
      @group(0) @binding(2) var<uniform> u : NormalUniforms;

      @fragment
      fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
        var info = textureSample(waterTexture, waterSampler, uv);
        
        let val_dx = textureSample(waterTexture, waterSampler, vec2f(uv.x + u.delta.x, uv.y)).r;
        let val_dy = textureSample(waterTexture, waterSampler, vec2f(uv.x, uv.y + u.delta.y)).r;
        
        let dx = vec3f(u.delta.x, val_dx - info.r, 0.0);
        let dy = vec3f(0.0, val_dy - info.r, u.delta.y);
        
        let normal = normalize(cross(dy, dx));
        info.b = normal.x;
        info.a = normal.z;
        
        return info;
      }
    `, 16);

    // Sphere Interaction Shader
    this.spherePipeline = this.createPipeline('Sphere', fullscreenQuadVS, `
      @group(0) @binding(0) var waterTexture : texture_2d<f32>;
      @group(0) @binding(1) var waterSampler : sampler;
      
      struct SphereUniforms {
        oldCenter : vec3f,
        radius : f32, 
        newCenter : vec3f,
        padding : f32,
      }
      @group(0) @binding(2) var<uniform> u : SphereUniforms;

      fn volumeInSphere(center : vec3f, uv : vec2f, radius : f32) -> f32 {
        let p = vec3f(uv.x * 2.0 - 1.0, 0.0, uv.y * 2.0 - 1.0);
        let dist = length(p - center);
        let t = dist / radius;
        
        let dy = exp(-pow(t * 1.5, 6.0));
        let ymin = min(0.0, center.y - dy);
        let ymax = min(max(0.0, center.y + dy), ymin + 2.0 * dy);
        return (ymax - ymin) * 0.1;
      }

      @fragment
      fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
        var info = textureSample(waterTexture, waterSampler, uv);
        
        info.r += volumeInSphere(u.oldCenter, uv, u.radius);
        info.r -= volumeInSphere(u.newCenter, uv, u.radius);
        
        return info;
      }
    `, 32); 
  }

  createPipeline(label, vsCode, fsCode, uniformSize) {
    const module = this.device.createShaderModule({
      label: label + ' Module',
      code: vsCode + fsCode
    });

    const pipeline = this.device.createRenderPipeline({
      label: label + ' Pipeline',
      layout: 'auto',
      vertex: {
        module: module,
        entryPoint: 'vs_main',
      },
      fragment: {
        module: module,
        entryPoint: 'fs_main',
        targets: [{ format: 'rgba32float' }]
      },
      primitive: {
        topology: 'triangle-list',
      }
    });

    return {
      pipeline,
      uniformSize,
      uniformBuffer: this.device.createBuffer({
        size: uniformSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    };
  }

  runPipeline(pipelineObj, uniformsData) {
    this.device.queue.writeBuffer(pipelineObj.uniformBuffer, 0, uniformsData);

    const bindGroup = this.device.createBindGroup({
      layout: pipelineObj.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.textureA.createView() },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: { buffer: pipelineObj.uniformBuffer } }
      ]
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.textureB.createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: { r: 0, g: 0, b: 0, a: 0 }
      }]
    });

    pass.setPipeline(pipelineObj.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6); 
    pass.end();

    this.device.queue.submit([encoder.finish()]);

    const temp = this.textureA;
    this.textureA = this.textureB;
    this.textureB = temp;
  }

  addDrop(x, y, radius, strength) {
    const data = new Float32Array(4); 
    data[0] = x; data[1] = y; 
    data[2] = radius;
    data[3] = strength;
    this.runPipeline(this.dropPipeline, data);
  }

  stepSimulation() {
    const data = new Float32Array(2);
    data[0] = 1.0 / this.width;
    data[1] = 1.0 / this.height;
    this.runPipeline(this.updatePipeline, data);
  }

  updateNormals() {
    const data = new Float32Array(2);
    data[0] = 1.0 / this.width;
    data[1] = 1.0 / this.height;
    this.runPipeline(this.normalPipeline, data);
  }

  moveSphere(oldCenter, newCenter, radius) {
    const data = new Float32Array(8);
    data[0] = oldCenter[0]; data[1] = oldCenter[1]; data[2] = oldCenter[2];
    data[3] = radius;
    data[4] = newCenter[0]; data[5] = newCenter[1]; data[6] = newCenter[2];
    data[7] = 0; 
    this.runPipeline(this.spherePipeline, data);
  }

  // --- Surface Rendering ---

  createSurfaceMesh() {
    const detail = 200;
    const positions = [];
    const indices = [];

    // Generate plane from -1 to 1 on X and Z (Y is up/down in shader logic)
    // Vertices
    for (let z = 0; z <= detail; z++) {
        const t = z / detail;
        for (let x = 0; x <= detail; x++) {
            const s = x / detail;
            // x: 2*s - 1
            // z: 2*t - 1 (mapped to Y in WebGL plane logic? No, mesh.plane makes XY plane)
            // renderer.js uses mesh.plane.
            // In shader: position = gl_Vertex.xzy; position.y += info.r;
            // So plane is on XY, swizzled to XZ.
            // Let's create XZ plane directly or swizzle in shader.
            // Keeping consistent with WebGL demo: Create XY plane, swizzle in shader.
            positions.push(2 * s - 1, 2 * t - 1, 0);
        }
    }

    // Indices
    for (let z = 0; z < detail; z++) {
        for (let x = 0; x < detail; x++) {
            const i = x + z * (detail + 1);
            indices.push(i, i + 1, i + detail + 1);
            indices.push(i + detail + 1, i + 1, i + detail + 2);
        }
    }

    this.vertexCount = indices.length;

    this.positionBuffer = this.device.createBuffer({
        label: 'Water Surface Vertices',
        size: positions.length * 4,
        usage: GPUBufferUsage.VERTEX,
        mappedAtCreation: true,
    });
    new Float32Array(this.positionBuffer.getMappedRange()).set(positions);
    this.positionBuffer.unmap();

    this.indexBuffer = this.device.createBuffer({
        label: 'Water Surface Indices',
        size: indices.length * 4,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
    });
    new Uint32Array(this.indexBuffer.getMappedRange()).set(indices);
    this.indexBuffer.unmap();
  }

  createSurfacePipeline() {
    const shaderModule = this.device.createShaderModule({
        label: 'Water Surface Shader',
        code: `
        struct CommonUniforms {
          viewProjectionMatrix : mat4x4f,
          eyePosition : vec3f,
        }
        @binding(0) @group(0) var<uniform> commonUniforms : CommonUniforms;

        struct LightUniforms {
           direction : vec3f,
        }
        @binding(1) @group(0) var<uniform> light : LightUniforms;

        struct SphereUniforms {
          center : vec3f,
          radius : f32,
        }
        @binding(2) @group(0) var<uniform> sphere : SphereUniforms;

        @binding(3) @group(0) var tileSampler : sampler;
        @binding(4) @group(0) var tileTexture : texture_2d<f32>;
        @binding(5) @group(0) var waterSampler : sampler;
        @binding(6) @group(0) var waterTexture : texture_2d<f32>;

        struct VertexOutput {
          @builtin(position) position : vec4f,
          @location(0) worldPos : vec3f,
        }

        @vertex
        fn vs_main(@location(0) position : vec3f) -> VertexOutput {
          var output : VertexOutput;
          
          // Get water info
          // Map position xy (-1..1) to UV (0..1)
          let uv = position.xy * 0.5 + 0.5;
          let info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
          
          // Swizzle to XZ plane and add height
          var pos = position.xzy; // x, 0, y -> x, z, y? No. X, Y, Z -> X, Z, Y.
          // Plane is created as XY. Z is 0.
          // In WebGL: gl_Vertex.xzy means (x, 0, y).
          // height is info.r.
          pos.y = info.r;
          
          output.worldPos = pos;
          output.position = commonUniforms.viewProjectionMatrix * vec4f(pos, 1.0);
          
          return output;
        }

        // Raytracing helpers
        fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
          let tMin = (cubeMin - origin) / ray;
          let tMax = (cubeMax - origin) / ray;
          let t1 = min(tMin, tMax);
          let t2 = max(tMin, tMax);
          let tNear = max(max(t1.x, t1.y), t1.z);
          let tFar = min(min(t2.x, t2.y), t2.z);
          return vec2f(tNear, tFar);
        }

        fn intersectSphere(origin: vec3f, ray: vec3f, sphereCenter: vec3f, sphereRadius: f32) -> f32 {
            let toSphere = origin - sphereCenter;
            let a = dot(ray, ray);
            let b = 2.0 * dot(toSphere, ray);
            let c = dot(toSphere, toSphere) - sphereRadius * sphereRadius;
            let discriminant = b*b - 4.0*a*c;
            if (discriminant > 0.0) {
              let t = (-b - sqrt(discriminant)) / (2.0 * a);
              if (t > 0.0) { return t; }
            }
            return 1.0e6;
        }
        
        fn getSphereColor(point: vec3f) -> vec3f {
            var color = vec3f(0.5);
            
            // Ambient Occlusion logic (simplified)
            // Need sphereRadius and pool bounds?
            let sphereRadius = sphere.radius;
            
            let dist_x = (1.0 + sphereRadius - abs(point.x)) / sphereRadius;
            let dist_z = (1.0 + sphereRadius - abs(point.z)) / sphereRadius;
            let dist_y = (point.y + 1.0 + sphereRadius) / sphereRadius;
            
            color *= 1.0 - 0.9 / pow(max(0.1, dist_x), 3.0);
            color *= 1.0 - 0.9 / pow(max(0.1, dist_z), 3.0);
            color *= 1.0 - 0.9 / pow(max(0.1, dist_y), 3.0);
            
            // Basic Diffuse
            let lightDir = normalize(light.direction); // Incoming? light struct usually has direction.
            // My light uniform is direction vector.
            // Sphere normal
            let sphereNormal = normalize(point - sphere.center);
            // Refracted light logic... let's simplify for sphere color under water
            // Just return shading.
            
            return color;
        }

        fn getWallColor(point: vec3f) -> vec3f {
            var wallColor : vec3f;
            
            // Planar mapping
            if (abs(point.x) > 0.999) {
                wallColor = textureSampleLevel(tileTexture, tileSampler, point.yz * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
            } else if (abs(point.z) > 0.999) {
                wallColor = textureSampleLevel(tileTexture, tileSampler, point.yx * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
            } else {
                wallColor = textureSampleLevel(tileTexture, tileSampler, point.xz * 0.5 + 0.5, 0.0).rgb;
            }
            
            // Simple shading for walls
            var scale = 0.5;
            // Just return wall color darker
            return wallColor * scale;
        }

        fn getSurfaceRayColor(origin: vec3f, ray: vec3f, waterColor: vec3f) -> vec3f {
            var color : vec3f;
            let poolHeight = 1.0;
            
            let q = intersectSphere(origin, ray, sphere.center, sphere.radius);
            if (q < 1.0e6) {
                color = getSphereColor(origin + ray * q);
            } else if (ray.y < 0.0) {
                let t = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
                color = getWallColor(origin + ray * t.y);
            } else {
                // Sky
                // Fake sky
                color = vec3f(0.3, 0.5, 0.9);
                // Sun spec
                let sunDir = normalize(light.direction);
                let spec = pow(max(0.0, dot(sunDir, ray)), 5000.0);
                color += vec3f(spec);
            }
            
            if (ray.y < 0.0) {
                color *= waterColor;
            }
            return color;
        }

        @fragment
        fn fs_main(@location(0) worldPos : vec3f) -> @location(0) vec4f {
            let IOR_AIR = 1.0;
            let IOR_WATER = 1.333;
            let abovewaterColor = vec3f(0.25, 1.0, 1.25);
            let underwaterColor = vec3f(0.4, 0.9, 1.0);
            
            // Sample Water Texture for Normal
            let uv = worldPos.xz * 0.5 + 0.5;
            let info = textureSample(waterTexture, waterSampler, uv);
            
            // Normal reconstruction
            // info.b = normal.x, info.a = normal.z
            // normal.y = sqrt(1 - x*x - z*z)
            let normal = vec3f(info.b, sqrt(max(0.0, 1.0 - info.b*info.b - info.a*info.a)), info.a);
            
            let viewDir = normalize(worldPos - commonUniforms.eyePosition); // View vector (Eye to Point)? 
            // Usually Incoming Ray is (Point - Eye).
            let incomingRay = normalize(worldPos - commonUniforms.eyePosition);
            
            // Reflection
            let reflectedRay = reflect(incomingRay, normal);
            let refractedRay = refract(incomingRay, normal, IOR_AIR / IOR_WATER);
            
            // Fresnel
            // mix(0.25, 1.0, pow(1.0 - dot(normal, -incomingRay), 3.0));
            let fresnel = mix(0.25, 1.0, pow(1.0 - dot(normal, -incomingRay), 3.0));
            
            let reflectedColor = getSurfaceRayColor(worldPos, reflectedRay, abovewaterColor);
            let refractedColor = getSurfaceRayColor(worldPos, refractedRay, abovewaterColor);
            
            let finalColor = mix(refractedColor, reflectedColor, fresnel);
            
            return vec4f(finalColor, 1.0);
        }
        `
    });

    this.surfacePipeline = this.device.createRenderPipeline({
        label: 'Water Surface Pipeline',
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
            targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }]
        },
        primitive: {
            topology: 'triangle-list',
            cullMode: 'none', // Render both sides just in case
        },
        depthStencil: {
            depthWriteEnabled: true,
            depthCompare: 'less',
            format: 'depth24plus',
        }
    });

    this.surfaceBindGroup = this.device.createBindGroup({
        layout: this.surfacePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: this.commonUniformBuffer } },
            { binding: 1, resource: { buffer: this.lightUniformBuffer } },
            { binding: 2, resource: { buffer: this.sphereUniformBuffer } },
            { binding: 3, resource: this.tileSampler },
            { binding: 4, resource: this.tileTexture.createView() },
            { binding: 5, resource: this.sampler },
            { binding: 6, resource: this.textureA.createView() } // Use textureA for reading in render
        ]
    });
  }

  renderSurface(passEncoder) {
    // Re-create bind group if texture swapped?
    // Simulation swaps textureA and textureB.
    // If I bind textureA once, it points to the GPU texture object.
    // Swapping `this.textureA` JS variable changes the reference.
    // I need to update the bind group or have two bind groups.
    
    // Easier: Just create bind group every frame or update it.
    // Creating bind group is cheap enough for now?
    // Or cache two bind groups.
    
    const bindGroup = this.device.createBindGroup({
        layout: this.surfacePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: this.commonUniformBuffer } },
            { binding: 1, resource: { buffer: this.lightUniformBuffer } },
            { binding: 2, resource: { buffer: this.sphereUniformBuffer } },
            { binding: 3, resource: this.tileSampler },
            { binding: 4, resource: this.tileTexture.createView() },
            { binding: 5, resource: this.sampler },
            { binding: 6, resource: this.textureA.createView() }
        ]
    });

    passEncoder.setPipeline(this.surfacePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setVertexBuffer(0, this.positionBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
    passEncoder.drawIndexed(this.vertexCount);
  }
}