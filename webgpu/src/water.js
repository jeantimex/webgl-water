export class Water {
  constructor(device, width, height, uniformBuffer, lightUniformBuffer, sphereUniformBuffer, tileTexture, tileSampler, skyTexture, skySampler) {
    this.device = device;
    this.width = width;
    this.height = height;
    
    // External resources
    this.commonUniformBuffer = uniformBuffer;
    this.lightUniformBuffer = lightUniformBuffer;
    this.sphereUniformBuffer = sphereUniformBuffer;
    this.tileTexture = tileTexture;
    this.tileSampler = tileSampler;
    this.skyTexture = skyTexture;
    this.skySampler = skySampler;

    // Physics state
    this.textureA = this.createTexture();
    this.textureB = this.createTexture();
    
    // Caustics Texture (Higher res for detail)
    this.causticsTexture = this.device.createTexture({
      size: [1024, 1024],
      format: 'rgba8unorm', // or rg8unorm? WebGL uses RGBA (or RGB).
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    this.sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });

    this.createPipelines();
    this.createSurfaceMesh();
    this.createSurfacePipeline();
    this.createCausticsPipeline();
  }

  createTexture() {
    const format = this.device.features.has('float32-filterable') ? 'rgba32float' : 'rgba16float';
    return this.device.createTexture({
      size: [this.width, this.height],
      format: format, 
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  createPipelines() {
    const format = this.device.features.has('float32-filterable') ? 'rgba32float' : 'rgba16float';
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
    `, 32, format); 

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
    `, 16, format); 

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
    `, 16, format);

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
    `, 32, format); 
  }

  createPipeline(label, vsCode, fsCode, uniformSize, format) {
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
        targets: [{ format: format }]
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

    // Generate plane from -1 to 1 on X and Z
    for (let z = 0; z <= detail; z++) {
        const t = z / detail;
        for (let x = 0; x <= detail; x++) {
            const s = x / detail;
            positions.push(2 * s - 1, 2 * t - 1, 0);
        }
    }

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
        
        // Skybox
        @binding(7) @group(0) var skySampler : sampler;
        @binding(8) @group(0) var skyTexture : texture_cube<f32>;

        struct VertexOutput {
          @builtin(position) position : vec4f,
          @location(0) worldPos : vec3f,
        }

        @vertex
        fn vs_main(@location(0) position : vec3f) -> VertexOutput {
          var output : VertexOutput;
          
          let uv = position.xy * 0.5 + 0.5;
          let info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
          
          var pos = position.xzy; 
          pos.y = info.r;
          
          output.worldPos = pos;
          output.position = commonUniforms.viewProjectionMatrix * vec4f(pos, 1.0);
          
          return output;
        }

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
            let sphereRadius = sphere.radius;
            let dist_x = (1.0 + sphereRadius - abs(point.x)) / sphereRadius;
            let dist_z = (1.0 + sphereRadius - abs(point.z)) / sphereRadius;
            let dist_y = (point.y + 1.0 + sphereRadius) / sphereRadius;
            color *= 1.0 - 0.9 / pow(max(0.1, dist_x), 3.0);
            color *= 1.0 - 0.9 / pow(max(0.1, dist_z), 3.0);
            color *= 1.0 - 0.9 / pow(max(0.1, dist_y), 3.0);
            return color;
        }

        fn getWallColor(point: vec3f) -> vec3f {
            var wallColor : vec3f;
            if (abs(point.x) > 0.999) {
                wallColor = textureSampleLevel(tileTexture, tileSampler, point.yz * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
            } else if (abs(point.z) > 0.999) {
                wallColor = textureSampleLevel(tileTexture, tileSampler, point.yx * 0.5 + vec2f(1.0, 0.5), 0.0).rgb;
            } else {
                wallColor = textureSampleLevel(tileTexture, tileSampler, point.xz * 0.5 + 0.5, 0.0).rgb;
            }
            return wallColor * 0.5;
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
                // Use Skybox
                color = textureSampleLevel(skyTexture, skySampler, ray, 0.0).rgb;
                // Add sun specular
                let sunDir = normalize(light.direction);
                let spec = pow(max(0.0, dot(sunDir, ray)), 5000.0);
                color += vec3f(spec) * vec3f(10.0, 8.0, 6.0);
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
            
            let uv = worldPos.xz * 0.5 + 0.5;
            let info = textureSample(waterTexture, waterSampler, uv);
            
            let normal = vec3f(info.b, sqrt(max(0.0, 1.0 - info.b*info.b - info.a*info.a)), info.a);
            
            let incomingRay = normalize(worldPos - commonUniforms.eyePosition);
            
            let reflectedRay = reflect(incomingRay, normal);
            let refractedRay = refract(incomingRay, normal, IOR_AIR / IOR_WATER);
            
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
            cullMode: 'none',
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
            { binding: 6, resource: this.textureA.createView() }, // Use textureA for reading in render
            { binding: 7, resource: this.skySampler },
            { binding: 8, resource: this.skyTexture.createView({dimension: 'cube'}) }
        ]
    });
  }

  renderSurface(passEncoder) {
    const bindGroup = this.device.createBindGroup({
        layout: this.surfacePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: this.commonUniformBuffer } },
            { binding: 1, resource: { buffer: this.lightUniformBuffer } },
            { binding: 2, resource: { buffer: this.sphereUniformBuffer } },
            { binding: 3, resource: this.tileSampler },
            { binding: 4, resource: this.tileTexture.createView() },
            { binding: 5, resource: this.sampler },
            { binding: 6, resource: this.textureA.createView() },
            { binding: 7, resource: this.skySampler },
            { binding: 8, resource: this.skyTexture.createView({dimension: 'cube'}) }
        ]
    });

    passEncoder.setPipeline(this.surfacePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.setVertexBuffer(0, this.positionBuffer);
    passEncoder.setIndexBuffer(this.indexBuffer, 'uint32');
    passEncoder.drawIndexed(this.vertexCount);
  }

  // --- Caustics ---

  createCausticsPipeline() {
    const shaderModule = this.device.createShaderModule({
        label: 'Caustics Shader',
        code: `
        struct LightUniforms {
           direction : vec3f,
        }
        @binding(0) @group(0) var<uniform> light : LightUniforms;

        struct SphereUniforms {
          center : vec3f,
          radius : f32,
        }
        @binding(1) @group(0) var<uniform> sphere : SphereUniforms;

        @binding(2) @group(0) var waterSampler : sampler;
        @binding(3) @group(0) var waterTexture : texture_2d<f32>;

        struct VertexOutput {
          @builtin(position) position : vec4f,
          @location(0) oldPos : vec3f,
          @location(1) newPos : vec3f,
          @location(2) ray : vec3f,
        }

        fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
          let tMin = (cubeMin - origin) / ray;
          let tMax = (cubeMax - origin) / ray;
          let t1 = min(tMin, tMax);
          let t2 = max(tMin, tMax);
          let tNear = max(max(t1.x, t1.y), t1.z);
          let tFar = min(min(t2.x, t2.y), t2.z);
          return vec2f(tNear, tFar);
        }

        fn project(origin: vec3f, ray: vec3f, refractedLight: vec3f) -> vec3f {
            let poolHeight = 1.0;
            var point = origin;
            let tcube = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
            point += ray * tcube.y;
            let tplane = (-point.y - 1.0) / refractedLight.y;
            return point + refractedLight * tplane;
        }

        @vertex
        fn vs_main(@location(0) position : vec3f) -> VertexOutput {
          var output : VertexOutput;
          let uv = position.xy * 0.5 + 0.5;
          
          let info = textureSampleLevel(waterTexture, waterSampler, uv, 0.0);
          
          let normal = vec3f(info.b, sqrt(max(0.0, 1.0 - info.b*info.b - info.a*info.a)), info.a);
          
          let IOR_AIR = 1.0;
          let IOR_WATER = 1.333;
          let lightDir = normalize(light.direction);
          
          // Note: WebGL demo uses -light for refraction?
          // "refract(-light, vec3(0.0, 1.0, 0.0), ...)"
          // My lightDir is direction.
          let refractedLight = refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
          let ray = refract(-lightDir, normal, IOR_AIR / IOR_WATER);
          
          let origin = vec3f(position.x, 0.0, position.y); // Plane on XZ in WebGL? Or XY swizzled?
          // In CreateSurfaceMesh, I pushed x, y (2s-1, 2t-1). Z=0.
          // VS Main logic: pos = position.xzy.
          // Caustics shader uses gl_Vertex.xzy.
          let pos = vec3f(position.x, 0.0, position.y); // Swizzle for calculation
          
          output.oldPos = project(pos, refractedLight, refractedLight);
          output.newPos = project(pos + vec3f(0.0, info.r, 0.0), ray, refractedLight);
          output.ray = ray;
          
          // Map to texture space
          // "gl_Position = vec4(0.75 * (newPos.xz + refractedLight.xz / refractedLight.y), 0.0, 1.0);"
          let projectedPos = 0.75 * (output.newPos.xz + refractedLight.xz / refractedLight.y);
          output.position = vec4f(projectedPos, 0.0, 1.0);
          
          return output;
        }

        @fragment
        fn fs_main(@location(0) oldPos : vec3f, @location(1) newPos : vec3f, @location(2) ray : vec3f) -> @location(0) vec4f {
            // Intensity = oldArea / newArea
            // Use dpdx, dpdy
            let oldArea = length(dpdx(oldPos)) * length(dpdy(oldPos));
            let newArea = length(dpdx(newPos)) * length(dpdy(newPos));
            
            var intensity = oldArea / newArea * 0.2;
            
            // Sphere Shadow Logic (Green channel)
            let IOR_AIR = 1.0;
            let IOR_WATER = 1.333;
            let lightDir = normalize(light.direction);
            let refractedLight = refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
            
            let dir = (sphere.center - newPos) / sphere.radius;
            let area = cross(dir, refractedLight);
            var shadow = dot(area, area);
            let dist = dot(dir, -refractedLight);
            
            shadow = 1.0 + (shadow - 1.0) / (0.05 + dist * 0.025);
            shadow = clamp(1.0 / (1.0 + exp(-shadow)), 0.0, 1.0);
            shadow = mix(1.0, shadow, clamp(dist * 2.0, 0.0, 1.0));
            
            // Rim shadow
            let poolHeight = 1.0;
            let t = intersectCube(newPos, -refractedLight, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
            intensity *= 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (newPos.y - refractedLight.y * t.y - 2.0 / 12.0)));
            
            return vec4f(intensity, shadow, 0.0, 1.0);
        }
        `
    });

    this.causticsPipeline = this.device.createRenderPipeline({
        label: 'Caustics Pipeline',
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
            targets: [{ 
                format: 'rgba8unorm',
                blend: {
                    color: {
                        operation: 'add',
                        srcFactor: 'one',
                        dstFactor: 'one',
                    },
                    alpha: {
                        operation: 'add',
                        srcFactor: 'one',
                        dstFactor: 'one',
                    }
                }
            }]
        },
        primitive: {
            topology: 'triangle-list',
        }
    });
  }

  updateCaustics() {
    const bindGroup = this.device.createBindGroup({
        layout: this.causticsPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: this.lightUniformBuffer } },
            { binding: 1, resource: { buffer: this.sphereUniformBuffer } },
            { binding: 2, resource: this.sampler },
            { binding: 3, resource: this.textureA.createView() }
        ]
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: this.causticsTexture.createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 0 }
        }]
    });

    pass.setPipeline(this.causticsPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.setVertexBuffer(0, this.positionBuffer); // Reuse surface mesh
    pass.setIndexBuffer(this.indexBuffer, 'uint32');
    pass.drawIndexed(this.vertexCount);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }
}