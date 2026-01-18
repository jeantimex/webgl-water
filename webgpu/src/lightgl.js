// Ported from lightgl.js and adapted for wgpu-matrix
// http://github.com/evanw/lightgl.js/

import { mat4, vec3, vec4 } from 'wgpu-matrix';

export class Vector {
  constructor(x, y, z) {
    this.x = x || 0;
    this.y = y || 0;
    this.z = z || 0;
  }

  negative() {
    return new Vector(-this.x, -this.y, -this.z);
  }

  add(v) {
    if (v instanceof Vector) return new Vector(this.x + v.x, this.y + v.y, this.z + v.z);
    else return new Vector(this.x + v, this.y + v, this.z + v);
  }

  subtract(v) {
    if (v instanceof Vector) return new Vector(this.x - v.x, this.y - v.y, this.z - v.z);
    else return new Vector(this.x - v, this.y - v, this.z - v);
  }

  multiply(v) {
    if (v instanceof Vector) return new Vector(this.x * v.x, this.y * v.y, this.z * v.z);
    else return new Vector(this.x * v, this.y * v, this.z * v);
  }

  divide(v) {
    if (v instanceof Vector) return new Vector(this.x / v.x, this.y / v.y, this.z / v.z);
    else return new Vector(this.x / v, this.y / v, this.z / v);
  }

  equals(v) {
    return this.x == v.x && this.y == v.y && this.z == v.z;
  }

  dot(v) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }

  cross(v) {
    return new Vector(
      this.y * v.z - this.z * v.y,
      this.z * v.x - this.x * v.z,
      this.x * v.y - this.y * v.x
    );
  }

  length() {
    return Math.sqrt(this.dot(this));
  }

  unit() {
    return this.divide(this.length());
  }

  min() {
    return Math.min(Math.min(this.x, this.y), this.z);
  }

  max() {
    return Math.max(Math.max(this.x, this.y), this.z);
  }

  toAngles() {
    return {
      theta: Math.atan2(this.z, this.x),
      phi: Math.asin(this.y / this.length())
    };
  }

  angleTo(a) {
    return Math.acos(this.dot(a) / (this.length() * a.length()));
  }

  toArray(n) {
    return [this.x, this.y, this.z].slice(0, n || 3);
  }

  clone() {
    return new Vector(this.x, this.y, this.z);
  }

  init(x, y, z) {
    this.x = x; this.y = y; this.z = z;
    return this;
  }

  static fromArray(a) {
    return new Vector(a[0], a[1], a[2]);
  }

  static lerp(a, b, t) {
    return a.add(b.subtract(a).multiply(t));
  }

  static min(a, b) {
    return new Vector(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.min(a.z, b.z));
  }

  static max(a, b) {
    return new Vector(Math.max(a.x, b.x), Math.max(a.y, b.y), Math.max(a.z, b.z));
  }

  static fromAngles(theta, phi) {
    return new Vector(Math.cos(phi) * Math.cos(theta), Math.sin(phi), Math.cos(phi) * Math.sin(theta));
  }
}

export class HitTest {
  constructor(t, hit, normal) {
    this.t = arguments.length ? t : Number.MAX_VALUE;
    this.hit = hit;
    this.normal = normal;
  }

  mergeWith(other) {
    if (other.t > 0 && other.t < this.t) {
      this.t = other.t;
      this.hit = other.hit;
      this.normal = other.normal;
    }
  }
}

export class Raytracer {
  constructor(viewMatrix, projectionMatrix, viewport) {
    // viewMatrix and projectionMatrix are Float32Array (wgpu-matrix)
    this.viewport = viewport;

    // Calculate Eye Position from View Matrix
    // Eye is origin (0,0,0) in Camera space, transformed to World.
    // World = View^-1 * Camera.
    // Eye = View^-1 * [0,0,0,1].
    const invView = mat4.invert(viewMatrix);
    const eyeVec = vec3.transformMat4([0,0,0], invView);
    this.eye = new Vector(eyeVec[0], eyeVec[1], eyeVec[2]);

    // Calculate View-Projection Inverse for unProject
    this.invViewProj = mat4.invert(mat4.multiply(projectionMatrix, viewMatrix));

    // Precalculate Corner Rays
    const minX = viewport[0], maxX = minX + viewport[2];
    const minY = viewport[1], maxY = minY + viewport[3];

    // unProject corners. Z=1 (Far).
    // Note: WebGPU Clip Z is 0..1.
    // For ray direction, we can unproject a point on Far plane.
    this.ray00 = this.unProject(minX, minY, 1).subtract(this.eye);
    this.ray10 = this.unProject(maxX, minY, 1).subtract(this.eye);
    this.ray01 = this.unProject(minX, maxY, 1).subtract(this.eye);
    this.ray11 = this.unProject(maxX, maxY, 1).subtract(this.eye);
  }

  unProject(winX, winY, winZ) {
    const v = this.viewport;
    // Map window (0..W, 0..H) to NDC (-1..1, -1..1)
    // Note: wgpu-matrix assumes Y-down for some things? 
    // But standard projection logic usually maps:
    // Bottom (-1) to Top (1) if Y-up.
    // Top (-1) to Bottom (1) if Y-down.
    // Let's assume standard GL behavior for now:
    // x: -1 to 1
    // y: -1 (bottom) to 1 (top)
    // But winY 0 is top. So we flip Y.
    
    const x = (winX - v[0]) / v[2] * 2 - 1;
    const y = (1 - (winY - v[1]) / v[3]) * 2 - 1; // Flip Y: 0->1, H->-1.
    const z = winZ; // 0..1

    const ndc = [x, y, z];
    const world = vec3.transformMat4(ndc, this.invViewProj);
    return new Vector(world[0], world[1], world[2]);
  }

  getRayForPixel(x, y) {
    // Interpolate precalculated rays
    // x, y are window coordinates
    const u = (x - this.viewport[0]) / this.viewport[2];
    const v = (y - this.viewport[1]) / this.viewport[3]; // 0 top, 1 bottom
    
    // ray00 is Top-Left?
    // In constructor: minY (0) -> unProject(..., 0).
    // unProject(..., 0) uses `y = (1 - 0) * 2 - 1 = 1` -> Top.
    // So ray00 is Top-Left (minX, minY).
    // ray01 is Bottom-Left (minX, maxY).
    // ray10 is Top-Right.
    // ray11 is Bottom-Right.
    
    // v goes 0 (Top) to 1 (Bottom).
    // Lerp(rayTop, rayBottom, v).
    const rayTop = Vector.lerp(this.ray00, this.ray10, u);
    const rayBottom = Vector.lerp(this.ray01, this.ray11, u);
    
    return Vector.lerp(rayTop, rayBottom, v).unit();
  }

  static hitTestBox(origin, ray, min, max) {
    var tMin = min.subtract(origin).divide(ray);
    var tMax = max.subtract(origin).divide(ray);
    var t1 = Vector.min(tMin, tMax);
    var t2 = Vector.max(tMin, tMax);
    var tNear = t1.max();
    var tFar = t2.min();

    if (tNear > 0 && tNear < tFar) {
      var epsilon = 1.0e-6, hit = origin.add(ray.multiply(tNear));
      min = min.add(epsilon);
      max = max.subtract(epsilon);
      return new HitTest(tNear, hit, new Vector(
        (hit.x > max.x) - (hit.x < min.x),
        (hit.y > max.y) - (hit.y < min.y),
        (hit.z > max.z) - (hit.z < min.z)
      ));
    }

    return null;
  }

  static hitTestSphere(origin, ray, center, radius) {
    var offset = origin.subtract(center);
    var a = ray.dot(ray);
    var b = 2 * ray.dot(offset);
    var c = offset.dot(offset) - radius * radius;
    var discriminant = b * b - 4 * a * c;

    if (discriminant > 0) {
      var t = (-b - Math.sqrt(discriminant)) / (2 * a), hit = origin.add(ray.multiply(t));
      return new HitTest(t, hit, hit.subtract(center).divide(radius));
    }

    return null;
  }

  static hitTestTriangle(origin, ray, a, b, c) {
    var ab = b.subtract(a);
    var ac = c.subtract(a);
    var normal = ab.cross(ac).unit();
    var t = normal.dot(a.subtract(origin)) / normal.dot(ray);

    if (t > 0) {
      var hit = origin.add(ray.multiply(t));
      var toHit = hit.subtract(a);
      var dot00 = ac.dot(ac);
      var dot01 = ac.dot(ab);
      var dot02 = ac.dot(toHit);
      var dot11 = ab.dot(ab);
      var dot12 = ab.dot(toHit);
      var divide = dot00 * dot11 - dot01 * dot01;
      var u = (dot11 * dot02 - dot01 * dot12) / divide;
      var v = (dot00 * dot12 - dot01 * dot02) / divide;
      if (u >= 0 && v >= 0 && u + v <= 1) return new HitTest(t, hit, normal);
    }

    return null;
  }
}