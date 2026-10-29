struct Params {
  full_width: u32,
  full_height: u32,
  visible_width: u32,
  visible_height: u32,
  crop_x: u32,
  crop_y: u32,
  // 3: RGB planes; 1: a luminance plane shown through `tint`; 2: luminance
  // plus a plane of object-outline logits.
  channels: u32,
  // Mono display: 0 = grayscale, 1 = green phosphor.
  tint: u32,
  // 0 = off, 1 = learned (outline plane), 2 = classic (Sobel on the output).
  outline_mode: u32,
  outline_strength: f32,
  // "Pop" for luminance output: contrast around the midtones and local
  // contrast against a 5x5 neighbourhood mean (0 = off).
  pop_contrast: f32,
  pop_local: f32,
}

// Green phosphor (P43-like), scaled so full luminance stays within range.
const PHOSPHOR = vec3f(0.42, 1.0, 0.36);

fn mono_color(value: f32) -> vec3f {
  if params.tint == 1u { return value * PHOSPHOR; }
  return vec3f(value);
}

@group(0) @binding(0) var<storage, read> image: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vertex_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  let positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var out: VertexOutput;
  let position = positions[index];
  out.position = vec4f(position, 0.0, 1.0);
  out.uv = vec2f((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
  return out;
}

fn channel_at(xy: vec2i, channel: u32) -> f32 {
  let x = u32(clamp(xy.x, 0, i32(params.full_width) - 1));
  let y = u32(clamp(xy.y, 0, i32(params.full_height) - 1));
  let plane = params.full_width * params.full_height;
  return image[channel * plane + y * params.full_width + x];
}

fn sample_bilinear(pixel: vec2f, channel: u32) -> f32 {
  let base = vec2i(floor(pixel));
  let fraction = fract(pixel);
  let top = mix(channel_at(base, channel), channel_at(base + vec2i(1, 0), channel), fraction.x);
  let bottom = mix(channel_at(base + vec2i(0, 1), channel), channel_at(base + vec2i(1, 1), channel), fraction.x);
  return mix(top, bottom, fraction.y);
}

fn luminance_at(xy: vec2i) -> f32 {
  if params.channels == 3u {
    return dot(vec3f(channel_at(xy, 0u), channel_at(xy, 1u), channel_at(xy, 2u)), vec3f(0.2126, 0.7152, 0.0722));
  }
  return channel_at(xy, 0u);
}

/// Outline strength in [0, 1] around an output pixel (see readback.wgsl).
fn outline_at(pixel: vec2f) -> f32 {
  if params.outline_mode == 1u && params.channels == 2u {
    // Only confident outlines (see readback.wgsl).
    return smoothstep(0.65, 0.9, 1.0 / (1.0 + exp(-sample_bilinear(pixel, 1u))));
  }
  if params.outline_mode == 2u {
    let p = vec2i(round(pixel));
    let gx = luminance_at(p + vec2i(1, -1)) + 2.0 * luminance_at(p + vec2i(1, 0)) + luminance_at(p + vec2i(1, 1))
      - luminance_at(p + vec2i(-1, -1)) - 2.0 * luminance_at(p + vec2i(-1, 0)) - luminance_at(p + vec2i(-1, 1));
    let gy = luminance_at(p + vec2i(-1, 1)) + 2.0 * luminance_at(p + vec2i(0, 1)) + luminance_at(p + vec2i(1, 1))
      - luminance_at(p + vec2i(-1, -1)) - 2.0 * luminance_at(p + vec2i(0, -1)) - luminance_at(p + vec2i(1, -1));
    return smoothstep(0.12, 0.4, length(vec2f(gx, gy)));
  }
  return 0.0;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
  let visible = vec2f(vec2u(params.visible_width, params.visible_height));
  let crop = vec2f(vec2u(params.crop_x, params.crop_y));
  let pixel = crop + input.uv * visible - vec2f(0.5);
  var value = sample_bilinear(pixel, 0u);
  if params.pop_contrast != 0.0 || params.pop_local != 0.0 {
    // See readback.wgsl: local contrast against the 5x5 mean, then S-curve.
    let p = vec2i(round(pixel));
    var mean = 0.0;
    for (var dy = -2; dy <= 2; dy += 1) {
      for (var dx = -2; dx <= 2; dx += 1) { mean += luminance_at(p + vec2i(dx, dy)); }
    }
    mean /= 25.0;
    let local = value + params.pop_local * (value - mean);
    value = clamp(0.5 + (local - 0.5) * (1.0 + params.pop_contrast), 0.0, 1.0);
  }
  var color = mono_color(value);
  if params.channels == 3u {
    color = vec3f(sample_bilinear(pixel, 0u), sample_bilinear(pixel, 1u), sample_bilinear(pixel, 2u));
  }
  // White outlines over green phosphor, amber over grayscale or colour.
  let ink = select(vec3f(1.0, 0.75, 0.2), vec3f(1.0), params.tint == 1u && params.channels != 3u);
  color = mix(color, ink, clamp(outline_at(pixel) * params.outline_strength, 0.0, 1.0));
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
}
