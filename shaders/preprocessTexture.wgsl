struct Params {
  out_width: u32,
  out_height: u32,
  visible_width: u32,
  visible_height: u32,
  mode: u32,
  collect_stats: u32,
  gain: f32,
  quarter_turns: u32,
  uv_scale_x: f32,
  uv_scale_y: f32,
  _pad1: f32,
  _pad2: f32,
}

@group(0) @binding(0) var camera_frame: texture_2d<f32>;
@group(0) @binding(1) var camera_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> histogram: array<atomic<u32>, 257>;

// The copied camera texture is rgba8unorm-srgb, so every sample (and the
// bilinear filter itself) is already in linear light.
fn srgb_encode(value: f32) -> f32 {
  if value <= 0.0031308 {
    return value * 12.92;
  }
  return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

// Maps upright display UV to the copied buffer, which Safari may deliver as the
// unrotated sensor image. quarter_turns is the clockwise display rotation.
fn buffer_uv(uv: vec2f) -> vec2f {
  switch params.quarter_turns {
    case 1u: { return vec2f(uv.y, 1.0 - uv.x); }
    case 2u: { return vec2f(1.0) - uv; }
    case 3u: { return vec2f(1.0 - uv.y, uv.x); }
    default: { return uv; }
  }
}

// Area-filtered sample of one output pixel. A single bilinear tap at 4x
// decimation reads only a 2x2 corner of each 4x4 camera block, aliasing and
// discarding most of the free noise averaging; this tiles the whole footprint
// with bilinear taps (~2x2 texels each), all averaged in linear light.
fn linear_rgb(uv: vec2f) -> vec3f {
  let scale = vec2f(params.uv_scale_x, params.uv_scale_y);
  let footprint = scale / vec2f(vec2u(params.visible_width, params.visible_height));
  var texels = vec2f(textureDimensions(camera_frame));
  if params.quarter_turns % 2u == 1u { texels = texels.yx; }
  let taps = vec2u(clamp(ceil(footprint * texels * 0.5), vec2f(1.0), vec2f(4.0)));
  let center = (uv - vec2f(0.5)) * scale + vec2f(0.5);
  var sum = vec3f(0.0);
  for (var j = 0u; j < taps.y; j += 1u) {
    for (var i = 0u; i < taps.x; i += 1u) {
      let offset = ((vec2f(vec2u(i, j)) + vec2f(0.5)) / vec2f(taps) - vec2f(0.5)) * footprint;
      let covered = center + offset;
      sum += textureSampleLevel(camera_frame, camera_sampler, buffer_uv(covered), 0.0).rgb;
    }
  }
  return sum / f32(taps.x * taps.y);
}

fn record_luminance(rgb: vec3f) {
  if params.collect_stats == 0u { return; }
  let luminance = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
  // sqrt-spaced bins resolve the dark scenes auto gain must expose: linear
  // 1/255-wide bins put every median below ~sRGB 13/255 into bin 0.
  let bin = min(u32(sqrt(clamp(luminance, 0.0, 1.0)) * 255.0), 255u);
  atomicAdd(&histogram[bin], 1u);
  if all(rgb == vec3f(0.0)) { atomicAdd(&histogram[256], 1u); }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if gid.x >= params.out_width || gid.y >= params.out_height { return; }
  let plane = params.out_width * params.out_height;
  let index = gid.y * params.out_width + gid.x;

  if params.mode == 0u {
    let crop = (vec2f(vec2u(params.out_width, params.out_height)) -
                vec2f(vec2u(params.visible_width, params.visible_height))) * 0.5;
    let pixel = vec2f(gid.xy) - crop;
    let uv = (pixel + vec2f(0.5)) / vec2f(vec2u(params.visible_width, params.visible_height));
    let rgb = linear_rgb(uv);
    output[index] = rgb.r;
    output[plane + index] = rgb.g;
    output[2u * plane + index] = rgb.b;
    record_luminance(rgb);
    return;
  }

  // Packed modes: each invocation covers one 2x2 cell of the padded frame.
  let visible = vec2f(vec2u(params.visible_width, params.visible_height));
  let padded = vec2f(vec2u(params.out_width * 2u, params.out_height * 2u));
  let crop = (padded - visible) * 0.5;
  let base = vec2f(gid.xy * 2u) - crop;
  let p00 = linear_rgb((base + vec2f(0.5, 0.5)) / visible);
  let p01 = linear_rgb((base + vec2f(1.5, 0.5)) / visible);
  let p10 = linear_rgb((base + vec2f(0.5, 1.5)) / visible);
  let p11 = linear_rgb((base + vec2f(1.5, 1.5)) / visible);
  record_luminance((p00 + p01 + p10 + p11) * 0.25);

  if params.mode == 2u {
    // Direct models: PixelUnshuffle(2) layout, channel-major then (00, 01, 10, 11).
    // Gain is applied after temporal accumulation, so values stay unamplified.
    var cells = array<vec3f, 4>(p00, p01, p10, p11);
    for (var k = 0u; k < 4u; k += 1u) {
      output[k * plane + index] = cells[k].r;
      output[(4u + k) * plane + index] = cells[k].g;
      output[(8u + k) * plane + index] = cells[k].b;
    }
    return;
  }

  // Deterministic RGGB baseline (R00, G01, B11, G10), amplified here.
  output[index] = clamp(p00.r * params.gain, 0.0, 1.0);
  output[plane + index] = clamp(p01.g * params.gain, 0.0, 1.0);
  output[2u * plane + index] = clamp(p11.b * params.gain, 0.0, 1.0);
  output[3u * plane + index] = clamp(p10.g * params.gain, 0.0, 1.0);
}

// Orientation calibration: 16x16 cell averages of the copied buffer in buffer
// space (no cover crop or rotation), compared on the CPU against an upright
// Canvas2D reference. Samples are re-encoded to sRGB to match getImageData().
@group(0) @binding(5) var<storage, read_write> thumbnail_cells: array<f32, 768>;

@compute @workgroup_size(16, 16)
fn thumbnail(@builtin(local_invocation_id) lid: vec3u) {
  var sum = vec3f(0.0);
  for (var j = 0u; j < 8u; j += 1u) {
    for (var i = 0u; i < 8u; i += 1u) {
      let uv = (vec2f(lid.xy) + (vec2f(vec2u(i, j)) + vec2f(0.5)) / 8.0) / 16.0;
      let linear = textureSampleLevel(camera_frame, camera_sampler, uv, 0.0).rgb;
      sum += vec3f(srgb_encode(linear.r), srgb_encode(linear.g), srgb_encode(linear.b));
    }
  }
  let cell = (lid.y * 16u + lid.x) * 3u;
  let average = sum / 64.0;
  thumbnail_cells[cell] = average.r;
  thumbnail_cells[cell + 1u] = average.g;
  thumbnail_cells[cell + 2u] = average.b;
}
