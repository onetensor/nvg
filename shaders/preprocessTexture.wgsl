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

fn linearize(value: f32) -> f32 {
  if value <= 0.04045 {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
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

fn linear_rgb(uv: vec2f) -> vec3f {
  let covered_uv = (uv - vec2f(0.5)) * vec2f(params.uv_scale_x, params.uv_scale_y) + vec2f(0.5);
  let encoded = textureSampleLevel(camera_frame, camera_sampler, buffer_uv(covered_uv), 0.0).rgb;
  return vec3f(linearize(encoded.r), linearize(encoded.g), linearize(encoded.b));
}

fn record_luminance(rgb: vec3f) {
  if params.collect_stats == 0u { return; }
  let luminance = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
  let bin = u32(clamp(luminance, 0.0, 1.0) * 255.0);
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

  let visible = vec2f(vec2u(params.visible_width, params.visible_height));
  let padded = vec2f(vec2u(params.out_width * 2u, params.out_height * 2u));
  let crop = (padded - visible) * 0.5;
  let base = vec2f(gid.xy * 2u) - crop;
  let p00 = linear_rgb((base + vec2f(0.5, 0.5)) / visible);
  let p01 = linear_rgb((base + vec2f(1.5, 0.5)) / visible);
  let p10 = linear_rgb((base + vec2f(0.5, 1.5)) / visible);
  let p11 = linear_rgb((base + vec2f(1.5, 1.5)) / visible);

  output[index] = clamp(p00.r * params.gain, 0.0, 1.0);
  output[plane + index] = clamp(p01.g * params.gain, 0.0, 1.0);
  output[2u * plane + index] = clamp(p11.b * params.gain, 0.0, 1.0);
  output[3u * plane + index] = clamp(p10.g * params.gain, 0.0, 1.0);
  record_luminance((p00 + p01 + p10 + p11) * 0.25);
}

// Orientation calibration: 16x16 cell averages of the copied buffer in buffer
// space (no cover crop or rotation), compared on the CPU against an upright
// Canvas2D reference. Values stay sRGB-encoded to match getImageData().
@group(0) @binding(5) var<storage, read_write> thumbnail_cells: array<f32, 768>;

@compute @workgroup_size(16, 16)
fn thumbnail(@builtin(local_invocation_id) lid: vec3u) {
  var sum = vec3f(0.0);
  for (var j = 0u; j < 8u; j += 1u) {
    for (var i = 0u; i < 8u; i += 1u) {
      let uv = (vec2f(lid.xy) + (vec2f(vec2u(i, j)) + vec2f(0.5)) / 8.0) / 16.0;
      sum += textureSampleLevel(camera_frame, camera_sampler, uv, 0.0).rgb;
    }
  }
  let cell = (lid.y * 16u + lid.x) * 3u;
  let average = sum / 64.0;
  thumbnail_cells[cell] = average.r;
  thumbnail_cells[cell + 1u] = average.g;
  thumbnail_cells[cell + 2u] = average.b;
}
