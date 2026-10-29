struct Params {
  out_width: u32,
  out_height: u32,
  visible_width: u32,
  visible_height: u32,
  mode: u32,
  collect_stats: u32,
  gain: f32,
  _pad0: f32,
}

@group(0) @binding(0) var camera_frame: texture_external;
@group(0) @binding(1) var camera_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> histogram: array<atomic<u32>, 256>;

fn linearize(value: f32) -> f32 {
  if value <= 0.04045 {
    return value / 12.92;
  }
  return pow((value + 0.055) / 1.055, 2.4);
}

fn linear_rgb(uv: vec2f) -> vec3f {
  let encoded = textureSampleBaseClampToEdge(camera_frame, camera_sampler, uv).rgb;
  return vec3f(linearize(encoded.r), linearize(encoded.g), linearize(encoded.b));
}

fn record_luminance(rgb: vec3f) {
  if params.collect_stats == 0u { return; }
  let luminance = dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
  let bin = u32(clamp(luminance, 0.0, 1.0) * 255.0);
  atomicAdd(&histogram[bin], 1u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if gid.x >= params.out_width || gid.y >= params.out_height { return; }
  let plane = params.out_width * params.out_height;
  let index = gid.y * params.out_width + gid.x;

  if params.mode == 0u {
    // Learned adapter input: full-resolution, inverse-sRGB linear NCHW.
    let crop = (vec2f(params.out_width, params.out_height) -
                vec2f(params.visible_width, params.visible_height)) * 0.5;
    let pixel = vec2f(gid.xy) - crop;
    let uv = (pixel + vec2f(0.5)) / vec2f(params.visible_width, params.visible_height);
    let rgb = linear_rgb(uv);
    output[index] = rgb.r;
    output[plane + index] = rgb.g;
    output[2u * plane + index] = rgb.b;
    record_luminance(rgb);
    return;
  }

  // Deterministic RGGB baseline. Each invocation emits one packed Bayer cell.
  let visible = vec2f(params.visible_width, params.visible_height);
  let padded = vec2f(params.out_width * 2u, params.out_height * 2u);
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
