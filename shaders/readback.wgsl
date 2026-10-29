struct Params {
  full_width: u32,
  full_height: u32,
  visible_width: u32,
  visible_height: u32,
  crop_x: u32,
  crop_y: u32,
  // 3: RGB planes; 1: a single luminance plane, shown through `tint`.
  channels: u32,
  // Mono display: 0 = grayscale, 1 = green phosphor.
  tint: u32,
}

// Green phosphor (P43-like), scaled so full luminance stays within range.
const PHOSPHOR = vec3f(0.42, 1.0, 0.36);

fn mono_color(value: f32) -> vec3f {
  if params.tint == 1u { return value * PHOSPHOR; }
  return vec3f(value);
}

@group(0) @binding(0) var<storage, read> image: array<f32>;
@group(0) @binding(1) var<storage, read_write> rgba: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if gid.x >= params.visible_width || gid.y >= params.visible_height { return; }

  let source_x = gid.x + params.crop_x;
  let source_y = gid.y + params.crop_y;
  let source = source_y * params.full_width + source_x;
  let plane = params.full_width * params.full_height;
  var rgb = vec3f(image[source]);
  if params.channels == 1u {
    rgb = mono_color(image[source]);
  } else {
    rgb = vec3f(image[source], image[plane + source], image[2u * plane + source]);
  }
  let encoded = vec3u(round(clamp(rgb, vec3f(0.0), vec3f(1.0)) * 255.0));
  let r = encoded.r;
  let g = encoded.g;
  let b = encoded.b;

  // WebGPU and the browser expose mapped bytes in little-endian order.
  rgba[gid.y * params.visible_width + gid.x] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
