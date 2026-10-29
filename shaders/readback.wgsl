struct Params {
  full_width: u32,
  full_height: u32,
  visible_width: u32,
  visible_height: u32,
  crop_x: u32,
  crop_y: u32,
  _pad0: u32,
  _pad1: u32,
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
  let r = u32(round(clamp(image[source], 0.0, 1.0) * 255.0));
  let g = u32(round(clamp(image[plane + source], 0.0, 1.0) * 255.0));
  let b = u32(round(clamp(image[2u * plane + source], 0.0, 1.0) * 255.0));

  // WebGPU and the browser expose mapped bytes in little-endian order.
  rgba[gid.y * params.visible_width + gid.x] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
