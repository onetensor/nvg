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

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
  let visible = vec2f(params.visible_width, params.visible_height);
  let crop = vec2f(params.crop_x, params.crop_y);
  let pixel = crop + input.uv * visible - vec2f(0.5);
  let rgb = vec3f(sample_bilinear(pixel, 0u), sample_bilinear(pixel, 1u), sample_bilinear(pixel, 2u));
  return vec4f(clamp(rgb, vec3f(0.0), vec3f(1.0)), 1.0);
}
