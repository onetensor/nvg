struct Params {
  count: u32,
  gain: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if gid.x >= params.count { return; }
  output[gid.x] = clamp(input[gid.x] * params.gain, 0.0, 1.0);
}
