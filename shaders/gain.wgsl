struct Params {
  count: u32,
  gain: f32,
  // Elements after `count` up to `total` are copied unamplified (the direct
  // model's noise-level plane).
  total: u32,
  // Non-zero: scale the copied noise plane by sqrt(gain) * sigma[0], the
  // per-frame noise estimate (models trained with noise_input "estimate").
  scale_noise: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> sigma: array<f32, 4>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if gid.x >= params.total { return; }
  if gid.x >= params.count {
    let scale = select(1.0, sqrt(params.gain) * sigma[0], params.scale_noise != 0u);
    output[gid.x] = input[gid.x] * scale;
    return;
  }
  output[gid.x] = clamp(input[gid.x] * params.gain, 0.0, 1.0);
}
