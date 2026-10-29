struct Params {
  count: u32,
  plane: u32,
  initialized: u32,
  _pad0: u32,
  history: f32,
  _pad1: f32,
  _pad2: f32,
  _pad3: f32,
}

@group(0) @binding(0) var<storage, read> current: array<f32>;
@group(0) @binding(1) var<storage, read_write> previous: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if gid.x >= params.plane { return; }
  let i = gid.x;
  let now = vec3f(current[i], current[params.plane + i], current[2u * params.plane + i]);
  let old = vec3f(previous[i], previous[params.plane + i], previous[2u * params.plane + i]);
  let delta = max(abs(now.r - old.r), max(abs(now.g - old.g), abs(now.b - old.b)));
  let static_weight = 1.0 - smoothstep(0.018, 0.11, delta);
  let weight = select(0.0, params.history * static_weight, params.initialized != 0u);
  let value = mix(now, old, weight);
  previous[i] = value.r;
  previous[params.plane + i] = value.g;
  previous[2u * params.plane + i] = value.b;
}
