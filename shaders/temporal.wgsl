struct Params {
  plane: u32,
  channels: u32,
  initialized: u32,
  _pad0: u32,
  max_frames: f32,
  motion_low: f32,
  motion_high: f32,
  _pad1: f32,
}

// Motion-aware accumulation of the linear model input. Static pixels average
// up to max_frames frames (noise std falls by sqrt(N)); moving pixels restart
// from the current frame so motion does not ghost. The result replaces the
// frame in place so the model consumes the accumulated image.
@group(0) @binding(0) var<storage, read_write> frame: array<f32>;
@group(0) @binding(1) var<storage, read_write> history: array<f32>;
@group(0) @binding(2) var<storage, read_write> frames: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if i >= params.plane { return; }

  if params.initialized == 0u {
    for (var c = 0u; c < params.channels; c += 1u) {
      history[c * params.plane + i] = frame[c * params.plane + i];
    }
    frames[i] = 1.0;
    return;
  }

  // sqrt roughly stabilizes the variance of linear-light noise, so one
  // threshold behaves similarly in shadows and highlights.
  var delta = 0.0;
  for (var c = 0u; c < params.channels; c += 1u) {
    let now = sqrt(max(frame[c * params.plane + i], 0.0));
    let old = sqrt(max(history[c * params.plane + i], 0.0));
    delta = max(delta, abs(now - old));
  }
  let still = 1.0 - smoothstep(params.motion_low, params.motion_high, delta);
  let count = 1.0 + (min(frames[i] + 1.0, params.max_frames) - 1.0) * still;
  for (var c = 0u; c < params.channels; c += 1u) {
    let index = c * params.plane + i;
    let value = history[index] + (frame[index] - history[index]) / count;
    history[index] = value;
    frame[index] = value;
  }
  frames[i] = count;
}
