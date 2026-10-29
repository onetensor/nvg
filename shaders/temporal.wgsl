struct Params {
  plane: u32,
  channels: u32,
  initialized: u32,
  // Non-zero: write 1/sqrt(frames averaged) to the plane after `channels`,
  // the noise-level input of 13-channel direct models.
  noise_plane: u32,
  max_frames: f32,
  motion_low: f32,
  motion_high: f32,
  width: u32,
}

// Motion-aware accumulation of the linear model input. Static pixels average
// up to max_frames frames (noise std falls by sqrt(N)); moving pixels restart
// from the current frame so motion does not ghost. The result replaces the
// frame in place so the model consumes the accumulated image.
@group(0) @binding(0) var<storage, read_write> frame: array<f32>;
@group(0) @binding(1) var<storage, read_write> history: array<f32>;
@group(0) @binding(2) var<storage, read_write> frames: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> motion_score: array<f32>;

// Pass 1 (reads frame and history, writes only motion_score): the signed
// change in sqrt space averaged over every channel of the 3x3 cell
// neighbourhood. sqrt roughly stabilizes linear-light noise, and averaging
// ~100 noisy differences lets real motion stand out where a per-channel
// maximum is triggered by noise alone in dark frames.
@compute @workgroup_size(256)
fn motion(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if i >= params.plane { return; }
  let height = params.plane / params.width;
  let x = i32(i % params.width);
  let y = i32(i / params.width);
  var sum = 0.0;
  var count = 0.0;
  for (var dy = -1; dy <= 1; dy += 1) {
    for (var dx = -1; dx <= 1; dx += 1) {
      let nx = u32(clamp(x + dx, 0, i32(params.width) - 1));
      let ny = u32(clamp(y + dy, 0, i32(height) - 1));
      let cell = ny * params.width + nx;
      for (var c = 0u; c < params.channels; c += 1u) {
        let index = c * params.plane + cell;
        sum += sqrt(max(frame[index], 0.0)) - sqrt(max(history[index], 0.0));
        count += 1.0;
      }
    }
  }
  motion_score[i] = abs(sum / count);
}

// Pass 2: accumulate each cell using its motion score.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if i >= params.plane { return; }

  if params.initialized == 0u {
    for (var c = 0u; c < params.channels; c += 1u) {
      history[c * params.plane + i] = frame[c * params.plane + i];
    }
    frames[i] = 1.0;
    if params.noise_plane != 0u { frame[params.channels * params.plane + i] = 1.0; }
    return;
  }

  let still = 1.0 - smoothstep(params.motion_low, params.motion_high, motion_score[i]);
  let count = 1.0 + (min(frames[i] + 1.0, params.max_frames) - 1.0) * still;
  for (var c = 0u; c < params.channels; c += 1u) {
    let index = c * params.plane + i;
    let value = history[index] + (frame[index] - history[index]) / count;
    history[index] = value;
    frame[index] = value;
  }
  frames[i] = count;
  if params.noise_plane != 0u { frame[params.channels * params.plane + i] = inverseSqrt(count); }
}
