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
//
// The history is read at a global shift found by align.wgsl (zero when
// alignment is off) and written to a second buffer, so hand shake moves the
// history with the scene instead of resetting it. Cells whose shifted source
// lies outside the previous frame start over.
@group(0) @binding(0) var<storage, read_write> frame: array<f32>;
@group(0) @binding(1) var<storage, read> history_in: array<f32>;
@group(0) @binding(2) var<storage, read> frames_in: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> motion_score: array<f32>;
@group(0) @binding(5) var<storage, read> shift: array<i32, 4>;
@group(0) @binding(6) var<storage, read_write> history_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> frames_out: array<f32>;

/// The history cell aligned with cell `i`, or -1 outside the previous frame.
fn source_cell(i: u32) -> i32 {
  let height = params.plane / params.width;
  let x = i32(i % params.width) + shift[0];
  let y = i32(i / params.width) + shift[1];
  if x < 0 || y < 0 || x >= i32(params.width) || y >= i32(height) { return -1; }
  return y * i32(params.width) + x;
}

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
      let source = source_cell(cell);
      if source < 0 { continue; }
      for (var c = 0u; c < params.channels; c += 1u) {
        sum += sqrt(max(frame[c * params.plane + cell], 0.0)) -
          sqrt(max(history_in[c * params.plane + u32(source)], 0.0));
        count += 1.0;
      }
    }
  }
  motion_score[i] = select(abs(sum / count), 1e9, count == 0.0);
}

// Pass 2: accumulate each cell using its motion score.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if i >= params.plane { return; }
  let source = source_cell(i);

  if params.initialized == 0u || source < 0 {
    for (var c = 0u; c < params.channels; c += 1u) {
      history_out[c * params.plane + i] = frame[c * params.plane + i];
    }
    frames_out[i] = 1.0;
    if params.noise_plane != 0u { frame[params.channels * params.plane + i] = 1.0; }
    return;
  }

  let still = 1.0 - smoothstep(params.motion_low, params.motion_high, motion_score[i]);
  let count = 1.0 + (min(frames_in[u32(source)] + 1.0, params.max_frames) - 1.0) * still;
  for (var c = 0u; c < params.channels; c += 1u) {
    let index = c * params.plane + i;
    let previous = history_in[c * params.plane + u32(source)];
    let value = previous + (frame[index] - previous) / count;
    history_out[index] = value;
    frame[index] = value;
  }
  frames_out[i] = count;
  if params.noise_plane != 0u { frame[params.channels * params.plane + i] = inverseSqrt(count); }
}
