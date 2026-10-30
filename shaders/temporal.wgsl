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
  // Non-zero: read per-tile shifts (align.wgsl tile_search) instead of the global one.
  use_tiles: u32,
  tile_size: u32,
  _pad0: u32,
  _pad1: u32,
}

// Motion-aware accumulation of the linear model input. Static pixels average
// up to max_frames frames (noise std falls by sqrt(N)); moving pixels restart
// from the current frame so motion does not ghost. The result replaces the
// frame in place so the model consumes the accumulated image.
//
// The history is read where align.wgsl says each cell's content was (a global
// shift, or tile shifts interpolated into a smooth warp; zero when alignment
// is off) and written to a second buffer, so hand shake and walking move the
// history with the scene instead of resetting it. Cells whose source lies
// outside the previous frame start over.
@group(0) @binding(0) var<storage, read_write> frame: array<f32>;
@group(0) @binding(1) var<storage, read> history_in: array<f32>;
@group(0) @binding(2) var<storage, read> frames_in: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> motion_score: array<f32>;
@group(0) @binding(5) var<storage, read> shift: array<i32, 4>;
@group(0) @binding(6) var<storage, read_write> history_out: array<f32>;
@group(0) @binding(7) var<storage, read_write> frames_out: array<f32>;
@group(0) @binding(8) var<storage, read> tile_shift: array<i32>;

/// Where cell `i`'s content was in the previous frame, in (fractional) cells.
/// With tiles, the per-tile shifts are interpolated bilinearly between tile
/// centres, so rotation and zoom become a smooth warp instead of steps.
fn source_position(i: u32) -> vec2f {
  let p = vec2f(f32(i % params.width), f32(i / params.width));
  if params.use_tiles == 0u { return p + vec2f(f32(shift[0]), f32(shift[1])); }
  let height = params.plane / params.width;
  let tiles = vec2i(i32((params.width + params.tile_size - 1u) / params.tile_size),
                    i32((height + params.tile_size - 1u) / params.tile_size));
  let t = (p + vec2f(0.5)) / f32(params.tile_size) - vec2f(0.5);
  let base = vec2i(floor(t));
  let f = t - floor(t);
  var offset = vec2f(0.0);
  for (var k = 0; k < 4; k += 1) {
    let corner = clamp(base + vec2i(k % 2, k / 2), vec2i(0), tiles - vec2i(1));
    let tile = u32(corner.y * tiles.x + corner.x);
    let weight = select(1.0 - f.x, f.x, k % 2 == 1) * select(1.0 - f.y, f.y, k / 2 == 1);
    offset += weight * vec2f(f32(tile_shift[2u * tile]), f32(tile_shift[2u * tile + 1u]));
  }
  return p + offset;
}

fn inside(position: vec2f) -> bool {
  let height = params.plane / params.width;
  return all(position >= vec2f(0.0)) && position.x <= f32(params.width - 1u) && position.y <= f32(height - 1u);
}

/// Bilinear sample of history plane `c` at an in-bounds position.
fn history_at(c: u32, position: vec2f) -> f32 {
  let height = params.plane / params.width;
  let base = vec2u(floor(position));
  let next = min(base + vec2u(1u), vec2u(params.width - 1u, height - 1u));
  let f = position - floor(position);
  let row0 = c * params.plane + base.y * params.width;
  let row1 = c * params.plane + next.y * params.width;
  let top = mix(history_in[row0 + base.x], history_in[row0 + next.x], f.x);
  let bottom = mix(history_in[row1 + base.x], history_in[row1 + next.x], f.x);
  return mix(top, bottom, f.y);
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
      let source = source_position(cell);
      if !inside(source) { continue; }
      for (var c = 0u; c < params.channels; c += 1u) {
        sum += sqrt(max(frame[c * params.plane + cell], 0.0)) - sqrt(max(history_at(c, source), 0.0));
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
  let source = source_position(i);

  if params.initialized == 0u || !inside(source) {
    for (var c = 0u; c < params.channels; c += 1u) {
      history_out[c * params.plane + i] = frame[c * params.plane + i];
    }
    frames_out[i] = 1.0;
    if params.noise_plane != 0u { frame[params.channels * params.plane + i] = 1.0; }
    return;
  }

  let still = 1.0 - smoothstep(params.motion_low, params.motion_high, motion_score[i]);
  let nearest = vec2u(round(source));
  let count = 1.0 + (min(frames_in[nearest.y * params.width + nearest.x] + 1.0, params.max_frames) - 1.0) * still;
  for (var c = 0u; c < params.channels; c += 1u) {
    let index = c * params.plane + i;
    let previous = history_at(c, source);
    let value = previous + (frame[index] - previous) / count;
    history_out[index] = value;
    frame[index] = value;
  }
  frames_out[i] = count;
  if params.noise_plane != 0u { frame[params.channels * params.plane + i] = inverseSqrt(count); }
}
