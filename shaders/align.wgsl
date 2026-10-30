struct Params {
  width: u32,          // packed cells per row
  height: u32,         // packed cell rows
  channels: u32,       // image planes (excludes a noise-level plane)
  coarse_width: u32,   // width / 4
  coarse_height: u32,  // height / 4
  coarse_radius: i32,  // coarse search radius in coarse pixels
  fine_radius: i32,    // fine refinement radius in cells
  tile_size: u32,      // cells per side of a local-alignment tile
}

// Global (translation-only) alignment of the incoming frame to the temporal
// history, so a hand-held phone keeps accumulating instead of resetting at
// every edge. All passes stay on the GPU; `shift` ends up holding the history
// offset (dx, dy) in packed cells such that history(p + shift) ~ frame(p).
@group(0) @binding(0) var<storage, read> frame: array<f32>;
@group(0) @binding(1) var<storage, read> history: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
// Luma maps, sqrt-stabilized: [cell frame | cell history | coarse frame | coarse history].
@group(0) @binding(3) var<storage, read_write> maps: array<f32>;
@group(0) @binding(4) var<storage, read_write> costs: array<f32>;
@group(0) @binding(5) var<storage, read_write> shift: array<i32, 4>;
// Per-tile shifts (dx, dy) in packed cells, refined around the global shift.
@group(0) @binding(6) var<storage, read_write> tile_shift: array<i32>;

fn cell_luma(source: u32, cell: u32) -> f32 {
  let plane = params.width * params.height;
  var sum = 0.0;
  for (var c = 0u; c < params.channels; c += 1u) {
    let value = select(history[c * plane + cell], frame[c * plane + cell], source == 0u);
    sum += sqrt(max(value, 0.0));
  }
  return sum / f32(params.channels);
}

// Pass 1: per-cell luma of frame and history.
@compute @workgroup_size(256)
fn cell_maps(@builtin(global_invocation_id) gid: vec3u) {
  let plane = params.width * params.height;
  if gid.x >= plane { return; }
  maps[gid.x] = cell_luma(0u, gid.x);
  maps[plane + gid.x] = cell_luma(1u, gid.x);
}

// Pass 2: 4x4-cell averages for the coarse search.
@compute @workgroup_size(256)
fn coarse_maps(@builtin(global_invocation_id) gid: vec3u) {
  let plane = params.width * params.height;
  let coarse = params.coarse_width * params.coarse_height;
  if gid.x >= coarse { return; }
  let cx = gid.x % params.coarse_width;
  let cy = gid.x / params.coarse_width;
  var current = 0.0;
  var previous = 0.0;
  for (var y = 0u; y < 4u; y += 1u) {
    for (var x = 0u; x < 4u; x += 1u) {
      let cell = (cy * 4u + y) * params.width + cx * 4u + x;
      current += maps[cell];
      previous += maps[plane + cell];
    }
  }
  maps[2u * plane + gid.x] = current / 16.0;
  maps[2u * plane + coarse + gid.x] = previous / 16.0;
}

var<workgroup> partial: array<f32, 64>;
var<workgroup> partial_count: array<f32, 64>;

fn candidate(index: u32, radius: i32) -> vec2i {
  let side = u32(2 * radius + 1);
  return vec2i(i32(index % side) - radius, i32(index / side) - radius);
}

// Pass 3: one workgroup per coarse shift; mean absolute difference over the overlap.
@compute @workgroup_size(64)
fn coarse_search(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
  let offset = candidate(group.x, params.coarse_radius);
  let plane = params.width * params.height;
  let coarse = params.coarse_width * params.coarse_height;
  var sum = 0.0;
  var count = 0.0;
  for (var i = lane; i < coarse; i += 64u) {
    let p = vec2i(i32(i % params.coarse_width), i32(i / params.coarse_width));
    let q = p + offset;
    if q.x < 0 || q.y < 0 || q.x >= i32(params.coarse_width) || q.y >= i32(params.coarse_height) { continue; }
    sum += abs(maps[2u * plane + i] - maps[2u * plane + coarse + u32(q.y) * params.coarse_width + u32(q.x)]);
    count += 1.0;
  }
  partial[lane] = sum;
  partial_count[lane] = count;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if lane < stride {
      partial[lane] += partial[lane + stride];
      partial_count[lane] += partial_count[lane + stride];
    }
    workgroupBarrier();
  }
  if lane == 0u {
    // Require a substantial overlap so large shifts cannot win on a sliver.
    let enough = partial_count[0] >= 0.5 * f32(coarse);
    costs[group.x] = select(1e9, partial[0] / max(partial_count[0], 1.0), enough);
  }
}

// Pass 4: pick the best coarse shift (single invocation).
@compute @workgroup_size(1)
fn coarse_pick() {
  let side = 2 * params.coarse_radius + 1;
  var best = 0u;
  var best_cost = 1e10;
  for (var i = 0u; i < u32(side * side); i += 1u) {
    // Prefer zero shift on ties so static scenes never drift.
    let cost = costs[i] * select(1.0, 0.999, all(candidate(i, params.coarse_radius) == vec2i(0)));
    if cost < best_cost { best_cost = cost; best = i; }
  }
  let coarse_shift = candidate(best, params.coarse_radius) * 4;
  shift[2] = coarse_shift.x;
  shift[3] = coarse_shift.y;
}

// Pass 5: one workgroup per fine shift around the coarse result, on cell maps.
@compute @workgroup_size(64)
fn fine_search(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
  let offset = vec2i(shift[2], shift[3]) + candidate(group.x, params.fine_radius);
  let plane = params.width * params.height;
  var sum = 0.0;
  var count = 0.0;
  // Every other cell in each direction keeps the cost low at 360p.
  let half_width = (params.width + 1u) / 2u;
  let samples = half_width * ((params.height + 1u) / 2u);
  for (var i = lane; i < samples; i += 64u) {
    let p = vec2i(i32((i % half_width) * 2u), i32((i / half_width) * 2u));
    let q = p + offset;
    if q.x < 0 || q.y < 0 || q.x >= i32(params.width) || q.y >= i32(params.height) { continue; }
    sum += abs(maps[u32(p.y) * params.width + u32(p.x)] - maps[plane + u32(q.y) * params.width + u32(q.x)]);
    count += 1.0;
  }
  partial[lane] = sum;
  partial_count[lane] = count;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if lane < stride {
      partial[lane] += partial[lane + stride];
      partial_count[lane] += partial_count[lane + stride];
    }
    workgroupBarrier();
  }
  if lane == 0u {
    let enough = partial_count[0] >= 0.5 * f32(samples);
    costs[128u + group.x] = select(1e9, partial[0] / max(partial_count[0], 1.0), enough);
  }
}

// Pass 6: pick the final shift.
@compute @workgroup_size(1)
fn fine_pick() {
  let side = 2 * params.fine_radius + 1;
  var best = 0u;
  var best_cost = 1e10;
  let coarse_shift = vec2i(shift[2], shift[3]);
  for (var i = 0u; i < u32(side * side); i += 1u) {
    let total = coarse_shift + candidate(i, params.fine_radius);
    let cost = costs[128u + i] * select(1.0, 0.999, all(total == vec2i(0)));
    if cost < best_cost { best_cost = cost; best = i; }
  }
  let final_shift = coarse_shift + candidate(best, params.fine_radius);
  shift[0] = final_shift.x;
  shift[1] = final_shift.y;
}

const TILE_RADIUS = 2;

// Pass 7: one workgroup per tile. Walking adds rotation, zoom and parallax
// that one global translation cannot follow; each tile searches +/-2 cells
// around the global shift and keeps its own only when it matches clearly
// better (flat or noisy tiles stay on the global shift instead of jittering).
@compute @workgroup_size(64)
fn tile_search(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
  let tiles_x = (params.width + params.tile_size - 1u) / params.tile_size;
  let origin = vec2u(group.x % tiles_x, group.x / tiles_x) * params.tile_size;
  let global = vec2i(shift[0], shift[1]);
  let plane = params.width * params.height;
  let side = 2 * TILE_RADIUS + 1;
  var best = global;
  var best_cost = 1e10;
  var global_cost = 1e10;
  for (var candidate = 0; candidate < side * side; candidate += 1) {
    let offset = global + vec2i(candidate % side - TILE_RADIUS, candidate / side - TILE_RADIUS);
    var sum = 0.0;
    var count = 0.0;
    for (var i = lane; i < params.tile_size * params.tile_size; i += 64u) {
      let p = vec2i(origin + vec2u(i % params.tile_size, i / params.tile_size));
      let q = p + offset;
      if p.x >= i32(params.width) || p.y >= i32(params.height) { continue; }
      if q.x < 0 || q.y < 0 || q.x >= i32(params.width) || q.y >= i32(params.height) { continue; }
      sum += abs(maps[u32(p.y) * params.width + u32(p.x)] - maps[plane + u32(q.y) * params.width + u32(q.x)]);
      count += 1.0;
    }
    partial[lane] = sum;
    partial_count[lane] = count;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride >>= 1u) {
      if lane < stride {
        partial[lane] += partial[lane + stride];
        partial_count[lane] += partial_count[lane + stride];
      }
      workgroupBarrier();
    }
    // Every lane reads the reduced totals, then all wait before the next reuse.
    let cost = select(1e9, partial[0] / max(partial_count[0], 1.0),
                      partial_count[0] >= 0.5 * f32(params.tile_size * params.tile_size));
    workgroupBarrier();
    if all(offset == global) { global_cost = cost; }
    if cost < best_cost { best_cost = cost; best = offset; }
  }
  if lane == 0u {
    let chosen = select(global, best, best_cost < 0.95 * global_cost);
    tile_shift[2u * group.x] = chosen.x;
    tile_shift[2u * group.x + 1u] = chosen.y;
  }
}
