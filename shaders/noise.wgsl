struct Params {
  width: u32,        // packed cells per row
  height: u32,       // packed cell rows
  channels: u32,     // image planes
  x0: u32,           // visible cell rectangle (padding excluded)
  y0: u32,
  x1: u32,
  y1: u32,
  _pad0: u32,
}

// Per-frame noise estimate, mirroring training.sid_model.estimate_noise: the
// mean over the packed planes of sqrt(value), a 4-neighbour high-pass, and the
// median absolute value (via a histogram) scaled to a standard deviation.
// Runs on the current frame before temporal accumulation; the result feeds
// the noise-level plane of direct models trained with noise_input "estimate".
const BINS = 2048u;
const RANGE = 0.064;

@group(0) @binding(0) var<storage, read> frame: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>, 2048>;
@group(0) @binding(3) var<storage, read_write> sigma: array<f32, 4>;

fn cell(x: u32, y: u32) -> f32 {
  let plane = params.width * params.height;
  let index = y * params.width + x;
  var sum = 0.0;
  for (var c = 0u; c < params.channels; c += 1u) { sum += sqrt(max(frame[c * plane + index], 0.0)); }
  return sum / f32(params.channels);
}

@compute @workgroup_size(256)
fn histogram_pass(@builtin(global_invocation_id) gid: vec3u) {
  let inner_width = params.x1 - params.x0 - 2u;
  let inner_height = params.y1 - params.y0 - 2u;
  if gid.x >= inner_width * inner_height { return; }
  let x = params.x0 + 1u + gid.x % inner_width;
  let y = params.y0 + 1u + gid.x / inner_width;
  let neighbours = (cell(x - 1u, y) + cell(x + 1u, y) + cell(x, y - 1u) + cell(x, y + 1u)) * 0.25;
  let high_pass = abs(cell(x, y) - neighbours);
  atomicAdd(&histogram[min(u32(high_pass / RANGE * f32(BINS)), BINS - 1u)], 1u);
}

// Single invocation: median bin to sigma, then clear the histogram for the next frame.
@compute @workgroup_size(1)
fn pick_pass() {
  var total = 0u;
  for (var i = 0u; i < BINS; i += 1u) { total += atomicLoad(&histogram[i]); }
  var cumulative = 0u;
  var median = 0u;
  var found = false;
  for (var i = 0u; i < BINS; i += 1u) {
    cumulative += atomicLoad(&histogram[i]);
    if !found && cumulative * 2u >= total { median = i; found = true; }
    atomicStore(&histogram[i], 0u);
  }
  sigma[0] = (f32(median) + 0.5) * RANGE / f32(BINS) / (0.6745 * 1.118);
}
