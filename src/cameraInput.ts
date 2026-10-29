import type { CameraInputPath } from './types';

// Direct media copies are the fast paths but on iPhone they can deliver the
// unrotated (landscape) sensor buffer behind a portrait <video>. Their
// orientation is therefore calibrated against a Canvas2D reference at startup,
// and canvas-copy (correct by construction, but it synchronizes the GPU) is the
// last resort.
export const IPHONE_CAMERA_INPUT_PATHS = [
  'video-copy',
  'video-frame-copy',
  'canvas-copy',
] as const satisfies readonly CameraInputPath[];

export function nextCameraInputPath(path: CameraInputPath): CameraInputPath | null {
  const index = IPHONE_CAMERA_INPUT_PATHS.indexOf(path as (typeof IPHONE_CAMERA_INPUT_PATHS)[number]);
  return index >= 0 && index + 1 < IPHONE_CAMERA_INPUT_PATHS.length
    ? IPHONE_CAMERA_INPUT_PATHS[index + 1]!
    : null;
}

/** `?cameraPath=canvas-copy` forces one ingestion path for on-device diagnosis. */
export function cameraPathOverride(search = globalThis.location?.search ?? ''): CameraInputPath | null {
  const value = new URLSearchParams(search).get('cameraPath');
  return IPHONE_CAMERA_INPUT_PATHS.find((path) => path === value) ?? null;
}

export function cameraInputPaths(search?: string): readonly CameraInputPath[] {
  const forced = cameraPathOverride(search);
  return forced ? [forced] : IPHONE_CAMERA_INPUT_PATHS;
}

/** Clockwise quarter turns needed to display a buffer of the given size upright. */
export function sourceQuarterTurns(
  bufferWidth: number,
  bufferHeight: number,
  displayWidth: number,
  displayHeight: number,
  reportedRotation?: number,
): number {
  if (typeof reportedRotation === 'number' && reportedRotation % 90 === 0) {
    return (((reportedRotation / 90) % 4) + 4) % 4;
  }
  // Without metadata, a landscape buffer behind a portrait <video> is the
  // iPhone rear camera held upright, which needs a 90° clockwise turn.
  return (bufferWidth > bufferHeight) !== (displayWidth > displayHeight) ? 1 : 0;
}

export const CALIBRATION_GRID = 16;

/** Mirrors buffer_uv() in preprocessTexture.wgsl on a square cell grid. */
function bufferCell(x: number, y: number, quarterTurns: number, grid: number): number {
  const u = (x + 0.5) / grid;
  const v = (y + 0.5) / grid;
  const [bu, bv] = quarterTurns === 1 ? [v, 1 - u]
    : quarterTurns === 2 ? [1 - u, 1 - v]
    : quarterTurns === 3 ? [1 - v, u]
    : [u, v];
  return Math.min(grid - 1, Math.floor(bv * grid)) * grid + Math.min(grid - 1, Math.floor(bu * grid));
}

export interface OrientationEstimate {
  quarterTurns: number;
  correlation: number;
  /** Best correlation minus the runner-up. */
  margin: number;
  /** Standard deviation of the upright reference luma. */
  contrast: number;
  /** Fraction of lit reference cells that are black in the copied buffer. */
  missing: number;
}

function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const n = a.length;
  let meanA = 0;
  let meanB = 0;
  for (let i = 0; i < n; i += 1) { meanA += a[i]!; meanB += b[i]!; }
  meanA /= n;
  meanB /= n;
  let covariance = 0;
  let varianceA = 0;
  let varianceB = 0;
  for (let i = 0; i < n; i += 1) {
    const da = a[i]! - meanA;
    const db = b[i]! - meanB;
    covariance += da * db;
    varianceA += da * da;
    varianceB += db * db;
  }
  return varianceA > 0 && varianceB > 0 ? covariance / Math.sqrt(varianceA * varianceB) : 0;
}

/**
 * Compares an upright luma grid (from Canvas2D drawImage, which applies the
 * camera track rotation) with a grid of the copied GPU buffer in buffer space,
 * and finds the quarter turn that aligns them.
 */
export function estimateOrientation(
  upright: ArrayLike<number>,
  buffer: ArrayLike<number>,
  grid = CALIBRATION_GRID,
): OrientationEstimate {
  const cells = grid * grid;
  let mean = 0;
  for (let i = 0; i < cells; i += 1) mean += upright[i]!;
  mean /= cells;
  let variance = 0;
  for (let i = 0; i < cells; i += 1) variance += (upright[i]! - mean) ** 2;

  const scores = [0, 1, 2, 3].map((turns) => {
    const aligned = new Float32Array(cells);
    for (let y = 0; y < grid; y += 1) {
      for (let x = 0; x < grid; x += 1) aligned[y * grid + x] = buffer[bufferCell(x, y, turns, grid)]!;
    }
    return { turns, aligned, score: correlation(upright, aligned) };
  }).sort((a, b) => b.score - a.score);
  const best = scores[0]!;

  let lit = 0;
  let missing = 0;
  for (let i = 0; i < cells; i += 1) {
    if (upright[i]! < 0.03) continue;
    lit += 1;
    if (best.aligned[i]! < 0.004) missing += 1;
  }
  return {
    quarterTurns: best.turns,
    correlation: best.score,
    margin: best.score - scores[1]!.score,
    contrast: Math.sqrt(variance / cells),
    missing: lit ? missing / lit : 0,
  };
}

export type OrientationVerdict =
  | { kind: 'accept'; quarterTurns: number }
  | { kind: 'reject'; reason: string }
  | { kind: 'inconclusive' };

export function orientationVerdict(estimate: OrientationEstimate): OrientationVerdict {
  if (estimate.missing > 0.1) {
    return { kind: 'reject', reason: `${Math.round(estimate.missing * 100)}% of the camera frame is missing` };
  }
  // A flat or symmetric scene cannot distinguish orientations; try a later frame.
  if (estimate.contrast < 0.025) return { kind: 'inconclusive' };
  if (estimate.correlation < 0.3) {
    return { kind: 'reject', reason: `copied frame does not match the camera (r=${estimate.correlation.toFixed(2)})` };
  }
  if (estimate.correlation < 0.5 || estimate.margin < 0.1) return { kind: 'inconclusive' };
  return { kind: 'accept', quarterTurns: estimate.quarterTurns };
}
