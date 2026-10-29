import { describe, expect, it } from 'vitest';
import {
  CALIBRATION_GRID,
  cameraInputPaths,
  cameraPathOverride,
  estimateOrientation,
  IPHONE_CAMERA_INPUT_PATHS,
  nextCameraInputPath,
  orientationVerdict,
  sourceQuarterTurns,
} from '../src/cameraInput';

const N = CALIBRATION_GRID;

/** An asymmetric upright scene: bright top-left blob plus a horizontal ramp. */
function uprightScene(): Float32Array {
  const grid = new Float32Array(N * N);
  for (let y = 0; y < N; y += 1) {
    for (let x = 0; x < N; x += 1) grid[y * N + x] = 0.1 + 0.3 * x / N + (x < 5 && y < 4 ? 0.5 : 0);
  }
  return grid;
}

/** What a buffer looks like when it must be turned `turns` clockwise to be upright. */
function unrotate(upright: Float32Array, turns: number): Float32Array {
  let grid = upright;
  for (let t = 0; t < turns; t += 1) {
    // Turn counter-clockwise once: buffer(x, y) = upright(N-1-y, x).
    const next = new Float32Array(N * N);
    for (let y = 0; y < N; y += 1) for (let x = 0; x < N; x += 1) next[y * N + x] = grid[x * N + (N - 1 - y)]!;
    grid = next;
  }
  return grid;
}

describe('iPhone camera ingestion fallback', () => {
  it('tries direct copies before the GPU-synchronizing canvas copy', () => {
    expect(IPHONE_CAMERA_INPUT_PATHS).toEqual(['video-copy', 'video-frame-copy', 'canvas-copy']);
    expect(nextCameraInputPath('video-copy')).toBe('video-frame-copy');
    expect(nextCameraInputPath('video-frame-copy')).toBe('canvas-copy');
    expect(nextCameraInputPath('canvas-copy')).toBeNull();
  });

  it('accepts only known query overrides', () => {
    expect(cameraPathOverride('?cameraPath=canvas-copy')).toBe('canvas-copy');
    expect(cameraPathOverride('?cameraPath=bogus')).toBeNull();
    expect(cameraInputPaths('?cameraPath=video-frame-copy')).toEqual(['video-frame-copy']);
    expect(cameraInputPaths('')).toEqual(IPHONE_CAMERA_INPUT_PATHS);
  });

  it('guesses a quarter turn for a landscape sensor buffer shown in portrait', () => {
    expect(sourceQuarterTurns(1280, 720, 720, 1280)).toBe(1);
    expect(sourceQuarterTurns(720, 1280, 720, 1280)).toBe(0);
    expect(sourceQuarterTurns(1280, 720, 720, 1280, 270)).toBe(3);
    expect(sourceQuarterTurns(1280, 720, 1280, 720, 0)).toBe(0);
  });
});

describe('camera orientation calibration', () => {
  it.each([0, 1, 2, 3])('recovers a %i quarter-turn buffer', (turns) => {
    const upright = uprightScene();
    const estimate = estimateOrientation(upright, unrotate(upright, turns));
    expect(estimate.quarterTurns).toBe(turns);
    expect(estimate.correlation).toBeCloseTo(1, 5);
    expect(orientationVerdict(estimate)).toEqual({ kind: 'accept', quarterTurns: turns });
  });

  it('rejects a copy that leaves part of the frame black', () => {
    const upright = uprightScene();
    const partial = unrotate(upright, 1).map((value, index) => (index >= N * N * 0.55 ? 0 : value));
    expect(orientationVerdict(estimateOrientation(upright, partial)).kind).toBe('reject');
  });

  it('defers judgement on a flat scene', () => {
    const flat = new Float32Array(N * N).fill(0.2);
    expect(orientationVerdict(estimateOrientation(flat, flat)).kind).toBe('inconclusive');
  });
});
