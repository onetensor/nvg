import type { CameraInputPath } from './types';

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
