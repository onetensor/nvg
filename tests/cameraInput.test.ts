import { describe, expect, it } from 'vitest';
import { IPHONE_CAMERA_INPUT_PATHS, nextCameraInputPath } from '../src/cameraInput';

describe('iPhone camera ingestion fallback', () => {
  it('falls back from the fastest copy to the most compatible copy', () => {
    expect(IPHONE_CAMERA_INPUT_PATHS).toEqual(['video-copy', 'video-frame-copy', 'canvas-copy']);
    expect(nextCameraInputPath('video-copy')).toBe('video-frame-copy');
    expect(nextCameraInputPath('video-frame-copy')).toBe('canvas-copy');
    expect(nextCameraInputPath('canvas-copy')).toBeNull();
  });
});
