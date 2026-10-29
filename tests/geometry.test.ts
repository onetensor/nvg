import { describe, expect, it } from 'vitest';
import {
  AdaptiveResolution,
  coverUvScale,
  modelGeometry,
  orientResolution,
  RESOLUTIONS,
  resolutionByLabel,
} from '../src/geometry';

describe('model geometry', () => {
  it.each(RESOLUTIONS)('pads $label for four packed-domain pooling stages', (resolution) => {
    const geometry = modelGeometry(resolution);
    expect(geometry.paddedWidth % 32).toBe(0);
    expect(geometry.paddedHeight % 32).toBe(0);
    expect(geometry.rawWidth % 16).toBe(0);
    expect(geometry.rawHeight % 16).toBe(0);
    expect(geometry.cropX * 2 + geometry.width).toBe(geometry.paddedWidth);
    expect(geometry.cropY * 2 + geometry.height).toBe(geometry.paddedHeight);
  });

  it('requires sustained evidence before changing auto resolution', () => {
    const policy = new AdaptiveResolution();
    expect(policy.current.label).toBe('360p');
    let changed = null;
    for (let i = 0; i < 90; i += 1) changed = policy.observe(10, 33);
    expect(changed?.label).toBe('540p');
  });

  it('orients processing geometry to match a portrait camera', () => {
    const portrait = orientResolution(resolutionByLabel('180p'), true);
    expect(portrait).toMatchObject({ width: 180, height: 320 });
    expect(modelGeometry(portrait)).toMatchObject({ paddedWidth: 192, paddedHeight: 320 });
  });

  it('computes cover sampling without stretching the source', () => {
    expect(coverUvScale(720, 1280, 320, 180)).toEqual({ x: 1, y: 0.31640625 });
    expect(coverUvScale(1920, 1080, 320, 180)).toEqual({ x: 1, y: 1 });
    expect(coverUvScale(1920, 1080, 180, 320)).toEqual({ x: 0.31640625, y: 1 });
  });
});
