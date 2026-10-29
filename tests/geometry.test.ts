import { describe, expect, it } from 'vitest';
import { AdaptiveResolution, modelGeometry, RESOLUTIONS } from '../src/geometry';

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
});
