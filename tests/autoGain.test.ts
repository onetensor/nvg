import { describe, expect, it, vi } from 'vitest';
import { AutoGainController, gainToSlider, sliderToGain } from '../src/autoGain';

describe('gain mapping', () => {
  it('maps the logarithmic slider across 1x to 300x', () => {
    expect(sliderToGain(0)).toBeCloseTo(1);
    expect(sliderToGain(1000)).toBeCloseTo(300);
    expect(gainToSlider(sliderToGain(412))).toBeCloseTo(412);
  });

  it('smooths automatic gain in log space', () => {
    vi.spyOn(performance, 'now').mockReturnValue(0);
    const controller = new AutoGainController();
    controller.reset(10);
    const next = controller.update({
      mean: 0.01,
      p10: 0.001,
      p50: 0.01,
      p90: 0.04,
      blackFraction: 0.2,
      exactZeroFraction: 0.1,
    }, 500);
    expect(next).toBeGreaterThan(10);
    expect(next).toBeLessThan(18);
    vi.restoreAllMocks();
  });
});
