import type { LuminanceStats } from './types';

export const MIN_GAIN = 1;
export const MAX_GAIN = 300;

export function sliderToGain(value: number): number {
  const position = Math.min(1, Math.max(0, value / 1000));
  return Math.exp(Math.log(MIN_GAIN) + position * Math.log(MAX_GAIN / MIN_GAIN));
}

export function gainToSlider(gain: number): number {
  const safe = Math.min(MAX_GAIN, Math.max(MIN_GAIN, gain));
  return 1000 * Math.log(safe / MIN_GAIN) / Math.log(MAX_GAIN / MIN_GAIN);
}

export class AutoGainController {
  private smoothed = 30;
  private lastUpdate = performance.now();

  get value(): number {
    return this.smoothed;
  }

  reset(gain = 30): void {
    this.smoothed = Math.min(MAX_GAIN, Math.max(MIN_GAIN, gain));
    this.lastUpdate = performance.now();
  }

  update(stats: LuminanceStats, now = performance.now()): number {
    // Aim the median at 18% gray while keeping the bright tail below clipping.
    const medianTarget = 0.18 / Math.max(stats.p50, 0.0006);
    const highlightLimit = 0.82 / Math.max(stats.p90, 0.001);
    let target = Math.min(medianTarget, highlightLimit * 1.35);
    if (stats.blackFraction > 0.55) target *= 1.15;
    target = Math.min(MAX_GAIN, Math.max(MIN_GAIN, target));

    const dt = Math.min(1, Math.max(0, (now - this.lastUpdate) / 1000));
    this.lastUpdate = now;
    const timeConstant = target > this.smoothed ? 0.75 : 1.25;
    const amount = 1 - Math.exp(-dt / timeConstant);
    const logValue = Math.log(this.smoothed) + amount * (Math.log(target) - Math.log(this.smoothed));
    this.smoothed = Math.exp(logValue);
    return this.smoothed;
  }
}
