import type { ModelGeometry, Resolution } from './types';

export const RESOLUTIONS: readonly Resolution[] = [
  { label: '180p', width: 320, height: 180 },
  { label: '360p', width: 640, height: 360 },
  { label: '540p', width: 960, height: 540 },
  { label: '720p', width: 1280, height: 720 },
  { label: '1080p', width: 1920, height: 1080 },
] as const;

const align = (value: number, multiple: number): number => Math.ceil(value / multiple) * multiple;

export function modelGeometry(resolution: Resolution): ModelGeometry {
  // Four SID pooling stages operate on packed (half-resolution) input. Padding
  // full-resolution video to 32 pixels avoids mismatched U-Net skip tensors.
  const paddedWidth = align(resolution.width, 32);
  const paddedHeight = align(resolution.height, 32);
  return {
    ...resolution,
    paddedWidth,
    paddedHeight,
    rawWidth: paddedWidth / 2,
    rawHeight: paddedHeight / 2,
    cropX: (paddedWidth - resolution.width) / 2,
    cropY: (paddedHeight - resolution.height) / 2,
  };
}

export function resolutionByLabel(label: Resolution['label']): Resolution {
  const resolution = RESOLUTIONS.find((candidate) => candidate.label === label);
  if (!resolution) throw new Error(`Unknown resolution: ${label}`);
  return resolution;
}

export function orientResolution(resolution: Resolution, portrait: boolean): Resolution {
  if (!portrait || resolution.height >= resolution.width) return resolution;
  return { ...resolution, width: resolution.height, height: resolution.width };
}

export function coverUvScale(
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): { x: number; y: number } {
  if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
    throw new Error('Cover sampling dimensions must all be positive.');
  }
  const sourceAspect = sourceWidth / sourceHeight;
  const targetAspect = targetWidth / targetHeight;
  return sourceAspect > targetAspect
    ? { x: targetAspect / sourceAspect, y: 1 }
    : { x: 1, y: sourceAspect / targetAspect };
}

export class AdaptiveResolution {
  private index = 1;
  private slowFrames = 0;
  private fastFrames = 0;
  private cooldown = 0;

  get current(): Resolution {
    return RESOLUTIONS[this.index]!;
  }

  reset(label: Resolution['label'] = '360p'): void {
    this.index = Math.max(0, RESOLUTIONS.findIndex((entry) => entry.label === label));
    this.slowFrames = 0;
    this.fastFrames = 0;
    this.cooldown = 0;
  }

  observe(totalMs: number, targetMs: number): Resolution | null {
    if (this.cooldown > 0) {
      this.cooldown -= 1;
      return null;
    }

    if (totalMs > targetMs * 1.18) {
      this.slowFrames += 1;
      this.fastFrames = 0;
    } else if (totalMs < targetMs * 0.62) {
      this.fastFrames += 1;
      this.slowFrames = 0;
    } else {
      this.fastFrames = Math.max(0, this.fastFrames - 1);
      this.slowFrames = Math.max(0, this.slowFrames - 1);
    }

    if (this.slowFrames >= 12 && this.index > 0) {
      this.index -= 1;
      this.resetCounters();
      return this.current;
    }
    if (this.fastFrames >= 90 && this.index < RESOLUTIONS.length - 1) {
      this.index += 1;
      this.resetCounters();
      return this.current;
    }
    return null;
  }

  private resetCounters(): void {
    this.slowFrames = 0;
    this.fastFrames = 0;
    this.cooldown = 60;
  }
}
