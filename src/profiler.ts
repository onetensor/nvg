import type { DebugSnapshot, TimingSample } from './types';

const EMPTY_TIMING: TimingSample = { preprocess: 0, adapter: 0, sid: 0, postprocess: 0, total: 0 };

class RollingAverage {
  private readonly values: number[] = [];
  private total = 0;

  constructor(private readonly capacity = 45) {}

  add(value: number): void {
    this.values.push(value);
    this.total += value;
    if (this.values.length > this.capacity) this.total -= this.values.shift()!;
  }

  get value(): number {
    return this.values.length ? this.total / this.values.length : 0;
  }
}

export class Profiler {
  readonly frames = { received: 0, processed: 0, skipped: 0 };
  private readonly averages: Record<keyof TimingSample, RollingAverage> = {
    preprocess: new RollingAverage(),
    adapter: new RollingAverage(),
    sid: new RollingAverage(),
    postprocess: new RollingAverage(),
    total: new RollingAverage(),
  };
  private readonly cameraIntervals = new RollingAverage(60);
  private readonly processedIntervals = new RollingAverage(30);
  private lastCameraAt = 0;
  private lastProcessedAt = 0;

  frameReceived(now: number): void {
    this.frames.received += 1;
    if (this.lastCameraAt) this.cameraIntervals.add(now - this.lastCameraAt);
    this.lastCameraAt = now;
  }

  frameSkipped(): void {
    this.frames.skipped += 1;
  }

  frameProcessed(sample: TimingSample, now = performance.now()): void {
    this.frames.processed += 1;
    for (const key of Object.keys(EMPTY_TIMING) as (keyof TimingSample)[]) this.averages[key].add(sample[key]);
    if (this.lastProcessedAt) this.processedIntervals.add(now - this.lastProcessedAt);
    this.lastProcessedAt = now;
  }

  get totalMs(): number {
    return this.averages.total.value;
  }

  snapshot(base: Omit<DebugSnapshot, 'cameraFps' | 'processedFps' | 'timings' | 'framesReceived' | 'framesProcessed' | 'framesSkipped'>): DebugSnapshot {
    return {
      ...base,
      cameraFps: this.cameraIntervals.value ? 1000 / this.cameraIntervals.value : 0,
      processedFps: this.processedIntervals.value ? 1000 / this.processedIntervals.value : 0,
      timings: Object.fromEntries(
        (Object.keys(this.averages) as (keyof TimingSample)[]).map((key) => [key, this.averages[key].value]),
      ) as unknown as TimingSample,
      framesReceived: this.frames.received,
      framesProcessed: this.frames.processed,
      framesSkipped: this.frames.skipped,
    };
  }
}

const fixed = (value: number, digits = 1): string => Number.isFinite(value) ? value.toFixed(digits) : '—';

export function formatDebug(snapshot: DebugSnapshot): string {
  const { timings: t, processing: p } = snapshot;
  return [
    `camera       ${snapshot.cameraWidth}×${snapshot.cameraHeight}  ${fixed(snapshot.cameraFps)} fps`,
    `processing   ${p.width}×${p.height} (pad ${p.paddedWidth}×${p.paddedHeight})`,
    `processed    ${fixed(snapshot.processedFps)} fps`,
    '',
    `preprocess   ${fixed(t.preprocess, 2)} ms`,
    `adapter      ${fixed(t.adapter, 2)} ms`,
    `SID          ${fixed(t.sid, 2)} ms`,
    `postprocess  ${fixed(t.postprocess, 2)} ms`,
    `total        ${fixed(t.total, 2)} ms`,
    '',
    `backend      ${snapshot.backend}`,
    `WebGPU       ${snapshot.webgpu ? 'yes' : 'no'}`,
    `shader-f16   ${snapshot.fp16 ? 'yes' : 'no'}`,
    `SID model    ${snapshot.modelVariant}`,
    `adapter      ${snapshot.adapterMode}`,
    `SID gain     ${fixed(snapshot.gain)}×${snapshot.automaticGain ? ' (auto)' : ''}`,
    ...(snapshot.outputStats ? [
      `output       ${fixed(snapshot.outputStats.min, 3)} min`,
      `             ${fixed(snapshot.outputStats.mean, 3)} mean`,
      `             ${fixed(snapshot.outputStats.max, 3)} max`,
      `non-finite   ${snapshot.outputStats.nonFinite}`,
    ] : []),
    '',
    `frames       ${snapshot.framesReceived} received`,
    `             ${snapshot.framesProcessed} processed`,
    `             ${snapshot.framesSkipped} skipped`,
  ].join('\n');
}
