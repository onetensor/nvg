/**
 * Saves consecutive camera frames exactly as the app receives them (upright,
 * 8-bit sRGB, native resolution) so real failure cases can be examined and the
 * training simulation calibrated offline. Frames are drawn synchronously into
 * canvases on consecutive video-frame callbacks and only encoded afterwards, so
 * the capture covers adjacent frames.
 */

const encoder = new TextEncoder();

function tarHeader(name: string, size: number): Uint8Array<ArrayBuffer> {
  const header = new Uint8Array(new ArrayBuffer(512));
  const field = (offset: number, length: number, value: string): void => {
    header.set(encoder.encode(value).subarray(0, length), offset);
  };
  field(0, 100, name);
  field(100, 8, '0000644\0');
  field(108, 8, '0000000\0');
  field(116, 8, '0000000\0');
  field(124, 12, `${size.toString(8).padStart(11, '0')}\0`);
  field(136, 12, `${Math.floor(Date.now() / 1000).toString(8).padStart(11, '0')}\0`);
  field(148, 8, '        ');
  header[156] = '0'.charCodeAt(0);
  field(257, 6, 'ustar\0');
  field(263, 2, '00');
  let checksum = 0;
  for (const byte of header) checksum += byte;
  field(148, 8, `${checksum.toString(8).padStart(6, '0')}\0 `);
  return header;
}

async function tarBlob(files: { name: string; data: Blob }[]): Promise<Blob> {
  const parts: BlobPart[] = [];
  for (const file of files) {
    parts.push(tarHeader(file.name, file.data.size), file.data);
    const padding = (512 - (file.data.size % 512)) % 512;
    if (padding) parts.push(new Uint8Array(padding));
  }
  parts.push(new Uint8Array(1024));
  return new Blob(parts, { type: 'application/x-tar' });
}

function canvasPng(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('PNG encoding failed.'))), 'image/png');
  });
}

export class FrameCapture {
  private canvases: HTMLCanvasElement[] = [];
  private timestamps: number[] = [];
  private remaining = 0;
  private details: Record<string, unknown> = {};

  get active(): boolean {
    return this.remaining > 0;
  }

  start(frames: number, details: Record<string, unknown>): void {
    this.canvases = [];
    this.timestamps = [];
    this.remaining = frames;
    this.details = details;
  }

  /** Call once per camera frame; resolves to the archive after the last frame. */
  async grab(video: HTMLVideoElement, now: number): Promise<Blob | null> {
    if (!this.active || !video.videoWidth) return null;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d', { alpha: false });
    if (!context) throw new Error('Could not create a capture canvas.');
    context.drawImage(video, 0, 0);
    this.canvases.push(canvas);
    this.timestamps.push(now);
    this.remaining -= 1;
    if (this.remaining > 0) return null;

    const files: { name: string; data: Blob }[] = [];
    for (const [index, frame] of this.canvases.entries()) {
      files.push({ name: `frame-${String(index).padStart(2, '0')}.png`, data: await canvasPng(frame) });
    }
    const metadata = {
      ...this.details,
      frames: this.canvases.length,
      width: this.canvases[0]!.width,
      height: this.canvases[0]!.height,
      timestampsMs: this.timestamps,
      userAgent: navigator.userAgent,
      capturedAt: new Date().toISOString(),
    };
    files.push({ name: 'metadata.json', data: new Blob([JSON.stringify(metadata, null, 2)]) });
    this.canvases = [];
    return tarBlob(files);
  }
}

export function downloadBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
