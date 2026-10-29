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

export interface CaptureFile {
  name: string;
  data: Blob;
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

  /** Call once per camera frame; resolves to the capture's files after the last frame. */
  async grab(video: HTMLVideoElement, now: number): Promise<CaptureFile[] | null> {
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

    const files: CaptureFile[] = [];
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
    return files;
  }
}

const DATABASE = 'sid-live-captures';
const STORE = 'captures';

/**
 * Keeps captures on the device until they are exported together. IndexedDB
 * survives the reloads iOS forces on background tabs; if it is unavailable
 * (for example in private browsing) captures stay in memory for the session.
 */
export class CaptureStore {
  private readonly memory: CaptureFile[][] = [];
  private database: Promise<IDBDatabase | null> | null = null;

  private open(): Promise<IDBDatabase | null> {
    this.database ??= new Promise((resolve) => {
      try {
        const request = indexedDB.open(DATABASE, 1);
        request.onupgradeneeded = () => request.result.createObjectStore(STORE, { autoIncrement: true });
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => resolve(null);
      } catch {
        resolve(null);
      }
    });
    return this.database;
  }

  private async transaction<T>(mode: IDBTransactionMode, run: (store: IDBObjectStore) => IDBRequest<T>): Promise<T | null> {
    const database = await this.open();
    if (!database) return null;
    return new Promise((resolve, reject) => {
      const request = run(database.transaction(STORE, mode).objectStore(STORE));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed.'));
    });
  }

  async add(files: CaptureFile[]): Promise<void> {
    try {
      if (await this.transaction('readwrite', (store) => store.add(files)) !== null) return;
    } catch (error) {
      console.warn('Could not persist the capture; keeping it in memory.', error);
    }
    this.memory.push(files);
  }

  async all(): Promise<CaptureFile[][]> {
    let stored: CaptureFile[][] = [];
    try {
      stored = (await this.transaction('readonly', (store) => store.getAll() as IDBRequest<CaptureFile[][]>)) ?? [];
    } catch (error) {
      console.warn('Could not read stored captures.', error);
    }
    return [...stored, ...this.memory];
  }

  async count(): Promise<number> {
    let stored = 0;
    try {
      stored = (await this.transaction('readonly', (store) => store.count())) ?? 0;
    } catch {
      stored = 0;
    }
    return stored + this.memory.length;
  }

  async clear(): Promise<void> {
    this.memory.length = 0;
    try {
      await this.transaction('readwrite', (store) => store.clear());
    } catch (error) {
      console.warn('Could not clear stored captures.', error);
    }
  }
}

/** One archive holding every capture in its own folder. */
export function captureArchive(captures: CaptureFile[][]): Promise<Blob> {
  return tarBlob(captures.flatMap((files, index) =>
    files.map((file) => ({ name: `capture-${String(index + 1).padStart(2, '0')}/${file.name}`, data: file.data }))));
}

/** Opens the share sheet (AirDrop, Files, …) when available, else downloads. */
export async function shareOrDownload(blob: Blob, name: string): Promise<void> {
  const file = new File([blob], name, { type: 'application/x-tar' });
  if (navigator.canShare?.({ files: [file] })) {
    await navigator.share({ files: [file], title: name });
    return;
  }
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
