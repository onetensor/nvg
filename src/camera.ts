export interface CameraInfo {
  width: number;
  height: number;
  frameRate: number;
  facingMode: string;
}

type FrameHandler = (now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) => Promise<void>;

export class CameraController {
  private stream: MediaStream | null = null;
  private callbackId: number | null = null;
  private running = false;
  private processing = false;

  constructor(private readonly video: HTMLVideoElement) {}

  async start(): Promise<CameraInfo> {
    if (!window.isSecureContext) throw new Error('Camera access requires HTTPS (localhost is also allowed).');
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('This browser does not provide getUserMedia().');

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    return this.info();
  }

  /**
   * Re-requests the capture size (long side, in the stream's current
   * orientation) and frame-rate ceiling on the live track. applyConstraints
   * replaces every constraint, so the size is always restated. A lower ceiling
   * lets the camera deliver fewer frames; browsers may ignore it, so callers
   * still pace processing themselves.
   */
  async configure(longSide: number, maxFrameRate: number): Promise<CameraInfo> {
    const track = this.stream?.getVideoTracks()[0];
    if (!track) throw new Error('The camera is not running.');
    const current = this.info();
    const shortSide = Math.round(longSide * 9 / 16);
    const portrait = current.height > current.width;
    await track.applyConstraints({
      facingMode: { ideal: 'environment' },
      width: { ideal: portrait ? shortSide : longSide },
      height: { ideal: portrait ? longSide : shortSide },
      frameRate: { ideal: maxFrameRate, max: maxFrameRate },
    });
    return this.info();
  }

  private info(): CameraInfo {
    const settings = this.stream?.getVideoTracks()[0]?.getSettings();
    return {
      width: settings?.width ?? this.video.videoWidth,
      height: settings?.height ?? this.video.videoHeight,
      frameRate: settings?.frameRate ?? 0,
      facingMode: settings?.facingMode ?? 'unknown',
    };
  }

  pump(handler: FrameHandler, onDrop: () => void): void {
    this.running = true;
    const schedule = (): void => {
      if (!this.running) return;
      this.callbackId = this.video.requestVideoFrameCallback(async (now, metadata) => {
        schedule();
        if (this.processing) {
          onDrop();
          return;
        }
        this.processing = true;
        try {
          await handler(now, metadata);
        } finally {
          this.processing = false;
        }
      });
    };
    schedule();
  }

  stop(): void {
    this.running = false;
    if (this.callbackId !== null) this.video.cancelVideoFrameCallback(this.callbackId);
    this.callbackId = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }
}
