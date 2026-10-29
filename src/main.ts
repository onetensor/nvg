import './styles.css';
import { AutoGainController, sliderToGain } from './autoGain';
import { CameraController, type CameraInfo } from './camera';
import { captureArchive, CaptureStore, FrameCapture, shareOrDownload } from './capture';
import { AdaptiveResolution, modelGeometry, orientResolution, resolutionByLabel } from './geometry';
import { preloadModel } from './inference/onnxWebgpu';
import { SIDPipeline } from './inference/sid';
import { isAppleMobileWebKit } from './platform';
import { formatDebug, Profiler } from './profiler';
import type { AdapterMode, DirectModelAsset, LuminanceStats, MonoTint, ModelGeometry, ModelMetadata, ModelVariant, PipelineResult, ResolutionChoice, TargetMode } from './types';
import { applyView, controls, selectedValues, setStatus } from './ui/controls';

const ui = controls();
const camera = new CameraController(ui.video);
const profiler = new Profiler();
const autoGain = new AutoGainController();
const adaptiveResolution = new AdaptiveResolution();
const capture = new FrameCapture();
const captureStore = new CaptureStore();
const CAPTURE_FRAMES = 16;

let metadata: ModelMetadata | null = null;
let pipeline: SIDPipeline | null = null;
let cameraInfo: CameraInfo = { width: 0, height: 0, frameRate: 0, facingMode: 'unknown' };
let geometry: ModelGeometry = modelGeometry(adaptiveResolution.current);
let manualGain = sliderToGain(Number(ui.gain.value));
let currentGain = manualGain;
let lastDebugUpdate = 0;
let frameErrors = 0;
let activeModel: ModelVariant = 'joint';
let metadataPromise: Promise<ModelMetadata> | null = null;
let outputStats: PipelineResult['outputStats'];
let inputStats: LuminanceStats | undefined;
let cameraInputPath: PipelineResult['cameraInputPath'];
let presentation: PipelineResult['presentation'];

function modelUrl(url: string): string {
  if (url.startsWith('./')) return `${import.meta.env.BASE_URL}${url.slice(2)}`;
  return url;
}

function resolveFixedAsset(asset: ModelMetadata['sid']): void {
  if (!asset.fixed360p) return;
  asset.fixed360p.url = modelUrl(asset.fixed360p.url);
  if (asset.fixed360p.fp16) asset.fixed360p.fp16.url = modelUrl(asset.fixed360p.fp16.url);
}

async function loadMetadata(): Promise<ModelMetadata> {
  const response = await fetch(`${import.meta.env.BASE_URL}models/metadata.json`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load model metadata (${response.status}).`);
  const value = await response.json() as ModelMetadata;
  value.sid.url = modelUrl(value.sid.url);
  value.adapter.url = modelUrl(value.adapter.url);
  if (value.sidJoint) value.sidJoint.url = modelUrl(value.sidJoint.url);
  if (value.sid.fp16) value.sid.fp16.url = modelUrl(value.sid.fp16.url);
  if (value.sidJoint?.fp16) value.sidJoint.fp16.url = modelUrl(value.sidJoint.fp16.url);
  if (value.adapter.fp16) value.adapter.fp16.url = modelUrl(value.adapter.fp16.url);
  for (const asset of [value.sidDirect, ...Object.values(value.directVariants ?? {})]) {
    if (!asset) continue;
    asset.url = modelUrl(asset.url);
    if (asset.fp16) asset.fp16.url = modelUrl(asset.fp16.url);
  }
  resolveFixedAsset(value.sid);
  if (value.sidJoint) resolveFixedAsset(value.sidJoint);
  resolveFixedAsset(value.adapter);
  return value;
}

function selectedMetadata(): Promise<ModelMetadata> {
  metadataPromise ??= loadMetadata();
  return metadataPromise;
}

const modelQuery = new URLSearchParams(window.location.search).get('model');

/**
 * `?model=joint` selects the adapter pair, `?model=<variant>` a named direct
 * model, and `?model=direct` the production direct model; otherwise the
 * metadata's production default applies.
 */
function requestedVariant(value: ModelMetadata): ModelVariant {
  const variant = modelQuery ? value.directVariants?.[modelQuery] : undefined;
  const wanted = variant ? 'direct' : modelQuery === 'direct' || modelQuery === 'joint' ? modelQuery : value.production ?? 'joint';
  if (wanted === 'direct' && (variant ?? value.sidDirect)?.available) return 'direct';
  return value.sidJoint?.available ? 'joint' : 'original';
}

/** The direct model to run: a `?model=` variant, else the production one. */
function selectedDirect(value: ModelMetadata): DirectModelAsset {
  const variant = modelQuery ? value.directVariants?.[modelQuery] : undefined;
  return variant?.available ? variant : value.sidDirect!;
}

async function preloadSelectedModels(): Promise<boolean> {
  const gpuAdapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!gpuAdapter?.features.has('shader-f16')) return false;
  const value = await selectedMetadata();
  if (requestedVariant(value) === 'direct') {
    const asset = selectedDirect(value);
    await preloadModel(asset.fp16?.available ? asset.fp16.url : asset.url);
    return true;
  }
  const selectedSid = value.sidJoint?.available ? value.sidJoint : value.sid;
  await Promise.all([preloadModel(selectedSid.url), preloadModel(value.adapter.url)]);
  return true;
}

function updateRequirements(): void {
  const secure = window.isSecureContext;
  const webgpu = Boolean(navigator.gpu);
  const notes = [secure ? 'HTTPS ready' : 'HTTPS required', webgpu ? 'WebGPU detected' : 'WebGPU unavailable'];
  ui.requirements.textContent = notes.join(' · ');
}

function selectedResolution(choice: ResolutionChoice): ModelGeometry {
  const resolution = choice === 'auto' ? adaptiveResolution.current : resolutionByLabel(choice);
  return modelGeometry(orientResolution(resolution, cameraInfo.height > cameraInfo.width));
}

function targetBudget(mode: TargetMode): number {
  if (mode === '30') return 1000 / 30;
  if (mode === '15') return 1000 / 15;
  return 82;
}

function setGeometry(choice: ResolutionChoice): void {
  geometry = selectedResolution(choice);
  pipeline?.resetTemporal();
}

function updateGainUi(): void {
  const displayed = ui.autoGain.checked ? currentGain : manualGain;
  ui.gain.disabled = ui.autoGain.checked;
  ui.gainValue.value = `${displayed < 10 ? displayed.toFixed(1) : displayed.toFixed(0)}×`;
}

function updateDebug(now: number, mode: AdapterMode): void {
  if (!ui.debug.checked || now - lastDebugUpdate < 250) return;
  lastDebugUpdate = now;
  ui.debugOverlay.textContent = formatDebug(profiler.snapshot({
    cameraWidth: cameraInfo.width,
    cameraHeight: cameraInfo.height,
    processing: geometry,
    backend: pipeline?.backendName ?? 'not initialized',
    webgpu: Boolean(navigator.gpu),
    fp16: pipeline?.fp16Available ?? false,
    modelVariant: activeModel,
    adapterMode: mode,
    gain: currentGain,
    automaticGain: ui.autoGain.checked ? currentGain : 0,
    ...(cameraInputPath ? { cameraInputPath } : {}),
    ...(inputStats ? { inputStats } : {}),
    ...(outputStats ? { outputStats } : {}),
    ...(presentation ? { presentation } : {}),
  }));
}

async function initializePipeline(): Promise<void> {
  metadata = await selectedMetadata();
  if (!metadata.sid.available) {
    ui.view.value = 'original';
    applyView(ui, 'original');
    throw new Error('Official SID model is not packaged yet; camera is running in Original mode.');
  }
  const jointOption = ui.model.querySelector<HTMLOptionElement>('option[value="joint"]');
  const directOption = ui.model.querySelector<HTMLOptionElement>('option[value="direct"]');
  if (jointOption) jointOption.disabled = metadata.sidJoint?.available !== true;
  if (directOption) directOption.disabled = metadata.sidDirect?.available !== true;
  activeModel = requestedVariant(metadata);
  ui.model.value = activeModel;
  const direct = activeModel === 'direct';
  const gpuAdapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  if (!gpuAdapter) throw new Error('WebGPU could not acquire a GPU adapter on this device.');
  if (!gpuAdapter.features.has('shader-f16')) {
    throw new Error('This optimized build requires WebGPU shader-f16; FP32 is intentionally not downloaded.');
  }
  const directAsset = direct ? selectedDirect(metadata) : undefined;
  const selectedSid = directAsset ?? (activeModel === 'joint' ? metadata.sidJoint! : metadata.sid);
  if (directAsset?.defaultResolution &&
      ui.resolution.querySelector(`option[value="${directAsset.defaultResolution}"]`)) {
    ui.resolution.value = directAsset.defaultResolution;
  }
  const requested = selectedValues(ui);
  geometry = selectedResolution(requested.resolution);
  const appleMobileWebKit = isAppleMobileWebKit();
  const useFixed360 = !direct && requested.resolution === '360p' &&
    selectedSid.fixed360p?.available === true &&
    metadata.adapter.fixed360p?.available === true;
  const sidFixed = useFixed360 ? selectedSid.fixed360p! : undefined;
  const adapterFixed = useFixed360 ? metadata.adapter.fixed360p! : undefined;
  const runtimeMetadata: ModelMetadata = {
    schemaVersion: metadata.schemaVersion,
    sid: {
      ...selectedSid,
      ...(sidFixed ? { url: sidFixed.url, sha256: sidFixed.sha256 } : {}),
      fp16: sidFixed?.fp16 ? { ...sidFixed.fp16 } : selectedSid.fp16 ? { ...selectedSid.fp16 } : undefined,
    },
    adapter: {
      ...metadata.adapter,
      // Direct models need no adapter, so it is never downloaded.
      ...(direct ? { available: false } : {}),
      ...(adapterFixed ? { url: adapterFixed.url, sha256: adapterFixed.sha256 } : {}),
      fp16: adapterFixed?.fp16 ? { ...adapterFixed.fp16 } : metadata.adapter.fp16 ? { ...metadata.adapter.fp16 } : undefined,
    },
    preprocessing: metadata.preprocessing,
    validation: metadata.validation,
  };
  if (runtimeMetadata.sid.fp16?.available) runtimeMetadata.sid.url = runtimeMetadata.sid.fp16.url;
  if (runtimeMetadata.adapter.fp16?.available) runtimeMetadata.adapter.url = runtimeMetadata.adapter.fp16.url;
  const useGraphCapture = useFixed360 && !appleMobileWebKit;
  setStatus(ui, appleMobileWebKit
    ? `iPhone compatibility mode · loading ${direct ? 'direct' : 'joint'} model…`
    : `Loading ${direct ? 'direct' : activeModel === 'joint' ? 'joint-tuned' : 'original'} SID model…`);
  pipeline = new SIDPipeline(
    runtimeMetadata,
    ui.canvas,
    useGraphCapture,
    useGraphCapture ? sidFixed?.outputShape : undefined,
    useGraphCapture ? adapterFixed?.outputShape : undefined,
    appleMobileWebKit,
  );
  await pipeline.initialize((message) => setStatus(ui, message));
  pipeline.tint = ui.tint.value as MonoTint;
  // The tint only applies to luminance-only models.
  ui.tint.disabled = pipeline.outputChannels !== 1;
  ui.model.disabled = true;
  if (useFixed360) {
    ui.resolution.disabled = true;
    ui.adapter.disabled = true;
  }
  const learnedOption = ui.adapter.querySelector<HTMLOptionElement>('option[value="learned"]');
  if (learnedOption) learnedOption.disabled = !pipeline.learnedAdapterAvailable;
  if (!pipeline.learnedAdapterAvailable && ui.adapter.value === 'learned') ui.adapter.value = 'manual';
  if (direct) {
    // The direct model consumes packed RGB itself; there is no adapter choice.
    ui.adapter.value = 'direct';
    ui.adapter.disabled = true;
  }
  const label = direct
    ? `${directAsset!.name ?? 'Direct SID'} (width ${directAsset!.width})`
    : activeModel === 'joint' ? 'Joint-tuned SID' : 'Original SID';
  const speed = useGraphCapture
    ? ' · static graph capture'
    : appleMobileWebKit ? ' · iPhone visible-canvas fallback' : '';
  setStatus(ui, metadata.validation.verified ? `${label}${speed} · WebGPU ready` : `${label}${speed} · model unverified`, metadata.validation.verified ? 'ok' : 'normal');
}

/** Feeds every camera frame, processed or skipped, to an active capture. */
function captureFrame(now: number): void {
  if (!capture.active) return;
  capture.grab(ui.video, now).then(async (files) => {
    if (!files) return;
    await captureStore.add(files);
    await refreshCaptureUi();
    setStatus(ui, `Capture kept on this device (${await captureStore.count()} total)`, 'ok');
  }).catch((error: unknown) => {
    void refreshCaptureUi();
    setStatus(ui, `Capture failed: ${error instanceof Error ? error.message : String(error)}`, 'error');
  });
}

async function refreshCaptureUi(): Promise<void> {
  const count = await captureStore.count();
  ui.capture.disabled = capture.active;
  ui.capture.textContent = capture.active ? 'Capturing…' : count ? `Capture (${count})` : 'Capture';
  ui.captureExport.hidden = count === 0;
  ui.captureClear.hidden = count === 0;
}

async function processFrame(now: number): Promise<void> {
  captureFrame(now);
  profiler.frameReceived(now);
  if (!pipeline || selectedValues(ui).view === 'original') {
    updateDebug(now, selectedValues(ui).adapter);
    return;
  }
  const state = selectedValues(ui);
  if (!pipeline.ready) {
    // The GPU has not finished the frames already queued. Skipping here costs
    // nothing; processing anyway would only grow the queue and drop the result.
    profiler.frameSkipped();
    updateDebug(now, state.adapter);
    return;
  }
  try {
    const result = await pipeline.process(ui.video, geometry, state.adapter, currentGain, state.temporal);
    if (result.outputStats) outputStats = result.outputStats;
    cameraInputPath = result.cameraInputPath;
    presentation = result.presentation;
    profiler.frameProcessed(result.timing);
    frameErrors = 0;
    result.stats?.then((stats) => {
      inputStats = stats;
      if (!ui.autoGain.checked) return;
      currentGain = autoGain.update(stats);
      updateGainUi();
    }).catch((error: unknown) => {
      // A camera-path failure is stored by the preprocessor and rethrown by the
      // next frame, which routes it through the normal error handling below.
      console.warn('Input statistics readback failed.', error);
    });
    if (state.resolution === 'auto') {
      const changed = adaptiveResolution.observe(result.timing.total, targetBudget(state.target));
      if (changed) {
        geometry = modelGeometry(orientResolution(changed, cameraInfo.height > cameraInfo.width));
        pipeline.resetTemporal();
        setStatus(ui, `Auto resolution: ${changed.label}`);
      }
    }
    updateDebug(now, state.adapter);
  } catch (error) {
    frameErrors += 1;
    const message = error instanceof Error ? error.message : String(error);
    setStatus(ui, message, 'error');
    if (frameErrors >= 3) {
      ui.view.value = 'original';
      applyView(ui, 'original');
    }
  }
}

async function start(): Promise<void> {
  ui.start.disabled = true;
  setStatus(ui, 'Requesting rear camera…');
  try {
    cameraInfo = await camera.start();
    ui.emptyState.hidden = true;
    setStatus(ui, `Camera ${cameraInfo.width}×${cameraInfo.height}`);
    try {
      await initializePipeline();
    } catch (error) {
      console.error(error);
      const message = error instanceof Error ? error.message : String(error);
      const compatibleMessage = /GPUBufferBinding\.buffer|instance of GPUBuffer/i.test(message)
        ? 'Safari rejected an ONNX WebGPU buffer. Original camera remains available; reload after updating iOS.'
        : message;
      ui.view.value = 'original';
      applyView(ui, 'original');
      setStatus(ui, compatibleMessage, 'error');
    }
    camera.pump(processFrame, () => {
      captureFrame(performance.now());
      profiler.frameReceived(performance.now());
      profiler.frameSkipped();
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setStatus(ui, message, 'error');
    ui.start.disabled = false;
  }
}

ui.start.addEventListener('click', () => { void start(); });
ui.view.addEventListener('change', () => {
  applyView(ui, selectedValues(ui).view);
  pipeline?.resetTemporal();
});
ui.adapter.addEventListener('change', () => pipeline?.resetTemporal());
ui.resolution.addEventListener('change', () => setGeometry(selectedValues(ui).resolution));
ui.temporal.addEventListener('change', () => pipeline?.resetTemporal());
ui.tint.addEventListener('change', () => {
  if (pipeline) pipeline.tint = ui.tint.value as MonoTint;
});
ui.capture.addEventListener('click', () => {
  if (capture.active || !ui.video.videoWidth) return;
  capture.start(CAPTURE_FRAMES, {
    camera: cameraInfo,
    trackSettings: ui.video.srcObject instanceof MediaStream
      ? ui.video.srcObject.getVideoTracks()[0]?.getSettings() ?? null
      : null,
    model: activeModel,
    gain: currentGain,
    autoGain: ui.autoGain.checked,
    processing: geometry,
    temporal: selectedValues(ui).temporal,
    inputStats: inputStats ?? null,
  });
  void refreshCaptureUi();
});
ui.captureExport.addEventListener('click', () => {
  ui.captureExport.disabled = true;
  void (async () => {
    try {
      const captures = await captureStore.all();
      if (!captures.length) return;
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      await shareOrDownload(await captureArchive(captures), `sid-live-captures-${stamp}.tar`);
      setStatus(ui, `Exported ${captures.length} capture${captures.length === 1 ? '' : 's'}; Clear removes them`, 'ok');
    } catch (error) {
      // Dismissing the share sheet rejects with AbortError; nothing is lost.
      if (!(error instanceof DOMException && error.name === 'AbortError')) {
        setStatus(ui, `Export failed: ${error instanceof Error ? error.message : String(error)}`, 'error');
      }
    } finally {
      ui.captureExport.disabled = false;
    }
  })();
});
ui.captureClear.addEventListener('click', () => {
  if (!window.confirm('Delete all captures stored on this device?')) return;
  void captureStore.clear().then(refreshCaptureUi);
});
ui.debug.addEventListener('change', () => { ui.debugOverlay.hidden = !ui.debug.checked; });
ui.autoGain.addEventListener('change', () => {
  if (!ui.autoGain.checked) currentGain = manualGain;
  else autoGain.reset(currentGain);
  updateGainUi();
});
ui.gain.addEventListener('input', () => {
  manualGain = sliderToGain(Number(ui.gain.value));
  if (!ui.autoGain.checked) currentGain = manualGain;
  updateGainUi();
});
ui.panelToggle.addEventListener('click', () => {
  const collapsed = ui.panel.dataset.collapsed !== 'true';
  ui.panel.dataset.collapsed = String(collapsed);
  ui.panelToggle.setAttribute('aria-expanded', String(!collapsed));
});

let draggingSplit = false;
ui.stage.addEventListener('pointerdown', (event) => {
  if (selectedValues(ui).view !== 'split') return;
  draggingSplit = true;
  ui.stage.setPointerCapture(event.pointerId);
});
ui.stage.addEventListener('pointermove', (event) => {
  if (!draggingSplit) return;
  const position = Math.min(85, Math.max(15, event.clientX / window.innerWidth * 100));
  ui.stage.style.setProperty('--split', `${position}%`);
});
ui.stage.addEventListener('pointerup', () => { draggingSplit = false; });
window.addEventListener('pagehide', () => {
  camera.stop();
  void pipeline?.dispose();
});

applyView(ui, selectedValues(ui).view);
void refreshCaptureUi();
updateRequirements();
updateGainUi();
void preloadSelectedModels().then((cached) => {
  ui.requirements.textContent = `${ui.requirements.textContent} · ${cached ? 'selected FP16 model cached' : 'shader-f16 unavailable'}`;
}).catch((error) => {
  console.warn('Selected model preload failed; initialization will retry it.', error);
});
