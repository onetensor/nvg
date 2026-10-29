import './styles.css';
import { AutoGainController, sliderToGain } from './autoGain';
import { CameraController, type CameraInfo } from './camera';
import { AdaptiveResolution, modelGeometry, resolutionByLabel } from './geometry';
import { SIDPipeline } from './inference/sid';
import { formatDebug, Profiler } from './profiler';
import type { AdapterMode, ModelGeometry, ModelMetadata, ModelVariant, ResolutionChoice, TargetMode } from './types';
import { applyView, controls, selectedValues, setStatus } from './ui/controls';

const ui = controls();
const camera = new CameraController(ui.video);
const profiler = new Profiler();
const autoGain = new AutoGainController();
const adaptiveResolution = new AdaptiveResolution();

let metadata: ModelMetadata | null = null;
let pipeline: SIDPipeline | null = null;
let cameraInfo: CameraInfo = { width: 0, height: 0, frameRate: 0, facingMode: 'unknown' };
let geometry: ModelGeometry = modelGeometry(adaptiveResolution.current);
let manualGain = sliderToGain(Number(ui.gain.value));
let currentGain = manualGain;
let lastDebugUpdate = 0;
let frameErrors = 0;
let activeModel: ModelVariant = 'joint';

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
  resolveFixedAsset(value.sid);
  if (value.sidJoint) resolveFixedAsset(value.sidJoint);
  resolveFixedAsset(value.adapter);
  return value;
}

function updateRequirements(): void {
  const secure = window.isSecureContext;
  const webgpu = Boolean(navigator.gpu);
  const notes = [secure ? 'HTTPS ready' : 'HTTPS required', webgpu ? 'WebGPU detected' : 'WebGPU unavailable'];
  ui.requirements.textContent = notes.join(' · ');
}

function selectedResolution(choice: ResolutionChoice): ModelGeometry {
  if (choice === 'auto') return modelGeometry(adaptiveResolution.current);
  return modelGeometry(resolutionByLabel(choice));
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
  }));
}

async function initializePipeline(): Promise<void> {
  metadata = await loadMetadata();
  if (!metadata.sid.available) {
    ui.view.value = 'original';
    applyView(ui, 'original');
    throw new Error('Official SID model is not packaged yet; camera is running in Original mode.');
  }
  const jointOption = ui.model.querySelector<HTMLOptionElement>('option[value="joint"]');
  const jointAvailable = metadata.sidJoint?.available === true;
  if (jointOption) jointOption.disabled = !jointAvailable;
  activeModel = selectedValues(ui).model;
  if (activeModel === 'joint' && !jointAvailable) {
    activeModel = 'original';
    ui.model.value = 'original';
  }
  const gpuAdapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
  const selectedSid = activeModel === 'joint' ? metadata.sidJoint! : metadata.sid;
  const requested = selectedValues(ui);
  const useFixed360 = requested.resolution === '360p' &&
    selectedSid.fixed360p?.available === true &&
    metadata.adapter.fixed360p?.available === true;
  const sidFixed = useFixed360 ? selectedSid.fixed360p! : undefined;
  const adapterFixed = useFixed360 ? metadata.adapter.fixed360p! : undefined;
  const runtimeMetadata: ModelMetadata = {
    ...metadata,
    sid: {
      ...selectedSid,
      ...(sidFixed ? { url: sidFixed.url, sha256: sidFixed.sha256 } : {}),
      fp16: sidFixed?.fp16 ? { ...sidFixed.fp16 } : selectedSid.fp16 ? { ...selectedSid.fp16 } : undefined,
    },
    adapter: {
      ...metadata.adapter,
      ...(adapterFixed ? { url: adapterFixed.url, sha256: adapterFixed.sha256 } : {}),
      fp16: adapterFixed?.fp16 ? { ...adapterFixed.fp16 } : metadata.adapter.fp16 ? { ...metadata.adapter.fp16 } : undefined,
    },
  };
  if (gpuAdapter?.features.has('shader-f16')) {
    if (runtimeMetadata.sid.fp16?.available) runtimeMetadata.sid.url = runtimeMetadata.sid.fp16.url;
    if (runtimeMetadata.adapter.fp16?.available) runtimeMetadata.adapter.url = runtimeMetadata.adapter.fp16.url;
  }
  setStatus(ui, `Loading ${activeModel === 'joint' ? 'joint-tuned' : 'original'} SID model…`);
  pipeline = new SIDPipeline(
    runtimeMetadata,
    ui.canvas,
    useFixed360,
    sidFixed?.outputShape,
    adapterFixed?.outputShape,
  );
  await pipeline.initialize();
  ui.model.disabled = true;
  if (useFixed360) {
    ui.resolution.disabled = true;
    ui.adapter.disabled = true;
  }
  const learnedOption = ui.adapter.querySelector<HTMLOptionElement>('option[value="learned"]');
  if (learnedOption) learnedOption.disabled = !pipeline.learnedAdapterAvailable;
  if (!pipeline.learnedAdapterAvailable && ui.adapter.value === 'learned') ui.adapter.value = 'manual';
  const label = activeModel === 'joint' ? 'Joint-tuned SID' : 'Original SID';
  const speed = useFixed360 ? ' · static graph capture' : '';
  setStatus(ui, metadata.validation.verified ? `${label}${speed} · WebGPU ready` : `${label}${speed} · model unverified`, metadata.validation.verified ? 'ok' : 'normal');
}

async function processFrame(now: number): Promise<void> {
  profiler.frameReceived(now);
  if (!pipeline || selectedValues(ui).view === 'original') {
    updateDebug(now, selectedValues(ui).adapter);
    return;
  }
  const state = selectedValues(ui);
  try {
    const result = await pipeline.process(ui.video, geometry, state.adapter, currentGain, state.temporal);
    profiler.frameProcessed(result.timing);
    frameErrors = 0;
    if (result.stats && ui.autoGain.checked) {
      currentGain = autoGain.update(result.stats);
      updateGainUi();
    }
    if (state.resolution === 'auto') {
      const changed = adaptiveResolution.observe(result.timing.total, targetBudget(state.target));
      if (changed) {
        geometry = modelGeometry(changed);
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
      const message = error instanceof Error ? error.message : String(error);
      setStatus(ui, message, 'error');
    }
    camera.pump(processFrame, () => {
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
updateRequirements();
updateGainUi();
