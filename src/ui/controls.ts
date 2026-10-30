import type { AdapterMode, ModelVariant, ResolutionChoice, FpsCap, TemporalLevel, ViewMode } from '../types';

function required<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing required UI element: ${selector}`);
  return element;
}

export interface Controls {
  stage: HTMLElement;
  video: HTMLVideoElement;
  canvas: HTMLCanvasElement;
  start: HTMLButtonElement;
  emptyState: HTMLElement;
  requirements: HTMLElement;
  status: HTMLElement;
  panel: HTMLElement;
  panelToggle: HTMLButtonElement;
  view: HTMLSelectElement;
  model: HTMLSelectElement;
  adapter: HTMLSelectElement;
  resolution: HTMLSelectElement;
  fpsCap: HTMLSelectElement;
  temporal: HTMLSelectElement;
  tint: HTMLSelectElement;
  outline: HTMLSelectElement;
  pop: HTMLSelectElement;
  debug: HTMLInputElement;
  debugOverlay: HTMLElement;
  capture: HTMLButtonElement;
  captureExport: HTMLButtonElement;
  captureClear: HTMLButtonElement;
  autoGain: HTMLInputElement;
  gain: HTMLInputElement;
  gainValue: HTMLOutputElement;
  splitDivider: HTMLElement;
  originalLabel: HTMLElement;
  enhancedLabel: HTMLElement;
}

export function controls(): Controls {
  return {
    stage: required('#stage'),
    video: required('#camera'),
    canvas: required('#enhanced'),
    start: required('#start-camera'),
    emptyState: required('#empty-state'),
    requirements: required('#requirements'),
    status: required('#status'),
    panel: required('#control-panel'),
    panelToggle: required('#controls-toggle'),
    view: required('#view-mode'),
    model: required('#model-variant'),
    adapter: required('#adapter-mode'),
    resolution: required('#resolution'),
    fpsCap: required('#fps-cap'),
    temporal: required('#temporal'),
    tint: required('#tint'),
    outline: required('#outline'),
    pop: required('#pop'),
    debug: required('#debug'),
    debugOverlay: required('#debug-overlay'),
    capture: required('#capture'),
    captureExport: required('#capture-export'),
    captureClear: required('#capture-clear'),
    autoGain: required('#auto-gain'),
    gain: required('#gain'),
    gainValue: required('#gain-value'),
    splitDivider: required('#split-divider'),
    originalLabel: required('#original-label'),
    enhancedLabel: required('#enhanced-label'),
  };
}

export function selectedValues(ui: Controls): {
  view: ViewMode;
  model: ModelVariant;
  adapter: AdapterMode;
  resolution: ResolutionChoice;
  fpsCap: FpsCap;
  temporal: TemporalLevel;
} {
  return {
    view: ui.view.value as ViewMode,
    model: ui.model.value as ModelVariant,
    adapter: ui.adapter.value as AdapterMode,
    resolution: ui.resolution.value as ResolutionChoice,
    fpsCap: ui.fpsCap.value as FpsCap,
    temporal: ui.temporal.value as TemporalLevel,
  };
}

export function setStatus(ui: Controls, message: string, level: 'normal' | 'ok' | 'error' = 'normal'): void {
  ui.status.textContent = message;
  ui.status.title = message;
  ui.status.dataset.level = level;
}

export function applyView(ui: Controls, view: ViewMode): void {
  ui.stage.dataset.view = view;
  const split = view === 'split';
  ui.splitDivider.hidden = !split;
  ui.originalLabel.hidden = !split;
  ui.enhancedLabel.hidden = !split;
}
