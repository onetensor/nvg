struct Params {
  full_width: u32,
  full_height: u32,
  visible_width: u32,
  visible_height: u32,
  crop_x: u32,
  crop_y: u32,
  // 3: RGB planes; 1: a luminance plane shown through `tint`; 2: luminance
  // plus a plane of object-outline logits.
  channels: u32,
  // Mono display: 0 = grayscale, 1 = green phosphor.
  tint: u32,
  // 0 = off, 1 = learned (outline plane), 2 = classic (Sobel on the output).
  outline_mode: u32,
  outline_strength: f32,
  // "Pop" for luminance output: contrast around the midtones and local
  // contrast against a 5x5 neighbourhood mean (0 = off).
  pop_contrast: f32,
  pop_local: f32,
}

// Green phosphor (P43-like), scaled so full luminance stays within range.
const PHOSPHOR = vec3f(0.42, 1.0, 0.36);

@group(0) @binding(0) var<storage, read> image: array<f32>;
@group(0) @binding(1) var<storage, read_write> rgba: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

fn mono_color(value: f32) -> vec3f {
  if params.tint == 1u { return value * PHOSPHOR; }
  return vec3f(value);
}

fn luminance_at(x: i32, y: i32) -> f32 {
  let cx = u32(clamp(x, 0, i32(params.full_width) - 1));
  let cy = u32(clamp(y, 0, i32(params.full_height) - 1));
  let index = cy * params.full_width + cx;
  if params.channels == 3u {
    let plane = params.full_width * params.full_height;
    return dot(vec3f(image[index], image[plane + index], image[2u * plane + index]), vec3f(0.2126, 0.7152, 0.0722));
  }
  return image[index];
}

/// Outline strength in [0, 1] at a full-image pixel.
fn outline_at(x: i32, y: i32) -> f32 {
  if params.outline_mode == 1u && params.channels == 2u {
    let plane = params.full_width * params.full_height;
    // Only confident outlines: the network's best F-score threshold is ~0.8,
    // and uncertain regions sit near 0.5.
    return smoothstep(0.65, 0.9, 1.0 / (1.0 + exp(-image[plane + u32(y) * params.full_width + u32(x)])));
  }
  if params.outline_mode == 2u {
    let gx = luminance_at(x + 1, y - 1) + 2.0 * luminance_at(x + 1, y) + luminance_at(x + 1, y + 1)
      - luminance_at(x - 1, y - 1) - 2.0 * luminance_at(x - 1, y) - luminance_at(x - 1, y + 1);
    let gy = luminance_at(x - 1, y + 1) + 2.0 * luminance_at(x, y + 1) + luminance_at(x + 1, y + 1)
      - luminance_at(x - 1, y - 1) - 2.0 * luminance_at(x, y - 1) - luminance_at(x + 1, y - 1);
    return smoothstep(0.12, 0.4, length(vec2f(gx, gy)));
  }
  return 0.0;
}

/// Luminance with "pop": local contrast against the 5x5 mean, then an
/// S-shaped contrast boost around mid-grey.
fn popped(x: i32, y: i32, value: f32) -> f32 {
  if params.pop_contrast == 0.0 && params.pop_local == 0.0 { return value; }
  var mean = 0.0;
  for (var dy = -2; dy <= 2; dy += 1) {
    for (var dx = -2; dx <= 2; dx += 1) { mean += luminance_at(x + dx, y + dy); }
  }
  mean /= 25.0;
  let local = value + params.pop_local * (value - mean);
  return clamp(0.5 + (local - 0.5) * (1.0 + params.pop_contrast), 0.0, 1.0);
}

fn display_color(x: i32, y: i32) -> vec3f {
  let index = u32(y) * params.full_width + u32(x);
  var color = mono_color(popped(x, y, image[index]));
  if params.channels == 3u {
    let plane = params.full_width * params.full_height;
    color = vec3f(image[index], image[plane + index], image[2u * plane + index]);
  }
  // White outlines over green phosphor, amber over grayscale or colour.
  let ink = select(vec3f(1.0, 0.75, 0.2), vec3f(1.0), params.tint == 1u && params.channels != 3u);
  return mix(color, ink, clamp(outline_at(x, y) * params.outline_strength, 0.0, 1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if gid.x >= params.visible_width || gid.y >= params.visible_height { return; }

  let color = display_color(i32(gid.x + params.crop_x), i32(gid.y + params.crop_y));
  let encoded = vec3u(round(clamp(color, vec3f(0.0), vec3f(1.0)) * 255.0));

  // WebGPU and the browser expose mapped bytes in little-endian order.
  rgba[gid.y * params.visible_width + gid.x] = encoded.r | (encoded.g << 8u) | (encoded.b << 16u) | (255u << 24u);
}
