// Port of the wavelet detector (src/detection/wavelet_detector.py) with subpixel = False, i.e. detections at the
// peak pixels (the Gaussian-fit refinement uses scipy's curve_fit and runs only in Python). Float32 arithmetic is
// mirrored with Math.fround where numpy works in float32 (the movie, the filtered image, the noise estimate).
// Peak finding follows skimage.feature.peak_local_max(min_distance, threshold_abs, exclude_border=True).
// Positions: x = column, y = row, pixel-centre convention. Tested against Python in engines.test.mjs.
const f32 = Math.fround;

export function mexicanHatKernel(sigma) {
  let size = Math.trunc(6 * sigma + 1);
  if (size % 2 === 0) size += 1;
  const half = size >> 1, k = new Float64Array(size * size);
  let mean = 0;
  for (let y = -half; y <= half; y++) for (let x = -half; x <= half; x++) {
    const r2 = x * x + y * y, v = (1 - r2 / (2 * sigma ** 2)) * Math.exp(-r2 / (2 * sigma ** 2));
    k[(y + half) * size + x + half] = v; mean += v;
  }
  mean /= k.length;
  let abs = 0;
  for (let i = 0; i < k.length; i++) { k[i] -= mean; abs += Math.abs(k[i]); }
  for (let i = 0; i < k.length; i++) k[i] /= abs;
  return { k, size, half };
}

const reflect = (i, n) => { while (i < 0 || i >= n) i = i < 0 ? -i - 1 : 2 * n - i - 1; return i; };

// scipy.ndimage.convolve(image, kernel, mode='reflect') for a symmetric kernel; float32 output.
export function convolve(img, h, w, { k, size, half }) {
  const out = new Float32Array(h * w);
  const rows = Array.from({ length: h + 2 * half }, (_, i) => reflect(i - half, h));
  const cols = Array.from({ length: w + 2 * half }, (_, i) => reflect(i - half, w));
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = 0; j < size; j++) {
      const r = rows[y + j] * w;
      for (let i = 0; i < size; i++) s += k[j * size + i] * img[r + cols[x + i]];
    }
    out[y * w + x] = s;
  }
  return out;
}

function median32(a) {
  const s = Float32Array.from(a).sort(), n = s.length, m = n >> 1;
  return n % 2 ? s[m] : f32(f32(s[m - 1] + s[m]) / 2);
}

// Median absolute deviation noise estimate (float32, as numpy computes it on a float32 image).
export function estimateNoise(img) {
  const med = median32(img), dev = new Float32Array(img.length);
  for (let i = 0; i < img.length; i++) dev[i] = Math.abs(f32(img[i] - med));
  return f32(1.4826 * median32(dev));
}

// skimage peak_local_max(image, min_distance, threshold_abs) -> [[row, col], ...] brightest first.
export function peakLocalMax(img, h, w, minDistance, threshold) {
  const r = minDistance, mx = new Float32Array(h * w), tmp = new Float32Array(h * w);
  const cl = (v, n) => (v < 0 ? 0 : v >= n ? n - 1 : v);   // mode='nearest'
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = -Infinity;
    for (let d = -r; d <= r; d++) m = Math.max(m, img[y * w + cl(x + d, w)]);
    tmp[y * w + x] = m;
  }
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = -Infinity;
    for (let d = -r; d <= r; d++) m = Math.max(m, tmp[cl(y + d, h) * w + x]);
    mx[y * w + x] = m;
  }
  let trivial = true;
  for (let i = 0; i < img.length && trivial; i++) if (img[i] !== mx[i]) trivial = false;
  const cand = [];
  if (!trivial) {
    for (let y = r; y < h - r; y++) for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      if (img[i] === mx[i] && img[i] > threshold) cand.push([y, x, img[i]]);
    }
  }
  cand.sort((a, b) => b[2] - a[2]);   // stable: ties keep row-major order
  if (minDistance <= 1) return cand.map(c => [c[0], c[1]]);
  const kept = [];
  for (const c of cand) {
    if (kept.every(k => Math.max(Math.abs(k[0] - c[0]), Math.abs(k[1] - c[1])) >= minDistance)) kept.push(c);
  }
  return kept.map(c => [c[0], c[1]]);
}

// WaveletDetector.detect(frame) with subpixel False -> [[x, y, intensity, snr], ...]
export function detectFrame(frame, h, w, { sigma = 1.5, threshold = 3.0, min_distance = 3 } = {}) {
  const img = frame instanceof Float32Array ? frame : Float32Array.from(frame);
  const wav = convolve(img, h, w, mexicanHatKernel(sigma));
  const noise = estimateNoise(wav);
  const thr = f32(noise * threshold);
  return peakLocalMax(wav, h, w, Math.trunc(min_distance), thr).map(([y, x]) => {
    const intensity = img[y * w + x];
    return [x, y, intensity, noise > 0 ? f32(intensity / noise) : 0];
  });
}

// Every frame of a movie (Uint16Array T*H*W) -> per-frame detections.
export function detectMovie(data, [T, H, W], params) {
  const out = [];
  for (let t = 0; t < T; t++) out.push(detectFrame(Float32Array.from(data.subarray(t * H * W, (t + 1) * H * W)), H, W, params));
  return out;
}
