// Canvas viewer for an image stack with detection, track and ground-truth overlays.
// Pixel-centre convention (as the detectors and the simulator): a particle at x = 10 sits in the middle of column 10.
import { trackColor } from './data.js';

export class Viewer {
  // host: element; size: [H, W]; scale: canvas pixels per image pixel for the overlay layer.
  constructor(host, [h, w], scale = 4) {
    Object.assign(this, { h, w, scale });
    host.classList.add('viewer');
    host.style.aspectRatio = `${w} / ${h}`;
    this.img = Object.assign(document.createElement('canvas'), { width: w, height: h, className: 'img' });
    this.ov = Object.assign(document.createElement('canvas'), { width: w * scale, height: h * scale });
    this.img.setAttribute('aria-hidden', 'true');
    this.ov.setAttribute('role', 'img');
    host.append(this.img, this.ov);
  }

  // Draw one frame (Uint16Array or Float array of H*W) with a linear lo..hi grey map.
  drawFrame(px, lo, hi) {
    const g = this.img.getContext('2d'), im = g.createImageData(this.w, this.h);
    const k = 255 / Math.max(1, hi - lo);
    for (let i = 0; i < px.length; i++) {
      const v = Math.max(0, Math.min(255, (px[i] - lo) * k));
      im.data[4 * i] = v * 0.75; im.data[4 * i + 1] = v; im.data[4 * i + 2] = v * 0.85; im.data[4 * i + 3] = 255;
    }
    g.putImageData(im, 0, 0);
  }

  drawImage(image) {
    this.img.getContext('2d').drawImage(image, 0, 0, this.w, this.h);
  }

  // o: { frame, particles: [[x, y, ...]] for this frame, tracks: [{track_id, frames, x, y}], tail, colorOf(track),
  //      truth: [{x, y}] for this frame, label }
  overlay({ frame = null, particles = [], tracks = [], tail = 12, colorOf = t => trackColor(t.track_id), truth = [], label = '' }) {
    const g = this.ov.getContext('2d'), s = this.scale, c = v => (v + 0.5) * s;
    g.clearRect(0, 0, this.ov.width, this.ov.height);
    g.lineWidth = 1.6;
    for (const t of tracks) {
      g.strokeStyle = colorOf(t); g.beginPath();
      let started = false;
      t.frames.forEach((f, i) => {
        if (frame !== null && (f > frame || f < frame - tail)) return;
        if (!started) { g.moveTo(c(t.x[i]), c(t.y[i])); started = true; } else g.lineTo(c(t.x[i]), c(t.y[i]));
      });
      g.stroke();
    }
    g.strokeStyle = '#ffffff'; g.lineWidth = 1.2;
    for (const p of particles) { g.beginPath(); g.arc(c(p[0]), c(p[1]), 3.2 * s / 2, 0, 2 * Math.PI); g.stroke(); }
    g.strokeStyle = '#ff6b8b'; g.lineWidth = 1.4;
    for (const p of truth) {
      const x = c(p.x), y = c(p.y), r = 4;
      g.beginPath(); g.moveTo(x - r, y - r); g.lineTo(x + r, y + r); g.moveTo(x + r, y - r); g.lineTo(x - r, y + r); g.stroke();
    }
    this.ov.setAttribute('aria-label', label);
  }
}
