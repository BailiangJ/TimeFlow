<script setup>
import { ref, computed } from 'vue';

// 21 steps from t=0.00 to t=2.00 in 0.10 increments
const steps = Array.from({ length: 21 }, (_, i) => (i * 0.1).toFixed(2));

const sliderVal = ref(0);
const tStr = computed(() => sliderVal.value.toFixed(2));
const isExtrapolation = computed(() => sliderVal.value > 1.0);

const ticks = [0.0, 0.5, 1.0, 1.5, 2.0];
</script>

<template>
  <div class="section-wrapper">
    <hr class="section-divider" />
    <h2 class="section-heading">Brain Aging Timeline</h2>
    <p class="section-desc">
      TimeFlow takes just two brain scans from one person — a baseline and a follow-up — and predicts brain anatomy at any future point in time.
      Drag the slider to watch the brain change:
      the <span class="label-interp">blue region</span> shows predictions within the observed window (<em>interpolation</em>),
      and the <span class="label-extrap">gold region</span> shows forecasts beyond the last scan (<em>extrapolation</em>).
      The left panel shows the warped baseline; the right shows the deformation field.
    </p>

    <!-- Static ground-truth reference images -->
    <div class="gt-row">
      <div class="gt-item">
        <div class="rot-wrap gt-wrap">
          <img class="gt-img" src="/frames/gt_t0.png" alt="Baseline MRI" />
        </div>
        <span class="gt-label">Baseline <em>(source)</em></span>
      </div>
      <div class="gt-item">
        <div class="rot-wrap gt-wrap">
          <img class="gt-img" src="/frames/gt_t4.png" alt="Target MRI" />
        </div>
        <span class="gt-label">Target <em>(~4 yrs later)</em></span>
      </div>
      <div class="gt-item extrap-ref">
        <div class="rot-wrap gt-wrap">
          <img class="gt-img" src="/frames/gt_t8.png" alt="Extrapolation endpoint" />
        </div>
        <span class="gt-label">Prospective Observation <em>(~8 yrs)*</em></span>
      </div>
    </div>
    <p class="gt-note">*Ground truth at ~8 yrs shown for reference; TimeFlow predicts this without seeing it during training.</p>

    <!-- Slider -->
    <div class="slider-section">
      <div class="slider-labels" style="display: flex; width: 100%; margin-bottom: 8px;">
        <div style="width: 50%; text-align: center;">
          <span class="mode-badge interp">Interpolation</span>
        </div>
        <div style="width: 50%; text-align: center;">
          <span class="mode-badge extrap">Extrapolation</span>
        </div>
      </div>

      <div class="slider-wrap">
        <div class="track-bg"></div>
        <div class="track-fill-interp" :style="{ width: Math.min(sliderVal, 1.0) / 2 * 100 + '%' }"></div>
        <div class="track-fill-extrap" v-if="isExtrapolation"
             :style="{ left: '50%', width: (sliderVal - 1.0) / 2 * 100 + '%' }"></div>
        <input
          type="range"
          min="0" max="2" step="0.1"
          v-model.number="sliderVal"
          class="timeline-slider"
        />
      </div>

      <!-- Tick marks -->
      <div class="ticks-row">
        <div v-for="tick in ticks" :key="tick" class="tick" :style="{ left: (tick/2*100) + '%' }">
          <div class="tick-line"></div>
          <div class="tick-text">{{ tick.toFixed(1) }}</div>
        </div>
      </div>
      <p class="slider-note">
        t&nbsp;=&nbsp;1.0 corresponds to the target visit (registered endpoint);
        t&nbsp;&gt;&nbsp;1.0 is forecasting beyond observed data.
      </p>
    </div>

    <!-- Side-by-side predicted frames (all in DOM, instant opacity toggle) -->
    <div class="frame-row">
      <div class="frame-panel">
        <div class="frame-label">Deformed Baseline to time t</div>
        <div class="rot-wrap warped-wrap">
          <img
            v-for="step in steps"
            :key="step"
            :src="`/frames/warped_t${step}.png`"
            :class="['stacked-img', { active: tStr === step }]"
            :alt="`Predicted brain at t=${step}`"
          />
        </div>
      </div>
      <div class="frame-panel">
        <div class="frame-label">Deformation Field at time t</div>
        <div class="rot-wrap grid-wrap">
          <img
            v-for="step in steps"
            :key="step"
            :src="`/frames/grid_t${step}.png`"
            :class="['stacked-img', { active: tStr === step }]"
            :alt="`Deformation field at t=${step}`"
          />
        </div>
      </div>
    </div>

  </div>
</template>

<style scoped>
.section-wrapper {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 0 24px 8px 24px;
}

.section-divider {
  border: none;
  border-top: 1px solid var(--color-divider);
  margin: 8px 0 32px 0;
}

.section-heading {
  font-family: "MyFont", Verdana, sans-serif;
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--color-text-primary);
  margin: 0 0 12px 0;
}

.section-desc {
  font-size: 18px;
  line-height: 1.7;
  color: var(--color-text-primary);
  margin: 0 0 28px 0;
}

.label-interp { color: var(--color-accent); font-weight: 600; }
.label-extrap  { color: #8a7340; font-weight: 600; }

/* ── Rotation wrappers ── */
/*
 * Images are landscape (600×500 warped/gt, 600×426 grid).
 * After rotate(-90deg) they appear portrait.
 * Container uses the post-rotation aspect ratio (H_i / W_i).
 * Image inside is sized to exactly fill the container after rotation:
 *   CSS width  = H_i/W_i × container_width  (= container_height)
 *   CSS height = W_i/H_i × container_height (= container_width)
 *   Position: centered with translate(-50%,-50%) + rotate(-90deg)
 */
.rot-wrap {
  position: relative;
  width: 100%;
  overflow: hidden;
  border-radius: 6px;
  border: 1px solid var(--color-border);
}

/* all brain/grid images: 600×500 → post-rotation aspect = 500/600 */
.gt-wrap,
.warped-wrap,
.grid-wrap {
  aspect-ratio: 500 / 600;
}

/* Single gt image (not stacked) */
/* Container aspect-ratio 500/600 → C_h = C_w × 600/500.
   After -90deg: visual_w = CSS height, visual_h = CSS width.
   To fill: CSS width = C_h → 120% of C_w; CSS height = C_w → 83.33% of C_h. */
.gt-img {
  position: absolute;
  width:  calc(600 / 500 * 100%);  /* = W_i/H_i × 100% of container width = C_h */
  height: calc(500 / 600 * 100%);  /* = H_i/W_i × 100% of container height = C_w */
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%) rotate(-90deg);
  transform-origin: center center;
  display: block;
}

.gt-item.extrap-ref .rot-wrap {
  border-color: #c4a35a;
}

.gt-item.extrap-ref .gt-img {
  filter: brightness(1.2) contrast(0.8);
}

/* Stacked frame images — shared base */
.stacked-img {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%) rotate(-90deg);
  transform-origin: center center;
  display: block;
  opacity: 0;
  will-change: opacity;
}
.stacked-img.active {
  opacity: 1;
}

/* Warped + grid panel images: 600×500, same as gt */
.warped-wrap .stacked-img,
.grid-wrap .stacked-img {
  width:  calc(600 / 500 * 100%);  /* = C_h */
  height: calc(500 / 600 * 100%);  /* = C_w */
}

/* ── Ground-truth reference row ── */
.gt-row {
  display: flex;
  gap: 16px;
  justify-content: center;
  margin-bottom: 4px;
}

.gt-item {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  max-width: 260px;
}

.gt-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
  text-align: center;
}

.gt-note {
  font-size: 11px;
  color: var(--color-text-muted);
  text-align: center;
  margin: 0 0 24px 0;
  font-style: italic;
}

/* ── Slider ── */
.slider-section {
  margin-bottom: 28px;
}

.slider-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}

.mode-badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  padding: 3px 10px;
  border-radius: 12px;
}
.mode-badge.interp {
  background: rgba(42, 111, 168, 0.1);
  color: var(--color-accent);
}
.mode-badge.extrap {
  background: rgba(196, 163, 90, 0.15);
  color: #8a7340;
}

.t-label {
  font-family: 'JetBrains Mono', 'Consolas', monospace;
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.slider-wrap {
  position: relative;
  height: 20px;
  display: flex;
  align-items: center;
}

.track-bg,
.track-fill-interp,
.track-fill-extrap {
  position: absolute;
  height: 4px;
  border-radius: 2px;
  pointer-events: none;
}

.track-bg {
  left: 0; right: 0;
  background: var(--color-border);
}

.track-fill-interp {
  left: 0;
  background: var(--color-accent);
  opacity: 0.6;
  z-index: 1;
}

.track-fill-extrap {
  background: #c4a35a;
  opacity: 0.6;
  z-index: 1;
}

.timeline-slider {
  position: relative;
  z-index: 2;
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  height: 4px;
  border-radius: 2px;
  background: transparent;
  outline: none;
  cursor: pointer;
}

.timeline-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--color-accent);
  cursor: pointer;
  border: 2px solid #fff;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}

.timeline-slider::-moz-range-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: var(--color-accent);
  cursor: pointer;
  border: 2px solid #fff;
  box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}

/* Tick marks */
.ticks-row {
  position: relative;
  height: 28px;
  margin-top: 2px;
}

.tick {
  position: absolute;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.tick-line {
  width: 1px;
  height: 6px;
  background: var(--color-text-muted);
  opacity: 0.5;
}

.tick-text {
  font-size: 11px;
  color: var(--color-text-muted);
  margin-top: 2px;
}

.slider-note {
  font-size: 12px;
  color: var(--color-text-muted);
  margin: 6px 0 0 0;
  font-style: italic;
  text-align: center;
}

/* ── Side-by-side frame panels ── */
.frame-row {
  display: flex;
  gap: 16px;
  justify-content: center;
}

.frame-panel {
  flex: 1;
  max-width: 260px;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  gap: 8px;
}

.frame-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-muted);
  text-align: center;
}

.frame-caption {
  font-size: 11px;
  color: var(--color-text-muted);
  text-align: center;
  font-style: italic;
}

/* ── Mobile ── */
@media (max-width: 768px) {
  .gt-row {
    flex-direction: column;
    align-items: center;
  }
  .gt-item {
    max-width: 100%;
    width: 80%;
  }
  .frame-row {
    flex-direction: column;
  }
}
</style>
