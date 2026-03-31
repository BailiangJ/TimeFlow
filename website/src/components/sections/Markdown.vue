<script setup>
import page from '../mds/md.mdx';
</script>

<template>
  <div class="section-wrapper">
    <page />
  </div>
</template>

<style scoped>
.section-wrapper {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 0 24px 8px 24px;
}

.section-wrapper :deep(h1) {
  font-family: "MyFont", Verdana, sans-serif;
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--color-text-primary);
  margin: 32px 0 16px 0;
}

.section-wrapper :deep(p) {
  text-align: justify;
}

.section-wrapper :deep(hr) {
  border: none;
  border-top: 1px solid var(--color-divider);
  margin: 36px 0;
}

.section-wrapper :deep(img) {
  border-radius: 6px;
  border: 1px solid var(--color-border);
}
</style>
