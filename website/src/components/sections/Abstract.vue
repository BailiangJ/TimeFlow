<script setup>
import abstract from '../mds/abstract.mdx';
</script>

<template>
  <div class="section-wrapper">
    <hr class="section-divider" />
    <abstract />
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

.section-wrapper :deep(h1) {
  font-family: "MyFont", Verdana, sans-serif;
  font-size: 1.75rem;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--color-text-primary);
  margin: 0 0 16px 0;
}

.section-wrapper :deep(p) {
  text-align: justify;
}

.section-wrapper :deep(p > img) {
  width: 100%;
  margin: 8px auto 16px auto;
  border-radius: 6px;
  border: 1px solid var(--color-border);
  box-shadow: none;
}
</style>
