<script lang="ts" setup>
const links = {
    template: "https://github.com/JunyaoHu/academic-project-page-template-vue",
    nerfies: "https://github.com/nerfies/nerfies.github.io",
}
</script>

<template>
  <div class="footer-wrapper">
    <div class="footer-content">
      <p class="footer-text">
        Page template adapted from
        <a :href="links.template" target="_blank" rel="noopener">Academic Project Page Template</a>,
        originally based on
        <a :href="links.nerfies" target="_blank" rel="noopener">Nerfies</a>.
      </p>
    </div>
  </div>
</template>

<style scoped>

.footer-wrapper {
  background: var(--color-bg-alt);
  border-top: 1px solid var(--color-border);
  margin-top: 40px;
}

.footer-content {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 24px;
  text-align: center;
}

.footer-text {
  font-family: var(--font-body);
  font-size: 13px;
  color: var(--color-text-muted);
  margin: 0;
  line-height: 1.6;
}

.footer-text a {
  color: var(--color-text-muted);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.footer-text a:hover {
  color: var(--color-accent);
}

</style>
