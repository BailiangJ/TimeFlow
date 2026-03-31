<script>

import Main from './components/Main.vue'
import Footer from './components/Footer.vue'

export default {
  components: {
    Main,
    Footer,
  }
}

</script>

<template>
    <div class="page-wrapper">
      <main class="page-main">
        <Main/>
      </main>
      <footer class="page-footer">
        <Footer/>
      </footer>

      <!-- Back to top -->
      <el-backtop :right="40" :bottom="80" />
    </div>
</template>

<style>

/* ===== Design Tokens ===== */
:root {
  --color-text-primary: #1a1a2e;
  --color-text-secondary: #4a4a6a;
  --color-text-muted: #7a7a96;
  --color-accent: #2a6fa8;
  --color-accent-hover: #1e5a8a;
  --color-bg-white: #ffffff;
  --color-bg-alt: #f8f9fb;
  --color-border: #e4e7ed;
  --color-divider: #dcdfe6;

  --font-heading: 'Inter', 'MyFont', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-body: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-serif: 'Source Serif 4', 'Georgia', serif;

  --max-width: 960px;
  --section-padding: 48px 24px;
  --card-radius: 8px;
  --card-shadow: 0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04);
  --card-shadow-hover: 0 4px 12px rgba(0, 0, 0, 0.08);
}

/* ===== Reset & Base ===== */
*, *::before, *::after {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  background: var(--color-bg-white);
  color: var(--color-text-primary);
  font-family: var(--font-body);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ===== Typography ===== */
h1 {
  font-family: var(--font-heading);
  font-weight: 700;
  font-size: 1.75rem;
  line-height: 1.3;
  letter-spacing: -0.01em;
  margin: 0;
  color: var(--color-text-primary);
}

h2 {
  font-family: var(--font-heading);
  font-weight: 600;
  font-size: 1.5rem;
  letter-spacing: -0.01em;
  color: var(--color-text-primary);
  margin: 0 0 16px 0;
}

h3, h4, h5, h6 {
  font-family: var(--font-heading);
  font-weight: 600;
  letter-spacing: -0.005em;
  color: var(--color-text-primary);
}

p, li {
  font-family: var(--font-body);
  font-size: 18px;
  line-height: 1.75;
  color: var(--color-text-primary);
}

code {
  font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
  border-radius: 4px;
  padding: 2px 6px;
  background: var(--color-bg-alt);
  color: var(--color-text-primary);
  font-size: 0.9em;
}

pre {
  border-radius: var(--card-radius);
  padding: 16px;
  background: var(--color-bg-alt);
  line-height: 1.5;
  border: 1px solid var(--color-border);
}

pre:not(pre[id=bibtex]) {
  overflow-x: auto;
}

pre code {
  font-size: 14px;
  color: var(--color-text-primary);
  border: none;
  padding: 0;
  background: none;
}

blockquote {
  border-left: 3px solid var(--color-accent);
  padding: 12px 20px;
  margin: 16px 0;
  background: var(--color-bg-alt);
  border-radius: 0 var(--card-radius) var(--card-radius) 0;
}

blockquote > p {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
  color: var(--color-text-primary);
}

/* ===== Links ===== */
a {
  color: var(--color-accent);
  text-decoration: none;
  transition: color 0.2s ease;
}

a:hover {
  color: var(--color-accent-hover);
  text-decoration: underline;
  border-bottom: none;
}

/* ===== Images ===== */
p > img {
  width: 90%;
  display: block;
  margin: 0 auto;
  border-radius: var(--card-radius);
  border: 1px solid var(--color-border);
}

/* ===== Tables ===== */
table {
  border-collapse: collapse;
  width: max-content;
  max-width: 100%;
  margin: 0 auto;
  display: block;
  overflow-x: auto;
  font-size: 14px;
}

thead {
  border-bottom: 2px solid var(--color-text-primary);
}

tbody tr:last-child {
  border-bottom: 2px solid var(--color-text-primary);
}

th, td {
  padding: 8px 16px;
}

/* ===== KaTeX ===== */
.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
}

/* ===== Layout ===== */
.page-wrapper {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

.page-main {
  flex: 1;
}

.page-footer {
  margin-top: auto;
}

/* ===== Element Plus Overrides ===== */
.el-header {
  padding: 0;
}

.el-footer {
  padding: 0;
}

</style>
