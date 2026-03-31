<script>
export default {
  data() {
    return {
      bibtex: [
        "@article{jian2026temporal,",
        "    title={Temporal Conditioning for Longitudinal Brain MRI Registration and Aging Analysis},",
        "    author={Jian, Bailiang and Pan, Jiazhen and Li, Yitong and Bongratz, Fabian",
        "            and Li, Ruochen and Rueckert, Daniel and Wiestler, Benedikt and Wachinger, Christian},",
        "    journal={IEEE Transactions on Medical Imaging},",
        "    year={2026},",
        "    publisher={IEEE}",
        "}",
      ],
    }
  },
  methods: {
    copyVal() {
      let text = document.getElementById('bibtex').innerText;
      navigator.clipboard.writeText(text).then(() => {
        this.$message.success('Copied to clipboard');
      }).catch(() => {
        // Fallback
        let oInput = document.createElement('textarea');
        oInput.value = text;
        document.body.appendChild(oInput);
        oInput.select();
        document.execCommand('Copy');
        this.$message.success('Copied to clipboard');
        oInput.remove();
      });
    }
  }
}

</script>

<template>
  <div class="section-wrapper">
    <hr class="section-divider" />
    <h2 class="section-heading">BibTeX</h2>

    <div class="bibtex-box" @click="copyVal()">
      <div class="copy-hint">📋 Click to copy</div>
      <pre id="bibtex"><code v-for="b in bibtex">{{ b }}<br/></code></pre>
    </div>
  </div>
</template>

<style scoped>

.section-wrapper {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 0 24px 32px 24px;
}

.section-divider {
  border: none;
  border-top: 1px solid var(--color-divider);
  margin: 8px 0 32px 0;
}

.section-heading {
  font-family: "MyFont", Verdana, sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: 1px;
  color: var(--color-text-primary);
  margin: 0 0 16px 0;
}

.bibtex-box {
  background: var(--color-bg-alt);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 16px 20px;
  cursor: pointer;
  max-width: 640px;
  margin: 0 auto;
  transition: border-color 0.2s ease;
}

.bibtex-box:hover {
  border-color: var(--color-accent);
}

.bibtex-box:active {
  background: #f0f2f5;
}

.copy-hint {
  text-align: center;
  color: var(--color-text-muted);
  font-size: 0.85rem;
  margin-bottom: 8px;
  font-family: var(--font-body);
}

pre {
  border: none;
  border-radius: 0;
  padding: 8px 0;
  background: none;
  margin: 0;
  overflow-x: auto;
  white-space: pre;
}

pre code {
  font-size: 13px;
  color: var(--color-text-secondary);
  background: none;
  font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
}

@media (max-width: 768px) {
  pre code {
    font-size: 11px;
  }
}

</style>
