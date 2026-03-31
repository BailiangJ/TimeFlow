<script lang="ts" setup>

import { ElIcon } from 'element-plus'
import { Document, Files } from '@element-plus/icons-vue'

// Title
const title = 'TimeFlow: Temporal Conditioning for Longitudinal Brain MRI Registration and Aging Analysis'

// Button color
const btn_color = '#2a6fa8'

// Authors
const authors_line1 = [
  { name: "Bailiang Jian", homepage: "", flags: "1,2,3" },
  { name: "Jiazhen Pan", homepage: "", flags: "1,2,3" },
  { name: "Yitong Li", homepage: "", flags: "1,2,3" },
  { name: "Fabian Bongratz", homepage: "", flags: "1,2,3" },
  { name: "Ruochen Li", homepage: "", flags: "1" },
]
const authors_line2 = [
  { name: "Daniel Rueckert", homepage: "", flags: "1,2,3,4,5" },
  { name: "Benedikt Wiestler", homepage: "", flags: "1,2,3,4,†" },
  { name: "Christian Wachinger", homepage: "", flags: "1,2,3,4,†" }
]

// Affiliations
const addresses_line1 = [
  { flag: "1", name: "Technical University of Munich" },
  { flag: "2", name: "TUM University Hospital" },
  { flag: "3", name: "Munich Center for Machine Learning (MCML)" },
]
const addresses_line2 = [
  { flag: "4", name: "Munich Data Science Institute (MDSI)" },
  { flag: "5", name: "Imperial College London" }
]

const note = "†: Equal Senior Supervisors."

const venue = "IEEE Transactions on Medical Imaging (TMI)"

// Buttons
const buttons = [
  {
    disabled: true,
    name: "Paper",
    component: Document,
  },
  {
    disabled: false,
    name: "Code",
    link: "https://github.com/BailiangJ/TimeFlow",
    component: Files,
  },
]

</script>

<template>
  <div class="title-section">

    <!-- Paper Title -->
    <h1 class="paper-title">
      <img src="/TimeFlow_icon.png" class="title-logo" alt="TimeFlow Logo" />
      {{ title }}
    </h1>

    <!-- Authors Line 1 -->
    <div class="authors-row">
      <span v-for="(author, idx) in authors_line1" :key="idx" class="author-block">
        <a v-if="author.homepage" :href="author.homepage" class="author-name">{{ author.name }}</a>
        <span v-else class="author-name">{{ author.name }}</span><sup>{{ author.flags }}</sup><span v-if="idx < authors_line1.length - 1">,&nbsp;</span>
      </span>
    </div>
    <!-- Authors Line 2 -->
    <div class="authors-row">
      <span v-for="(author, idx) in authors_line2" :key="idx" class="author-block">
        <a v-if="author.homepage" :href="author.homepage" class="author-name">{{ author.name }}</a>
        <span v-else class="author-name">{{ author.name }}</span><sup>{{ author.flags }}</sup><span v-if="idx < authors_line2.length - 1">,&nbsp;</span>
      </span>
    </div>

    <!-- Affiliations Line 1 -->
    <div class="affiliations-row">
      <span v-for="(addr, idx) in addresses_line1" :key="idx" class="affiliation-block">
        <sup>{{ addr.flag }}</sup>{{ addr.name }}<span v-if="idx < addresses_line1.length - 1">&nbsp;&nbsp;&nbsp;</span>
      </span>
    </div>
    <!-- Affiliations Line 2 -->
    <div class="affiliations-row">
      <span v-for="(addr, idx) in addresses_line2" :key="idx" class="affiliation-block">
        <sup>{{ addr.flag }}</sup>{{ addr.name }}<span v-if="idx < addresses_line2.length - 1">&nbsp;&nbsp;&nbsp;</span>
      </span>
    </div>

    <!-- Note -->
    <div class="note-row">{{ note }}</div>

    <!-- Venue -->
    <div class="venue-row">{{ venue }}</div>

    <!-- Buttons -->
    <div class="buttons-row">
      <a :href="button.link" v-for="button in buttons" :key="button.name">
        <el-button class="guidance-button" size="default" :color="btn_color" :disabled="button.disabled" round>
          <el-icon :size="16">
            <component :is="button.component" />
          </el-icon>
          <span class="btn-text">{{ button.name }}</span>
        </el-button>
      </a>
    </div>

  </div>
</template>

<style scoped>

.title-section {
  max-width: var(--max-width);
  margin: 0 auto;
  padding: 56px 24px 24px 24px;
  text-align: center;
}

.paper-title {
  font-family: "MyFont", Verdana, sans-serif;
  font-weight: 700;
  font-size: 2rem;
  line-height: 1.35;
  margin: 0 0 24px 0;
  letter-spacing: 1px;
  display: flex;
  align-items: center;
  justify-content: center;
  text-align: left;
  color: var(--color-text-primary);
}

.title-logo {
  height: 3.9em;
  margin-right: 20px;
  flex-shrink: 0;
}

.authors-row {
  font-size: 1rem;
  line-height: 1.7;
  margin: 0 0 2px 0;
  color: var(--color-text-secondary);
}

.author-name {
  color: var(--color-accent);
}

.author-block a.author-name {
  color: var(--color-accent);
  text-decoration: none;
}

.author-block a.author-name:hover {
  text-decoration: underline;
  border-bottom: none;
}

.author-block sup {
  font-size: 0.65rem;
  color: var(--color-text-muted);
  margin-left: 1px;
}

.affiliations-row {
  font-size: 0.85rem;
  line-height: 1.6;
  margin: 0 0 2px 0;
  color: var(--color-text-muted);
}

.affiliation-block sup {
  font-size: 0.6rem;
  color: var(--color-text-muted);
  margin-right: 1px;
}

.note-row {
  font-size: 0.8rem;
  color: var(--color-text-muted);
  margin: 10px 0 6px 0;
  font-style: italic;
}

.venue-row {
  font-size: 1rem;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 8px 0 20px 0;
  letter-spacing: -0.005em;
}

.buttons-row {
  display: flex;
  justify-content: center;
  gap: 12px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.guidance-button {
  box-shadow: var(--card-shadow);
  border: none;
}

.guidance-button:hover:not(:disabled) {
  box-shadow: var(--card-shadow-hover);
}

.btn-text {
  font-family: var(--font-heading);
  font-size: 14px;
  font-weight: 500;
  color: #ffffff;
  margin-left: 5px;
}

/* Remove link underline decorations for buttons */
.buttons-row a {
  text-decoration: none;
  color: inherit;
}

.buttons-row a:hover {
  color: inherit;
  border-bottom: none;
  text-decoration: none;
}

</style>
