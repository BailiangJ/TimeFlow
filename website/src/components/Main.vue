<script>
import Title from './sections/Title.vue'
import Abstract from './sections/Abstract.vue'
import Markdown from './sections/Markdown.vue'
import Timeline from './sections/Timeline.vue'
import BibTeX from './sections/BibTeX.vue'

export default {
  components: {
    Title,
    Abstract,
    Timeline,
    Markdown,
    BibTeX
  }
}
</script>

<template>
  <Title/>
  <Timeline/>
  <Abstract/>
  <Markdown/>
  <BibTeX/>
</template>
