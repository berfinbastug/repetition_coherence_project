import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Replace 'your-repo-name' with your actual GitHub repository name
export default defineConfig({
  plugins: [react()],
  base: 'https://github.com/berfinbastug/repetition_coherence_project.git',
})
