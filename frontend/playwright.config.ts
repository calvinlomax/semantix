import {defineConfig, devices} from '@playwright/test'

export default defineConfig({
  testDir: './e2e', timeout: 30000, workers: 1, fullyParallel: false,
  use: {baseURL: 'http://127.0.0.1:5173', channel: 'chrome', trace: 'retain-on-failure'},
  projects: [
    {name: 'desktop', use: {...devices['Desktop Chrome'], viewport: {width: 1440, height: 1100}}},
    {name: 'mobile', use: {...devices['iPhone 13'], browserName: 'chromium'}},
  ],
  webServer: [
    {command: '../.venv/bin/python -m uvicorn backend.app.main:app --app-dir .. --host 127.0.0.1 --port 8000', url: 'http://127.0.0.1:8000/api/health', reuseExistingServer: true},
    {command: 'npm run dev', url: 'http://127.0.0.1:5173', reuseExistingServer: true},
  ],
})
