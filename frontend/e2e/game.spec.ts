import {test, expect} from '@playwright/test'
import type {Page} from '@playwright/test'
import {readFileSync} from 'node:fs'
import type {Game, MovePreview} from '../src/lib/types'

const catalog = JSON.parse(readFileSync(new URL('../../data/levels/catalog.json', import.meta.url), 'utf8'))
async function open(page: Page): Promise<Game> {
  const response = page.waitForResponse(response => response.url().endsWith('/api/game/daily') && response.status() === 200)
  await page.goto('/')
  const game = await (await response).json()
  await expect(page.getByRole('dialog', {name: 'Welcome to Semantix.'})).toBeVisible()
  await page.getByRole('button', {name: 'Let’s play'}).click()
  await expect(page.getByRole('dialog', {name: 'Welcome to Semantix.'})).toHaveCount(0)
  await expect(page.getByLabel('Where will your next word take you?')).toBeEnabled()
  return game
}
async function submit(page: Page, word: string) {
  await page.getByLabel('Where will your next word take you?').fill(word)
  const response = page.waitForResponse(response => response.url().endsWith('/api/game/move'))
  await page.getByRole('button', {name: 'Make a move', exact: true}).click()
  return await response
}

test('real Google game: invalid entry, victory, statistics, refresh and daily deduplication', async ({page}, info) => {
  const errors: string[] = []; page.on('pageerror', error => errors.push(error.message))
  const game = await open(page)
  const solution = catalog.levels.find((level: {id: string}) => level.id === game.level_id).solution
  if (info.project.name === 'desktop') {
    await expect(page.locator('.map-3d canvas')).toBeVisible()
    await expect(page.locator('.map-label.start')).toBeVisible()
  }
  await page.screenshot({path: `test-results/${info.project.name}-initial.png`, fullPage: true})
  const invalid = await submit(page, 'asdfnotaword')
  expect(invalid.status()).toBe(400)
  await expect(page.getByRole('alert')).toContainText('vocabulary')
  await expect(page.locator('.moves-remaining strong')).toHaveText('06')
  for (const word of solution) {
    const response = await submit(page, word)
    expect(response.ok()).toBeTruthy()
    if ((await response.json()).status === 'won') break
  }
  await expect(page.getByText('You found your way.')).toBeVisible()
  await expect(page.locator('.history-table tbody tr')).toHaveCount(solution.length)
  await page.getByRole('button', {name: 'Replay journey'}).click()
  await page.getByRole('button', {name: /Your journey/}).click()
  await expect(page.locator('#history-content')).toBeHidden()
  await page.getByRole('button', {name: /Your journey/}).click()
  await expect(page.locator('#history-content')).toBeVisible()
  await page.getByRole('button', {name: 'Statistics', exact: true}).click()
  await expect(page.locator('.stats-grid > div').first()).toContainText('1')
  await page.getByRole('button', {name: 'Close dialog'}).click()
  await page.reload()
  await expect(page.getByText('You found your way.')).toBeVisible()
  await page.getByRole('button', {name: 'Retry route'}).click()
  for (const word of solution) {const response = await submit(page, word); if ((await response.json()).status === 'won') break}
  await expect(page.getByText('You found your way.')).toBeVisible()
  await page.getByRole('button', {name: 'Statistics', exact: true}).click()
  await expect(page.locator('.stats-grid > div').first()).toContainText('1')
  expect(errors).toEqual([])
})

test('free play loss and retry, hints, map controls, help and settings', async ({page}, info) => {
  await open(page)
  await page.getByRole('button', {name: 'Free play', exact: true}).click()
  await page.getByLabel('Difficulty').selectOption('easy')
  await expect(page.locator('.moves-remaining strong')).toHaveText('08')
  await page.getByRole('button', {name: /Need a nudge/}).click()
  await expect(page.locator('.hint-message')).toContainText('Consider the word')
  await page.getByRole('button', {name: 'Zoom in', exact: true}).click()
  await page.getByRole('button', {name: 'Reset map view'}).click()
  await page.getByRole('button', {name: 'How to play', exact: true}).click()
  await expect(page.getByRole('dialog')).toContainText('300-dimensional')
  await page.getByRole('button', {name: 'Close dialog'}).click()
  await page.getByRole('button', {name: 'Settings', exact: true}).click()
  await page.getByRole('switch', {name: /Reduced motion/}).check()
  await page.getByRole('button', {name: 'Close dialog'}).click()
  const start = await page.locator('.word-card strong').first().innerText()
  for (let i = 0; i < 8; i++) expect((await submit(page, start)).ok()).toBeTruthy()
  await expect(page.getByText('Every word is a new direction.')).toBeVisible()
  await page.getByRole('button', {name: 'Retry route'}).click()
  await expect(page.locator('.moves-remaining strong')).toHaveText('08')
  await expect(page.locator('.word-card strong').first()).toHaveText(start)
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy()
  await page.screenshot({path: `test-results/${info.project.name}-free-play.png`, fullPage: true})
  await page.reload()
  await expect(page.locator('.moves-remaining strong')).toHaveText('08')
  await page.getByRole('button', {name: 'Settings', exact: true}).click()
  await expect(page.getByRole('switch', {name: /Reduced motion/})).toBeChecked()
})


test('welcome, preview preferences, normalization and consistent movement', async ({page}, info) => {
  const game = await open(page)
  const field = page.getByLabel('Where will your next word take you?')
  if (info.project.name === 'desktop') await expect(field).toBeFocused()
  let previewRequests = 0
  page.on('request', request => {if (request.url().endsWith('/api/game/preview')) previewRequests++})
  await field.fill('prepared')
  await page.waitForTimeout(500)
  expect(previewRequests).toBe(0)
  await page.getByRole('button', {name: 'Clear word'}).click()
  await expect(field).toHaveValue('')
  await page.getByRole('button', {name: 'Settings', exact: true}).click()
  await expect(page.getByRole('switch', {name: /Daily Challenge previews/})).not.toBeChecked()
  await expect(page.getByRole('switch', {name: /Free Play previews/})).toBeChecked()
  await page.getByRole('switch', {name: /Daily Challenge previews/}).check()
  await page.getByRole('button', {name: 'Close dialog'}).click()
  const previewResponse = page.waitForResponse(response => response.url().endsWith('/api/game/preview') && response.ok())
  await field.fill('Prepared')
  const preview: MovePreview = await (await previewResponse).json()
  expect(preview.resolved_word).toBe('prepared')
  expect(preview.resolution).toBe('normalized')
  await expect(page.locator('.preview-feedback')).toContainText('"Prepared" accepted using "prepared"')
  await expect(page.locator('.moves-remaining strong')).toHaveText('06')
  const state = await (await page.request.get(`/api/game/history/${game.id}`)).json()
  expect(state.moves_used).toBe(0)
  const moveResponse = page.waitForResponse(response => response.url().endsWith('/api/game/move'))
  await field.press('Enter')
  const updated: Game = await (await moveResponse).json()
  expect(updated.history[0].position).toEqual(preview.position)
  expect(updated.history[0].position3d).toEqual(preview.position3d)
  expect(updated.history[0].delta).toBe(preview.delta)
  await expect(page.locator('.history-word')).toContainText('Prepared')
  await expect(page.locator('.move-feedback')).toContainText('"Prepared" accepted using "prepared"')
  await expect(page.locator('.history-normalization')).toContainText('prepared')
  expect((await submit(page, game.target_word)).status()).toBe(400)
  await expect(page.getByRole('alert')).toContainText('target')
  await expect(page.locator('.moves-remaining strong')).toHaveText('05')
  await page.reload()
  await expect(field).toBeEnabled()
  await expect(page.getByRole('dialog')).toHaveCount(0)
  await expect(page.locator('.history-word')).toContainText('Prepared')
})

test('new input cancels stale previews and map modes remain accessible', async ({page}) => {
  await open(page)
  await page.getByRole('button', {name: 'Free play', exact: true}).click()
  const field = page.getByLabel('Where will your next word take you?')
  await expect(field).toBeEnabled()
  await page.getByRole('button', {name: '2D', exact: true}).click()
  await expect(page.locator('.semantic-map-shell')).toHaveAttribute('data-view-mode', '2d')
  const three = page.getByRole('button', {name: '3D', exact: true})
  if (await three.isEnabled()) {
    await three.click()
    await expect.poll(async () => await three.getAttribute('aria-pressed') === 'true' || await three.isDisabled()).toBe(true)
  }
  await page.getByRole('button', {name: '2D', exact: true}).click()
  await page.route('**/api/game/preview', async route => {
    const response = await route.fetch()
    if (route.request().postDataJSON().word === 'prepared') await new Promise(resolve => setTimeout(resolve, 700))
    await route.fulfill({response})
  })
  const first = page.waitForRequest(request => request.url().endsWith('/api/game/preview') && request.postDataJSON().word === 'prepared')
  await field.fill('prepared')
  await first
  await field.fill('WALKED')
  await expect(page.locator('.preview-feedback')).toContainText('"WALKED" accepted using "walked"')
  await page.waitForTimeout(750)
  await expect(page.locator('.preview-feedback')).not.toContainText('prepared')
  await page.getByRole('button', {name: 'Clear word'}).click()
  await expect(page.locator('.preview-feedback')).toHaveCount(0)
  await page.unrouteAll({behavior: 'wait'})
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy()
})

test('2D fallback remains playable without WebGL', async ({page}) => {
  await page.addInitScript(() => {
    const original = HTMLCanvasElement.prototype.getContext
    HTMLCanvasElement.prototype.getContext = function (...args: Parameters<typeof original>) {
      if (String(args[0]).includes('webgl')) return null
      return original.apply(this, args)
    } as typeof original
  })
  const game = await open(page)
  const three = page.getByRole('button', {name: '3D', exact: true})
  if (await three.isEnabled()) await three.click()
  await expect(page.locator('.semantic-map-shell')).toHaveAttribute('data-view-mode', '2d')
  await expect(page.locator('.map-fallback')).toContainText('WebGL is unavailable')
  expect((await submit(page, game.start_word)).ok()).toBeTruthy()
  await expect(page.locator('.moves-remaining strong')).toHaveText('05')
})

test('broader guesses work while puzzle words stay curated', async ({page}) => {
  const game = await open(page)
  const oldWords = new Set(readFileSync(new URL('../../data/vocabulary/words.txt', import.meta.url), 'utf8').split(/\s+/))
  const status = await (await page.request.get('/api/model/status')).json()
  expect(status.vocabulary_size).toBeGreaterThan(50_000)
  expect(status.target_vocabulary_size).toBeLessThan(10_000)
  expect(status.memory_mapped).toBe(true)
  expect(oldWords.has(game.target_word)).toBe(true)
  expect(oldWords.has('photosynthesis')).toBe(false)
  const response = await submit(page, 'photosynthesis')
  expect(response.ok()).toBeTruthy()
  const updated: Game = await response.json()
  expect(updated.history[0].resolved_word).toBe('photosynthesis')
  expect(updated.history[0].resolution).toBe('exact')
  expect(updated.history[0].semantic?.rank).toBeGreaterThan(0)
  expect(updated.history[0].semantic?.vocabulary_size).toBeGreaterThan(50_000)
  await expect(page.locator('.move-feedback')).toContainText('Word rank #')
  expect(status.hint_vocabulary_size).toBeGreaterThan(2000)
  await expect(page.locator('.history-word')).toContainText('photosynthesis')
  expect((await submit(page, 'fucking')).status()).toBe(400)
  await expect(page.locator('.moves-remaining strong')).toHaveText('05')
})
