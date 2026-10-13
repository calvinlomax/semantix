import {test, expect} from '@playwright/test'
import type {Page} from '@playwright/test'
import {readFileSync} from 'node:fs'
import type {Game} from '../src/lib/types'

const catalog = JSON.parse(readFileSync(new URL('../../data/levels/catalog.json', import.meta.url), 'utf8'))
async function open(page: Page): Promise<Game> {
  const response = page.waitForResponse(response => response.url().endsWith('/api/game/daily') && response.status() === 200)
  await page.goto('/')
  const game = await (await response).json()
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
  await expect(page.getByRole('status')).toContainText('Consider the word')
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
