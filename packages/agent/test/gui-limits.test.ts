import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { stripTypeScriptTypes } from 'node:module'
import { test } from 'node:test'
import { Script } from 'node:vm'

// Exercise the actual embedded page and request sanitizer without starting a
// desktop process, joining a network, or touching the user's saved settings.
const source = readFileSync(new URL('../src/gui.ts', import.meta.url), 'utf8')
const pageStart = source.indexOf('function page(')
const pageEnd = source.indexOf('// ------------------------------------------------------------------ the app', pageStart)
const page = stripTypeScriptTypes(source.slice(pageStart, pageEnd))
const html = new Function(`${page}; return page('test')`)() as string
const script = /<script>([\s\S]*?)<\/script>/.exec(html)![1]!
const readLimits = /function readLimits\(s\) \{[\s\S]*?\n\}/.exec(script)![0]
const sanitizer = stripTypeScriptTypes(source.slice(source.indexOf('function sanitiseLimits('), pageStart))
const sanitize = new Function('raw', `${sanitizer}; return sanitiseLimits(raw)`)

function saved(workloads: string[], days: number[], from = '19:00', to = '07:00') {
  const available = ['echo', 'walker_evolution']
  const fields: Record<string, unknown> = {
    workloadChoices: { querySelectorAll: () => available.map(adapter => ({ checked: workloads.includes(adapter), dataset: { adapter } })) },
    dayChoices: { querySelectorAll: () => Array.from({ length: 7 }, (_, day) => ({ checked: days.includes(day), dataset: { day: String(day) } })) },
    budgetMinutes: { value: '0' }, fromTime: { value: from }, toTime: { value: to },
    maxLoad: { value: '0' }, minFreeRam: { value: '0' }, batteryBtn: { textContent: 'Off' },
  }
  const result = new Function('$', 's', `${readLimits}; return readLimits(s)`)((id: string) => fields[id], { adapters: available })
  return sanitize(result)
}

test('the generated GUI script parses', () => {
  assert.doesNotThrow(() => new Script(script))
})

test('disabling every workload survives the page and request sanitizer', () => {
  assert.deepEqual(saved([], [1, 2, 3, 4, 5]).workloads, [])
})

test('disabling every schedule day survives even when the hours span all day', () => {
  assert.deepEqual(saved(['echo'], [], '00:00', '00:00').schedule, { from: '00:00', to: '00:00', days: [] })
})

test('weekday overnight limits retain both the hours and explicit selected days', () => {
  assert.deepEqual(saved(['echo'], [1, 2, 3, 4, 5]), {
    workloads: ['echo'], schedule: { from: '19:00', to: '07:00', days: [1, 2, 3, 4, 5] },
  })
})

test('selecting every workload and day with all-day hours clears admission restrictions', () => {
  assert.deepEqual(saved(['echo', 'walker_evolution'], [0, 1, 2, 3, 4, 5, 6], '00:00', '00:00'), {})
})
