import assert from 'node:assert/strict'
import { test } from 'node:test'
import { dsin, dcos, dtanh, dlog, dexp } from '@dwp/protocol/dmath.js'
import { evaluate, perturb, randomGenome, GENOME_SIZE } from '@dwp/protocol/walker.js'
import { runEcho } from '../src/adapters/echo.ts'
import { runWalker } from '../src/adapters/walker.ts'

function bits(value: number): bigint {
  const bytes = Buffer.alloc(8)
  bytes.writeDoubleBE(value)
  return bytes.readBigUInt64BE()
}

function fromBits(value: bigint): number {
  const bytes = Buffer.alloc(8)
  bytes.writeBigUInt64BE(value)
  return bytes.readDoubleBE()
}

// Pinned in ios/DWPAgentKit/Tests/DWPAgentKitTests/WalkerConformanceTests.swift.
// Exact IEEE-754 comparisons catch drift hidden by rounded output or numeric tolerances.
const mathVectors = [
  [0x0n, 0x0n, 0x3ff0000000000000n, 0x0n, 0xc034b927f32bffb8n, 0x3ff0000000000000n],
  [0x3ff0000000000000n, 0x3feaed548f090ceen, 0x3fe14a280fb5068cn, 0x3fe85efab514f394n, 0x3e112e0bffdb1b44n, 0x4005bf0a8b14576an],
  [0xbfe0000000000000n, 0xbfdeaee8744b05f0n, 0x3fec1528065b7d50n, 0xbfdd9353d7568af3n, 0xbfe62e42fde75931n, 0x3fe368b2fc6f960an],
  [0x401921fb54442d18n, 0xbcb1a62633145c07n, 0x3ff0000000000000n, 0x3feffff15f81f9abn, 0x3ffd67f1c86fae97n, 0x4080bbeee9177e18n],
  [0xc059000000000000n, 0x3fe03425b78c4db8n, 0x3feb981dbf665fdfn, 0xbff0000000000000n, 0x40126bb1bbb58111n, 0x36ea8c1f14e2af5dn],
] as const

const gaitVectors = [
  { genomeSeed: 1, sigma: 0, seed: 0, steps: 300, fitness: 0x3ff6404ea4a8c155n, distance: 0xbfe1eb851eb851ecn, ticks: 52, fell: true },
  { genomeSeed: 1, sigma: 0, seed: 0, steps: 1800, fitness: 0x3fcf2b020c49ba5en, distance: 0xbfe1eb851eb851ecn, ticks: 52, fell: true },
  { genomeSeed: 7, sigma: 0.2, seed: 100001, steps: 1800, fitness: 0x3fd2631f8a0902den, distance: 0xbfc6872b020c49ban, ticks: 61, fell: true },
  { genomeSeed: 7, sigma: 0.2, seed: 100002, steps: 1800, fitness: 0x3fe7f559b3d07c85n, distance: 0xbfe116872b020c4an, ticks: 140, fell: true },
  { genomeSeed: 23, sigma: 0.05, seed: 900007, steps: 900, fitness: 0x3fca29c779a6b50bn, distance: 0x3fa1eb851eb851ecn, ticks: 19, fell: true },
  { genomeSeed: 3, sigma: 0.9, seed: 424242, steps: 1200, fitness: 0x3fc8c49ba5e353f8n, distance: 0xbfcdd2f1a9fbe76dn, ticks: 31, fell: true },
] as const

test('deterministic math matches pinned iOS reference bits', () => {
  for (const [input, sin, cos, tanh, log, exp] of mathVectors) {
    const x = fromBits(input)
    assert.deepEqual([
      bits(dsin(x)), bits(dcos(x)), bits(dtanh(x)),
      bits(dlog(Math.abs(x) + 1e-9)), bits(dexp(Math.max(-700, Math.min(700, x)))),
    ], [sin, cos, tanh, log, exp], `math input ${x}`)
  }
})

test('genome generation and all six gait fixtures match pinned iOS reference bits', () => {
  const genome = randomGenome(11)
  assert.equal(genome.length, GENOME_SIZE)
  assert.deepEqual(genome.slice(0, 8).map(bits), [
    0x3fecbe52256271f2n, 0x3f8c066d300830c5n, 0x3fc7e8f19b4d7f43n, 0x3fce721ad3ab3e09n,
    0xbf77e4ababe68259n, 0xbfa7ee6a5336619fn, 0xbfe9045d1f15c658n, 0x3fbd919aef2f741an,
  ])
  for (const vector of gaitVectors) {
    const result = evaluate(perturb(randomGenome(vector.genomeSeed), vector.sigma, vector.seed), vector.steps)
    assert.equal(bits(result.fitness), vector.fitness, `fitness seed ${vector.seed}`)
    assert.equal(bits(result.distance), vector.distance, `distance seed ${vector.seed}`)
    assert.equal(result.ticks, vector.ticks)
    assert.equal(result.fell, vector.fell)
  }
})

test('real walker adapter returns the correct worker identity, generation and pinned scores', async () => {
  const output = await runWalker({
    generation: 3, parent: randomGenome(7), sigma: 0.2, seeds: [100001, 100002], steps: 1800,
  }, 'worker-fixture', new AbortController().signal)
  assert.equal(output.hostId, 'worker-fixture')
  assert.equal(output.generation, 3)
  assert.deepEqual(output.results.map(result => ({
    seed: result.seed, fitness: bits(result.fitness), distance: bits(result.distance), ticks: result.ticks, fell: result.fell,
  })), gaitVectors.slice(2, 4).map(({ seed, fitness, distance, ticks, fell }) => ({ seed, fitness, distance, ticks, fell })))
})

test('real echo adapter returns its nonce and cancels outstanding work', async () => {
  const output = await runEcho({ nonce: 'fixture-nonce', sleepMs: 0 }, 'worker-fixture', new AbortController().signal)
  assert.equal(output.nonce, 'fixture-nonce')
  assert.equal(output.hostId, 'worker-fixture')
  assert.equal(output.os, process.platform)
  const controller = new AbortController()
  const pending = runEcho({ nonce: 'cancel-me', sleepMs: 60_000 }, 'worker-fixture', controller.signal)
  controller.abort()
  await assert.rejects(pending, { name: 'AbortError' })
})

test('real walker adapter stops both pre-cancelled and already-running slices', async () => {
  const input = { generation: 1, parent: randomGenome(7), sigma: 0.2, seeds: Array.from({ length: 32 }, (_, i) => i + 1), steps: 1800 }
  const alreadyCancelled = new AbortController()
  alreadyCancelled.abort()
  await assert.rejects(runWalker(input, 'worker-fixture', alreadyCancelled.signal), /cancelled/)

  const controller = new AbortController()
  // runWalker yields after eight evaluations, allowing the local kill switch to run.
  const pending = runWalker(input, 'worker-fixture', controller.signal)
  controller.abort()
  await assert.rejects(pending, /cancelled/)
})
