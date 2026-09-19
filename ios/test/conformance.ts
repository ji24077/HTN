/**
 * Does the Swift agent speak the same bytes the TypeScript protocol expects?
 *
 *   node ios/test/conformance.ts
 *
 * This is the highest-value test in the iOS suite, because the failure it catches does
 * not look like a bug. A signature the control service cannot verify is recorded as
 * `result.signature_invalid` — the event the system emits when it believes someone is
 * forging results — so a formatting difference here would be investigated as an attack.
 *
 * Nothing is mocked. Swift produces the values with its real code path and the real
 * `@dwp/protocol` verifier checks them, in both directions.
 */
import { execFileSync } from 'node:child_process'
import { generateKeyPairSync, createPublicKey, randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { verifyAssertion, verifyAttestation, hashOutput } from '@dwp/protocol'
import { randomGenome, perturb, evaluate } from '@dwp/protocol/walker.js'
import { dsin, dcos, dtanh, dlog } from '@dwp/protocol/dmath.js'

const HERE = dirname(fileURLToPath(import.meta.url))
const AGENT = join(HERE, '..', 'DWPAgentKit', '.build', 'debug', 'dwpagent')

const GREEN = '\x1b[32m', RED = '\x1b[31m', DIM = '\x1b[2m', BOLD = '\x1b[1m', RESET = '\x1b[0m'
let passed = 0, failed = 0

function check(name: string, ok: boolean, detail = ''): void {
  if (ok) { passed += 1; console.log(`  ${GREEN}ok${RESET}   ${name}${detail ? `  ${DIM}${detail}${RESET}` : ''}`) }
  else { failed += 1; console.log(`  ${RED}FAIL${RESET} ${name}${detail ? `  ${RED}${detail}${RESET}` : ''}`) }
}

function swift(args: string[], stdin = ''): string {
  return execFileSync(AGENT, args, { input: stdin, encoding: 'utf8', timeout: 60_000 }).trim()
}

if (!existsSync(AGENT)) {
  console.error(`\nNo Swift agent binary at ${AGENT}\n  Build it first:  cd ios/DWPAgentKit && swift build\n`)
  process.exit(1)
}

console.log(`\n${BOLD}Cross-language conformance${RESET}  ${DIM}Swift signs, TypeScript verifies${RESET}\n`)

// ------------------------------------------------------- signing, Swift → Node

{
  console.log(`${BOLD}signatures${RESET}`)
  // A key minted by Node, handed to Swift. If the PKCS#8 reader or the SPKI writer is
  // wrong in either direction, nothing below verifies.
  const { privateKey, publicKey } = generateKeyPairSync('ed25519')
  const pem = privateKey.export({ type: 'pkcs8', format: 'pem' }) as string
  const nodeSpki = publicKey.export({ type: 'spki', format: 'der' }).toString('base64')

  const hostId = randomUUID()
  const outputHash = 'a'.repeat(64)
  const startedAt = '2026-09-18T10:00:00.000Z'
  const finishedAt = '2026-09-18T10:00:02.500Z'

  const vectors = JSON.parse(swift([
    'selftest-vectors',
    '--host', hostId, '--task', 'task-xyz', '--attempt', '3',
    '--hash', outputHash, '--started', startedAt, '--finished', finishedAt,
  ], pem))

  check('Swift derives the same SPKI public key as Node',
    vectors.publicKey === nodeSpki,
    vectors.publicKey === nodeSpki ? `${nodeSpki.slice(0, 20)}…` : `swift=${vectors.publicKey} node=${nodeSpki}`)

  const assertion = verifyAssertion(vectors.assertion, () => nodeSpki, () => false)
  check('the control service accepts a Swift-minted connection assertion',
    assertion.ok, assertion.ok ? `iss=${assertion.hostId.slice(0, 8)}…` : assertion.reason)

  check('a Swift assertion is bound to the host that minted it',
    assertion.ok && assertion.hostId === hostId)

  const attestationOk = verifyAttestation(nodeSpki, vectors.attestation, {
    taskId: 'task-xyz', attempt: 3, hostId, outputHash, startedAt, finishedAt,
  })
  check('the control service accepts a Swift-signed result attestation', attestationOk)

  // Each field must actually be covered, not merely present in the string.
  const tampered = [
    ['taskId', { taskId: 'task-other' }],
    ['attempt', { attempt: 4 }],
    ['hostId', { hostId: randomUUID() }],
    ['outputHash', { outputHash: 'b'.repeat(64) }],
    ['startedAt', { startedAt: '2026-09-18T10:00:00.001Z' }],
    ['finishedAt', { finishedAt: '2026-09-18T10:00:02.501Z' }],
  ] as const
  const base = { taskId: 'task-xyz', attempt: 3, hostId, outputHash, startedAt, finishedAt }
  for (const [field, override] of tampered) {
    const stillValid = verifyAttestation(nodeSpki, vectors.attestation, { ...base, ...override })
    check(`changing ${field} invalidates the attestation`, !stillValid)
  }

  // And a Swift-generated identity must be verifiable too, not only a Node-generated one.
  const swiftOnly = JSON.parse(swift(['selftest-vectors', '--host', 'h-swift'],
    (generateKeyPairSync('ed25519').privateKey.export({ type: 'pkcs8', format: 'pem' }) as string)))
  const reparsed = createPublicKey({
    key: Buffer.from(swiftOnly.publicKey, 'base64'), format: 'der', type: 'spki',
  })
  check('Node can import a Swift-reported public key',
    reparsed.asymmetricKeyType === 'ed25519')
}

// ------------------------------------------------------ serialization, both ways

{
  console.log(`\n${BOLD}serialization${RESET}`)

  // Swift must reproduce JSON.stringify byte for byte: the agent hashes its own output
  // and signs the hash, so a different rendering is a different hash.
  const documents: [string, unknown][] = [
    ['integral doubles', { a: 1.0, b: -4.0, c: 0, d: 1000 }],
    ['fractional', { a: 12.345, b: 0.1, c: -0.5, d: 1234.567 }],
    ['nested arrays', { p: [1, 2, 3], q: [[1], [2, [3]]], r: [] }],
    ['strings needing escapes', { s: 'a"b', t: 'line\nbreak', u: 'tab\there', v: 'a/b' }],
    ['non-ascii', { s: 'café', t: '日本語', u: '😀 emoji' }],
    ['booleans and null', { t: true, f: false, n: null }],
    ['empty containers', { o: {}, a: [] }],
    ['key order is not alphabetical', { zebra: 1, apple: 2, middle: 3 }],
    ['an echo output', {
      nonce: 'abc-123', hostId: randomUUID(), hostname: 'iPhone17,1',
      os: 'ios', arch: 'arm64', elapsedMs: 12.345,
    }],
    ['an inference output', {
      from: 0, count: 5, predictions: [7, 2, 1, 0, 4], correct: 5,
      logitChecksum: 47.115, hostId: randomUUID(), hostname: 'iPhone17,1',
      modelLoadMs: 84.2, inferenceMs: 33.7, itemsPerSecond: 148.4,
    }],
  ]

  for (const [name, doc] of documents) {
    const expected = JSON.stringify(doc)
    const actual = swift(['selftest-json'], expected)
    check(`stringify matches: ${name}`, actual === expected,
      actual === expected ? '' : `\n         node : ${expected}\n         swift: ${actual}`)
  }

  // The boundaries of the spec's number algorithm, pinned individually so a regression
  // names the case rather than a random index. 2^53 is the one that actually bit: below
  // it an integral double prints exactly, above it only the shortest round-trip is right.
  const edges: [string, number][] = [
    ['zero', 0], ['negative zero', -0], ['one', 1], ['a half', 1.5],
    ['smallest plain decimal', 0.000001], ['first exponent below', 1e-7],
    ['largest plain integer', 1e21 - 1], ['first exponent above', 1e21], ['well above', 1e22],
    ['past 2^53', 123456789012345680000], ['classic float error', 0.1 + 0.2],
    ['denormal minimum', 5e-324], ['double maximum', 1.7976931348623157e308],
    ['negative fractional', -0.0001], ['long mantissa', 1 / 3],
  ]
  for (const [name, value] of edges) {
    const expected = JSON.stringify(value)
    const actual = swift(['selftest-json'], expected)
    check(`number edge case: ${name}`, actual === expected,
      actual === expected ? expected : `node=${expected} swift=${actual}`)
  }

  // Fuzz the number formatter, which is where Swift and JavaScript genuinely disagree
  // by default (Swift writes 1.0 where JSON.stringify writes 1).
  const numbers: number[] = []
  for (let i = 0; i < 400; i++) {
    const pick = i % 4
    numbers.push(
      pick === 0 ? Math.round(Math.random() * 1e6) - 5e5
      : pick === 1 ? Number((Math.random() * 1000).toFixed(3))
      : pick === 2 ? Number((Math.random() * 1e-3).toFixed(6))
      : Number((Math.random() * 1e6).toFixed(1)),
    )
  }
  const expectedNumbers = JSON.stringify(numbers)
  const actualNumbers = swift(['selftest-json'], expectedNumbers)
  if (actualNumbers === expectedNumbers) {
    check(`number formatting agrees across ${numbers.length} values`, true)
  } else {
    const a = JSON.parse(actualNumbers) as number[]
    const first = numbers.findIndex((n, i) => String(n) !== String(a[i]))
    check(`number formatting agrees across ${numbers.length} values`, false,
      `first divergence at ${first}: node=${JSON.stringify(numbers[first])} swift=${JSON.stringify(a[first])}`)
  }

  // The hash is what actually travels, so check it directly rather than inferring it.
  const output = {
    from: 0, count: 3, predictions: [7, 2, 1], correct: 3, logitChecksum: 12.5,
    hostId: 'h', hostname: 'iPhone', modelLoadMs: 1, inferenceMs: 2.5, itemsPerSecond: 1200,
  }
  const canonical = JSON.stringify(output)
  const swiftCanonical = swift(['selftest-json'], canonical)
  check('output hashing agrees', hashOutput(output) === hashOutput(JSON.parse(swiftCanonical)),
    hashOutput(output).slice(0, 16) + '…')
  check('the hashed bytes are identical, not merely equivalent', swiftCanonical === canonical)
}

// ------------------------------------------------------ physics, both ways

{
  console.log(`\n${BOLD}deterministic physics${RESET}`)

  // The arithmetic first, so a failure says which function rather than "the stickman
  // disagreed". Bit patterns, because one ulp is exactly what goes wrong here.
  const probes: number[] = []
  for (let i = 0; i < 4000; i++) probes.push((i / 4000 - 0.5) * 360 + (i % 11) * 0.000137)
  const view = new DataView(new ArrayBuffer(8))
  const bits = (v: number): string => { view.setFloat64(0, v); return view.getBigUint64(0).toString() }
  const mathVectors = probes
    .map(x => `${bits(x)} ${bits(dsin(x))} ${bits(dcos(x))} ${bits(dtanh(x / 20))} ${bits(dlog(Math.abs(x) + 1e-9))}`)
    .join('\n')
  const mathOut = swift(['selftest-dmath'], mathVectors)
  check(`dmath is bit-identical across ${probes.length} inputs`, mathOut === 'ok',
    mathOut === 'ok' ? 'sin, cos, tanh, log' : mathOut)

  // Then whole simulations, which is the claim that actually matters. An untrained gait
  // falls in under a second and hides any divergence, so this evolves a survivor first:
  // with the platform's own libm these same genomes differed by 33 fitness and 6 metres.
  const STEPS = 1800

  /**
   * Restart until a full-run survivor appears, rather than trusting one search to find
   * one. Whether a fixed budget succeeds is luck, not effort: with the squared effort
   * term, 40 generations x 40 fell at 1411 ticks, 80 x 50 survived the full run, and
   * 120 x 60 fell at 1423. Budget is not monotonic here, so a budget chosen because it
   * happened to work is a test that fails whenever the fitness landscape shifts.
   *
   * Every seed is fixed, so this stays deterministic — it just does not bet the fixture
   * on one trajectory. The gait only has to live long enough for divergence to show;
   * finding a *good* gait is the experiment's job, not this test's.
   */
  const searchFrom = (offset: number): { parent: number[]; best: ReturnType<typeof evaluate> } => {
    let parent = randomGenome(1 + offset)
    let best = evaluate(parent, STEPS)
    for (let s = 2; s <= 40; s++) {
      const c = randomGenome(s + offset); const r = evaluate(c, STEPS)
      if (r.fitness > best.fitness) { best = r; parent = c }
    }
    for (let gen = 1; gen <= 60; gen++) {
      const sigma = Math.max(0.045, 0.22 * Math.pow(0.985, gen))
      let bp = parent, bf = evaluate(parent, STEPS)
      for (let k = 1; k <= 45; k++) {
        const g = perturb(parent, sigma, (gen + offset) * 100_000 + k)
        const r = evaluate(g, STEPS)
        if (r.fitness > bf.fitness) { bf = r; bp = g }
      }
      parent = bp; best = bf
    }
    return { parent, best }
  }

  let attempt = searchFrom(0)
  for (let restart = 1; restart <= 6 && attempt.best.ticks < STEPS; restart++) {
    attempt = searchFrom(restart * 1_000)
  }
  let parent = attempt.parent
  let best = attempt.best
  check('the reference evolved a gait that survives the full run',
    best.ticks === STEPS, `fitness ${best.fitness}, ${best.distance}m, ${best.ticks}/${STEPS} ticks`)

  const cases = Array.from({ length: 40 }, (_, i) =>
    ({ parent, sigma: 0.05, seed: i === 0 ? 0 : 900_000 + i }))
  const swiftScores = JSON.parse(swift(['selftest-walker', '--steps', String(STEPS)],
    cases.map(c => JSON.stringify(c)).join('\n'))) as
    { seed: number; fitness: number; distance: number; ticks: number; fell: boolean }[]

  let identical = 0, survivors = 0
  let worstFitness = 0, worstDistance = 0
  for (let i = 0; i < cases.length; i++) {
    const js = evaluate(perturb(cases[i]!.parent, cases[i]!.sigma, cases[i]!.seed), STEPS)
    const sw = swiftScores[i]!
    if (js.fitness === sw.fitness && js.distance === sw.distance
        && js.ticks === sw.ticks && js.fell === sw.fell) identical += 1
    if (js.ticks > STEPS * 0.8) survivors += 1
    worstFitness = Math.max(worstFitness, Math.abs(js.fitness - sw.fitness))
    worstDistance = Math.max(worstDistance, Math.abs(js.distance - sw.distance))
  }

  check(`Swift and JavaScript score ${cases.length} gaits identically`,
    identical === cases.length, `${identical}/${cases.length} exact, ${survivors} long survivors`)
  check('no drift in fitness', worstFitness === 0, `max |Δ| = ${worstFitness}`)
  check('no drift in distance', worstDistance === 0, `max |Δ| = ${worstDistance} m`)
}

console.log(`\n${passed} passed, ${failed} failed\n`)
process.exit(failed === 0 ? 0 : 1)
