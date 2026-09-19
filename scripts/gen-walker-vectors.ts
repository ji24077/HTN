/**
 * Regenerate the golden vectors in `WalkerConformanceTests.swift`.
 *
 *   node scripts/gen-walker-vectors.ts > ios/DWPAgentKit/Tests/DWPAgentKitTests/WalkerConformanceTests.swift
 *
 * The Swift walker must agree with the JavaScript one exactly, and the fast way to check
 * that is to pin what JavaScript actually produces. Run this only when the physics changes
 * on purpose — a diff here otherwise means the two implementations have drifted, which is
 * the thing the test exists to catch.
 */
import { randomGenome, perturb, evaluate } from '@dwp/protocol/walker.js'
import { dsin, dcos, dtanh, dlog, dexp } from '@dwp/protocol/dmath.js'

const view = new DataView(new ArrayBuffer(8))
const bits = (v: number): string => { view.setFloat64(0, v); return `0x${view.getBigUint64(0).toString(16)}` }

// Inputs chosen to cross every branch: quadrants either side of zero, the tanh
// saturation cut-off, sub-normal-ish logs, and arguments past a single period.
const probes = [
  0, 1, -1, 0.5, -0.5, 1e-9, 3.14159, -3.14159, 6.283185307179586,
  1.5707963267948966, -1.5707963267948966, 100, -100, 179.9, 0.7853981633974483,
  20.0000001, -20.0000001, 19.999, 0.3465735902799727, 2.718281828459045,
]

const mathRows = probes.map(x =>
  `        (${bits(x)}, ${bits(dsin(x))}, ${bits(dcos(x))}, ${bits(dtanh(x))}, ` +
  `${bits(dlog(Math.abs(x) + 1e-9))}, ${bits(dexp(Math.max(-700, Math.min(700, x))))}),`
).join('\n')

// Whole simulations: an untrained genome, and mutations of it at two step budgets.
type Case = { label: string; genomeSeed: number; sigma: number; seed: number; steps: number }
const cases: Case[] = [
  { label: 'untrained, short', genomeSeed: 1, sigma: 0, seed: 0, steps: 300 },
  { label: 'untrained, full', genomeSeed: 1, sigma: 0, seed: 0, steps: 1800 },
  { label: 'mutated a', genomeSeed: 7, sigma: 0.2, seed: 100_001, steps: 1800 },
  { label: 'mutated b', genomeSeed: 7, sigma: 0.2, seed: 100_002, steps: 1800 },
  { label: 'mutated c', genomeSeed: 23, sigma: 0.05, seed: 900_007, steps: 900 },
  { label: 'large sigma', genomeSeed: 3, sigma: 0.9, seed: 424_242, steps: 1200 },
]

const walkerRows = cases.map(c => {
  const r = evaluate(perturb(randomGenome(c.genomeSeed), c.sigma, c.seed), c.steps)
  return `        (genomeSeed: ${c.genomeSeed}, sigma: ${c.sigma}, seed: ${c.seed}, steps: ${c.steps},\n` +
         `         fitness: ${bits(r.fitness)}, distance: ${bits(r.distance)}, ticks: ${r.ticks}, ` +
         `fell: ${r.fell}),   // ${c.label}`
}).join('\n')

// A genome's first values, so a broken PRNG is caught before the physics muddies it.
const genomeHead = randomGenome(11).slice(0, 8).map(bits).join(', ')

process.stdout.write(`import XCTest
@testable import DWPAgentKit

/**
 * The Swift walker must agree with the JavaScript one exactly — generated file.
 *
 * Regenerate with:
 *   node scripts/gen-walker-vectors.ts > ios/DWPAgentKit/Tests/DWPAgentKitTests/WalkerConformanceTests.swift
 *
 * These are bit patterns, not values, because one ulp is precisely what goes wrong here:
 * V8 and Apple's libm disagree on roughly 4% of transcendental calls, and over 1800 steps
 * of stiff contact physics that compounds into a 33-point fitness difference — one
 * stickman walking six metres while the other falls on its face. Before \`dmath\`, the same
 * genomes scored 18/60 identical across the two languages; after it, 60/60.
 *
 * \`ios/test/conformance.ts\` checks the same property live against a running Node, which is
 * stronger. This file exists so that \`swift test\` alone catches a Swift-side regression,
 * with no Node, no server, and no network — the case where someone edits \`JSMath.swift\`
 * and runs only the Swift suite.
 *
 * A failure here means the implementations have drifted, not that the vectors are stale.
 * Regenerate only when the physics changes deliberately.
 */
final class WalkerConformanceTests: XCTestCase {

    // (x, sin, cos, tanh, log(|x|+1e-9), exp(clamped x))
    private let mathVectors: [(Double, UInt64, UInt64, UInt64, UInt64, UInt64)] = [
${mathRows}
    ].map { (Double(bitPattern: $0.0), $0.1, $0.2, $0.3, $0.4, $0.5) }

    func testDeterministicMathMatchesJavaScriptBitForBit() {
        for (x, s, c, t, l, e) in mathVectors {
            XCTAssertEqual(JSMath.sin(x).bitPattern, s, "sin(\\(x))")
            XCTAssertEqual(JSMath.cos(x).bitPattern, c, "cos(\\(x))")
            XCTAssertEqual(JSMath.tanh(x).bitPattern, t, "tanh(\\(x))")
            XCTAssertEqual(JSMath.log(abs(x) + 1e-9).bitPattern, l, "log(|\\(x)|+1e-9)")
            XCTAssertEqual(JSMath.exp(Swift.max(-700, Swift.min(700, x))).bitPattern, e, "exp(\\(x))")
        }
    }

    /// The PRNG first: JavaScript's \`>>\` is arithmetic over a value just coerced unsigned,
    /// which a plain UInt32 shift is not. Get it wrong and every genome differs from the
    /// first mutation, while still looking entirely plausible.
    func testGenomeGenerationMatchesJavaScript() {
        let expected: [UInt64] = [${genomeHead}]
        let genome = Walker.randomGenome(seed: 11)
        XCTAssertEqual(genome.count, Walker.genomeSize)
        for (i, want) in expected.enumerated() {
            XCTAssertEqual(genome[i].bitPattern, want, "genome[\\(i)]")
        }
    }

    private let gaits: [(genomeSeed: Int, sigma: Double, seed: Int, steps: Int,
                         fitness: UInt64, distance: UInt64, ticks: Int, fell: Bool)] = [
${walkerRows}
    ]

    func testWholeSimulationsMatchJavaScript() {
        for g in gaits {
            let genome = Walker.perturb(parent: Walker.randomGenome(seed: g.genomeSeed),
                                        sigma: g.sigma, seed: g.seed)
            let r = Walker.evaluate(genome, steps: g.steps)
            let label = "genome \\(g.genomeSeed), seed \\(g.seed), \\(g.steps) steps"
            XCTAssertEqual(r.ticks, g.ticks, "ticks — \\(label)")
            XCTAssertEqual(r.fell, g.fell, "fell — \\(label)")
            XCTAssertEqual(r.fitness.bitPattern, g.fitness,
                           "fitness \\(r.fitness) vs \\(Double(bitPattern: g.fitness)) — \\(label)")
            XCTAssertEqual(r.distance.bitPattern, g.distance,
                           "distance \\(r.distance) vs \\(Double(bitPattern: g.distance)) — \\(label)")
        }
    }

    /// Seed 0 means "the parent itself", which the driver relies on so a generation can
    /// never score worse than the one before it.
    func testSeedZeroReturnsTheParentUnchanged() {
        let parent = Walker.randomGenome(seed: 5)
        XCTAssertEqual(Walker.perturb(parent: parent, sigma: 0.3, seed: 0), parent)
        XCTAssertNotEqual(Walker.perturb(parent: parent, sigma: 0.3, seed: 1), parent)
    }

    /// Determinism within Swift, separately from agreement with JavaScript: the same seed
    /// must give the same score every time, or fleet-wide comparison means nothing.
    func testEvaluationIsRepeatable() {
        let genome = Walker.perturb(parent: Walker.randomGenome(seed: 2), sigma: 0.15, seed: 777)
        let first = Walker.evaluate(genome, steps: 1200)
        let second = Walker.evaluate(genome, steps: 1200)
        XCTAssertEqual(first.fitness, second.fitness)
        XCTAssertEqual(first.distance, second.distance)
        XCTAssertEqual(first.ticks, second.ticks)
    }
}
`)
