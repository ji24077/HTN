import XCTest
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
 * stickman walking six metres while the other falls on its face. Before `dmath`, the same
 * genomes scored 18/60 identical across the two languages; after it, 60/60.
 *
 * `ios/test/conformance.ts` checks the same property live against a running Node, which is
 * stronger. This file exists so that `swift test` alone catches a Swift-side regression,
 * with no Node, no server, and no network — the case where someone edits `JSMath.swift`
 * and runs only the Swift suite.
 *
 * A failure here means the implementations have drifted, not that the vectors are stale.
 * Regenerate only when the physics changes deliberately.
 */
final class WalkerConformanceTests: XCTestCase {

    // (x, sin, cos, tanh, log(|x|+1e-9), exp(clamped x))
    private let mathVectors: [(Double, UInt64, UInt64, UInt64, UInt64, UInt64)] = [
        (0x0, 0x0, 0x3ff0000000000000, 0x0, 0xc034b927f32bffb8, 0x3ff0000000000000),
        (0x3ff0000000000000, 0x3feaed548f090cee, 0x3fe14a280fb5068c, 0x3fe85efab514f394, 0x3e112e0bffdb1b44, 0x4005bf0a8b14576a),
        (0xbff0000000000000, 0xbfeaed548f090cee, 0x3fe14a280fb5068c, 0xbfe85efab514f394, 0x3e112e0bffdb1b44, 0x3fd78b56362cef38),
        (0x3fe0000000000000, 0x3fdeaee8744b05f0, 0x3fec1528065b7d50, 0x3fdd9353d7568af4, 0xbfe62e42fde75931, 0x3ffa61298e1e069c),
        (0xbfe0000000000000, 0xbfdeaee8744b05f0, 0x3fec1528065b7d50, 0xbfdd9353d7568af3, 0xbfe62e42fde75931, 0x3fe368b2fc6f960a),
        (0x3e112e0be826d695, 0x3e112e0be826d695, 0x3ff0000000000000, 0x3e112e0be826d695, 0xc03407b5db342de9, 0x3ff000000044b830),
        (0x400921f9f01b866e, 0x3ec6428a6aa44cd1, 0xbfefffffffff8420, 0x3fefe175ef8f055d, 0x3ff250cf66409f60, 0x4037240068789162),
        (0xc00921f9f01b866e, 0xbec6428a6aa44cd1, 0xbfefffffffff8420, 0xbfefe175ef8f055e, 0x3ff250cf66409f60, 0x3fa62026546050af),
        (0x401921fb54442d18, 0xbcb1a62633145c07, 0x3ff0000000000000, 0x3feffff15f81f9ab, 0x3ffd67f1c86fae97, 0x4080bbeee9177e18),
        (0x3ff921fb54442d18, 0x3ff0000000000000, 0x3c91a62633145c07, 0x3fed594fdae482ba, 0x3fdce6bb26591138, 0x40133dedc855935f),
        (0xbff921fb54442d18, 0xbff0000000000000, 0x3c91a62633145c07, 0xbfed594fdae482b9, 0x3fdce6bb26591138, 0x3fca9bcc46f767e0),
        (0x4059000000000000, 0xbfe03425b78c4db8, 0x3feb981dbf665fdf, 0x3ff0000000000000, 0x40126bb1bbb58111, 0x48f3494a9b171bf5),
        (0xc059000000000000, 0x3fe03425b78c4db8, 0x3feb981dbf665fdf, 0xbff0000000000000, 0x40126bb1bbb58111, 0x36ea8c1f14e2af5d),
        (0x40667ccccccccccd, 0xbfe798d00e3e1d64, 0xbfe59d4da6df1a86, 0x3ff0000000000000, 0x4014c504ce00948e, 0x502746ee5de15634),
        (0x3fe921fb54442d18, 0x3fe6a09e667f3bcc, 0x3fe6a09e667f3bcd, 0x3fe4fc441fa6d6d6, 0xbfceeb95add8c90b, 0x40018bd669471caa),
        (0x4034000001ad7f2a, 0x3fed36d90b45dfd1, 0x3fda1e03d75dcf10, 0x3ff0000000000000, 0x4007f7427c2167d6, 0x41bceb08bbed22d6),
        (0xc034000001ad7f2a, 0xbfed36d90b45dfd1, 0x3fda1e03d75dcf10, 0xbff0000000000000, 0x4007f7427c2167d6, 0x3e21b486383f21ec),
        (0x4033ffbe76c8b439, 0x3fed338030f1f6df, 0x3fda2cf889a42410, 0x3ff0000000000000, 0x4007f7284467bccd, 0x41bce3a250b754c0),
        (0x3fd62e42fefa39f0, 0x3fd5bd451fe0fa0e, 0x3fee18ebc184046e, 0x3fd5555555555555, 0xbff0f45e25ae2f8d, 0x3ff6a09e667f3bcd),
        (0x4005bf0a8b145769, 0x3fda4a3d9c2131de, 0xbfed2cec9a554006, 0x3fefb8f76b1e2ab6, 0x3ff00000001947ce, 0x402e4efb75e4527a),
    ].map { (Double(bitPattern: $0.0), $0.1, $0.2, $0.3, $0.4, $0.5) }

    func testDeterministicMathMatchesJavaScriptBitForBit() {
        for (x, s, c, t, l, e) in mathVectors {
            XCTAssertEqual(JSMath.sin(x).bitPattern, s, "sin(\(x))")
            XCTAssertEqual(JSMath.cos(x).bitPattern, c, "cos(\(x))")
            XCTAssertEqual(JSMath.tanh(x).bitPattern, t, "tanh(\(x))")
            XCTAssertEqual(JSMath.log(abs(x) + 1e-9).bitPattern, l, "log(|\(x)|+1e-9)")
            XCTAssertEqual(JSMath.exp(Swift.max(-700, Swift.min(700, x))).bitPattern, e, "exp(\(x))")
        }
    }

    /// The PRNG first: JavaScript's `>>` is arithmetic over a value just coerced unsigned,
    /// which a plain UInt32 shift is not. Get it wrong and every genome differs from the
    /// first mutation, while still looking entirely plausible.
    func testGenomeGenerationMatchesJavaScript() {
        let expected: [UInt64] = [0x3fecbe52256271f2, 0x3f8c066d300830c5, 0x3fc7e8f19b4d7f43, 0x3fce721ad3ab3e09, 0xbf77e4ababe68259, 0xbfa7ee6a5336619f, 0xbfe9045d1f15c658, 0x3fbd919aef2f741a]
        let genome = Walker.randomGenome(seed: 11)
        XCTAssertEqual(genome.count, Walker.genomeSize)
        for (i, want) in expected.enumerated() {
            XCTAssertEqual(genome[i].bitPattern, want, "genome[\(i)]")
        }
    }

    private let gaits: [(genomeSeed: Int, sigma: Double, seed: Int, steps: Int,
                         fitness: UInt64, distance: UInt64, ticks: Int, fell: Bool)] = [
        (genomeSeed: 1, sigma: 0, seed: 0, steps: 300,
         fitness: 0x3ff6404ea4a8c155, distance: 0xbfe1eb851eb851ec, ticks: 52, fell: true),   // untrained, short
        (genomeSeed: 1, sigma: 0, seed: 0, steps: 1800,
         fitness: 0x3fcf2b020c49ba5e, distance: 0xbfe1eb851eb851ec, ticks: 52, fell: true),   // untrained, full
        (genomeSeed: 7, sigma: 0.2, seed: 100001, steps: 1800,
         fitness: 0x3fd2631f8a0902de, distance: 0xbfc6872b020c49ba, ticks: 61, fell: true),   // mutated a
        (genomeSeed: 7, sigma: 0.2, seed: 100002, steps: 1800,
         fitness: 0x3fe7f559b3d07c85, distance: 0xbfe116872b020c4a, ticks: 140, fell: true),   // mutated b
        (genomeSeed: 23, sigma: 0.05, seed: 900007, steps: 900,
         fitness: 0x3fca29c779a6b50b, distance: 0x3fa1eb851eb851ec, ticks: 19, fell: true),   // mutated c
        (genomeSeed: 3, sigma: 0.9, seed: 424242, steps: 1200,
         fitness: 0x3fc8c49ba5e353f8, distance: 0xbfcdd2f1a9fbe76d, ticks: 31, fell: true),   // large sigma
    ]

    func testWholeSimulationsMatchJavaScript() {
        for g in gaits {
            let genome = Walker.perturb(parent: Walker.randomGenome(seed: g.genomeSeed),
                                        sigma: g.sigma, seed: g.seed)
            let r = Walker.evaluate(genome, steps: g.steps)
            let label = "genome \(g.genomeSeed), seed \(g.seed), \(g.steps) steps"
            XCTAssertEqual(r.ticks, g.ticks, "ticks — \(label)")
            XCTAssertEqual(r.fell, g.fell, "fell — \(label)")
            XCTAssertEqual(r.fitness.bitPattern, g.fitness,
                           "fitness \(r.fitness) vs \(Double(bitPattern: g.fitness)) — \(label)")
            XCTAssertEqual(r.distance.bitPattern, g.distance,
                           "distance \(r.distance) vs \(Double(bitPattern: g.distance)) — \(label)")
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
