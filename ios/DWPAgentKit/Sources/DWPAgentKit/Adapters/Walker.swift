import Foundation

/**
 * A 2D stickman, a controller for it, and the physics it lives in.
 *
 * A line-by-line port of `packages/protocol/src/walker.js`. It is duplicated rather than
 * shared because the phone has no JavaScript engine in this app, and duplication is only
 * defensible while something checks the copies still agree. Two things do:
 * `WalkerConformanceTests` scores whole simulations against bit patterns captured from the
 * JavaScript, so `swift test` alone catches a drift with no Node and no server; and
 * `ios/test/conformance.ts` re-runs the comparison against a live Node, which is stronger
 * because the vectors cannot go stale without the diff showing it.
 *
 * The delicate part is not the physics, it is JavaScript's integer semantics. The PRNG
 * relies on `>>` being an *arithmetic* shift over a value that has just been coerced to
 * unsigned by `>>> 0`, which is not what a UInt32 shift does in Swift. Get that wrong and
 * every genome differs from the first mutation onwards, while still looking plausible.
 */
public enum Walker {
    public static let numJoints = 4
    public static let obsSize = 14
    public static let hidden = 16
    public static let genomeSize = obsSize * hidden + hidden + hidden * numJoints + numJoints

    static let dt = 1.0 / 60.0
    static let gravity = -9.81
    static let thigh = 0.45
    static let shin = 0.45
    static let hipSpan = 0.16
    static let torsoMass = 12.0
    static let torsoInertia = 1.6
    static let contactK = 9000.0
    static let contactC = 90.0
    static let friction = 1.4
    static let heel = 0.05
    static let toe = 0.15
    static let jointRate = 9.0
    static let maxHip = 1.0
    static let maxKnee = 1.5

    // -------------------------------------------------------------------- rng

    /**
     * xorshift32, with JavaScript's exact coercion behaviour.
     *
     * `s ^= s >> 17` in JS applies ToInt32 first, so the shift is arithmetic and sign bits
     * propagate; a plain UInt32 shift in Swift is logical and would diverge for every seed
     * with the high bit set — which is half of them.
     */
    public struct RNG {
        private var s: UInt32
        public init(seed: UInt32) { s = seed == 0 ? 1 : seed }

        public mutating func next() -> Double {
            s ^= s << 13
            s ^= UInt32(bitPattern: Int32(bitPattern: s) >> 17)
            s ^= s << 5
            return Double(s) / 4294967296.0
        }
    }

    /// Box-Muller, for mutation noise.
    public static func gaussian(_ rng: inout RNG) -> Double {
        let u = max(rng.next(), 1e-9)
        return (-2 * JSMath.log(u)).squareRoot() * JSMath.cos(2 * Double.pi * rng.next())
    }

    /// genome = parent + sigma * noise(seed). Seed 0 means "the parent itself".
    public static func perturb(parent: [Double], sigma: Double, seed: Int) -> [Double] {
        if seed == 0 { return parent }
        var rng = RNG(seed: UInt32(truncatingIfNeeded: seed))
        var out = [Double](repeating: 0, count: parent.count)
        for i in 0..<parent.count { out[i] = parent[i] + sigma * gaussian(&rng) }
        return out
    }

    public static func randomGenome(seed: Int) -> [Double] {
        var rng = RNG(seed: UInt32(truncatingIfNeeded: seed))
        var g = [Double](repeating: 0, count: genomeSize)
        for i in 0..<genomeSize { g[i] = gaussian(&rng) * 0.5 }
        return g
    }

    // ----------------------------------------------------------------- policy

    /// One hidden layer, tanh throughout.
    static func policy(_ genome: [Double], _ obs: [Double], _ out: inout [Double]) {
        var p = 0
        var h = [Double](repeating: 0, count: hidden)
        for k in 0..<hidden {
            var sum = 0.0
            for i in 0..<obsSize { sum += genome[p] * obs[i]; p += 1 }
            sum += genome[p]; p += 1
            h[k] = JSMath.tanh(sum)
        }
        for j in 0..<numJoints {
            var sum = 0.0
            for k in 0..<hidden { sum += genome[p] * h[k]; p += 1 }
            sum += genome[p]; p += 1
            out[j] = JSMath.tanh(sum)
        }
    }

    // ------------------------------------------------------------------ state

    public struct State {
        public var x = 0.0, y = 0.92, th = 0.0
        public var vx = 0.0, vy = 0.0, vth = 0.0
        public var j: [Double] = [0.25, -0.25, -0.25, -0.15]
        public var jv: [Double] = [0, 0, 0, 0]
        public var t = 0.0
        public var alive = true
        public init() {}
    }

    struct Foot { var hx = 0.0, hy = 0.0, kx = 0.0, ky = 0.0, fx = 0.0, fy = 0.0, heelX = 0.0, toeX = 0.0 }

    static func footOf(_ s: State, _ side: Int) -> Foot {
        let hipAngle = s.j[side * 2]
        let kneeAngle = s.j[side * 2 + 1]
        let dx = (side == 0 ? -hipSpan : hipSpan) / 2
        let hx = s.x + dx * JSMath.cos(s.th)
        let hy = s.y + dx * JSMath.sin(s.th)
        let thighA = s.th + hipAngle
        let kx = hx + thigh * JSMath.sin(thighA)
        let ky = hy - thigh * JSMath.cos(thighA)
        let shinA = thighA + kneeAngle
        let fx = kx + shin * JSMath.sin(shinA)
        let fy = ky - shin * JSMath.cos(shinA)
        return Foot(hx: hx, hy: hy, kx: kx, ky: ky, fx: fx, fy: fy, heelX: fx - heel, toeX: fx + toe)
    }

    /// Advance the world one tick.
    public static func step(_ s: inout State, _ genome: [Double], _ targets: inout [Double]) {
        let obs: [Double] = [
            s.th, s.vth, s.vx, s.vy, s.y - 0.9,
            s.j[0], s.j[1], s.j[2], s.j[3],
            s.jv[0] * 0.1, s.jv[1] * 0.1, s.jv[2] * 0.1, s.jv[3] * 0.1,
            JSMath.sin(s.t * 6),
        ]
        policy(genome, obs, &targets)

        for j in 0..<numJoints {
            let limit = j % 2 == 0 ? maxHip : maxKnee
            let target = targets[j] * limit
            let delta = target - s.j[j]
            let rate = max(-jointRate, min(jointRate, delta * 12))
            s.jv[j] = rate
            s.j[j] += rate * dt
            if j % 2 == 1 { s.j[j] = min(0, max(-maxKnee, s.j[j])) }
            else { s.j[j] = max(-maxHip, min(maxHip, s.j[j])) }
        }

        var fxTotal = 0.0
        var fyTotal = torsoMass * gravity
        var torque = 0.0

        for side in 0..<2 {
            let f = footOf(s, side)
            if f.fy >= 0 { continue }

            for cx in [f.heelX, f.toeX] {
                let rx = cx - s.x
                let ry = f.fy - s.y
                let vfx = s.vx - s.vth * ry
                let vfy = s.vy + s.vth * rx

                let penetration = -f.fy
                var normal = (contactK * penetration - contactC * vfy) * 0.5
                if normal < 0 { normal = 0 }

                var tangential = -vfx * 110
                let cap = friction * normal
                if tangential > cap { tangential = cap }
                if tangential < -cap { tangential = -cap }

                fxTotal += tangential
                fyTotal += normal
                torque += rx * normal - ry * tangential
            }
        }

        s.vx += (fxTotal / torsoMass) * dt
        s.vy += (fyTotal / torsoMass) * dt
        s.vth += (torque / torsoInertia) * dt
        s.vth *= 0.985
        s.x += s.vx * dt
        s.y += s.vy * dt
        s.th += s.vth * dt
        s.t += dt

        if s.y < 0.45 || abs(s.th) > 1.1 { s.alive = false }
    }

    // --------------------------------------------------------------- evaluate

    public struct Score {
        public let fitness: Double
        public let distance: Double
        public let ticks: Int
        public let fell: Bool
    }

    public static func evaluate(_ genome: [Double], steps: Int = 900) -> Score {
        var s = State()
        var targets = [Double](repeating: 0, count: numJoints)
        var uprightTicks = 0
        var effort = 0.0

        for _ in 0..<steps {
            step(&s, genome, &targets)
            if !s.alive { break }
            uprightTicks += 1
            // Squared, not absolute — must match walker.js exactly.
            for j in 0..<numJoints { effort += s.jv[j] * s.jv[j] }
        }

        let distance = s.x
        let aliveFraction = Double(uprightTicks) / Double(steps)
        let survival = aliveFraction * aliveFraction
        // Squared joint velocity at 0.04. The linear 0.02 term was measured at ~1% of the
        // distance term, so it constrained nothing and the champion evolved a vibration:
        // joints pinned to their speed limit for essentially every step, crossing the
        // ground at 0.40 m/s. Squaring separates a saturated gait (12.8 points) from a
        // smooth one (0.6). Same constants and same operation order as walker.js, because
        // an iPhone and a laptop scoring the same gait differently is not a small bug --
        // it surfaces as result.signature_invalid, i.e. a suspected forgery.
        let fitness = aliveFraction * 15 + distance * 5 * survival - (effort / Double(steps)) * 0.04

        return Score(
            fitness: jsToFixed(fitness, 4),
            distance: jsToFixed(distance, 3),
            ticks: uprightTicks,
            fell: !s.alive)
    }
}

/// `Number(x.toFixed(n))` — the rounding the JavaScript scorer applies before reporting.
func jsToFixed(_ value: Double, _ places: Int) -> Double {
    guard value.isFinite else { return 0 }
    let factor = pow(10.0, Double(places))
    return (value * factor).rounded() / factor
}
