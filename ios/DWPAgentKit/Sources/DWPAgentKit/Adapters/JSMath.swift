import Foundation

/**
 * A verbatim port of `packages/protocol/src/dmath.js`.
 *
 * Deliberately built from nothing but addition, multiplication, division, comparison and
 * exact power-of-two scaling — every one of which IEEE 754 pins down exactly. The platform
 * `sin`/`cos`/`tanh`/`log` are *not* pinned down: Apple's libm and V8 disagree on about 4%
 * of inputs by a single ulp, which over 1800 steps of contact physics is the difference
 * between a stickman that walks and one that falls on its face.
 *
 * Every constant, coefficient and operation order here must match the JavaScript line for
 * line. Two tests hold that: `WalkerConformanceTests` pins both the arithmetic and whole
 * 1800-step simulations to bit patterns taken from the JavaScript, and
 * `ios/test/conformance.ts` recomputes the same comparison against a live Node. Both check
 * simulations rather than arithmetic alone, because arithmetic agreeing is not the claim
 * that matters — a gait that falls early agrees even with the platform's own libm, so a
 * test built only on short runs passes while proving nothing.
 */
public enum JSMath {

    static let pio2Hi = 1.5707963267948966
    static let pio2Lo = 6.123233995736766e-17
    static let ln2Hi = 0.6931471803691238
    static let ln2Lo = 1.9082149292705877e-10
    static let invLn2 = 1.4426950408889634

    /// Multiply by 2^k exactly. A loop, because `pow` is another unspecified function.
    static func scale2(_ v: Double, _ k: Int) -> Double {
        var out = v
        if k > 0 { for _ in 0..<k { out *= 2 } }
        else { for _ in 0..<(-k) { out *= 0.5 } }
        return out
    }

    static func sinKernel(_ r: Double) -> Double {
        let z = r * r
        let p = -1.66666666666666324348e-01 + z * (8.33333333332248946124e-03 +
            z * (-1.98412698298579493134e-04 + z * (2.75573137070700676789e-06 +
            z * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10))))
        return r + r * z * p
    }

    static func cosKernel(_ r: Double) -> Double {
        let z = r * r
        let p = 4.16666666666666019037e-02 + z * (-1.38888888888741095749e-03 +
            z * (2.48015872894767294178e-05 + z * (-2.75573143513906633035e-07 +
            z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11))))
        return 1 - 0.5 * z + z * z * p
    }

    /// `floor`, never `rounded()`: JavaScript and Swift break ties differently.
    static func reduce(_ x: Double) -> (q: Int, r: Double) {
        let k = (x * (1 / pio2Hi) + 0.5).rounded(.down)
        let r = (x - k * pio2Hi) - k * pio2Lo
        let ki = Int(k)
        return (((ki % 4) + 4) % 4, r)
    }

    public static func sin(_ x: Double) -> Double {
        guard x.isFinite else { return .nan }
        let (q, r) = reduce(x)
        if q == 0 { return sinKernel(r) }
        if q == 1 { return cosKernel(r) }
        if q == 2 { return -sinKernel(r) }
        return -cosKernel(r)
    }

    public static func cos(_ x: Double) -> Double {
        guard x.isFinite else { return .nan }
        let (q, r) = reduce(x)
        if q == 0 { return cosKernel(r) }
        if q == 1 { return -sinKernel(r) }
        if q == 2 { return -cosKernel(r) }
        return sinKernel(r)
    }

    public static func exp(_ x: Double) -> Double {
        if x > 709.78 { return .infinity }
        if x < -745.2 { return 0 }
        let k = (x * invLn2 + 0.5).rounded(.down)
        let r = (x - k * ln2Hi) - k * ln2Lo
        var e = 7.647163731819816e-13
        e = e * r + 1.1470745597729725e-11
        e = e * r + 1.6059043836821613e-10
        e = e * r + 2.08767569878681e-9
        e = e * r + 2.505210838544172e-8
        e = e * r + 2.7557319223985893e-7
        e = e * r + 2.755731922398589e-6
        e = e * r + 2.48015873015873e-5
        e = e * r + 1.9841269841269841e-4
        e = e * r + 1.3888888888888889e-3
        e = e * r + 8.333333333333333e-3
        e = e * r + 4.1666666666666664e-2
        e = e * r + 0.16666666666666666
        e = e * r + 0.5
        e = e * r + 1
        e = e * r + 1
        return scale2(e, Int(k))
    }

    /// exp(x) - 1, accurate near zero where `exp(x) - 1` would cancel its digits away.
    public static func expm1(_ x: Double) -> Double {
        if x > 0.5 || x < -0.5 { return exp(x) - 1 }
        var e = 7.647163731819816e-13
        e = e * x + 1.1470745597729725e-11
        e = e * x + 1.6059043836821613e-10
        e = e * x + 2.08767569878681e-9
        e = e * x + 2.505210838544172e-8
        e = e * x + 2.7557319223985893e-7
        e = e * x + 2.755731922398589e-6
        e = e * x + 2.48015873015873e-5
        e = e * x + 1.9841269841269841e-4
        e = e * x + 1.3888888888888889e-3
        e = e * x + 8.333333333333333e-3
        e = e * x + 4.1666666666666664e-2
        e = e * x + 0.16666666666666666
        e = e * x + 0.5
        e = e * x + 1
        return e * x
    }

    public static func tanh(_ x: Double) -> Double {
        if x > 20 { return 1 }
        if x < -20 { return -1 }
        let m = expm1(2 * x)
        return m / (m + 2)
    }

    public static func log(_ x: Double) -> Double {
        if x <= 0 { return x == 0 ? -.infinity : .nan }
        if !x.isFinite { return x }
        var m = x
        var e = 0.0
        while m >= 1.4142135623730951 { m *= 0.5; e += 1 }
        while m < 0.7071067811865476 { m *= 2; e -= 1 }

        let s = (m - 1) / (m + 1)
        let z = s * s
        var p = 1.0 / 23
        p = p * z + 1.0 / 21
        p = p * z + 1.0 / 19
        p = p * z + 1.0 / 17
        p = p * z + 1.0 / 15
        p = p * z + 1.0 / 13
        p = p * z + 1.0 / 11
        p = p * z + 1.0 / 9
        p = p * z + 1.0 / 7
        p = p * z + 1.0 / 5
        p = p * z + 1.0 / 3
        p = p * z + 1
        return e * ln2Hi + (e * ln2Lo + 2 * s * p)
    }
}
