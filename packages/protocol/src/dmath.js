/**
 * Transcendental functions that give the same answer on every platform.
 *
 * `Math.sin` and friends are not specified to the last bit. V8 and Apple's libm disagree
 * on roughly 4% of inputs by one ulp — harmless in isolation, and fatal here: over 1800
 * steps of stiff contact physics the error compounds until one machine's stickman walks
 * six metres and another's falls over. Measured, not assumed: a trained gait scored 45.4
 * under one implementation and 12.3 under the other.
 *
 * So the simulation uses these instead. Everything below is built from addition,
 * multiplication, division and comparison, all of which IEEE 754 defines exactly, plus
 * scaling by powers of two, which is exact. Two implementations that perform the same
 * operations in the same order therefore produce identical bits — which is what lets an
 * iPhone score a gait a laptop will later replay.
 *
 * Accuracy is about 1e-16 relative, comfortably past what the physics needs. Determinism
 * is the point; being the correctly-rounded result is not.
 */

const PIO2_HI = 1.5707963267948966
const PIO2_LO = 6.123233995736766e-17
const LN2_HI = 0.6931471803691238
const LN2_LO = 1.9082149292705877e-10
const INV_LN2 = 1.4426950408889634

/** Multiply by 2^k exactly. A loop, because `Math.pow` is another unspecified function. */
function scale2 (v, k) {
  let out = v
  if (k > 0) { for (let i = 0; i < k; i++) out *= 2 }
  else { for (let i = 0; i < -k; i++) out *= 0.5 }
  return out
}

/** sin on |r| <= pi/4, minimax polynomial. */
function sinKernel (r) {
  const z = r * r
  const p = -1.66666666666666324348e-01 + z * (8.33333333332248946124e-03 +
        z * (-1.98412698298579493134e-04 + z * (2.75573137070700676789e-06 +
        z * (-2.50507602534068634195e-08 + z * 1.58969099521155010221e-10))))
  return r + r * z * p
}

/** cos on |r| <= pi/4. */
function cosKernel (r) {
  const z = r * r
  const p = 4.16666666666666019037e-02 + z * (-1.38888888888741095749e-03 +
        z * (2.48015872894767294178e-05 + z * (-2.75573143513906633035e-07 +
        z * (2.08757232129817482790e-09 + z * -1.13596475577881948265e-11))))
  return 1 - 0.5 * z + z * z * p
}

/**
 * Reduce x to a quadrant and a remainder in [-pi/4, pi/4].
 *
 * Cody-Waite with a two-part pi/2: `Math.floor` is exact and identical everywhere, unlike
 * `Math.round`, which breaks ties differently in JavaScript and Swift.
 */
function reduce (x) {
  const k = Math.floor(x * (1 / PIO2_HI) + 0.5)
  const r = (x - k * PIO2_HI) - k * PIO2_LO
  return [((k % 4) + 4) % 4, r]
}

export function dsin (x) {
  if (!isFinite(x)) return NaN
  const [q, r] = reduce(x)
  if (q === 0) return sinKernel(r)
  if (q === 1) return cosKernel(r)
  if (q === 2) return -sinKernel(r)
  return -cosKernel(r)
}

export function dcos (x) {
  if (!isFinite(x)) return NaN
  const [q, r] = reduce(x)
  if (q === 0) return cosKernel(r)
  if (q === 1) return -sinKernel(r)
  if (q === 2) return -cosKernel(r)
  return sinKernel(r)
}

/** exp, by reduction to [-ln2/2, ln2/2] and a Taylor series. */
export function dexp (x) {
  if (x > 709.78) return Infinity
  if (x < -745.2) return 0
  const k = Math.floor(x * INV_LN2 + 0.5)
  const r = (x - k * LN2_HI) - k * LN2_LO
  let e = 7.647163731819816e-13                                 // 1/15!
  e = e * r + 1.1470745597729725e-11                            // 1/14!
  e = e * r + 1.6059043836821613e-10                            // 1/13!
  e = e * r + 2.08767569878681e-9                               // 1/12!
  e = e * r + 2.505210838544172e-8                              // 1/11!
  e = e * r + 2.7557319223985893e-7                             // 1/10!
  e = e * r + 2.755731922398589e-6                              // 1/9!
  e = e * r + 2.48015873015873e-5                               // 1/8!
  e = e * r + 1.9841269841269841e-4                             // 1/7!
  e = e * r + 1.3888888888888889e-3                             // 1/6!
  e = e * r + 8.333333333333333e-3                              // 1/5!
  e = e * r + 4.1666666666666664e-2                             // 1/4!
  e = e * r + 0.16666666666666666                               // 1/3!
  e = e * r + 0.5                                               // 1/2!
  e = e * r + 1                                                 // 1/1!
  e = e * r + 1                                                 // 1/0!
  return scale2(e, k)
}

/** exp(x) - 1, kept accurate near zero where `dexp(x) - 1` would cancel away its digits. */
export function dexpm1 (x) {
  if (x > 0.5 || x < -0.5) return dexp(x) - 1
  let e = 7.647163731819816e-13
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

/**
 * tanh, via expm1 so that small arguments keep their precision.
 *
 * `(exp(2x) - 1) / (exp(2x) + 1)` loses most of its significant digits near zero, which is
 * where the controller spends much of its time.
 */
export function dtanh (x) {
  if (x > 20) return 1
  if (x < -20) return -1
  const m = dexpm1(2 * x)
  return m / (m + 2)
}

/** log, by splitting off the exponent and using the atanh series on the mantissa. */
export function dlog (x) {
  if (x <= 0) return x === 0 ? -Infinity : NaN
  if (!isFinite(x)) return x
  let m = x
  let e = 0
  while (m >= 1.4142135623730951) { m *= 0.5; e += 1 }
  while (m < 0.7071067811865476) { m *= 2; e -= 1 }

  const s = (m - 1) / (m + 1)
  const z = s * s
  let p = 1 / 23
  p = p * z + 1 / 21
  p = p * z + 1 / 19
  p = p * z + 1 / 17
  p = p * z + 1 / 15
  p = p * z + 1 / 13
  p = p * z + 1 / 11
  p = p * z + 1 / 9
  p = p * z + 1 / 7
  p = p * z + 1 / 5
  p = p * z + 1 / 3
  p = p * z + 1
  return e * LN2_HI + (e * LN2_LO + 2 * s * p)
}

export function dsqrt (x) { return Math.sqrt(x) }   // IEEE-exact everywhere, unlike the rest
