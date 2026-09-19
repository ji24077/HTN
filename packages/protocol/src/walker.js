/**
 * A 2D stickman, a controller for it, and the physics it lives in.
 *
 * Written as plain JavaScript on purpose: the agents run this to evaluate candidate
 * gaits, and the browser runs the very same file to replay the winner. If the replay
 * used its own copy of the physics, the animation would drift from what was actually
 * scored, and you would be watching a lie.
 *
 * Everything here is deterministic. Given a genome and a seed you get identical motion
 * on any machine, which is what makes a fitness score comparable across a fleet.
 */

import { dsin, dcos, dtanh, dlog, dsqrt } from './dmath.js'

export const NUM_JOINTS = 4          // left hip, left knee, right hip, right knee
export const OBS_SIZE = 14
export const HIDDEN = 16
export const GENOME_SIZE = OBS_SIZE * HIDDEN + HIDDEN + HIDDEN * NUM_JOINTS + NUM_JOINTS

const DT = 1 / 60
const GRAVITY = -9.81
const THIGH = 0.45
const SHIN = 0.45
const HIP_SPAN = 0.16
const TORSO_MASS = 12
const TORSO_INERTIA = 1.6
const CONTACT_K = 9000      // ground stiffness
const CONTACT_C = 90        // ground damping
const FRICTION = 1.4
const HEEL = 0.05          // behind the ankle
const TOE = 0.15           // in front of it
const JOINT_RATE = 9        // how fast a joint can chase its target, rad/s
const MAX_HIP = 1.0
const MAX_KNEE = 1.5

/** Deterministic PRNG, so a seed fully determines a run. */
export function rng (seed) {
  let s = (seed >>> 0) || 1
  return () => {
    s ^= s << 13; s >>>= 0
    s ^= s >> 17
    s ^= s << 5; s >>>= 0
    return s / 4294967296
  }
}

/** Box-Muller, for mutation noise. */
export function gaussian (next) {
  const u = Math.max(next(), 1e-9)
  return dsqrt(-2 * dlog(u)) * dcos(2 * Math.PI * next())
}

/** genome = parent + sigma * noise(seed). Seed 0 means "the parent itself". */
export function perturb (parent, sigma, seed) {
  if (seed === 0) return parent.slice()
  const next = rng(seed)
  const out = new Array(parent.length)
  for (let i = 0; i < parent.length; i++) out[i] = parent[i] + sigma * gaussian(next)
  return out
}

export function randomGenome (seed) {
  const next = rng(seed)
  const g = new Array(GENOME_SIZE)
  for (let i = 0; i < GENOME_SIZE; i++) g[i] = gaussian(next) * 0.5
  return g
}

/** One hidden layer, tanh throughout. Small enough to evaluate thousands of times. */
function policy (genome, obs, out) {
  let p = 0
  const hidden = new Array(HIDDEN)
  for (let h = 0; h < HIDDEN; h++) {
    let sum = 0
    for (let i = 0; i < OBS_SIZE; i++) sum += genome[p++] * obs[i]
    sum += genome[p++]
    hidden[h] = dtanh(sum)
  }
  for (let j = 0; j < NUM_JOINTS; j++) {
    let sum = 0
    for (let h = 0; h < HIDDEN; h++) sum += genome[p++] * hidden[h]
    sum += genome[p++]
    out[j] = dtanh(sum)
  }
}

export function initialState () {
  return {
    x: 0, y: 0.92, th: 0,
    vx: 0, vy: 0, vth: 0,
    // left hip, left knee, right hip, right knee — the legs start out of phase so the
    // figure has somewhere to fall forwards from rather than standing perfectly still.
    j: [0.25, -0.25, -0.25, -0.15],
    jv: [0, 0, 0, 0],
    t: 0,
    alive: true,
  }
}

/** Where a foot ends up, given the torso pose and that leg's two joint angles. */
function footOf (s, side) {
  const hipAngle = s.j[side * 2]
  const kneeAngle = s.j[side * 2 + 1]
  const dx = (side === 0 ? -HIP_SPAN : HIP_SPAN) / 2
  const hx = s.x + dx * dcos(s.th)
  const hy = s.y + dx * dsin(s.th)
  const thighA = s.th + hipAngle
  const kx = hx + THIGH * dsin(thighA)
  const ky = hy - THIGH * dcos(thighA)
  const shinA = thighA + kneeAngle
  const fx = kx + SHIN * dsin(shinA)
  const fy = ky - SHIN * dcos(shinA)
  // A foot, not a point.
  //
  // With a single contact point the figure is an inverted pendulum balanced on a pin,
  // and evolution cannot find a gait because it cannot first find standing. Heel and toe
  // give it a base to stand on, which is what feet are for.
  return { hx, hy, kx, ky, fx, fy, heelX: fx - HEEL, toeX: fx + TOE }
}

/** Advance the world one tick. Mutates `s`. */
export function step (s, genome, targets) {
  const obs = [
    s.th, s.vth, s.vx, s.vy, s.y - 0.9,
    s.j[0], s.j[1], s.j[2], s.j[3],
    s.jv[0] * 0.1, s.jv[1] * 0.1, s.jv[2] * 0.1, s.jv[3] * 0.1,
    // A phase clock: locomotion is rhythmic, and without a sense of time the controller
    // has to invent its own oscillator, which it rarely manages.
    dsin(s.t * 6),
  ]
  policy(genome, obs, targets)

  for (let j = 0; j < NUM_JOINTS; j++) {
    const limit = j % 2 === 0 ? MAX_HIP : MAX_KNEE
    const target = targets[j] * limit
    const delta = target - s.j[j]
    const rate = Math.max(-JOINT_RATE, Math.min(JOINT_RATE, delta * 12))
    s.jv[j] = rate
    s.j[j] += rate * DT
    if (j % 2 === 1) s.j[j] = Math.min(0, Math.max(-MAX_KNEE, s.j[j]))   // knees only bend back
    else s.j[j] = Math.max(-MAX_HIP, Math.min(MAX_HIP, s.j[j]))
  }

  let fxTotal = 0
  let fyTotal = TORSO_MASS * GRAVITY
  let torque = 0

  for (let side = 0; side < 2; side++) {
    const f = footOf(s, side)
    if (f.fy >= 0) continue

    // Heel and toe each carry half the leg's contact, giving a support polygon.
    for (const cx of [f.heelX, f.toeX]) {
      const rx = cx - s.x
      const ry = f.fy - s.y
      const vfx = s.vx - s.vth * ry
      const vfy = s.vy + s.vth * rx

      const penetration = -f.fy
      let normal = (CONTACT_K * penetration - CONTACT_C * vfy) * 0.5
      if (normal < 0) normal = 0

      // Coulomb friction, opposing horizontal slip and capped by the normal force.
      let tangential = -vfx * 110
      const cap = FRICTION * normal
      if (tangential > cap) tangential = cap
      if (tangential < -cap) tangential = -cap

      fxTotal += tangential
      fyTotal += normal
      torque += rx * normal - ry * tangential
    }
  }

  s.vx += (fxTotal / TORSO_MASS) * DT
  s.vy += (fyTotal / TORSO_MASS) * DT
  s.vth += (torque / TORSO_INERTIA) * DT
  // Light damping keeps the integrator from exploding when contacts are stiff.
  s.vth *= 0.985
  s.x += s.vx * DT
  s.y += s.vy * DT
  s.th += s.vth * DT
  s.t += DT

  if (s.y < 0.45 || Math.abs(s.th) > 1.1) s.alive = false
  return s
}

/**
 * Score one genome.
 *
 * Distance travelled, with an upright bonus so that shuffling forward on its face does
 * not beat walking, and a cost on flailing so the result looks like locomotion rather
 * than a seizure.
 */
export function evaluate (genome, steps = 900) {
  const s = initialState()
  const targets = new Array(NUM_JOINTS).fill(0)
  let uprightTicks = 0
  let effort = 0

  for (let i = 0; i < steps; i++) {
    step(s, genome, targets)
    if (!s.alive) break
    uprightTicks += 1
    // Squared, not absolute: see the effort term below.
    for (let j = 0; j < NUM_JOINTS; j++) effort += s.jv[j] * s.jv[j]
  }

  const distance = s.x
  const aliveFraction = uprightTicks / steps

  /**
   * Standing has to be worth more than falling, or evolution never gets past the dive.
   *
   * A first attempt scored distance directly and produced a figure that threw itself
   * forwards, travelled 7.8m in 3.5 seconds, and could not be beaten — every mutation
   * that started to walk scored worse than the dive, so the run stalled at generation 1.
   *
   * Squaring the upright fraction makes a fall cost almost everything it earned, and the
   * survival term is large enough that simply staying up beats any dive. The result is a
   * two-stage curriculum that emerges on its own: learn to stand, then learn to travel.
   */
  const survival = aliveFraction * aliveFraction
  /**
   * The effort term is squared joint velocity, not absolute, and weighted to matter.
   *
   * With a linear term at 0.02 the penalty was measured at 0.7 points against 58.5 points
   * of distance — about 1%, which is no constraint at all. What evolved was not a walk:
   * the champion's joints sat at 8.9 rad/s each, and its *peak* summed joint speed (36.00)
   * was within 1% of its *average* (35.72), meaning every joint was pinned to its speed
   * limit for essentially all 1800 steps. It crossed the ground at 0.40 m/s — under a
   * third of walking pace — by vibrating. Optimising exactly what was asked for.
   *
   * Squaring is what distinguishes the two cases. A saturated gait pays 12.8 points here
   * while a smooth 2 rad/s gait pays 0.6 — a 20x separation that a linear term cannot
   * produce at any coefficient, because linear scales both alike. Standing still still
   * scores 15 and a shuffling traveller well above that, so this discourages frantic
   * motion without reintroducing the stall that the survival term exists to prevent.
   */
  const fitness = aliveFraction * 15 + distance * 5 * survival - (effort / steps) * 0.04
  return {
    fitness: Number(fitness.toFixed(4)),
    distance: Number(distance.toFixed(3)),
    ticks: uprightTicks,
    fell: !s.alive,
  }
}

/** Replay a genome and hand back every frame, for drawing. */
export function trace (genome, steps = 900) {
  const s = initialState()
  const targets = new Array(NUM_JOINTS).fill(0)
  const frames = []
  for (let i = 0; i < steps; i++) {
    step(s, genome, targets)
    const l = footOf(s, 0)
    const r = footOf(s, 1)
    frames.push({
      x: Number(s.x.toFixed(3)), y: Number(s.y.toFixed(3)), th: Number(s.th.toFixed(3)),
      l: [l.hx, l.hy, l.kx, l.ky, l.fx, l.fy, l.heelX, l.toeX].map(v => Number(v.toFixed(3))),
      r: [r.hx, r.hy, r.kx, r.ky, r.fx, r.fy, r.heelX, r.toeX].map(v => Number(v.toFixed(3))),
    })
    if (!s.alive) break
  }
  return frames
}
