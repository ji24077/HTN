/**
 * Types for walker.js.
 *
 * The implementation stays plain JavaScript so the browser can import the identical file
 * the agents run; this only describes it to TypeScript.
 */
export declare const NUM_JOINTS: number
export declare const OBS_SIZE: number
export declare const HIDDEN: number
export declare const GENOME_SIZE: number

export declare function rng(seed: number): () => number
export declare function gaussian(next: () => number): number
export declare function perturb(parent: number[], sigma: number, seed: number): number[]
export declare function randomGenome(seed: number): number[]

export type WalkerState = {
  x: number; y: number; th: number
  vx: number; vy: number; vth: number
  j: number[]; jv: number[]
  t: number; alive: boolean
}

export declare function initialState(): WalkerState
export declare function step(s: WalkerState, genome: number[], targets: number[]): WalkerState
export declare function evaluate(genome: number[], steps?: number): {
  fitness: number; distance: number; ticks: number; fell: boolean
}
export type WalkerFrame = { x: number; y: number; th: number; l: number[]; r: number[] }
export declare function trace(genome: number[], steps?: number): WalkerFrame[]
