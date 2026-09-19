/**
 * Deterministic replacements for the parts of `Math` that platforms disagree about.
 *
 * Identical results on every engine and language, which `Math.sin` and friends do not
 * promise — see the implementation for why one ulp matters here.
 */
export function dsin(x: number): number
export function dcos(x: number): number
export function dtanh(x: number): number
export function dlog(x: number): number
export function dexp(x: number): number
export function dexpm1(x: number): number
export function dsqrt(x: number): number
