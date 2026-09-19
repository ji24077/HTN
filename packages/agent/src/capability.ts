import { arch, cpus, freemem, hostname, platform, totalmem } from 'node:os'
import type { CapabilityRecord } from '@dwp/protocol'
import { AGENT_VERSION } from './paths.ts'
import { containerFreeBytes, containerLimits, imageReference, isContainer } from './runtime.ts'

const MB = 1024 * 1024

/**
 * What this machine can offer, which in a container is not what the host has.
 *
 * Docker does not virtualise /proc, so `cpus()` and `totalmem()` report the whole host
 * from inside a container capped at a fraction of it. Reporting those made every agent
 * in a four-container fleet claim 15 cores and 12 GB while each was limited to two and
 * one — figures a scheduler would use to decide who gets the big slices.
 */
export function logicalCores(): number {
  const { cpus: quota } = containerLimits()
  if (quota === null) return cpus().length
  /**
   * A fractional quota has to become an integer, because that is what the protocol
   * carries. Rounded rather than floored: `--cpus 1.5` is meaningfully more than one
   * core's worth of work, and flooring every fractional limit to 1 would make a machine
   * look like the smallest possible worker whatever it was given.
   */
  return Math.max(1, Math.round(quota))
}

export function totalRamMb(): number {
  const { memoryBytes } = containerLimits()
  return Math.round((memoryBytes ?? totalmem()) / MB)
}

export function freeRamMb(): number {
  return Math.round((containerFreeBytes() ?? freemem()) / MB)
}

/**
 * `release` is what this machine is actually running, not what it was compiled as.
 *
 * Reporting AGENT_VERSION made every machine in the fleet claim "0.4.0" whatever release
 * it held, because the compile-time stamp is identical in every build of a version. The
 * server therefore had no way to tell an updated machine from a stale one, and the only
 * way to check was to open each window in turn. Falling back to the stamp keeps a
 * machine that has never updated reporting something rather than nothing.
 */
export function probe(adapters: string[], release?: string | null): CapabilityRecord {
  const p = platform()
  return {
    agentVersion: version(release),
    os: p === 'darwin' || p === 'win32' || p === 'linux' ? p : 'linux',
    arch: arch(),
    cpuModel: cpus()[0]?.model ?? 'unknown',
    logicalCores: logicalCores(),
    totalRamMb: totalRamMb(),
    freeRamMb: freeRamMb(),
    adapters,
  }
}

/**
 * The version a container reports is its image, because that is what it is running.
 *
 * `installedRelease` is the release the *server* was offering when this machine paired.
 * Outside a container that is also what the agent then installed, so the two agree. A
 * container never installs it — its code is fixed at build time — so reporting it meant
 * every container in the fleet claimed the same version no matter which image it came
 * from, and "which machines still need updating" had no answer at all.
 */
function version(release?: string | null): string {
  const image = imageReference()
  if (image) return image
  if (isContainer()) return `${AGENT_VERSION} (image not declared; set DWP_IMAGE)`
  return release ?? AGENT_VERSION
}

export const machineName = (): string => hostname()
