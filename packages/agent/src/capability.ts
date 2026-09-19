import { arch, cpus, freemem, hostname, platform, totalmem } from 'node:os'
import type { CapabilityRecord } from '@dwp/protocol'
import { AGENT_VERSION } from './paths.ts'

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
    agentVersion: release ?? AGENT_VERSION,
    os: p === 'darwin' || p === 'win32' || p === 'linux' ? p : 'linux',
    arch: arch(),
    cpuModel: cpus()[0]?.model ?? 'unknown',
    logicalCores: cpus().length,
    totalRamMb: Math.round(totalmem() / 1024 / 1024),
    freeRamMb: Math.round(freemem() / 1024 / 1024),
    adapters,
  }
}

export const machineName = (): string => hostname()
