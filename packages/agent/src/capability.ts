import { arch, cpus, freemem, hostname, platform, totalmem } from 'node:os'
import type { CapabilityRecord } from '@dwp/protocol'
import { AGENT_VERSION } from './paths.ts'

export function probe(adapters: string[]): CapabilityRecord {
  const p = platform()
  return {
    agentVersion: AGENT_VERSION,
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
