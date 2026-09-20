import { arch, cpus, hostname, platform } from 'node:os'
import type { CapabilityRecord, RuntimePreference } from '@dwp/protocol'
import { detectAccelerator } from './accelerator.ts'
import { AGENT_VERSION } from './paths.ts'
import { freeRamMb, imageReference, isContainer, logicalCores, totalRamMb } from './runtime.ts'

/**
 * Re-exported rather than moved outright: these are what this machine can offer, which
 * is a capability question, and callers have always asked capability.ts for it. The
 * implementations live next to the cgroup reading they depend on.
 */
export { freeRamMb, logicalCores, totalRamMb }

/**
 * `release` is what this machine is actually running, not what it was compiled as.
 *
 * Reporting AGENT_VERSION made every machine in the fleet claim "0.4.0" whatever release
 * it held, because the compile-time stamp is identical in every build of a version. The
 * server therefore had no way to tell an updated machine from a stale one, and the only
 * way to check was to open each window in turn. Falling back to the stamp keeps a
 * machine that has never updated reporting something rather than nothing.
 */
export function probe(
  adapters: string[],
  release?: string | null,
  preference: RuntimePreference = 'auto',
): CapabilityRecord {
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
    // Probed on every hello rather than cached at start: a container can be recreated
    // with `--gpus` added, and a machine that only re-reports its devices on a fresh
    // install would go on claiming none until someone noticed.
    accelerator: detectAccelerator(preference),
    runtimePreference: preference,
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
