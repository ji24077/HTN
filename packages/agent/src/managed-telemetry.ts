import { WorkerTelemetry } from '@dwp/protocol'
import { loadConfig, saveConfig, type AgentConfig } from './config.ts'
import { initTelemetry } from './telemetry.ts'

/** Persist only telemetry, preserving settings changed by the GUI or updater. */
export function applyManagedTelemetry(config: AgentConfig, value: unknown): void {
  const parsed = WorkerTelemetry.safeParse(value)
  if (!parsed.success) return // Older servers omit this optional field.
  let latest
  try { latest = loadConfig() } catch { return }
  if (!latest || latest.hostId !== config.hostId || latest.server !== config.server) return
  config.telemetry = parsed.data
  if (JSON.stringify(latest.telemetry) !== JSON.stringify(parsed.data)) {
    try { saveConfig({ ...latest, telemetry: parsed.data }) } catch {
      // A read-only disk must not break the worker's connection or current reporting.
    }
  }
  initTelemetry(parsed.data)
}
