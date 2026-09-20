import { initTelemetry } from './telemetry.ts'
import { loadConfig } from './config.ts'

// This module is evaluated before the CLI imports the agent runtime.
let saved
try { saved = loadConfig()?.telemetry } catch { /* The normal CLI reports profile errors. */ }
initTelemetry(saved)
