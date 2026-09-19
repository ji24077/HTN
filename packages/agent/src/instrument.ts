import { initTelemetry } from './telemetry.ts'

// This module is evaluated before the CLI imports the agent runtime.
initTelemetry()
