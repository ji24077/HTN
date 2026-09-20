import { setTimeout as sleep } from 'node:timers/promises'
import { EchoInput, type EchoOutput } from '@dwp/protocol'
import { machineName, probe } from '../capability.ts'
import type { ExecutionReporter } from '../execution.ts'

/**
 * The Pass 1 adapter. It computes nothing useful on purpose — its whole job is to
 * prove that a specific physical machine executed a specific server-issued nonce.
 */
export async function runEcho(rawInput: unknown, hostId: string, signal: AbortSignal, report?: ExecutionReporter): Promise<EchoOutput> {
  const input = EchoInput.parse(rawInput)
  report?.step('Validated connection test input')
  const started = performance.now()
  if (input.sleepMs > 0) await sleep(input.sleepMs, undefined, { signal })
  const cap = probe([])
  report?.step('Collected machine capabilities')
  return {
    nonce: input.nonce,
    hostId,
    hostname: machineName(),
    os: cap.os,
    arch: cap.arch,
    elapsedMs: Number((performance.now() - started).toFixed(3)),
  }
}
