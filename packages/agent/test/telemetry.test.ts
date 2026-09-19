import assert from 'node:assert/strict'
import { test } from 'node:test'
import * as Sentry from '@sentry/node'
import { captureWorkloadFailure, rememberTelemetrySecret, scrubTelemetry, telemetryOptions } from '../src/telemetry.ts'

test('blank DSN leaves telemetry disabled', () => {
  assert.equal(telemetryOptions({}), null)
  assert.equal(telemetryOptions({ SENTRY_DSN: '  ' }), null)
})

test('scrubs nested credentials, PEM, invitation codes, header pairs and secrets in errors', () => {
  rememberTelemetrySecret('JOIN-9876')
  const env = { API_TOKEN: 'configured-secret-token' }
  const scrubbed = scrubTelemetry({
    private_key: 'raw-key-value',
    code: 'raw-code-value',
    headers: [['Authorization', 'secret-header-value']],
    nested: { error: new Error('failed JOIN-9876 configured-secret-token Bearer secret-bearer-value') },
    details: '-----BEGIN PRIVATE KEY-----\nPRIVATE-PEM-BYTES\n-----END PRIVATE KEY-----',
    url: 'https://alice:password@example.com/join?code=URL-CODE&token=URL-TOKEN',
    task_id: 'task-123',
  }, env)
  const text = JSON.stringify(scrubbed)
  for (const secret of ['raw-key-value', 'raw-code-value', 'secret-header-value', 'JOIN-9876',
    'configured-secret-token', 'secret-bearer-value', 'PRIVATE-PEM-BYTES', 'alice:password', 'URL-CODE', 'URL-TOKEN']) {
    assert.equal(text.includes(secret), false, `leaked ${secret}`)
  }
  assert.equal(scrubbed.task_id, 'task-123')
})

test('deep or circular values are redacted instead of bypassing the scrubber', () => {
  const value: Record<string, unknown> = {}
  value.self = value
  assert.doesNotThrow(() => JSON.stringify(scrubTelemetry(value, {})))
  assert.ok(JSON.stringify(scrubTelemetry(value, {})).includes('[redacted]'))
})

test('caught workload failures reach Sentry with task tags and no secrets', async () => {
  const envelopes: unknown[] = []
  const options = telemetryOptions({
    SENTRY_DSN: 'https://test@example.invalid/1', SENTRY_ENVIRONMENT: 'unit-test', SENTRY_RELEASE: 'test-release',
  })!
  Sentry.init({ ...options, integrations: [], transport: () => ({
    send: async envelope => { envelopes.push(envelope); return { statusCode: 200 } },
    flush: async () => true,
  }) })
  try {
    captureWorkloadFailure(new Error('inference failed Bearer do-not-send-this'), {
      workerId: 'worker-123', taskId: 'task-456', adapter: 'cpu_inference_batch', attempt: 2,
    })
    await Sentry.flush(2_000)
    const serialized = JSON.stringify(envelopes)
    assert.ok(envelopes.length > 0)
    for (const expected of ['worker-123', 'task-456', 'cpu_inference_batch', 'unit-test', 'test-release']) {
      assert.ok(serialized.includes(expected), `missing ${expected}`)
    }
    assert.equal(serialized.includes('do-not-send-this'), false)
  } finally {
    await Sentry.close(2_000)
  }
})
