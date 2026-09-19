import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { spawnSync } from 'node:child_process'
import { telemetryOptions } from '../src/telemetry.ts'

const remote = { dsn: 'https://public@example.invalid/123', environment: 'fleet-test', release: 'fleet-v1' }
const root = resolve(import.meta.dirname, '../../..')
const inspect = `
  import { createRequire } from 'node:module';
  const require = createRequire(new URL('./packages/agent/package.json', import.meta.url));
  const Sentry = require('@sentry/node');
  const {loadConfig, saveConfig} = await import('./packages/agent/src/config.ts');
`

function isolated(run: (home: string, child: (source: string) => any) => void) {
  const home = mkdtempSync(join(tmpdir(), 'managed-telemetry-'))
  const env: NodeJS.ProcessEnv = { ...process.env, DWP_HOME: home }
  for (const key of Object.keys(env)) if (key.startsWith('SENTRY_') || key === 'RELEASE') delete env[key]
  try {
    run(home, source => {
      const result = spawnSync(process.execPath, ['--input-type=module', '-e', inspect + source], {
        cwd: root, env, encoding: 'utf8', timeout: 15_000,
      })
      assert.equal(result.status, 0, result.stderr || String(result.error))
      return JSON.parse(result.stdout.trim())
    })
  } finally { rmSync(home, { recursive: true, force: true }) }
}

test('pairing provisions only public telemetry and a clean service restart loads it without env', () => {
  isolated((home, child) => {
    const paired = child(`
      globalThis.fetch = async () => new Response(JSON.stringify({
        hostId: 'test-device', label: 'Test device', wsUrl: 'wss://fleet.example/agent/connect',
        telemetry: {...${JSON.stringify(remote)}, authToken: 'must-not-persist'},
        databasePassword: 'must-not-persist',
      }), {status: 200});
      const {pairHost} = await import('./packages/agent/src/pair.ts');
      const result = await pairHost('https://fleet.example', 'test-pair-code');
      console.log(JSON.stringify({ok:result.ok, telemetry:loadConfig().telemetry, enabled:Sentry.isEnabled()}));
      await Sentry.close(100);
    `)
    assert.deepEqual(paired, { ok: true, telemetry: remote, enabled: true })
    assert.equal(readFileSync(join(home, 'config.json'), 'utf8').includes('must-not-persist'), false)
    const restarted = child(`
      await import('./packages/agent/src/instrument.ts');
      const options = Sentry.getClient().getOptions();
      console.log(JSON.stringify({enabled:Sentry.isEnabled(),dsn:options.dsn,environment:options.environment,release:options.release}));
      await Sentry.close(100);
    `)
    assert.deepEqual(restarted, { enabled: true, ...remote })
  })
})

test('reconnect config upgrades an existing profile, preserves local edits, rotates and disables telemetry', () => {
  isolated((_home, child) => {
    const result = child(`
      const config = {server:'https://fleet.example',wsUrl:'wss://fleet.example/agent/connect',hostId:'device',label:'Test',allowCompute:true,allowBrowser:false,maxConcurrency:1};
      saveConfig(config);
      const {applyManagedTelemetry} = await import('./packages/agent/src/managed-telemetry.ts');
      saveConfig({...config,allowCompute:false,pendingInstall:'new-release'});
      applyManagedTelemetry(config, ${JSON.stringify(remote)});
      const initial = loadConfig();
      const first = Sentry.getClient();
      applyManagedTelemetry(config, undefined);
      applyManagedTelemetry(config, {...${JSON.stringify(remote)},dsn:'https://public:secret@example.invalid/123'});
      const ignored = Sentry.getClient() === first;
      applyManagedTelemetry(config, {...${JSON.stringify(remote)},dsn:'https://next@example.invalid/456'});
      const rotated = Sentry.getClient().getOptions().dsn;
      applyManagedTelemetry(config, {...${JSON.stringify(remote)},dsn:null});
      console.log(JSON.stringify({initial,ignored,rotated,disabled:!Sentry.isEnabled(),saved:loadConfig().telemetry}));
    `)
    assert.equal(result.initial.allowCompute, false)
    assert.equal(result.initial.pendingInstall, 'new-release')
    assert.deepEqual(result.initial.telemetry, remote)
    assert.equal(result.ignored, true)
    assert.equal(result.rotated, 'https://next@example.invalid/456')
    assert.equal(result.disabled, true)
    assert.deepEqual(result.saved, { ...remote, dsn: null })
  })
})

test('a replaced device profile cannot receive a stale connection telemetry update', () => {
  isolated((_home, child) => {
    const result = child(`
      const config = {server:'https://old.example',hostId:'old-device'};
      saveConfig({server:'https://new.example',hostId:'new-device'});
      const {applyManagedTelemetry} = await import('./packages/agent/src/managed-telemetry.ts');
      applyManagedTelemetry(config, ${JSON.stringify(remote)});
      console.log(JSON.stringify(loadConfig()));
    `)
    assert.deepEqual(result, { server: 'https://new.example', hostId: 'new-device' })
  })
})

test('pairing with a legacy server clears the previous platform telemetry', () => {
  isolated((home, child) => {
    writeFileSync(join(home, 'config.json'), JSON.stringify({telemetry:remote}))
    const result = child(`
      await import('./packages/agent/src/instrument.ts');
      globalThis.fetch = async () => new Response(JSON.stringify({hostId:'new-device',label:'New',wsUrl:'wss://new.example/agent/connect'}));
      const {pairHost} = await import('./packages/agent/src/pair.ts');
      await pairHost('https://new.example','test-code');
      console.log(JSON.stringify({enabled:Sentry.isEnabled(),saved:loadConfig().telemetry ?? null}));
    `)
    assert.deepEqual(result, { enabled: false, saved: null })
  })
})

test('explicit local configuration overrides managed settings, including opting out', () => {
  assert.equal(telemetryOptions({SENTRY_DSN:''}, remote), null)
  const options = telemetryOptions({SENTRY_DSN:'https://local@example.invalid/9',SENTRY_ENVIRONMENT:'local',SENTRY_RELEASE:'local-v2'}, remote)!
  assert.equal(options.dsn, 'https://local@example.invalid/9')
  assert.equal(options.environment, 'local')
  assert.equal(options.release, 'local-v2')
  assert.equal(telemetryOptions({}, {...remote,dsn:'https://key:secret@example.invalid/1'}), null)
})
