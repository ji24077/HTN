import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtempSync, writeFileSync, chmodSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { enrolmentFromEnvironment, parseInvite } from '../src/pair.ts'

const ENROLMENT_VARS = ['DWP_INVITE', 'DWP_INVITE_FILE', 'DWP_SERVER', 'DWP_CODE', 'DWP_LABEL'] as const

/** Start from a clean environment every time: these variables are read, not passed in. */
function withEnv<T>(vars: Record<string, string | undefined>, body: () => T): T {
  const previous = new Map<string, string | undefined>()
  for (const key of ENROLMENT_VARS) {
    previous.set(key, process.env[key])
    delete process.env[key]
  }
  for (const [key, value] of Object.entries(vars)) {
    if (!previous.has(key)) previous.set(key, process.env[key])
    if (value !== undefined) process.env[key] = value
  }
  try {
    return body()
  } finally {
    for (const [key, value] of previous) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
  }
}

test('an invite link yields the origin and the code', () => {
  assert.deepEqual(
    parseInvite('https://fleet.example.com/join?code=ABC123'),
    { server: 'https://fleet.example.com', code: 'ABC123' },
  )
  // Whitespace is what a copied line actually carries.
  assert.deepEqual(
    parseInvite('  https://fleet.example.com/join?code=ABC123\n'),
    { server: 'https://fleet.example.com', code: 'ABC123' },
  )
})

test('a link with no code is refused with the reason, not a crash', () => {
  const parsed = parseInvite('https://fleet.example.com/join')
  assert.ok('error' in parsed)
  assert.match(parsed.error, /no invite code/)
})

test('a bare code needs somewhere to send it', () => {
  const alone = parseInvite('ABC123')
  assert.ok('error' in alone)
  assert.match(alone.error, /whole invite link/)

  assert.deepEqual(
    parseInvite('ABC123', 'https://fleet.example.com/'),
    { server: 'https://fleet.example.com', code: 'ABC123' },
  )
})

test('an empty string asks for the link rather than reporting a parse failure', () => {
  const parsed = parseInvite('   ')
  assert.ok('error' in parsed)
  assert.match(parsed.error, /Paste the invite link/)
})

test('no invite in the environment is not an error', () => {
  withEnv({}, () => assert.equal(enrolmentFromEnvironment(), null))
  // A server with no code cannot pair, and inventing one would burn a pairing attempt.
  withEnv({ DWP_SERVER: 'https://fleet.example.com' }, () =>
    assert.equal(enrolmentFromEnvironment(), null))
})

test('DWP_SERVER and DWP_CODE compose into the same link a person would paste', () => {
  withEnv({ DWP_SERVER: 'https://fleet.example.com/', DWP_CODE: 'ABC123', DWP_LABEL: 'box-7' }, () => {
    const found = enrolmentFromEnvironment()
    assert.ok(found)
    assert.equal(found.label, 'box-7')
    assert.deepEqual(parseInvite(found.invite), { server: 'https://fleet.example.com', code: 'ABC123' })
  })
})

test('a file beats the environment, because a secret should not be in docker inspect', () => {
  const dir = mkdtempSync(join(tmpdir(), 'dwp-enrol-'))
  const path = join(dir, 'invite')
  writeFileSync(path, 'https://from-file.example.com/join?code=FILE1\n', { mode: 0o600 })
  chmodSync(path, 0o600)

  withEnv({ DWP_INVITE_FILE: path, DWP_INVITE: 'https://from-env.example.com/join?code=ENV1' }, () => {
    const found = enrolmentFromEnvironment()
    assert.ok(found)
    assert.match(found.from, /DWP_INVITE_FILE/)
    assert.deepEqual(parseInvite(found.invite), { server: 'https://from-file.example.com', code: 'FILE1' })
  })
})

test('an unreadable secret falls through to the environment instead of failing the start', () => {
  withEnv({
    DWP_INVITE_FILE: join(tmpdir(), 'dwp-enrol-nothing-here', 'invite'),
    DWP_INVITE: 'https://from-env.example.com/join?code=ENV1',
  }, () => {
    const found = enrolmentFromEnvironment()
    assert.ok(found)
    assert.equal(found.from, 'DWP_INVITE')
  })
})

test('an empty label is absent, not an empty name for the machine', () => {
  withEnv({ DWP_INVITE: 'https://fleet.example.com/join?code=ABC123', DWP_LABEL: '   ' }, () => {
    assert.equal(enrolmentFromEnvironment()?.label, undefined)
  })
})
