import assert from 'node:assert/strict'
import { generateKeyPairSync } from 'node:crypto'
import { test } from 'node:test'
import { binaryReleasePayload, hashBytes, signRelease, verifyBinaryRelease, verifyRelease } from '../src/release.ts'

function fixture() {
  const { privateKey } = generateKeyPairSync('ed25519')
  const binaries = [{ target: 'win32-x64', file: 'agent.exe', sha256: hashBytes(Buffer.from('original binary')) }]
  const apps = [{ target: 'win32-x64', file: 'agent.zip', sha256: hashBytes(Buffer.from('original app')) }]
  const manifest = {
    version: 'test+bin.12345678', sha256: hashBytes(binaryReleasePayload(binaries, apps)),
    bytes: 100, createdAt: '2026-09-19T00:00:00Z', notes: '',
  }
  return { ...signRelease(privateKey, manifest), binaries, apps }
}

test('accepts the published signed binary and app set', () => {
  const release = fixture()
  assert.equal(verifyBinaryRelease(release, release.publicKey), true)
  assert.equal(binaryReleasePayload(release.binaries, release.apps).toString(),
    JSON.stringify([
      `win32-x64:${release.binaries[0]!.sha256}`,
      `app:win32-x64:${release.apps[0]!.sha256}`,
    ].sort()))
})

test('rejects substituted binary bytes even when the original manifest signature is valid', () => {
  const release = fixture()
  release.binaries[0]!.sha256 = hashBytes(Buffer.from('unsigned replacement'))
  assert.equal(verifyRelease(release, release.publicKey), true)
  assert.equal(verifyBinaryRelease(release, release.publicKey), false)
})

test('rejects modified app entries, targets, signatures and signing keys', () => {
  const release = fixture()
  const changedApp = structuredClone(release)
  changedApp.apps[0]!.sha256 = '0'.repeat(64)
  assert.equal(verifyBinaryRelease(changedApp, release.publicKey), false)
  const changedTarget = structuredClone(release)
  changedTarget.binaries[0]!.target = 'darwin-arm64'
  assert.equal(verifyBinaryRelease(changedTarget, release.publicKey), false)
  assert.equal(verifyBinaryRelease({ ...release, signature: 'invalid' }, release.publicKey), false)
  assert.equal(verifyBinaryRelease(release, fixture().publicKey), false)
})

test('rejects malformed indexes and unsafe download paths without throwing', () => {
  for (const value of [null, {}, { manifest: null }, { binaries: [] }]) {
    assert.equal(verifyBinaryRelease(value, 'invalid'), false)
  }
  const release = fixture()
  release.binaries[0]!.file = '../other-file'
  assert.equal(verifyBinaryRelease(release, release.publicKey), false)
})
