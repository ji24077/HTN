/**
 * Cross-compile the agent for every platform, hash it, and sign the set.
 *
 *   node scripts/build-binaries.ts
 *
 * Bun compiles all targets from this one machine, so there is no build matrix and no CI
 * to keep alive. The output is a signed manifest plus one executable per platform, which
 * the installers fetch and verify.
 *
 * Machine learning is deliberately absent from these binaries: the native runtime's
 * shared libraries cannot be carried inside a single file (see docs/06). Binaries cover
 * every pure-JavaScript workload; machines wanting inference use the Node install.
 */
import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync, statSync } from 'node:fs'
import { createPrivateKey } from 'node:crypto'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { hashBytes, signRelease, generateReleaseKey, binaryReleasePayload, type ReleaseManifest } from '@dwp/protocol'
import { packageApps, buildWindowsGuiVariant, type AppEntry } from './lib/apps.ts'

const OUT = join(process.env.DWP_RELEASES_DIR ?? 'releases', 'binaries')
/** Skip the five compiles and repackage from what is already in releases/binaries. */
const SKIP_COMPILE = process.argv.includes('--skip-compile')
const KEY_DIR = process.env.DWP_HOME ?? join(homedir(), '.dwp')
const KEY_PATH = join(KEY_DIR, 'release.key')

/** Bun's target names, and the platform strings an installer will detect. */
const TARGETS = [
  { bun: 'bun-darwin-arm64', os: 'darwin', arch: 'arm64' },
  { bun: 'bun-darwin-x64', os: 'darwin', arch: 'x64' },
  { bun: 'bun-linux-x64', os: 'linux', arch: 'x64' },
  { bun: 'bun-linux-arm64', os: 'linux', arch: 'arm64' },
  { bun: 'bun-windows-x64', os: 'win32', arch: 'x64' },
] as const

function loadKey(): ReturnType<typeof createPrivateKey> {
  mkdirSync(KEY_DIR, { recursive: true, mode: 0o700 })
  if (!existsSync(KEY_PATH)) {
    const { privateKeyPem } = generateReleaseKey()
    writeFileSync(KEY_PATH, privateKeyPem, { mode: 0o600 })
    console.log(`  created a release signing key at ${KEY_PATH}`)
  }
  return createPrivateKey(readFileSync(KEY_PATH, 'utf8'))
}

const privateKey = loadKey()
if (!SKIP_COMPILE) rmSync(OUT, { recursive: true, force: true })
mkdirSync(OUT, { recursive: true })

const agentPkg = JSON.parse(readFileSync('packages/agent/package.json', 'utf8')) as { version: string }
type Built = { target: string; os: string; arch: string; file: string; sha256: string; bytes: number }
const built: Built[] = []

console.log(SKIP_COMPILE
  ? `\n  Repackaging from ${OUT}/ without recompiling…\n`
  : `\n  Building the agent for ${TARGETS.length} platforms…\n`)

for (const t of TARGETS) {
  const suffix = t.os === 'win32' ? '.exe' : ''
  const name = `dwp-agent-${t.os}-${t.arch}${suffix}`
  const outfile = join(OUT, name)
  process.stdout.write(`    ${t.os}/${t.arch}`.padEnd(22))
  try {
    if (SKIP_COMPILE) {
      const bytes = readFileSync(outfile)
      built.push({
        target: `${t.os}-${t.arch}`, os: t.os, arch: t.arch, file: name,
        sha256: hashBytes(bytes), bytes: bytes.length,
      })
      console.log(`${(bytes.length / 1024 / 1024).toFixed(0)} MB (kept)`)
      continue
    }
    execFileSync('bun', [
      'build', '--compile', `--target=${t.bun}`,
      // The optional runtimes are not bundled: they cannot work inside a single file,
      // and pulling them in would only make every binary larger for no gain.
      '--external', 'playwright', '--external', 'onnxruntime-node',
      // So a binary can report its own version without a package.json beside it.
      '--define', `__DWP_VERSION__=${JSON.stringify(agentPkg.version)}`,
      'packages/agent/src/index.ts', '--outfile', outfile,
    ], { stdio: 'pipe' })

    const bytes = readFileSync(outfile)
    built.push({
      target: `${t.os}-${t.arch}`, os: t.os, arch: t.arch, file: name,
      sha256: hashBytes(bytes), bytes: bytes.length,
    })
    console.log(`${(statSync(outfile).size / 1024 / 1024).toFixed(0)} MB`)
  } catch (err) {
    console.log(`FAILED — ${err instanceof Error ? err.message.split('\n')[0] : String(err)}`)
  }
}

if (built.length === 0) {
  console.error('\n  Nothing built. Is bun installed?  brew install oven-sh/bun/bun\n')
  process.exit(1)
}

/**
 * Sign the set, not each file.
 *
 * The installer needs one signature to trust the whole list, and every hash inside the
 * signed payload is then trustworthy — so a swapped binary for one platform is caught
 * even though only one signature was checked.
 */
/**
 * The Windows app variant, made here rather than by the compiler.
 *
 * Measured, not assumed: bun 1.3.11 rejects every `--windows-*` flag when the host is
 * not Windows — `--windows-hide-console`, `--windows-icon` and the version metadata all
 * fail with "only available when compiling on Windows". Since building on a Mac is the
 * entire distribution story, the one that matters is done afterwards by editing the PE
 * header directly. The icon is not: embedding one means rewriting the resource directory
 * of a 116 MB executable, which is a lot of risk for decoration, on a platform this
 * machine cannot run to check the result.
 */
const guiExe = buildWindowsGuiVariant(built)
if (guiExe) built.push(guiExe)

let apps: AppEntry[] = []
try {
  apps = packageApps(built, agentPkg.version)
} catch (err) {
  console.error(`\n  Could not package the apps: ${err instanceof Error ? err.message : String(err)}\n`)
}

/**
 * The apps are inside the signed payload, not merely listed beside it.
 *
 * They are what a person actually downloads, so a hash nobody signed would leave the
 * one artefact that matters unprotected while the binaries it wraps were covered.
 */
const payload = binaryReleasePayload(built, apps)
const manifest: ReleaseManifest = {
  version: `${agentPkg.version}+bin.${hashBytes(Buffer.from(payload)).slice(0, 8)}`,
  sha256: hashBytes(Buffer.from(payload)),
  bytes: built.reduce((n, b) => n + b.bytes, 0),
  createdAt: new Date().toISOString(),
  notes: `standalone binaries for ${built.map(b => b.target).join(', ')}`
    + (apps.length > 0 ? `; desktop apps for ${apps.map(a => a.target).join(', ')}` : ''),
}
const signed = { ...signRelease(privateKey, manifest), binaries: built, apps }
writeFileSync(join(OUT, 'index.json'), JSON.stringify(signed, null, 2) + '\n')

console.log(`\n  ${manifest.version}`)
console.log(`  ${built.length} binaries, ${(manifest.bytes / 1024 / 1024).toFixed(0)} MB total`)
for (const a of apps) console.log(`  ${a.file.padEnd(28)} ${(a.bytes / 1024 / 1024).toFixed(0)} MB`)
console.log(`  signed and written to ${OUT}/\n`)
