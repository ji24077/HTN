import { execFile, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { promisify } from 'node:util'
import type { KeyObject } from 'node:crypto'
import { SignedRelease, SignedBinaryRelease, verifyRelease, verifyBinaryRelease, hashBytes, mintAssertion, createLogger } from '@dwp/protocol'
import { chmodSync, renameSync, unlinkSync } from 'node:fs'
import { platform, arch } from 'node:os'
import { request as httpRequest } from 'node:http'
import { request as httpsRequest } from 'node:https'
import { loadConfig, saveConfig, type AgentConfig } from './config.ts'
import { dnsFallbackEnabled, resolvePublicly, systemCanResolve } from './resolver.ts'
import { installRoot, isCompiledBinary, windowsSubsystem } from './paths.ts'
import { WORKLOADS, installWorkload, isInstalled } from './workloads.ts'

const exec = promisify(execFile)
const log = createLogger({ component: 'agent' })

const IS_WINDOWS = process.platform === 'win32'

/**
 * Errors that mean "something else is holding this file", not "this is broken".
 *
 * Windows is the reason this exists: it keeps an exclusive handle on every native addon
 * a process has loaded, so replacing one from inside the agent that loaded it fails. The
 * pattern is deliberately broad — the exact code varies between pnpm, the filesystem and
 * the Windows version — because the cost of matching too widely is one deferred install,
 * and the cost of matching too narrowly is an update that reports failure and stops.
 */
const FILE_IN_USE = /\bEBUSY\b|\bEPERM\b|\bEACCES\b|resource busy|being used by another process|access is denied|locked/i

/**
 * What an install would have to change, reduced to one hash.
 *
 * Most updates are source-only, and running `pnpm install` for those is pure cost — on
 * Windows it is also the one step that can fail. Comparing this before and after the
 * unpack tells us whether the dependency tree moved at all, so the common update does
 * not touch node_modules on any platform.
 */
function dependencyFingerprint(root: string): string {
  const h = createHash('sha256')
  for (const f of ['pnpm-lock.yaml', 'package.json', 'pnpm-workspace.yaml',
    'packages/agent/package.json', 'packages/protocol/package.json']) {
    h.update(f)
    try { h.update(readFileSync(join(root, f))) } catch { h.update('<absent>') }
  }
  return h.digest('hex')
}

/**
 * Install the way this tree was already installed.
 *
 * Anyone who joins through `join.sh` or `join.ps1` gets `--prod`, which is what keeps the
 * download near 40 MB instead of 350 MB. A plain `pnpm install` over that tree does not
 * merely differ — pnpm rejects it outright with ERR_PNPM_INCLUDED_DEPS_CONFLICT — so the
 * dependency step of every update would fail on exactly the machines this is built for,
 * while passing on the operator's own full checkout.
 */
function installFlags(root: string): string[] {
  try {
    const modules = readFileSync(join(root, 'node_modules', '.modules.yaml'), 'utf8')
    if (/devDependencies:\s*false/.test(modules)) return ['--prod']
  } catch {}
  return []
}

/** execFile puts the useful part in stderr and only "Command failed" in the message. */
function execMessage(err: unknown): string {
  const e = err as { message?: string; stderr?: string; stdout?: string }
  const detail = (e?.stderr ?? e?.stdout ?? '').toString().trim()
  const head = e?.message ?? String(err)
  return detail ? `${head.split('\n')[0]}: ${detail.split('\n').slice(0, 3).join(' ')}` : head
}

const pnpmInstall = (root: string): Promise<unknown> =>
  // shell on Windows only: pnpm is a .cmd shim there, which Node refuses to spawn
  // without one, and CreateProcess would only ever look for a .exe.
  exec('pnpm', ['install', '--silent', ...installFlags(root)],
    { cwd: root, timeout: 600_000, shell: IS_WINDOWS })


/**
 * The same public-resolver fallback the connection already has, for the update path.
 *
 * The agent installs a DNS fallback for Node's `fetch` at startup, but a Bun-compiled
 * binary ignores it — so a machine whose resolver cannot see a freshly created hostname
 * would connect and work, having dialled the address directly, and then be permanently
 * unable to fetch a release. That is the phone-tethering case this project keeps meeting,
 * and it is exactly the machine least likely to get a fix by hand.
 *
 * Only the address is supplied by a different resolver: SNI and Host stay the real
 * hostname, so certificate verification is unchanged.
 */
async function updateFetch(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  try {
    return await fetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs) })
  } catch (err) {
    if (!dnsFallbackEnabled()) throw err
    const { hostname } = new URL(url)
    // If the name resolves, the failure was something else and is the caller's to see.
    if (await systemCanResolve(hostname)) throw err
    log.info('update.direct_address', { host: hostname })
    return directRequest(url, init, timeoutMs)
  }
}

async function directRequest(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const target = new URL(url)
  const addresses = await resolvePublicly(target.hostname)
  if (addresses.length === 0) throw new Error(`no A record for ${target.hostname} from a public resolver`)

  const isHttps = target.protocol === 'https:'
  return new Promise<Response>((resolve, reject) => {
    const req = (isHttps ? httpsRequest : httpRequest)({
      host: addresses[0],
      port: target.port ? Number(target.port) : (isHttps ? 443 : 80),
      path: target.pathname + target.search,
      method: init.method ?? 'GET',
      servername: target.hostname,
      headers: { ...(init.headers as Record<string, string> | undefined), host: target.hostname },
      timeout: timeoutMs,
    }, res => {
      const chunks: Buffer[] = []
      res.on('data', (c: Buffer) => chunks.push(c))
      res.on('end', () => {
        const status = res.statusCode ?? 502
        // 204 and 304 may not carry a body; Response rejects one.
        const body = status === 204 || status === 304 ? null : Buffer.concat(chunks)
        resolve(new Response(body, { status }))
      })
    })
    req.on('timeout', () => req.destroy(new Error(`timed out after ${timeoutMs}ms`)))
    req.on('error', reject)
    req.end()
  })
}

export type UpdateResult =
  | { status: 'current'; version: string }
  | { status: 'updated'; from: string | null; to: string }
  | { status: 'unavailable'; reason: string }
  | { status: 'refused'; reason: string }

/**
 * Fetch, verify and install the release the server is offering.
 *
 * The verification order is deliberate: signature first, then hash, then unpack. A
 * bundle that fails either check is never written anywhere it could be executed, and
 * the signature is checked against the key pinned at pairing rather than the one the
 * release arrives with — otherwise the signature would be decorative.
 */
/**
 * Replace a compiled binary with a newer one.
 *
 * A single-file install has no package manager to run, so updating means swapping the
 * executable itself. The running process keeps its own open file, so renaming over it is
 * safe on macOS and Linux; Windows refuses, so the old file is moved aside first and
 * cleaned up on the next start.
 */
async function updateBinary(cfg: AgentConfig, opts: { force?: boolean }): Promise<UpdateResult> {
  let index: SignedBinaryRelease
  try {
    const res = await updateFetch(`${cfg.server}/release/binaries`, {}, 20_000)
    if (res.status === 404) return { status: 'unavailable', reason: 'this server is not offering binaries' }
    if (!res.ok) return { status: 'unavailable', reason: `HTTP ${res.status}` }
    const parsed = SignedBinaryRelease.safeParse(await res.json())
    if (!parsed.success) return { status: 'refused', reason: 'the binary release index is malformed' }
    index = parsed.data
  } catch (err) {
    return { status: 'unavailable', reason: err instanceof Error ? err.message : String(err) }
  }

  const pinned = cfg.releaseKey ?? index.publicKey
  if (!verifyBinaryRelease(index, pinned)) {
    log.error('update.signature_rejected', { version: index.manifest.version })
    return { status: 'refused', reason: 'the binary list or its signature does not match the pinned release key — refusing to install it' }
  }

  if (!opts.force && cfg.installedRelease === index.manifest.version) {
    return { status: 'current', version: index.manifest.version }
  }

  /**
   * Ask for the same kind of build we already are.
   *
   * On Windows there are two: one with a console, which the command-line install uses
   * and needs for its output, and one without, which is the desktop app. They differ by
   * two bytes in the PE header and nothing else, so an update that fetched the wrong one
   * would still run — it would just silently gain or lose a command prompt.
   */
  const base = `${platform()}-${arch() === 'arm64' ? 'arm64' : 'x64'}`
  const want = windowsSubsystem() === 2 ? `${base}-gui` : base
  const entry = index.binaries.find(b => b.target === want)
  // A server that publishes no windowless build has nothing this app can take: the
  // console build would run, but it would sprout a command prompt the owner never had.
  // Staying put and saying so is the honest outcome.
  if (!entry) return { status: 'unavailable', reason: `no binary published for ${want}` }

  const res = await updateFetch(`${cfg.server}/download/${entry.file}`, {}, 600_000)
  if (!res.ok) return { status: 'unavailable', reason: `downloading the binary failed: HTTP ${res.status}` }
  const bytes = Buffer.from(await res.arrayBuffer())
  if (hashBytes(bytes) !== entry.sha256) {
    return { status: 'refused', reason: 'the downloaded binary does not match the signed hash' }
  }

  const target = process.execPath
  /**
   * Stage beside the target, not in the temp directory.
   *
   * The last step is a rename, and rename cannot cross filesystems. `/tmp` frequently is
   * a different one — tmpfs on most Linux distributions — so staging there fails with
   * EXDEV at the very end, after the entire download and hash check. It cannot reproduce
   * on macOS, where the temp directory and the home directory share a volume.
   *
   * Staging in the install directory also means the rename is atomic, so there is no
   * moment where the agent's own executable is a half-written file.
   */
  const staged = join(dirname(target), `.dwp-agent-${entry.sha256.slice(0, 8)}`)
  try {
    writeFileSync(staged, bytes, { mode: 0o755 })
    chmodSync(staged, 0o755)

    if (platform() === 'win32') {
      // Windows will not overwrite a running executable; move it aside instead.
      const parked = `${target}.old`
      try { unlinkSync(parked) } catch {}
      renameSync(target, parked)
    }
    renameSync(staged, target)

    const previous = cfg.installedRelease ?? null
    saveConfig({ ...cfg, installedRelease: index.manifest.version, releaseKey: pinned })
    log.info('update.installed', { from: previous, to: index.manifest.version, mode: 'binary' })
    return { status: 'updated', from: previous, to: index.manifest.version }
  } catch (err) {
    try { unlinkSync(staged) } catch {}
    return { status: 'unavailable', reason: err instanceof Error ? err.message : String(err) }
  }
}

export async function applyUpdate(
  cfg: AgentConfig,
  privateKey: KeyObject,
  opts: { force?: boolean; trustOnFirstUse?: boolean } = {},
): Promise<UpdateResult> {
  // A single-file install and a source install update in completely different ways.
  if (isCompiledBinary()) {
    if (!cfg.releaseKey && !opts.trustOnFirstUse) {
      return {
        status: 'refused',
        reason: 'this computer pinned no release key when it paired, so it cannot verify updates',
      }
    }
    return updateBinary(cfg, opts)
  }
  // An agent that paired before this server offered signed releases has nothing to
  // verify against. It can adopt the key it is offered, but only on a deliberate,
  // explicit instruction — never silently, and never as part of an automatic update.
  if (!cfg.releaseKey && !opts.trustOnFirstUse) {
    return {
      status: 'refused',
      reason: 'this computer pinned no release key when it paired, so it cannot verify updates.\n' +
        '  Run  pnpm agent trust-updates  to see the key and decide, or pair again to pin it automatically',
    }
  }

  let signed: SignedRelease
  try {
    const res = await updateFetch(`${cfg.server}/release/latest`, {}, 20_000)
    if (res.status === 404) return { status: 'unavailable', reason: 'the server is not offering a release' }
    if (!res.ok) return { status: 'unavailable', reason: `HTTP ${res.status}` }
    signed = SignedRelease.parse(await res.json())
  } catch (err) {
    return { status: 'unavailable', reason: err instanceof Error ? err.message : String(err) }
  }

  const pinned = cfg.releaseKey ?? signed.publicKey
  if (!verifyRelease(signed, pinned)) {
    // Either the server is serving something the operator did not sign, or the operator
    // rotated keys. Both need a human; neither justifies running the code.
    log.error('update.signature_rejected', { version: signed.manifest.version })
    return {
      status: 'refused',
      reason: 'the release was not signed by the key this computer pinned when it paired — refusing to install it',
    }
  }

  if (!opts.force && cfg.installedRelease === signed.manifest.version) {
    return { status: 'current', version: signed.manifest.version }
  }

  const res = await updateFetch(`${cfg.server}/release/${signed.manifest.sha256}`, {
    headers: { authorization: `Bearer ${mintAssertion(cfg.hostId, privateKey)}` },
  }, 180_000)
  if (!res.ok) return { status: 'unavailable', reason: `downloading the bundle failed: HTTP ${res.status}` }

  const bytes = Buffer.from(await res.arrayBuffer())
  if (hashBytes(bytes) !== signed.manifest.sha256) {
    return { status: 'refused', reason: 'the downloaded bundle does not match the hash that was signed' }
  }

  const staging = mkdtempSync(join(tmpdir(), 'dwp-update-'))
  try {
    const archive = join(staging, 'release.tar.gz')
    writeFileSync(archive, bytes)
    const root = installRoot()

    // Unpack over the install, then reconcile dependencies. Only source is replaced;
    // ~/.dwp — keys, config, cached artifacts — is a separate directory and untouched.
    //
    // tar is bsdtar on Windows 10 1803 and newer, so the same invocation works there.
    await exec('tar', ['-xzf', archive, '-C', root])

    // Both sides of this comparison are bundle-side values: what the previous release
    // shipped, against what this one ships. An agent that has never recorded one installs
    // unconditionally, which is the safe direction to be wrong in.
    const shipped = dependencyFingerprint(root)
    const dependenciesMoved = cfg.depFingerprint == null || shipped !== cfg.depFingerprint

    /**
     * The dependency step, which is the one place the platforms genuinely differ.
     *
     * On POSIX a file being replaced while something reads it is unremarkable, so the
     * install happens here and the agent restarts into a finished tree. Windows keeps an
     * exclusive handle on every native addon this process has loaded — `onnxruntime-node`
     * and Playwright both qualify — so the same call fails, and it fails *after* the new
     * source is already on disk, which is the worst moment to give up.
     *
     * So on Windows it is deferred: recorded in the config, run at the next start by a
     * process that has loaded none of those files yet. The restart was happening anyway,
     * so this costs nothing but ordering.
     */
    let pendingInstall: string | null = null
    if (!dependenciesMoved) {
      // Worth saying out loud: on Windows this is the difference between an update that
      // touches node_modules and one that cannot possibly hit a locked file.
      log.info('update.dependencies_unchanged', { version: signed.manifest.version })
    } else if (IS_WINDOWS) {
      pendingInstall = signed.manifest.version
      log.info('update.install_deferred', { version: pendingInstall, why: 'windows holds loaded binaries open' })
    } else {
      try {
        await pnpmInstall(root)
        await restoreWorkloads(cfg)
      } catch (err) {
        /**
         * The new source is already on disk at this point, so failing the update here is
         * the one thing that must not happen: the agent would report `unavailable`, keep
         * the old version number, and re-download and re-unpack the same release on every
         * reconnect, forever. Deferring records the version, keeps the retry, and says so.
         *
         * A lock is not the only way this fails on POSIX either — a mismatched install
         * mode does too, which is how this was found.
         */
        pendingInstall = signed.manifest.version
        const why = execMessage(err)
        log.warn('update.install_deferred', { version: pendingInstall, why })
      }
    }

    const previous = cfg.installedRelease ?? null
    saveConfig({
      ...cfg,
      installedRelease: signed.manifest.version,
      releaseKey: pinned,
      pendingInstall,
      depFingerprint: shipped,
    })
    log.info('update.installed', { from: previous, to: signed.manifest.version, pendingInstall })
    return { status: 'updated', from: previous, to: signed.manifest.version }
  } catch (err) {
    return { status: 'unavailable', reason: err instanceof Error ? err.message : String(err) }
  } finally {
    rmSync(staging, { recursive: true, force: true })
  }
}

/**
 * Put back the optional runtimes this machine had chosen.
 *
 * `pnpm agent enable ml` records itself in packages/agent/package.json, and an update
 * replaces that file with the one from the bundle — which does not list it. The install
 * that follows therefore *removes* the runtime, and the machine quietly stops being able
 * to do the work it was enrolled for. The config remembers the choice precisely so this
 * can put it back.
 */
async function restoreWorkloads(cfg: AgentConfig): Promise<void> {
  for (const id of cfg.enabledWorkloads ?? []) {
    const workload = WORKLOADS.find(w => w.id === id)
    if (!workload || isInstalled(workload.package)) continue
    log.info('update.workload_restoring', { workload: id })
    try {
      await installWorkload(id, { quiet: true })
      log.info('update.workload_restored', { workload: id })
    } catch (err) {
      // Never fatal: an agent that can still run echo and the walker is worth more than
      // one that refuses to start because an 85 MB optional download failed.
      log.error('update.workload_restore_failed', { workload: id, err })
      console.error(`\n  Could not reinstall ${workload.describe} after the update.` +
        `\n  Re-enable it with:  pnpm agent enable ${id}\n`)
    }
  }
}

/**
 * Finish an update whose dependency step was deferred.
 *
 * Called at startup, before anything loads a native addon, which is the whole point:
 * this process holds none of the files the install needs to replace. On POSIX there is
 * normally nothing to do here — the install already happened inline.
 */
export async function completePendingInstall(): Promise<void> {
  const cfg = loadConfig()
  if (!cfg?.pendingInstall) return

  const root = installRoot()
  console.log(`  Finishing the update to ${cfg.pendingInstall} — reconciling dependencies…`)
  try {
    await pnpmInstall(root)
  } catch (err) {
    const message = execMessage(err)
    // The process that deferred this exits moments before we start, so its handles can
    // still be closing. One retry covers that overlap; anything longer is a real lock.
    if (!FILE_IN_USE.test(message)) return failPending(cfg.pendingInstall, message)
    log.warn('update.install_retrying', { version: cfg.pendingInstall, why: message.split('\n')[0] })
    await new Promise(r => setTimeout(r, 2000))
    try {
      await pnpmInstall(root)
    } catch (retryErr) {
      return failPending(cfg.pendingInstall, execMessage(retryErr))
    }
  }

  // Re-read: the deferred install may have taken long enough for something else to have
  // written the config, and clobbering a newer one would lose it.
  const latest = loadConfig() ?? cfg
  saveConfig({ ...latest, pendingInstall: null })
  log.info('update.install_completed', { version: cfg.pendingInstall })
  console.log('  Dependencies are up to date.\n')

  await restoreWorkloads(latest)
}

/**
 * The flag is deliberately left set, so the next start tries again.
 *
 * The alternative — clearing it and carrying on — produces a machine that believes it is
 * up to date while running new source against old dependencies, which is far harder to
 * notice than a message every time it starts.
 */
function failPending(version: string, message: string): void {
  log.error('update.install_failed', { version, message: message.split('\n')[0] })
  console.error(
    `\n  Could not finish the dependency step of the update to ${version}.` +
    `\n    ${message.split('\n')[0]}` +
    `\n  The agent will still run, but a workload needing a new dependency may not.` +
    `\n  Fix it by running  pnpm install  in the project folder, or just start the agent again.\n`,
  )
}

/**
 * Restart into the freshly installed code.
 *
 * Detached, so the new process outlives this one rather than dying with it, and with
 * stdio inherited so whoever is watching the window keeps seeing output.
 */
export function restartIntoNewVersion(): never {
  // `require` does not exist in an ES module. An earlier version called it here, so the
  // update installed correctly and then the restart threw — leaving the agent updated
  // and dead, which is the worst of both outcomes and invisible to whoever owns the
  // machine. Import it properly instead.
  //
  // `detached` is POSIX-only on purpose. There it is what lets the replacement escape
  // this process group and survive; on Windows a child already outlives its parent, and
  // detaching additionally means DETACHED_PROCESS — no console — so an agent restarted
  // that way would keep running with its output going nowhere, in the one window its
  // owner is watching.
  /**
   * What counts as "the arguments" is not the same for a script and for a binary.
   *
   * Node's argv is [node, script, ...args], so slice(1) is right there. A Bun standalone
   * executable's is ["bun", "/$bunfs/root/<name>", ...args]: argv[0] is the literal
   * string "bun" and argv[1] a path inside a virtual filesystem — neither exists on disk.
   * slice(1) therefore handed the replacement that virtual path as its first argument,
   * which it read as a command it did not recognise, printed the help for, and exited 0.
   *
   * The effect was that every binary update installed correctly and then failed to come
   * back, leaving the machine on the new version and not running — and exiting 0 meant
   * launchd and systemd both saw a clean shutdown and did not restart it either. Silent,
   * and visible only as a host that went offline around the time a release went out.
   */
  const args = isCompiledBinary() ? process.argv.slice(2) : process.argv.slice(1)
  const child = spawn(process.execPath, args, {
    detached: !IS_WINDOWS,
    stdio: 'inherit',
    cwd: process.cwd(),
    env: process.env,
  })
  child.unref()
  process.exit(0)
}

export function currentVersion(): string | null {
  const cfg = loadConfig()
  if (cfg?.installedRelease) return cfg.installedRelease
  try {
    return (JSON.parse(readFileSync(join(installRoot(), 'packages/agent/package.json'), 'utf8')) as { version: string }).version
  } catch {
    return null
  }
}
