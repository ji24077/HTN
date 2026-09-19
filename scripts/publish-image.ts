/**
 * Build the agent image for every architecture a contributor might have, and push it.
 *
 *   node scripts/publish-image.ts                     # build, push, print the digest
 *   node scripts/publish-image.ts --dry-run           # build both arches, push nothing
 *   node scripts/publish-image.ts --registry ghcr.io/you/dwp-agent
 *
 * Without this there is no answer to "how does someone else get the container": the
 * compose files build from this repository, so joining would mean cloning it first —
 * which is the barrier the image was supposed to remove.
 *
 * Both architectures matter and neither is optional. An Apple Silicon laptop is arm64, a
 * desktop or a cloud box is usually amd64, and a single-arch image does not fail
 * helpfully on the wrong one: on Docker Desktop it silently runs under emulation at a
 * fraction of the speed, which reads as "that machine is slow" rather than "that image
 * is for a different CPU".
 */
import { execFile, execFileSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { promisify } from 'node:util'

const exec = promisify(execFile)

const argv = process.argv.slice(2)
const has = (name: string): boolean => argv.includes(`--${name}`)
const opt = (name: string, fallback: string): string => {
  const i = argv.indexOf(`--${name}`)
  return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1]! : fallback
}

const DRY_RUN = has('dry-run')
const PLATFORMS = opt('platforms', 'linux/amd64,linux/arm64')
const TARGET = opt('target', 'agent')

/**
 * Where the image lives, worked out from the git remote rather than hardcoded.
 *
 * A constant here would be wrong for every fork, and the first thing anyone running this
 * in their own copy would have to edit. GHCR's path is the repository's owner, which the
 * remote already knows.
 */
function defaultRegistry(): string {
  if (process.env.DWP_AGENT_IMAGE) return process.env.DWP_AGENT_IMAGE.replace(/:[^:/]+$/, '')
  try {
    /**
     * Any GitHub remote, not `origin` specifically.
     *
     * This repository's remote is called `htn`, so asking for `origin` printed "No such
     * remote" to stderr and fell through to an unqualified name that cannot be pushed
     * anywhere — a default that looks like it worked and produces an image nobody else
     * can pull.
     */
    const remotes = execFileSync('git', ['remote', '-v'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
    const owner = /github\.com[:/]([^/\s]+)\//.exec(remotes)?.[1]
    if (owner) return `ghcr.io/${owner.toLowerCase()}/dwp-agent`
  } catch { /* not a git checkout, or no remotes */ }
  return 'dwp-agent'
}

const REGISTRY = opt('registry', defaultRegistry())

function version(): string {
  const pkg = JSON.parse(readFileSync('packages/agent/package.json', 'utf8')) as { version: string }
  return pkg.version
}

function gitSha(): string | null {
  try {
    return execFileSync('git', ['rev-parse', '--short=12', 'HEAD'], { encoding: 'utf8' }).trim()
  } catch {
    return null
  }
}

/**
 * Three tags, each answering a different question.
 *
 * `latest` is what a person copies out of the docs. The version is what an operator pins
 * so a `docker compose pull` cannot move them somewhere they did not choose. The commit
 * is what makes a running container traceable to a line of code — the question the
 * binary releases answered with a signed manifest and this answers with a tag.
 */
function tags(): string[] {
  const sha = gitSha()
  return [
    `${REGISTRY}:latest`,
    `${REGISTRY}:${version()}`,
    ...(sha ? [`${REGISTRY}:${sha}`] : []),
  ]
}

async function builderReady(): Promise<string> {
  /**
   * The default `docker` driver cannot build more than one platform at a time, and says
   * so only at the end of an otherwise successful build. Use a container builder, making
   * one if this machine has never done a multi-arch build before.
   */
  const name = 'dwp-agent-builder'
  const { stdout } = await exec('docker', ['buildx', 'ls']).catch(() => ({ stdout: '' }))
  if (!stdout.includes(name)) {
    console.log(`  creating a multi-architecture builder (${name})…`)
    await exec('docker', ['buildx', 'create', '--name', name, '--driver', 'docker-container', '--bootstrap'])
  }
  return name
}

async function main(): Promise<void> {
  const list = tags()
  console.log(`\n  Building ${REGISTRY}`)
  console.log(`    platforms  ${PLATFORMS}`)
  console.log(`    target     ${TARGET}`)
  console.log(`    tags       ${list.map(t => t.split(':').pop()).join(', ')}`)

  if (!DRY_RUN) {
    /**
     * Check we can push before spending several minutes building.
     *
     * A push that fails on credentials after the build has finished is a long way to
     * travel for "unauthorized", and the fix — `docker login ghcr.io` — is not obvious
     * from the error the daemon returns.
     */
    const probe = await exec('docker', ['buildx', 'imagetools', 'inspect', `${REGISTRY}:latest`])
      .then(() => 'exists').catch((err: { stderr?: string }) => err.stderr ?? '')
    if (typeof probe === 'string' && /unauthorized|denied|authentication required/i.test(probe)) {
      console.error(`\n  Not signed in to the registry holding ${REGISTRY}.\n`)
      console.error(`      echo <a GitHub token with write:packages> | docker login ghcr.io -u <you> --password-stdin\n`)
      process.exit(1)
    }
  }

  const builder = await builderReady()
  const started = Date.now()
  await exec('docker', [
    'buildx', 'build',
    '--builder', builder,
    '--platform', PLATFORMS,
    '--target', TARGET,
    '-f', 'deploy/Dockerfile.agent',
    ...list.flatMap(t => ['-t', t]),
    // Provenance and SBOM attestations, so the published image can be traced back.
    '--provenance', 'true',
    DRY_RUN ? '--output=type=cacheonly' : '--push',
    '.',
  ], { maxBuffer: 64 * 1024 * 1024, stdio: 'inherit' } as never)

  console.log(`\n  ${DRY_RUN ? 'Built' : 'Pushed'} in ${((Date.now() - started) / 1000).toFixed(0)}s`)
  if (DRY_RUN) {
    console.log(`  --dry-run: nothing was pushed.\n`)
    return
  }

  const { stdout } = await exec('docker', ['buildx', 'imagetools', 'inspect', `${REGISTRY}:latest`])
  const digest = /Digest:\s*(sha256:[a-f0-9]{64})/.exec(stdout)?.[1]
  const arches = [...stdout.matchAll(/Platform:\s*(\S+)/g)].map(m => m[1])
  console.log(`  digest     ${digest ?? 'unknown'}`)
  console.log(`  platforms  ${arches.join(', ') || 'unknown'}`)
  console.log(`\n  Anyone can now join with:\n`)
  console.log(`      docker run -d --restart unless-stopped -v dwp-agent-data:/data \\`)
  console.log(`        -p 127.0.0.1:43117:43117 -e DWP_INVITE='<their invite link>' \\`)
  console.log(`        ${REGISTRY}:latest\n`)
  console.log(`  Pin it in production with the digest above:  ${REGISTRY}@${digest ?? 'sha256:…'}\n`)
}

main().catch((err: unknown) => {
  console.error(`\n  ${err instanceof Error ? err.message : String(err)}\n`)
  process.exit(1)
})
