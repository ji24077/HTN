/**
 * Wrap the compiled agent into something a person can double-click, and zip it.
 *
 * There is no app framework here and nothing to install: the "app" is the same binary
 * the command-line install uses, plus the small amount of per-platform packaging each
 * operating system needs to treat an executable as an application. macOS wants a bundle
 * directory with a plist and an icon; Windows wants the icon and the console-free flag
 * compiled in, which `bun build` does, so it wants only a folder and a readme.
 *
 * Both are produced from one Mac. That is the whole reason this shape was chosen over
 * Tauri, which cannot cross-compile to Windows, and over Electron, which would have
 * added a runtime larger than the payload.
 */
import { execFileSync } from 'node:child_process'
import { chmodSync, copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { hashBytes } from '@dwp/protocol'

const BINARIES = join('releases', 'binaries')
const STAGE = join('releases', 'apps')

export type AppEntry = {
  target: 'macos' | 'windows-x64'
  file: string
  sha256: string
  bytes: number
  /** What the person sees after unzipping, so a download page can say it exactly. */
  opens: string
}

type Built = { target: string; os: string; arch: string; file: string; sha256: string; bytes: number }

/**
 * Turn a console executable into a windowless one by flipping two bytes.
 *
 * The PE Subsystem field decides whether Windows gives a process a console: 3 is
 * IMAGE_SUBSYSTEM_WINDOWS_CUI, which means a command prompt opens behind the app and
 * closing it kills the agent; 2 is _GUI, which means no console at all. It sits at
 * offset 68 of the Optional Header in both PE32 and PE32+, because the fields that
 * differ between them all come earlier.
 *
 * The header's CheckSum is left stale on purpose. Windows verifies it for drivers and
 * for binaries loaded into privileged processes, not for an ordinary user-mode program,
 * and recomputing it would mean implementing the PE checksum for a 116 MB file to
 * satisfy a check that will not run.
 */
function toWindowsGuiSubsystem(bytes: Buffer): Buffer {
  const peOffset = bytes.readUInt32LE(0x3c)
  if (bytes.toString('ascii', peOffset, peOffset + 4) !== 'PE\u0000\u0000') {
    throw new Error('not a PE executable')
  }
  const optional = peOffset + 24
  const magic = bytes.readUInt16LE(optional)
  if (magic !== 0x10b && magic !== 0x20b) throw new Error(`unknown optional header magic ${magic.toString(16)}`)

  const subsystemAt = optional + 68
  const current = bytes.readUInt16LE(subsystemAt)
  if (current === 2) return bytes
  if (current !== 3) throw new Error(`refusing to change subsystem ${current}; expected 3 (console)`)

  const patched = Buffer.from(bytes)
  patched.writeUInt16LE(2, subsystemAt)
  return patched
}

/**
 * Produce the windowless Windows build, as its own release target.
 *
 * It has to be a separate download rather than a replacement: the command-line install
 * (`install.ps1`) runs the agent from a terminal and needs its output, which a GUI
 * subsystem process does not have. So both are published, and an agent updating itself
 * asks for whichever kind it already is — see `windowsSubsystem()` in the agent.
 */
export function buildWindowsGuiVariant(built: Built[]): Built | null {
  const win = built.find(b => b.target === 'win32-x64')
  if (!win) return null
  const file = 'dwp-agent-win32-x64-gui.exe'
  const out = join(BINARIES, file)
  try {
    const patched = toWindowsGuiSubsystem(readFileSync(join(BINARIES, win.file)))
    writeFileSync(out, patched)
    console.log(`    win32/x64 (no console)  ${(patched.length / 1024 / 1024).toFixed(0)} MB`)
    return { target: 'win32-x64-gui', os: 'win32', arch: 'x64', file, sha256: hashBytes(patched), bytes: patched.length }
  } catch (err) {
    console.log(`    win32/x64 (no console)  FAILED — ${err instanceof Error ? err.message : String(err)}`)
    return null
  }
}

const WINDOWS_README = `DWP Agent for Windows
=====================

1. Unzip this folder anywhere you like — your Desktop is fine.
2. Double-click "DWP Agent.exe".

Windows will show a blue box: "Windows protected your PC".

  Click "More info", then "Run anyway".

That warning appears because this app is not code-signed, which costs money we
have not spent. It is not a virus warning and it does not mean anything is wrong
— Windows shows it for every program it has not seen many times before. If you
would rather not take our word for it, the download page lists a SHA-256 hash
you can check with:

    Get-FileHash "DWP Agent.exe"

What happens next
-----------------

A small window opens and asks for the invite link you were sent. Paste it in and
press Join. That is the whole setup.

The window is just a control panel. Closing it leaves the agent running in the
background; use Quit inside the window to stop it properly. Turn on "Start
automatically when I log in" and it will rejoin by itself after a restart.

Updates install themselves over the internet. You will not need to download this
file again.

If nothing happens when you open it
-----------------------------------

Run "DWP Agent (with console).exe" instead. It is the same program, but it keeps
a black text window open so you can see what it says. Send that text back — it
will name the problem exactly.

(Leave that window open while you use it: closing it stops the agent. That is
why it is the fallback and not the main one.)
`


function infoPlist(version: string): string {
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>DWP Agent</string>
  <key>CFBundleDisplayName</key><string>DWP Agent</string>
  <key>CFBundleIdentifier</key><string>com.dwp.agent</string>
  <key>CFBundleExecutable</key><string>dwp-agent</string>
  <key>CFBundleIconFile</key><string>icon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${version}</string>
  <key>CFBundleVersion</key><string>${version}</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <!--
    No Dock icon and no menu bar. The window is a browser window, so a Dock tile for
    this process would be a second, dead representation of the same app — clicking it
    would do nothing, which is worse than its absence.
  -->
  <key>LSUIElement</key><true/>
</dict>
</plist>
`
}

function zipSize(path: string): { sha256: string; bytes: number } {
  const bytes = readFileSync(path)
  return { sha256: hashBytes(bytes), bytes: bytes.length }
}

export function packageApps(built: Built[], version: string): AppEntry[] {
  rmSync(STAGE, { recursive: true, force: true })
  mkdirSync(STAGE, { recursive: true })
  const entries: AppEntry[] = []

  // --------------------------------------------------------------- macOS
  const macArm = built.find(b => b.target === 'darwin-arm64')
  const macX64 = built.find(b => b.target === 'darwin-x64')
  if (macArm && macX64) {
    const app = join(STAGE, 'DWP Agent.app')
    const macos = join(app, 'Contents', 'MacOS')
    const resources = join(app, 'Contents', 'Resources')
    mkdirSync(macos, { recursive: true })
    mkdirSync(resources, { recursive: true })
    writeFileSync(join(app, 'Contents', 'Info.plist'), infoPlist(version))
    writeFileSync(join(app, 'Contents', 'PkgInfo'), 'APPL????')

    /**
     * One universal binary, not two thin ones behind a launcher script.
     *
     * The first version used a shell script as the bundle's executable, picking a build
     * by `uname -m`. It worked, and it could not be signed properly: a script is not a
     * Mach-O, so `codesign` has nowhere inside it to put a signature and writes a
     * *detached* one into an extended attribute instead. Stripping xattrs before
     * archiving then destroyed it, and not stripping them means the signature only
     * survives if the person unzips with Finder — `unzip` turns those attributes into
     * `._` files and the bundle arrives broken. Either way it fails on someone's machine
     * and not on this one.
     *
     * `lipo` merges the two into one file whose signature lives inside it, which
     * survives any archiver. Verified that Bun's appended bundle comes through the merge
     * intact — it is not obvious that it would, since the payload rides along after the
     * Mach-O the linker produced.
     */
    execFileSync('lipo', ['-create',
      join(BINARIES, macArm.file), join(BINARIES, macX64.file),
      '-output', join(macos, 'dwp-agent')])
    chmodSync(join(macos, 'dwp-agent'), 0o755)
    copyFileSync(join('assets', 'icon.icns'), join(resources, 'icon.icns'))

    /**
     * Ad-hoc sign the bundle. Free, and not optional on Apple Silicon.
     *
     * Bun linker-signs the executables it produces, but nothing signs the bundle around
     * them, and `spctl` rejects that with "no usable signature". Unsigned is not merely
     * a worse version of unnotarized: a quarantined bundle with no signature at all can
     * fail as *"is damaged and can't be opened. You should move it to the Trash"*, which
     * offers no way past it — where an ad-hoc signed one gets the ordinary "unidentified
     * developer" block, which Privacy & Security can release.
     *
     * This does not remove the warning; only notarization does, and that needs the paid
     * programme. It makes the warning the kind a person can get past.
     */
    execFileSync('codesign', ['--force', '--sign', '-', app], { stdio: 'pipe' })
    // Fail the build rather than ship a bundle whose seal is already broken here.
    execFileSync('codesign', ['--verify', '--strict', app], { stdio: 'pipe' })

    const file = 'DWP-Agent-macOS.zip'
    const zip = join(STAGE, file)
    /**
     * ditto, not zip: it preserves the bundle's executable bits, where a plain `zip`
     * produces an archive whose unpacked .app macOS refuses to launch — a failure that
     * looks like a broken download rather than a bad build.
     *
     * `--norsrc --noextattr` is what makes the signature survive, and it took two wrong
     * answers to find. macOS stamps every file it writes with com.apple.provenance;
     * ditto faithfully carries such attributes, and `unzip` cannot restore them, so it
     * materialises each one as a real `._name` file *inside the bundle*. Those extra
     * files are not in the signature's seal, so the app arrives reporting "a sealed
     * resource is missing or invalid" — for anyone who unzips from a terminal, while
     * working perfectly for anyone who uses Finder. Not carrying the attributes at all
     * removes the problem at its source rather than cleaning up after it.
     */
    execFileSync('ditto', ['-c', '-k', '--norsrc', '--noextattr', '--keepParent', app, zip])
    entries.push({ target: 'macos', file, ...zipSize(zip), opens: 'DWP Agent.app' })
  }

  // ------------------------------------------------------------- Windows
  const win = built.find(b => b.target === 'win32-x64')
  const winGui = built.find(b => b.target === 'win32-x64-gui')
  if (win) {
    const folder = join(STAGE, 'DWP Agent')
    mkdirSync(folder, { recursive: true })
    /**
     * Both builds ship, and the readme says to try the second if the first does nothing.
     *
     * The windowless build cannot be run on the machine that produced it, so its one
     * unverified assumption — that a Bun binary starts with no console attached — is
     * settled by whoever opens it first. A fallback in the same folder turns that from a
     * dead end into a one-click answer, and into a useful report either way.
     */
    copyFileSync(join(BINARIES, (winGui ?? win).file), join(folder, 'DWP Agent.exe'))
    if (winGui) copyFileSync(join(BINARIES, win.file), join(folder, 'DWP Agent (with console).exe'))
    writeFileSync(join(folder, 'Read me first.txt'), WINDOWS_README)

    const file = 'DWP-Agent-Windows-x64.zip'
    const zip = join(STAGE, file)
    // -X drops the macOS metadata that would otherwise litter the folder on Windows.
    rmSync(join(STAGE, file), { force: true })
    execFileSync('zip', ['-r', '-q', '-X', file, 'DWP Agent'], { cwd: STAGE })
    entries.push({ target: 'windows-x64', file, ...zipSize(zip), opens: 'DWP Agent\\DWP Agent.exe' })
  }

  return entries
}
