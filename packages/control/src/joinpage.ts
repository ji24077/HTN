import type { AppEntry } from './installer.ts'

const escape = (s: string): string =>
  s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!))

const mb = (bytes: number): string => `${Math.round(bytes / 1024 / 1024)} MB`

/**
 * The page a friend lands on.
 *
 * It leads with the app, because that is now the path that asks least of them: one
 * download, one warning to click through, and a box to paste a link into. The terminal
 * routes stay below it — the one-line installer for anyone comfortable there, and the
 * full checkout for machines doing machine-learning work, which the single-file builds
 * cannot carry.
 *
 * The warnings are on the page, in advance. An unsigned app produces a frightening
 * dialog, and someone who meets it unprepared closes it and does not come back; someone
 * who was told it is coming and what to click does not.
 */
export function joinPage(
  origin: string,
  code?: string,
  downloads?: { version: string; apps: AppEntry[] } | null,
): string {
  const o = escape(origin)
  const c = code ? escape(code.trim().toUpperCase()) : 'YOUR-CODE'
  const known = Boolean(code)
  const invite = `${o}/join?code=${c}`

  const mac = downloads?.apps.find(a => a.target === 'macos')
  const win = downloads?.apps.find(a => a.target === 'windows-x64')
  const hasApps = Boolean(mac || win)

  const appCard = !hasApps ? '' : `
<div class="card">
  <h2>1 &middot; Download the app</h2>
  <div class="dl">
    ${mac ? `<a class="button" href="/download/${escape(mac.file)}" download>
      <span class="os-name">macOS</span><span class="os-meta">${mb(mac.bytes)} &middot; Apple Silicon and Intel</span></a>` : ''}
    ${win ? `<a class="button" href="/download/${escape(win.file)}" download>
      <span class="os-name">Windows</span><span class="os-meta">${mb(win.bytes)} &middot; 64-bit</span></a>` : ''}
  </div>
  <p class="note">Unzip it, open it, and paste the invite link below. That is the whole
  setup &mdash; nothing else to install.</p>
</div>

<div class="card warn">
  <b>Your computer will warn you about this app, once.</b> It is not signed, because
  signing it costs money this project has not spent. The warning is about that and
  nothing else &mdash; here is exactly what to click.
  <p class="os">macOS</p>
  <p class="step">Double-click it and let it be blocked. Then open <b>System Settings &rarr;
  Privacy &amp; Security</b>, scroll down to <b>Security</b>, and click <b>Open Anyway</b>
  next to the message about this app. Once only; afterwards it opens normally.</p>
  <p class="note" style="margin-top:6px">Older advice says to right-click and choose Open.
  Apple removed that in macOS&nbsp;15, so on anything current it does nothing.</p>
  <p class="os">Windows</p>
  <p class="step">A blue box says &ldquo;Windows protected your PC&rdquo;. Click
  <b>More info</b>, then <b>Run anyway</b>.</p>
  <p class="note">If you would rather check the file first, compare it against the hash
  below. Both the file and the hash came from this server over the same connection, so
  this proves the download was not damaged &mdash; not that the server is honest.</p>
  ${mac ? `<p class="os">macOS</p><pre>shasum -a 256 ~/Downloads/${escape(mac.file)}
${mac.sha256}</pre>` : ''}
  ${win ? `<p class="os">Windows (PowerShell)</p><pre>Get-FileHash ~\\Downloads\\${escape(win.file)}
${win.sha256.toUpperCase()}</pre>` : ''}
</div>

<div class="card">
  <h2>2 &middot; Paste this into the app</h2>
  <pre id="invite">${invite}</pre>
  ${known
    ? '<p class="note">This link works once and expires ten minutes after it was made. Ask for a fresh one if it is refused.</p>'
    : '<p class="note">Ask for your invite link &mdash; it carries a code that expires ten minutes after it is created.</p>'}
</div>
`

  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Join this compute network</title>
<style>
  :root{--bg:#f4f6f7;--card:#fff;--ink:#10161b;--ink2:#46555f;--rule:#d5dbe0;--accent:#0b6e8c;--code:#eef1f3}
  @media (prefers-color-scheme:dark){:root{--bg:#0e1316;--card:#151b1f;--ink:#e4eaed;--ink2:#aebac3;--rule:#2a343a;--accent:#4bb2d1;--code:#11181c}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;padding:32px 16px}
  main{max-width:640px;margin:0 auto}
  h1{font-size:26px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
  p.lede{color:var(--ink2);margin:0 0 28px}
  .card{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:20px 22px;margin:0 0 16px}
  h2{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);margin:0 0 12px;font-weight:600}
  pre{background:var(--code);border-radius:5px;padding:13px 15px;overflow-x:auto;margin:10px 0 0;
      font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace}
  ol{margin:0;padding-left:1.2em}
  li{margin:8px 0}
  code{background:var(--code);padding:1px 5px;border-radius:3px;font:13px ui-monospace,monospace}
  p.os{margin:16px 0 0;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);font-weight:600}
  p.os:first-of-type{margin-top:0}
  p.step{margin:4px 0 0}
  p.note{margin:12px 0 0;color:var(--ink2);font-size:14px}
  .warn{border-left:3px solid var(--accent);font-size:14.5px;color:var(--ink2)}
  .warn b{color:var(--ink)}
  .dl{display:flex;gap:10px;flex-wrap:wrap}
  .button{flex:1 1 220px;display:flex;flex-direction:column;gap:2px;text-decoration:none;
          background:var(--accent);color:#fff;border-radius:6px;padding:13px 16px}
  .button:hover{filter:brightness(1.08)}
  .os-name{font-size:16px;font-weight:600}
  .os-meta{font-size:12.5px;opacity:.85}
  details{margin:0 0 16px}
  summary{cursor:pointer;color:var(--ink2);font-size:14px;padding:6px 0}
  footer{color:var(--ink2);font-size:13px;margin-top:24px;text-align:center}
</style></head><body><main>

<h1>Join this compute network</h1>
<p class="lede">You have been invited to let this computer run work for a friend's network.
You stay in control and can stop at any time.</p>

${appCard}

<details${hasApps ? '' : ' open'}>
  <summary>${hasApps ? 'Prefer a terminal? Two other ways &rarr;' : 'How to join'}</summary>

  <div class="card">
    <h2>One line, no app</h2>
    <p class="os">macOS or Linux</p>
    <pre>curl -sSf ${o}/install | sh -s -- ${c}</pre>
    <p class="os">Windows (PowerShell)</p>
    <pre>irm ${o}/install.ps1 | iex</pre>
    <p class="note">Downloads one executable, checks it against a hash delivered with the
    script, and pairs this computer. No Node, no package manager, no project folder.</p>
  </div>

  <div class="card">
    <h2>Full install &mdash; for machine-learning work</h2>
    <p class="note">The single-file builds cannot carry the inference runtime&rsquo;s native
    libraries, so a machine doing that work needs the project itself: Node 24 or newer,
    pnpm, and a copy of the folder.</p>
    <pre>pnpm install
pnpm agent pair --server ${o} --code ${c}
pnpm agent enable ml
pnpm agent run</pre>
  </div>
</details>

<div class="card">
  <h2>${hasApps ? '3' : ''} &middot; Stopping</h2>
  <p class="note" style="margin-top:0">In the app, press <b>Pause</b> to stop taking work
  and <b>Quit</b> to stop altogether. From a terminal:</p>
  <pre>dwp-agent pause      # stop taking work, right now
dwp-agent resume     # start again</pre>
  <p class="note">Pause is a local switch. It works even if this server is unreachable,
  and anything running is cancelled immediately.</p>
</div>

<div class="card warn">
  <b>What this does and does not do.</b> Your computer connects out to the server &mdash;
  nothing is opened up for incoming connections, and you do not need to change any router
  settings. It runs only the specific task types this project ships; it cannot be sent
  arbitrary programs or shell commands. It does not touch your files, your browser
  profile, your camera, or other devices on your network.
  <p class="note">Updates arrive over that same connection and install themselves. Each one
  is signed by the person running the network, and your computer checks that signature
  against a key it recorded when it joined &mdash; so this server can send you bytes, but
  it cannot send you code you did not already agree to trust.</p>
</div>

<footer>Server: ${o}${downloads ? ` &middot; ${escape(downloads.version)}` : ''}</footer>
</main></body></html>`
}
