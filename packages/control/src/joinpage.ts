const escape = (s: string): string =>
  s.replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]!))

/**
 * The page a friend lands on. Its only job is to turn "someone sent me a link" into
 * two commands they can paste, with the server address and code already filled in.
 */
export function joinPage(origin: string, code?: string): string {
  const o = escape(origin)
  const c = code ? escape(code.trim().toUpperCase()) : 'YOUR-CODE'
  const known = Boolean(code)

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
  p.os{margin:14px 0 0;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);font-weight:600}
  p.os:first-of-type{margin-top:0}
  .warn{border-left:3px solid var(--accent);font-size:14.5px;color:var(--ink2)}
  .warn b{color:var(--ink)}
  footer{color:var(--ink2);font-size:13px;margin-top:24px;text-align:center}
</style></head><body><main>

<h1>Join this compute network</h1>
<p class="lede">You have been invited to let this computer run work for a friend's
${known ? 'network' : 'network'}. It takes two commands. You stay in control and can stop at any time.</p>

<div class="card">
  <h2>1 &middot; Requirements</h2>
  <ol>
    <li><b>Node.js 24 or newer</b> &mdash; check with <code>node -v</code></li>
    <li><b>pnpm</b> &mdash; install with <code>npm i -g pnpm</code></li>
    <li>A copy of the project folder</li>
  </ol>
</div>

<div class="card">
  <h2>2 &middot; Connect &mdash; one command</h2>
  <p class="os">macOS or Linux</p>
  <pre>./scripts/join.sh ${o} ${c}</pre>
  <p class="os">Windows (PowerShell)</p>
  <pre>.\\scripts\\join.ps1 ${o} ${c}</pre>
  <p style="margin:12px 0 0;color:var(--ink2);font-size:14px">Either one checks what you
  need, installs it, connects, and starts taking work. Run only the line for your
  computer.</p>
</div>

<div class="card">
  <h2>3 &middot; Or step by step</h2>
  <pre>pnpm install
pnpm agent pair --server ${o} --code ${c}
pnpm agent run</pre>
  ${known ? '' : '<p style="margin:12px 0 0;color:var(--ink2);font-size:14px">Ask for your pairing code &mdash; it expires 10 minutes after it is created.</p>'}
</div>

<div class="card">
  <h2>4 &middot; Stopping</h2>
  <pre>pnpm agent pause      # stop taking work, right now
pnpm agent resume     # start again</pre>
  <p style="margin:12px 0 0;color:var(--ink2);font-size:14px">Pause is a local switch. It works even
  if this server is unreachable, and running work is cancelled immediately.</p>
</div>

<div class="card warn">
  <b>What this does and does not do.</b> Your computer connects out to the server &mdash;
  nothing is opened up for incoming connections, and you do not need to change any router
  settings. It runs only the specific task types this project ships; it cannot be sent
  arbitrary programs or shell commands. It does not touch your files, your browser
  profile, your camera, or other devices on your network.
</div>

<footer>Server: ${o}</footer>
</main></body></html>`
}
