/**
 * Sign-in for the dashboard.
 *
 * The dashboard is reachable on the public address, so it needs a way in from a browser
 * rather than only a cookie set by curl. Deliberately plain: no account creation, no
 * password reset, no hints about whether an address exists.
 */
export const loginHtml = (next = '/dashboard'): string => `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sign in</title>
<style>
  :root{--bg:#f4f6f7;--card:#fff;--ink:#10161b;--ink2:#5a6a74;--rule:#dde3e7;--accent:#0b6e8c;--bad:#8c2f2f}
  @media (prefers-color-scheme:dark){:root{--bg:#0e1316;--card:#161c21;--ink:#e4eaed;--ink2:#a9b6bf;--rule:#28333a;--accent:#4bb2d1;--bad:#d98686}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);display:grid;place-items:center;min-height:100vh;
    padding:24px;font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
  form{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:26px;width:100%;max-width:360px}
  h1{font-size:18px;margin:0 0 4px}
  p.sub{color:var(--ink2);font-size:13.5px;margin:0 0 20px}
  label{display:block;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--ink2);margin:14px 0 5px}
  input{width:100%;padding:9px 11px;border:1px solid var(--rule);border-radius:6px;background:var(--bg);
    color:var(--ink);font-size:15px}
  input:focus{outline:2px solid var(--accent);outline-offset:1px}
  button{width:100%;margin-top:20px;padding:10px;border:0;border-radius:6px;background:var(--accent);
    color:#fff;font-size:15px;font-weight:600;cursor:pointer}
  button:disabled{opacity:.6;cursor:default}
  .err{color:var(--bad);font-size:13.5px;margin-top:14px;min-height:1.2em}
</style></head><body>
<form id="f">
  <h1>Sign in</h1>
  <p class="sub">Operator access to the fleet dashboard.</p>
  <label for="email">Email</label>
  <input id="email" name="email" type="email" autocomplete="username" required autofocus>
  <label for="password">Password</label>
  <input id="password" name="password" type="password" autocomplete="current-password" required>
  <button id="go" type="submit">Sign in</button>
  <div class="err" id="err"></div>
</form>
<script>
const f = document.getElementById('f')
f.addEventListener('submit', async e => {
  e.preventDefault()
  const go = document.getElementById('go'); const err = document.getElementById('err')
  go.disabled = true; err.textContent = ''
  try {
    const res = await fetch('/auth/login', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        email: document.getElementById('email').value,
        password: document.getElementById('password').value,
      }),
    })
    if (res.ok) { location.href = ${JSON.stringify(next)}; return }
    err.textContent = res.status === 429
      ? 'Too many attempts. Wait a few minutes.'
      : 'That email and password did not match.'
  } catch {
    err.textContent = 'Could not reach the server.'
  }
  go.disabled = false
})
</script></body></html>`
