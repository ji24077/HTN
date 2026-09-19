# Running it

Everything below has been rehearsed end to end on this machine. The only step that needs
a decision from you is §3, and only when you want the address to stop changing.

---

## 1. Host a network

**One time, ever:**

```bash
pnpm install
pnpm db:up       # the database container now restarts by itself after a reboot
```

**Every time you want to host:**

```bash
pnpm share       # that is the whole thing
```

`pnpm doctor` checks everything and names anything that needs fixing, if a session does
not start cleanly.

`pnpm share` generates a strong operator password into `.env` on first run, opens a public
tunnel, starts the server bound to loopback only, waits until the address genuinely
serves, and prints an invite link. Press Enter in that window for another invite, or from
a second terminal:

```bash
pnpm invite sams-laptop
```

## 2. Get a friend connected

### The one-time part, per computer

Send them an invite link. They run, once — on macOS or Linux:

```bash
./scripts/join.sh https://your-address ABCD-1234
```

or, on Windows, from PowerShell:

```powershell
.\scripts\join.ps1 https://your-address ABCD-1234
```

The invite page shows both and they run only their own. Either one checks their Node
version, installs dependencies, pairs, and starts taking work. Pairing generates a keypair
on their machine and stores it in `~/.dwp/` (`C:\Users\<name>\.dwp` on Windows).

If PowerShell refuses to run the script, that is the execution policy rather than the
script: `powershell -ExecutionPolicy Bypass -File .\scripts\join.ps1 <url> <code>`.

**A pairing code is needed once per computer, not once per session.** It is single-use and
expires in ten minutes.

### Every time after that

```bash
pnpm agent run
```

No code, no link, no setup. Verified: stopped and restarted an agent repeatedly, and
restarted the server underneath it — it rejoined by itself each time and carried on
working with nobody touching it.

They can stop with Ctrl-C, or `pnpm agent pause` to stay enrolled but idle.

**What they need first:** Node.js 24+ and a copy of this project folder — see §5.

## 3. Make the address permanent — the one thing needing your decision

Quick tunnel addresses change every time you restart, and an agent remembers the address
it paired with. With a handful of friends that becomes annoying quickly.

Pick one, then run `node scripts/share.ts --url https://your-address`:

| Option | Signup | Domain needed | Notes |
| --- | --- | --- | --- |
| **Tailscale Funnel** | Tailscale account | No | Gives a permanent `*.ts.net` address. Friends install nothing — it is an ordinary public URL. Easiest if you do not own a domain. |
| **Cloudflare named tunnel** | Cloudflare account | Yes (~$10/yr) | The path Cloudflare intends for anything beyond experimenting. |
| **ngrok** | ngrok account | No (one free reserved domain) | Simple and well documented. |

Each needs you to sign in once — that is the manual step, and it cannot be automated
because it is your account.

**Until then**, a rotating address is survivable: friends do not need to pair again, only
to re-point.

```bash
pnpm agent set-server --server https://your-new-address
```

Their identity, keys and history are unchanged. The agent also detects this case by itself
and prints that exact command after a few failed attempts.

## 4. When something goes wrong

```bash
pnpm logs                  # live
pnpm logs --story          # per-computer: joins, drops, refusals, with reasons
pnpm logs --level warn     # only problems
pnpm logs --since 30m --no-follow
pnpm doctor                # re-check the whole setup
```

The first question is always **did their request reach you at all?** If nothing appears in
the log, the problem is their network or the address, not the software. If
`agent.auth_rejected` appears, its `reason` field says precisely what was wrong.

Agents diagnose their own failures too: after three failed attempts they print whether the
cause is DNS, a refused connection, a dead tunnel or a broken certificate — and what to do
about it.

## 5. Known rough edges

- **A friend needs Node 24 and the project folder.** Packaging a signed installer is the
  obvious next step and has not been done.
- **The address changes on restart** until §3 is done.
- **One computer at a time has been tested.** Everything here ran on this Mac; a genuine
  second machine on a different network is still unproven, though eleven simulated network
  scenarios cover the software's behaviour.
- **Your own machine may not resolve your own tunnel address.** Observed here on phone
  tethering. It works for everyone else; agents on this machine need `DWP_DNS_FALLBACK=1`,
  which `join.sh` and `join.ps1` both set by default.
- **Windows hosts join, but have not been run end to end.** The agent, the join script and
  the tooling all have Windows paths now and a Windows host is just another host to the
  scheduler — nothing filters on OS. What is untested is the machine itself: everything
  below was rehearsed on macOS only.
- **The private key is protected differently on Windows.** Windows has no POSIX mode bits,
  so the `0600` check that guards the key on macOS and Linux cannot run there. The
  equivalent is an ACL granting only the current account, applied by `icacls` when the key
  is created. If that call fails the agent says so loudly and continues, where its POSIX
  counterpart would refuse to start — a deliberate difference, since the usual Windows case
  is a single-account laptop rather than a shared box.
- **The remote-browser debug-port assertion now runs on Windows too**, reading command
  lines from `Win32_Process` rather than `ps`. It is only skipped if PowerShell itself
  cannot be run, and says so when that happens instead of reporting a pass.
