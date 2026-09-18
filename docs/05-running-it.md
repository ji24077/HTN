# Running it

Everything below has been rehearsed end to end on this machine. The only step that needs
a decision from you is §3, and only when you want the address to stop changing.

---

## 1. Host a network

```bash
pnpm doctor      # checks everything, names anything that needs fixing
pnpm db:up       # start the database (once per reboot)
pnpm share       # public address + an invite link
```

`pnpm share` generates a strong operator password into `.env` on first run, opens a public
tunnel, starts the server bound to loopback only, waits until the address genuinely
serves, and prints an invite link. Press Enter in that window for another invite, or from
a second terminal:

```bash
pnpm invite sams-laptop
```

## 2. Get a friend connected

Send them the printed link. It opens a page showing exactly what to run. The short path,
once they have the project folder:

```bash
./scripts/join.sh https://your-address ABCD-1234
```

That checks their Node version, installs pnpm and dependencies if needed, pairs, and
starts taking work. They stop with Ctrl-C, or `pnpm agent pause` to stay connected but
idle.

**What they need first:** Node.js 24+ and a copy of this project folder. That is the one
part not yet automated — see §5.

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
  which `join.sh` sets by default.
