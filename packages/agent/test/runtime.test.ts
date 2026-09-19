import assert from 'node:assert/strict'
import { test } from 'node:test'
import { hostname } from 'node:os'
import {
  containerFreeBytes, containerLimits, guiBindHost, guiPort, hostAllowed, imageReference,
  isContainer, shouldOpenWindow, supervisorNote,
} from '../src/runtime.ts'
import { probe } from '../src/capability.ts'

/**
 * Every test here drives the explicit override rather than the filesystem sniff.
 *
 * The sniff is the part that cannot be tested honestly from inside a test runner — on
 * the machine this suite usually runs on there is no /.dockerenv, and on CI inside a
 * container there is. Pinning DWP_CONTAINER is what makes the *consequences* of each
 * answer testable in both places, which is the half that has behaviour in it.
 */
function withEnv<T>(vars: Record<string, string | undefined>, body: () => T): T {
  const previous = new Map<string, string | undefined>()
  for (const [key, value] of Object.entries(vars)) {
    previous.set(key, process.env[key])
    if (value === undefined) delete process.env[key]
    else process.env[key] = value
  }
  try {
    return body()
  } finally {
    for (const [key, value] of previous) {
      if (value === undefined) delete process.env[key]
      else process.env[key] = value
    }
  }
}

test('DWP_CONTAINER settles the question in both directions', () => {
  withEnv({ DWP_CONTAINER: '1' }, () => assert.equal(isContainer(), true))
  withEnv({ DWP_CONTAINER: 'true' }, () => assert.equal(isContainer(), true))
  // The off spellings matter: an operator who writes DWP_CONTAINER=0 to turn it off must
  // not get the opposite, which a plain truthiness check on the string would give them.
  for (const off of ['0', 'false', 'no', 'off', 'FALSE']) {
    withEnv({ DWP_CONTAINER: off }, () => assert.equal(isContainer(), false, off))
  }
})

test('the window server binds where its audience can reach it', () => {
  withEnv({ DWP_CONTAINER: '0', DWP_GUI_HOST: undefined }, () => {
    assert.equal(guiBindHost(), '127.0.0.1')
  })
  // Loopback inside a container is the container's own, which no published port reaches.
  withEnv({ DWP_CONTAINER: '1', DWP_GUI_HOST: undefined }, () => {
    assert.equal(guiBindHost(), '0.0.0.0')
  })
  withEnv({ DWP_CONTAINER: '1', DWP_GUI_HOST: '127.0.0.1' }, () => {
    assert.equal(guiBindHost(), '127.0.0.1')
  })
})

test('a container pins its port; a laptop is free to walk the range', () => {
  withEnv({ DWP_CONTAINER: '1', DWP_GUI_PORT: undefined }, () => assert.equal(guiPort(), 43117))
  withEnv({ DWP_CONTAINER: '0', DWP_GUI_PORT: undefined }, () => assert.equal(guiPort(), null))
  withEnv({ DWP_CONTAINER: '0', DWP_GUI_PORT: '9000' }, () => assert.equal(guiPort(), 9000))
  // Nonsense is not a port. Falling back to the range beats binding NaN.
  withEnv({ DWP_CONTAINER: '0', DWP_GUI_PORT: 'banana' }, () => assert.equal(guiPort(), null))
  withEnv({ DWP_CONTAINER: '0', DWP_GUI_PORT: '70000' }, () => assert.equal(guiPort(), null))
})

test('no window is opened where there is no screen', () => {
  withEnv({ DWP_CONTAINER: '1', DWP_NO_WINDOW: undefined }, () => assert.equal(shouldOpenWindow(), false))
  withEnv({ DWP_CONTAINER: '0', DWP_NO_WINDOW: undefined }, () => assert.equal(shouldOpenWindow(), true))
  withEnv({ DWP_CONTAINER: '0', DWP_NO_WINDOW: '1' }, () => assert.equal(shouldOpenWindow(), false))
})

test('the Host check accepts a published port and still refuses a rebound name', () => {
  withEnv({ DWP_CONTAINER: '1', DWP_GUI_ALLOWED_HOSTS: undefined }, () => {
    // The whole reason the exact-port comparison had to go: -p 43200:43117 arrives here
    // as a Host of localhost:43200 while the server is listening on 43117.
    assert.equal(hostAllowed('localhost:43200'), true)
    assert.equal(hostAllowed('127.0.0.1:43117'), true)
    assert.equal(hostAllowed('[::1]:43117'), true)
    assert.equal(hostAllowed('localhost'), true)

    // DNS rebinding: a name the attacker owns, pointed at an address that reaches us.
    assert.equal(hostAllowed('evil.example.com:43117'), false)
    assert.equal(hostAllowed('127.0.0.1.nip.io:43117'), false)
    assert.equal(hostAllowed(''), false)
    assert.equal(hostAllowed(undefined), false)
  })
})

test('a container answers to its own name, a laptop does not', () => {
  const own = hostname()
  withEnv({ DWP_CONTAINER: '1', DWP_GUI_ALLOWED_HOSTS: undefined }, () => {
    assert.equal(hostAllowed(own + ':43117'), true)
    assert.equal(hostAllowed(own.toUpperCase() + ':43117'), true)
  })
  withEnv({ DWP_CONTAINER: '0', DWP_GUI_ALLOWED_HOSTS: undefined }, () => {
    assert.equal(hostAllowed(own + ':43117'), false)
  })
})

test('extra hosts are opt-in, by name, and never implied', () => {
  withEnv({ DWP_CONTAINER: '1', DWP_GUI_ALLOWED_HOSTS: 'dwp-agent.tailnet.ts.net, box.lan' }, () => {
    assert.equal(hostAllowed('dwp-agent.tailnet.ts.net'), true)
    assert.equal(hostAllowed('BOX.LAN:8080'), true)
    assert.equal(hostAllowed('other.tailnet.ts.net'), false)
  })
  withEnv({ DWP_CONTAINER: '1', DWP_GUI_ALLOWED_HOSTS: '*' }, () => {
    assert.equal(hostAllowed('anything.at.all'), true)
  })
})

test('the window is told who keeps this agent alive', () => {
  withEnv({ DWP_CONTAINER: '1' }, () => assert.match(supervisorNote(), /container runtime restarts it/))
  withEnv({ DWP_CONTAINER: '0' }, () => assert.match(supervisorNote(), /restart, with no window open/))
})

test('a container with no limits keeps the host figures', () => {
  // The cgroup files do not exist on macOS, so this also covers "cannot read them".
  withEnv({ DWP_CONTAINER: '0' }, () => {
    const limits = containerLimits()
    assert.equal(limits.cpus, null)
    assert.equal(limits.memoryBytes, null)
    assert.equal(containerFreeBytes(), null)
  })
})

test('the image is reported only when the operator declared one, and only in a container', () => {
  withEnv({ DWP_CONTAINER: '1', DWP_IMAGE: 'ghcr.io/example/dwp-agent:1.2.3' }, () => {
    assert.equal(imageReference(), 'ghcr.io/example/dwp-agent:1.2.3')
  })
  withEnv({ DWP_CONTAINER: '1', DWP_IMAGE: '   ' }, () => assert.equal(imageReference(), null))
  withEnv({ DWP_CONTAINER: '1', DWP_IMAGE: undefined }, () => assert.equal(imageReference(), null))
  // On a laptop the variable means nothing and must not be reported as a version.
  withEnv({ DWP_CONTAINER: '0', DWP_IMAGE: 'ghcr.io/example/dwp-agent:1.2.3' }, () => {
    assert.equal(imageReference(), null)
  })
})

test('a container reports its image as its version, never the server release it paired against', () => {
  withEnv({ DWP_CONTAINER: '1', DWP_IMAGE: 'ghcr.io/example/dwp-agent:1.2.3' }, () => {
    // The release argument is what the *server* was offering; a container never runs it.
    assert.equal(probe(['echo'], '0.4.0+f4e80599').agentVersion, 'ghcr.io/example/dwp-agent:1.2.3')
  })
  withEnv({ DWP_CONTAINER: '1', DWP_IMAGE: undefined }, () => {
    assert.match(probe(['echo'], '0.4.0+f4e80599').agentVersion, /set DWP_IMAGE/)
  })
  withEnv({ DWP_CONTAINER: '0', DWP_IMAGE: undefined }, () => {
    assert.equal(probe(['echo'], '0.4.0+f4e80599').agentVersion, '0.4.0+f4e80599')
  })
})

test('capability stays within what the protocol accepts', () => {
  withEnv({ DWP_CONTAINER: '0' }, () => {
    const cap = probe(['echo', 'walker_evolution'])
    assert.ok(Number.isInteger(cap.logicalCores) && cap.logicalCores > 0, 'cores must be a positive integer')
    assert.ok(Number.isInteger(cap.totalRamMb) && cap.totalRamMb > 0, 'total RAM must be a positive integer')
    assert.ok(Number.isInteger(cap.freeRamMb) && cap.freeRamMb >= 0, 'free RAM must be a non-negative integer')
  })
})
