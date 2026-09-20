/**
 * The 3D space the fleet graph lives in: a small force-directed layout and the
 * camera that projects it onto a flat canvas.
 *
 * Hand-rolled rather than pulled from a library, so this view adds no dependency and
 * no bundle weight to a dashboard that ships without one. At fleet scale — tens of
 * machines — the O(n²) repulsion pass is a rounding error.
 */
export interface Body {
  id: string;
  x: number;
  y: number;
  z: number;
  vx: number;
  vy: number;
  vz: number;
  /** The control plane holds the origin; everything arranges itself around it. */
  pinned: boolean;
}

export interface Camera {
  yaw: number;
  pitch: number;
  zoom: number;
}

export const DEFAULT_CAMERA: Camera = { yaw: 0.6, pitch: -0.3, zoom: 1.05 };

const FOCAL = 1150;
const SPRING = 0.03;
/** Nothing may move further than this in one step; a stiff spring plus a long frame
 *  can otherwise fling a node out of the scene and never fully bring it back. */
const MAX_SPEED = 26;
const REPEL = 620_000;
const GRAVITY = 0.0013;
/** Pulled harder toward the horizon than sideways, so the cloud settles into a disc.
 *  A stage is far wider than it is tall; a sphere wastes that width and then has to
 *  be zoomed out to fit its own height. */
const FLATTEN = 3.4;
const DAMPING = 0.86;
const CONTROL_REST = 275;
const PEER_REST = 205;
/** Subagents sit close to the coordinator they report to, so the cluster reads as one. */
const AGENT_REST = 190;
const JOB_REST = 250;
const REST: Record<string, number> = {
  control: CONTROL_REST,
  peer: PEER_REST,
  agent: AGENT_REST,
  job: JOB_REST,
};

/** A deterministic point on a sphere, so a machine keeps its place between snapshots. */
function seat(index: number, total: number, radius: number) {
  const offset = 2 / Math.max(1, total);
  const y = index * offset - 1 + offset / 2;
  const ring = Math.sqrt(Math.max(0, 1 - y * y));
  const angle = index * 2.399963229728653;
  return {
    x: Math.cos(angle) * ring * radius,
    y: y * radius * 0.72,
    z: Math.sin(angle) * ring * radius,
  };
}

/**
 * Bring the body set in line with the node set, keeping the bodies that survive.
 * A machine that stays connected keeps its position; one that joins is seated at a
 * free point on the sphere rather than at the origin, so it never shoots across the view.
 */
export function reconcileBodies(
  bodies: Map<string, Body>,
  nodes: { id: string; kind: string }[],
) {
  const live = new Set(nodes.map((node) => node.id));
  for (const id of [...bodies.keys()]) if (!live.has(id)) bodies.delete(id);
  nodes.forEach((node, index) => {
    if (bodies.has(node.id)) return;
    const pinned = node.kind === "control";
    // Seat by kind, so an arriving subagent starts near its cluster rather than
    // flying in across the whole scene.
    const radius =
      node.kind === "agent"
        ? AGENT_REST
        : node.kind === "job"
          ? JOB_REST
          : CONTROL_REST;
    const point = pinned
      ? { x: 0, y: 0, z: 0 }
      : seat(index, nodes.length, radius);
    bodies.set(node.id, { id: node.id, ...point, vx: 0, vy: 0, vz: 0, pinned });
  });
  return bodies;
}

/**
 * One integration step. Returns how much the layout moved, so a settled graph can
 * stop spending frames on physics.
 */
export function stepLayout(
  bodies: Map<string, Body>,
  links: { source: string; target: string; kind: string }[],
  dt: number,
) {
  const list = [...bodies.values()];
  const step = Math.min(dt, 0.05);
  const spread = 1 + Math.min(0.5, list.length / 40);
  for (let a = 0; a < list.length; a += 1) {
    for (let b = a + 1; b < list.length; b += 1) {
      const one = list[a];
      const two = list[b];
      let dx = two.x - one.x;
      let dy = two.y - one.y;
      let dz = two.z - one.z;
      let distance = Math.hypot(dx, dy, dz);
      if (distance < 1) {
        // Two bodies exactly on top of each other have no direction to separate in.
        dx = (a % 3) - 1 || 0.7;
        dy = (b % 3) - 1 || 0.4;
        dz = 0.6;
        distance = Math.hypot(dx, dy, dz);
      }
      const push =
        Math.min((REPEL * spread) / (distance * distance), 2800 * spread) /
        distance;
      one.vx -= dx * push * step;
      one.vy -= dy * push * step;
      one.vz -= dz * push * step;
      two.vx += dx * push * step;
      two.vy += dy * push * step;
      two.vz += dz * push * step;
    }
  }
  for (const link of links) {
    const one = bodies.get(link.source);
    const two = bodies.get(link.target);
    if (!one || !two) continue;
    const dx = two.x - one.x;
    const dy = two.y - one.y;
    const dz = two.z - one.z;
    const distance = Math.max(1, Math.hypot(dx, dy, dz));
    // A bigger graph needs a bigger room: rest lengths grow with the node count so
    // twenty nodes are not packed into the space five were comfortable in.
    const rest = (REST[link.kind] ?? PEER_REST) * spread;
    const pull = ((distance - rest) * SPRING) / distance;
    one.vx += dx * pull * step * 60;
    one.vy += dy * pull * step * 60;
    one.vz += dz * pull * step * 60;
    two.vx -= dx * pull * step * 60;
    two.vy -= dy * pull * step * 60;
    two.vz -= dz * pull * step * 60;
  }
  let motion = 0;
  const damping = Math.pow(DAMPING, step * 60);
  for (const body of list) {
    if (body.pinned) {
      body.x = body.y = body.z = body.vx = body.vy = body.vz = 0;
      continue;
    }
    body.vx = (body.vx - body.x * GRAVITY * step * 60) * damping;
    body.vy = (body.vy - body.y * GRAVITY * FLATTEN * step * 60) * damping;
    body.vz = (body.vz - body.z * GRAVITY * step * 60) * damping;
    const speed = Math.hypot(body.vx, body.vy, body.vz);
    if (speed > MAX_SPEED) {
      const brake = MAX_SPEED / speed;
      body.vx *= brake;
      body.vy *= brake;
      body.vz *= brake;
    }
    body.x += body.vx * step * 60;
    body.y += body.vy * step * 60;
    body.z += body.vz * step * 60;
    motion += Math.abs(body.vx) + Math.abs(body.vy) + Math.abs(body.vz);
  }
  return motion;
}

export interface Projected {
  x: number;
  y: number;
  /** Perspective factor: 1 at the origin plane, smaller further away. */
  scale: number;
  /** Camera-space depth. Larger is further from the viewer. */
  depth: number;
}

/** World point to canvas point, through the camera's yaw, pitch and zoom. */
export function project(
  point: { x: number; y: number; z: number },
  camera: Camera,
  width: number,
  height: number,
): Projected {
  const cosYaw = Math.cos(camera.yaw);
  const sinYaw = Math.sin(camera.yaw);
  const cosPitch = Math.cos(camera.pitch);
  const sinPitch = Math.sin(camera.pitch);
  const x = point.x * cosYaw + point.z * sinYaw;
  const spun = point.z * cosYaw - point.x * sinYaw;
  const y = point.y * cosPitch - spun * sinPitch;
  const depth = point.y * sinPitch + spun * cosPitch;
  const scale = (FOCAL / Math.max(FOCAL * 0.3, FOCAL + depth)) * camera.zoom;
  return {
    x: width / 2 + x * scale,
    y: height / 2 + y * scale,
    scale,
    depth,
  };
}

export const lerp3 = (
  one: { x: number; y: number; z: number },
  two: { x: number; y: number; z: number },
  t: number,
) => ({
  x: one.x + (two.x - one.x) * t,
  y: one.y + (two.y - one.y) * t,
  z: one.z + (two.z - one.z) * t,
});

export const clampZoom = (zoom: number) => Math.min(3, Math.max(0.3, zoom));
export const clampPitch = (pitch: number) =>
  Math.min(1.35, Math.max(-1.35, pitch));

/**
 * A drag measured in screen pixels, turned into the world-space move that produces
 * it. The node travels in the plane facing the camera, at its own perspective scale,
 * so it tracks the pointer however the scene is currently turned.
 */
export function screenDelta(
  dx: number,
  dy: number,
  camera: Camera,
  scale: number,
) {
  const cx = dx / Math.max(0.0001, scale);
  const cy = dy / Math.max(0.0001, scale);
  const cosYaw = Math.cos(camera.yaw);
  const sinYaw = Math.sin(camera.yaw);
  const spun = -cy * Math.sin(camera.pitch);
  return {
    x: cx * cosYaw - spun * sinYaw,
    y: cy * Math.cos(camera.pitch),
    z: cx * sinYaw + spun * cosYaw,
  };
}
