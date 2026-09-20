import {
  Component,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import type { Snapshot } from "../api/types";
import {
  buildFleetGraph,
  type GraphLink,
  type GraphNode,
} from "../lib/fleetGraph";
import { useAnalysisAgents } from "../hooks/useAnalysisAgents";
import { useReplay } from "../hooks/useReplay";
import { replaySnapshot } from "../lib/fleetReplay";
import {
  clampPitch,
  clampZoom,
  DEFAULT_CAMERA,
  lerp3,
  project,
  reconcileBodies,
  screenDelta,
  stepLayout,
  type Body,
  type Camera,
} from "../lib/fleetScene";
import { eventText, time } from "../lib/format";
import { statusLabels } from "../lib/jobs";
import { Icon } from "./Icon";
import "./FleetNetwork.css";

/** Line and node colour per state, as RGB so alpha can vary with depth. */
const TONE: Record<string, [number, number, number]> = {
  busy: [17, 138, 99],
  result: [92, 128, 146],
  idle: [143, 158, 168],
  paused: [173, 124, 17],
  offline: [166, 177, 185],
  online: [17, 138, 99],
  unhealthy: [190, 82, 52],
  control: [31, 43, 51],
  agent: [104, 88, 196],
  job: [39, 105, 148],
};

/** Line opacity per state. Tuned for dark ink on a light ground. */
const EDGE_ALPHA: Record<string, number> = {
  busy: 0.85,
  result: 0.7,
  idle: 0.34,
  paused: 0.5,
  offline: 0.22,
};

const INK: [number, number, number] = [32, 43, 51];
const MUTED: [number, number, number] = [102, 115, 124];

const rgba = (tone: [number, number, number], alpha: number) =>
  `rgba(${tone[0]}, ${tone[1]}, ${tone[2]}, ${Math.max(0, Math.min(1, alpha)).toFixed(3)})`;

const clip = (text: string, max = 38) =>
  text.length > max ? `${text.slice(0, max - 1)}…` : text;

type Rect = { x: number; y: number; w: number; h: number };
const overlaps = (one: Rect, two: Rect) =>
  one.x < two.x + two.w &&
  one.x + one.w > two.x &&
  one.y < two.y + two.h &&
  one.y + one.h > two.y;

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    return;
  }
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/**
 * Node silhouettes. The agent layer is drawn as squares and diamonds so it reads as
 * a different kind of thing from the round machines, before any colour is applied.
 */
function shape(
  ctx: CanvasRenderingContext2D,
  kind: string,
  x: number,
  y: number,
  r: number,
) {
  if (kind === "job") {
    roundRect(ctx, x - r, y - r, r * 2, r * 2, r * 0.34);
    return;
  }
  if (kind === "agent") {
    ctx.beginPath();
    ctx.moveTo(x, y - r);
    ctx.lineTo(x + r, y);
    ctx.lineTo(x, y + r);
    ctx.lineTo(x - r, y);
    ctx.closePath();
    return;
  }
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
}

const RADIUS: Record<string, number> = {
  control: 15,
  worker: 8.5,
  agent: 7,
  job: 8.5,
};
const nodeTone = (node: GraphNode): [number, number, number] =>
  node.state === "offline"
    ? TONE.offline
    : node.kind === "control" || node.kind === "agent" || node.kind === "job"
      ? TONE[node.kind]
      : TONE[node.state] || TONE.offline;
const nodeRadius = (node: GraphNode) =>
  RADIUS[node.kind] +
  (node.kind === "worker" ? Math.min(node.load, 4) * 1.2 : 0);

const STATE_TEXT: Record<string, string> = {
  online: "connected",
  paused: "paused",
  unhealthy: "no heartbeat",
  offline: "offline",
};

function Scene({ snapshot }: { snapshot: Snapshot }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const popupRef = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [allLabels, setAllLabels] = useState(false);
  const [spin, setSpin] = useState(true);
  const [logOpen, setLogOpen] = useState(false);
  const [full, setFull] = useState(false);

  const [replayOpen, setReplayOpen] = useState(false);
  const film = useReplay(snapshot);
  // In replay the graph is built from a reconstructed snapshot, so every other part
  // of this view — nodes, routes, popups — works on the past run unchanged.
  const live = useAnalysisAgents(snapshot, !film.replay);
  // A simulation job's stages are not in the snapshot, so they are folded in here
  // alongside its root row. The root keeps the job its name; the stages are what put
  // it on a machine.
  const source =
    film.replay && film.step
      ? replaySnapshot(
          film.replay.replay,
          film.step,
          film.replay.group,
          snapshot.workers,
        )
      : live.stages.length
        ? { ...snapshot, tasks: [...snapshot.tasks, ...live.stages] }
        : snapshot;
  const agents = film.replay ? film.replay.agents : live.agents;
  const moment = film.step?.at;
  const graph = useMemo(
    () => buildFleetGraph(source, moment ?? Date.now(), agents),
    [source, moment, agents],
  );

  // The animation loop reads the live values through refs. It is started once and
  // never restarted by a snapshot, so a fleet update never interrupts the motion.
  const graphRef = useRef(graph);
  graphRef.current = graph;
  const bodiesRef = useRef(new Map<string, Body>());
  const cameraRef = useRef<Camera>({ ...DEFAULT_CAMERA });
  const pointerRef = useRef<{
    x: number;
    y: number;
    dragging?: boolean;
  } | null>(null);
  const focusRef = useRef<{ hovered: string | null; selected: string | null }>({
    hovered: null,
    selected: null,
  });
  focusRef.current = { hovered, selected };
  const optionsRef = useRef({ allLabels, spin });
  optionsRef.current = { allLabels, spin };
  /** Where each node landed on screen last frame: what pointer hits are tested against. */
  const hitsRef = useRef<{ id: string; x: number; y: number; scale: number }[]>(
    [],
  );
  const heldRef = useRef<{ id: string; scale: number } | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    // No 2D context means no picture, which is survivable: the log popup carries
    // the same information as text.
    const context = canvas?.getContext?.("2d");
    if (!canvas || !stage || !context) return;
    const ctx: CanvasRenderingContext2D = context;

    const calm =
      typeof window.matchMedia === "function" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let width = 0;
    let height = 0;
    const resize = () => {
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      width = Math.max(320, stage.clientWidth);
      height = Math.max(260, stage.clientHeight);
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    };
    resize();
    const observer =
      typeof ResizeObserver === "function"
        ? new ResizeObserver(resize)
        : undefined;
    observer?.observe(stage);
    if (!observer) window.addEventListener("resize", resize);

    let raf = 0;
    let previous = 0;
    let clock = 0;
    let lastHover: string | null = null;
    // Keeps the whole graph in frame as it grows, without touching the zoom the
    // viewer set: it multiplies that, and eases so the picture never jumps.
    let fit = 1;

    function draw(elapsed: number) {
      raf = requestAnimationFrame(draw);
      if (document.hidden) return;
      const dt = previous ? Math.min((elapsed - previous) / 1000, 0.05) : 0.016;
      previous = elapsed;
      clock += dt;
      const { nodes, links } = graphRef.current;
      const camera = cameraRef.current;
      const { allLabels: showAll, spin: spinning } = optionsRef.current;
      const bodies = reconcileBodies(bodiesRef.current, nodes);
      stepLayout(bodies, links, dt);
      const held = heldRef.current;
      if (held) {
        // A node under the pointer is placed, not simulated: zero its velocity so the
        // springs do not tug it out from under the cursor.
        const body = bodies.get(held.id);
        if (body) body.vx = body.vy = body.vz = 0;
      }
      if (spinning && !calm && !pointerRef.current?.dragging && !held)
        camera.yaw += dt * 0.085;

      const lens = { ...camera, zoom: camera.zoom * fit };
      const screen = new Map<
        string,
        { x: number; y: number; scale: number; depth: number; node: GraphNode }
      >();
      const spans: number[] = [];
      for (const node of nodes) {
        const body = bodies.get(node.id);
        if (!body) continue;
        const point = project(body, lens, width, height);
        screen.set(node.id, { ...point, node });
        spans.push(
          Math.max(
            Math.abs(point.x - width / 2) / (width / 2),
            Math.abs(point.y - height / 2) / (height / 2),
          ),
        );
      }
      // The node most of the graph sits inside, not the single furthest one: one
      // stray node must not shrink everything else to nothing.
      spans.sort((one, two) => one - two);
      const reach = Math.max(0.2, spans[Math.floor(spans.length * 0.88)] ?? 1);
      // Shrink only far enough to keep the outermost node in frame, then drift back
      // to full size. Growing on demand would fight the viewer's own zoom.
      const wanted =
        reach > 0.96
          ? Math.max(0.5, fit * (0.92 / reach))
          : Math.min(1, fit * 1.02);
      fit += (wanted - fit) * Math.min(1, dt * 2);
      hitsRef.current = [...screen.entries()].map(([id, point]) => ({
        id,
        x: point.x,
        y: point.y,
        scale: point.scale,
      }));

      const pointer = pointerRef.current;
      let nearest: string | null = null;
      if (pointer && !pointer.dragging) {
        let best = 22;
        for (const [id, point] of screen) {
          const distance = Math.hypot(point.x - pointer.x, point.y - pointer.y);
          if (distance < best) {
            best = distance;
            nearest = id;
          }
        }
      }
      if (nearest !== lastHover) {
        lastHover = nearest;
        setHovered(nearest);
      }
      const focus = focusRef.current.selected || focusRef.current.hovered;

      // The popup is anchored in the loop rather than in React state: it tracks a
      // node that moves every frame, and a re-render per frame would be absurd.
      const popup = popupRef.current;
      if (popup) {
        const anchor = focus ? screen.get(focus) : undefined;
        if (anchor) {
          const x = Math.max(150, Math.min(width - 150, anchor.x));
          popup.style.transform = `translate3d(${Math.round(x)}px, ${Math.round(anchor.y - 22)}px, 0)`;
          popup.style.visibility = "visible";
        } else popup.style.visibility = "hidden";
      }

      // A light room: white in the middle, cooling off toward the edges.
      ctx.clearRect(0, 0, width, height);
      const backdrop = ctx.createRadialGradient(
        width / 2,
        height / 2,
        0,
        width / 2,
        height / 2,
        Math.max(width, height) * 0.8,
      );
      backdrop.addColorStop(0, "#ffffff");
      backdrop.addColorStop(0.6, "#f4f7f9");
      backdrop.addColorStop(1, "#e7edf1");
      ctx.fillStyle = backdrop;
      ctx.fillRect(0, 0, width, height);

      const drawn: Rect[] = [];
      const ordered = [...links]
        .map((link) => {
          const a = screen.get(link.source);
          const b = screen.get(link.target);
          return a && b ? { link, a, b, depth: (a.depth + b.depth) / 2 } : null;
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
        .sort((one, two) => two.depth - one.depth);

      for (const { link, a, b } of ordered) {
        const related =
          !focus || link.source === focus || link.target === focus;
        const lit = related ? 1 : 0.18;
        const tone = TONE[link.state] || TONE.idle;
        const scale = (a.scale + b.scale) / 2;
        ctx.save();
        ctx.lineCap = "round";
        if (link.state === "offline") ctx.setLineDash([3, 6]);
        if (link.kind === "peer") ctx.setLineDash([7, 5]);
        if (link.kind === "agent") ctx.setLineDash([2, 4]);
        // A job not yet executing draws the route it is about to take.
        if (link.kind === "job" && link.state === "idle")
          ctx.setLineDash([5, 5]);
        ctx.lineWidth =
          Math.max(0.7, (link.state === "busy" ? 1.8 : 1.2) * scale) *
          (focus && related ? 1.5 : 1);
        ctx.strokeStyle = rgba(tone, (EDGE_ALPHA[link.state] ?? 0.3) * lit);
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
        ctx.restore();

        // Traffic. Packets ride the 3D segment, so they keep the line's perspective.
        if (link.intensity > 0.05 && !calm) {
          const source = bodies.get(link.source);
          const target = bodies.get(link.target);
          if (source && target) {
            const count = 1 + Math.round(link.intensity * 3);
            const speed = 0.16 + link.intensity * 0.5;
            ctx.save();
            for (let index = 0; index < count; index += 1) {
              const phase = (clock * speed + index / count) % 1;
              const forward =
                link.flow === 0 ? index % 2 === 0 : link.flow === 1;
              const t = forward ? phase : 1 - phase;
              const point = project(
                lerp3(source, target, t),
                lens,
                width,
                height,
              );
              const radius = Math.max(1.4, 3 * point.scale);
              ctx.fillStyle = "#fff";
              ctx.beginPath();
              ctx.arc(point.x, point.y, radius + 1.4, 0, Math.PI * 2);
              ctx.fill();
              ctx.fillStyle = rgba(tone, lit);
              ctx.beginPath();
              ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
              ctx.fill();
            }
            ctx.restore();
          }
        }
      }

      const nodeOrder = [...screen.values()].sort(
        (one, two) => two.depth - one.depth,
      );
      for (const point of nodeOrder) {
        const { node } = point;
        const active = !focus || node.id === focus;
        const tone = nodeTone(node);
        const radius = Math.max(3.5, nodeRadius(node) * point.scale);
        ctx.save();
        ctx.shadowColor = "rgba(29, 45, 56, 0.22)";
        ctx.shadowBlur = 12 * point.scale;
        ctx.shadowOffsetY = 2 * point.scale;
        // A white collar keeps the node readable where lines pass behind it.
        ctx.fillStyle = "#fff";
        shape(ctx, node.kind, point.x, point.y, radius + 2.6 * point.scale);
        ctx.fill();
        ctx.restore();
        ctx.fillStyle = rgba(
          tone,
          node.state === "offline" ? 0.4 : active ? 1 : 0.35,
        );
        shape(ctx, node.kind, point.x, point.y, radius);
        ctx.fill();
        if (
          (node.kind === "agent" || node.kind === "job") &&
          node.state === "paused"
        ) {
          // Queued, not yet working: an outline rather than a solid.
          ctx.fillStyle = "#fff";
          shape(ctx, node.kind, point.x, point.y, radius - 2 * point.scale);
          ctx.fill();
        }
        if (node.load > 0 && node.state === "online") {
          // A breathing ring while this node has work in hand.
          const pulse = calm ? 0.5 : (Math.sin(clock * 2.4) + 1) / 2;
          ctx.lineWidth = 1.4 * point.scale;
          ctx.strokeStyle = rgba(
            tone,
            0.45 * (0.35 + pulse * 0.65) * (active ? 1 : 0.4),
          );
          shape(
            ctx,
            node.kind,
            point.x,
            point.y,
            radius + (7 + pulse * 5) * point.scale,
          );
          ctx.stroke();
        }
        if (node.id === focusRef.current.selected) {
          ctx.lineWidth = 1.6;
          ctx.strokeStyle = rgba(INK, 0.55);
          ctx.setLineDash([3, 3]);
          shape(ctx, node.kind, point.x, point.y, radius + 11 * point.scale);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      // Labels last, so no line is ever drawn over a word, and node names are
      // placed first so a busy line can never push a machine's name off screen.
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const point of nodeOrder) {
        const { node } = point;
        if (focus && node.id !== focus && node.kind !== "control") continue;
        // An idle agent is a diamond on a line to its machine; its name only earns
        // the room once it is working, or once the viewer asks for every label.

        const size = Math.max(10.5, Math.min(13, 12 * point.scale));
        ctx.font = `${node.kind === "control" ? 600 : 500} ${size}px "DM Sans", system-ui, sans-serif`;
        const offset = nodeRadius(node) * point.scale + 20;
        const text = clip(node.label, 24);
        const w = ctx.measureText(text).width + 4;
        const box = {
          x: point.x - w / 2,
          y: point.y + offset - 10,
          w: w + 10,
          h: 21,
        };
        if (drawn.some((other) => overlaps(box, other))) continue;
        drawn.push(box);
        const alpha = Math.max(0.5, Math.min(1, point.scale));
        ctx.fillStyle = rgba(INK, alpha);
        ctx.fillText(text, point.x, point.y + offset);
      }
      for (const { link, a, b } of ordered) {
        const related = link.source === focus || link.target === focus;
        // What a line is carrying is worth the room when nothing else says it. A
        // job already executing is announced on the machine's own line, so only the
        // routes a job has not taken yet — the part that says what happens next —
        // are labelled by default. Focus or "All labels" shows the rest.
        const worth =
          showAll ||
          related ||
          (link.kind === "job" && link.state === "idle") ||
          (link.kind === "control" &&
            (link.state === "busy" ||
              link.state === "result" ||
              link.state === "paused")) ||
          (link.kind === "peer" && link.state === "busy");
        if (!worth || (focus && !related)) continue;
        const scale = (a.scale + b.scale) / 2;
        const x = (a.x + b.x) / 2;
        const y = (a.y + b.y) / 2;
        const size = Math.max(10, Math.min(12.5, 11.5 * scale));
        ctx.font = `${size}px "DM Sans", system-ui, sans-serif`;
        const text = clip(link.label);
        const w = ctx.measureText(text).width + 16;
        const h = size + 11;
        const box = { x: x - w / 2, y: y - h / 2, w, h };
        if (drawn.some((other) => overlaps(box, other))) continue;
        drawn.push(box);
        const alpha = Math.max(0.5, Math.min(1, scale));
        const tone = TONE[link.state] || TONE.idle;
        ctx.save();
        ctx.shadowColor = "rgba(29, 45, 56, 0.14)";
        ctx.shadowBlur = 8;
        ctx.shadowOffsetY = 1;
        roundRect(ctx, box.x, box.y, w, h, 7);
        ctx.fillStyle = `rgba(255, 255, 255, ${0.96 * alpha})`;
        ctx.fill();
        ctx.restore();
        ctx.lineWidth = 1;
        ctx.strokeStyle = rgba(tone, 0.45 * alpha);
        ctx.stroke();
        ctx.fillStyle = rgba(
          link.state === "idle" || link.state === "offline" ? MUTED : tone,
          alpha,
        );
        ctx.fillText(text, x, y + 0.5);
      }
    }

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      observer?.disconnect();
      if (!observer) window.removeEventListener("resize", resize);
    };
  }, []);

  // Wheel has to be non-passive to keep the page from scrolling under the zoom.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const zoom = (event: WheelEvent) => {
      event.preventDefault();
      cameraRef.current.zoom = clampZoom(
        cameraRef.current.zoom * (event.deltaY > 0 ? 0.92 : 1.08),
      );
    };
    stage.addEventListener("wheel", zoom, { passive: false });
    return () => stage.removeEventListener("wheel", zoom);
  }, []);

  useEffect(() => {
    const sync = () => setFull(document.fullscreenElement === stageRef.current);
    document.addEventListener("fullscreenchange", sync);
    return () => document.removeEventListener("fullscreenchange", sync);
  }, []);

  useEffect(() => {
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setLogOpen(false);
      setSelected(null);
    };
    window.addEventListener("keydown", escape);
    return () => window.removeEventListener("keydown", escape);
  }, []);

  const dragRef = useRef<{ x: number; y: number; moved: number } | null>(null);
  const nodes = graph.nodes;
  const focused = nodes.find((node) => node.id === (selected || hovered));
  const focusLinks = focused
    ? graph.links.filter(
        (link) => link.source === focused.id || link.target === focused.id,
      )
    : [];
  const partner = (link: GraphLink) =>
    nodes.find(
      (node) =>
        node.id === (link.source === focused?.id ? link.target : link.source),
    );

  return (
    <div className="fg-stage" ref={stageRef}>
      <canvas
        ref={canvasRef}
        className="fg-canvas"
        role="img"
        aria-label={`Fleet topology: ${nodes.length - 1} machines connected to the control plane`}
      />
      <div
        className="fg-surface"
        style={{ cursor: hovered ? "grab" : "default" }}
        onPointerDown={(event) => {
          (event.target as HTMLElement).setPointerCapture?.(event.pointerId);
          const bounds = event.currentTarget.getBoundingClientRect();
          const x = event.clientX - bounds.left;
          const y = event.clientY - bounds.top;
          // A press that lands on a node moves that node; anywhere else orbits.
          let closest: { id: string; scale: number } | null = null;
          let best = 24;
          for (const hit of hitsRef.current) {
            const distance = Math.hypot(hit.x - x, hit.y - y);
            if (distance < best) {
              best = distance;
              closest = { id: hit.id, scale: hit.scale };
            }
          }
          heldRef.current = closest;
          dragRef.current = { x: event.clientX, y: event.clientY, moved: 0 };
        }}
        onPointerMove={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          const drag = dragRef.current;
          if (drag) {
            const dx = event.clientX - drag.x;
            const dy = event.clientY - drag.y;
            drag.moved += Math.abs(dx) + Math.abs(dy);
            drag.x = event.clientX;
            drag.y = event.clientY;
            const held = heldRef.current;
            const body = held ? bodiesRef.current.get(held.id) : undefined;
            if (held && body && !body.pinned) {
              const move = screenDelta(dx, dy, cameraRef.current, held.scale);
              body.x += move.x;
              body.y += move.y;
              body.z += move.z;
              body.vx = body.vy = body.vz = 0;
            } else if (!held) {
              cameraRef.current.yaw += dx * 0.006;
              cameraRef.current.pitch = clampPitch(
                cameraRef.current.pitch + dy * 0.006,
              );
            }
            pointerRef.current = {
              x: event.clientX - bounds.left,
              y: event.clientY - bounds.top,
              dragging: true,
            };
            return;
          }
          pointerRef.current = {
            x: event.clientX - bounds.left,
            y: event.clientY - bounds.top,
          };
        }}
        onPointerUp={() => {
          const drag = dragRef.current;
          const held = heldRef.current;
          dragRef.current = null;
          heldRef.current = null;
          if (pointerRef.current) pointerRef.current.dragging = false;
          if (drag && drag.moved < 6)
            setSelected((current) =>
              held && held.id !== current ? held.id : null,
            );
        }}
        onPointerLeave={() => {
          dragRef.current = null;
          heldRef.current = null;
          pointerRef.current = null;
        }}
        onDoubleClick={() => {
          cameraRef.current = { ...DEFAULT_CAMERA };
        }}
      />
      <div className="fg-toolbar">
        <button
          type="button"
          className={spin ? "fg-chip fg-on" : "fg-chip"}
          aria-pressed={spin}
          onClick={() => setSpin((value) => !value)}
        >
          Auto-rotate
        </button>
        <button
          type="button"
          className={allLabels ? "fg-chip fg-on" : "fg-chip"}
          aria-pressed={allLabels}
          onClick={() => setAllLabels((value) => !value)}
        >
          All labels
        </button>
        <button
          type="button"
          className={replayOpen || film.replay ? "fg-chip fg-on" : "fg-chip"}
          aria-expanded={replayOpen}
          onClick={() => setReplayOpen((value) => !value)}
        >
          Replay a run
        </button>
        <button
          type="button"
          className={logOpen ? "fg-chip fg-on" : "fg-chip"}
          aria-expanded={logOpen}
          onClick={() => setLogOpen((value) => !value)}
        >
          Activity log
        </button>
        <button
          type="button"
          className="fg-chip"
          onClick={() => {
            cameraRef.current = { ...DEFAULT_CAMERA };
            setSelected(null);
          }}
        >
          Reset view
        </button>
        <button
          type="button"
          className="fg-chip"
          onClick={() => {
            const stage = stageRef.current;
            if (document.fullscreenElement) void document.exitFullscreen?.();
            else void stage?.requestFullscreen?.();
          }}
        >
          {full ? "Exit full screen" : "Full screen"}
        </button>
      </div>
      <div className="fg-legend" aria-hidden="true">
        <span>
          <i className="fg-dot fg-busy" />
          transferring
        </span>
        <span>
          <i className="fg-dot fg-idle" />
          heartbeat
        </span>
        <span>
          <i className="fg-dot fg-paused" />
          paused
        </span>
        <span>
          <i className="fg-dot fg-offline" />
          offline
        </span>
        <span>
          <i className="fg-dot fg-job" />
          job
        </span>
        <span>
          <i className="fg-dot fg-agent" />
          analysis agent
        </span>
      </div>

      {/* Anchored to its node by the animation loop, which moves it every frame. */}
      <div
        ref={popupRef}
        className={selected ? "fg-popup fg-pinned" : "fg-popup"}
        style={{ visibility: "hidden" }}
        role={selected ? "dialog" : undefined}
        aria-label={
          selected && focused ? `${focused.label} details` : undefined
        }
      >
        {focused && (
          <>
            <header>
              <div>
                <strong>{focused.label}</strong>
                <span className={`fg-state fg-state-${focused.state}`}>
                  {STATE_TEXT[focused.state]}
                  {focused.kind === "worker" && focused.load > 0
                    ? ` · ${focused.load} active task${focused.load === 1 ? "" : "s"}`
                    : ""}
                </span>
              </div>
              {selected && (
                <button
                  type="button"
                  className="fg-close"
                  aria-label="Close details"
                  onClick={() => setSelected(null)}
                >
                  <Icon name="close" size={14} />
                </button>
              )}
            </header>
            <p className="fg-detail">{focused.detail}</p>
            <ul className="fg-links">
              {focusLinks.map((link) => (
                <li key={link.id} className={`fg-link fg-link-${link.state}`}>
                  <strong>{partner(link)?.label || "Unknown"}</strong>
                  <span>{link.label}</span>
                </li>
              ))}
              {focusLinks.length === 0 && (
                <li className="fg-link fg-link-offline">
                  <span>No connections</span>
                </li>
              )}
            </ul>
            {!selected && <p className="fg-tip">Click to pin this panel</p>}
          </>
        )}
      </div>

      <div
        className="fg-log"
        hidden={!logOpen}
        role="dialog"
        aria-label="Activity log"
      >
        <header>
          <strong>Activity log</strong>
          <button
            type="button"
            className="fg-close"
            aria-label="Close activity log"
            onClick={() => setLogOpen(false)}
          >
            <Icon name="close" size={14} />
          </button>
        </header>
        <h4 id="fg-links-heading">Connections</h4>
        <ul className="fg-log-list" aria-labelledby="fg-links-heading">
          {graph.links.map((link) => {
            const from = nodes.find((node) => node.id === link.source);
            const to = nodes.find((node) => node.id === link.target);
            return (
              <li key={link.id} className={`fg-link-${link.state}`}>
                <strong>
                  {from?.label} → {to?.label}
                </strong>
                <span>{link.label}</span>
              </li>
            );
          })}
          {graph.links.length === 0 && <li>No connections to show yet.</li>}
        </ul>
        <h4 id="fg-agents-heading">Analysis agents</h4>
        <ul className="fg-log-list" aria-labelledby="fg-agents-heading">
          {agents.flatMap((group) =>
            group.children.map((child) => (
              <li key={`${group.jobId}:${child.id}`}>
                <strong>
                  {group.title} · {child.role.replaceAll("_", " ")}
                </strong>
                <span>{child.status.replaceAll("_", " ")}</span>
              </li>
            )),
          )}
          {agents.length === 0 && <li>No analysis agents are running.</li>}
        </ul>
        <h4 id="fg-events-heading">Recent events</h4>
        <ul className="fg-log-list" aria-labelledby="fg-events-heading">
          {[...snapshot.events]
            .slice(-40)
            .reverse()
            .map((event) => (
              <li key={event.id}>
                <strong>{eventText(event, snapshot.tasks)}</strong>
                <span className="mono">{time(event.at)}</span>
              </li>
            ))}
          {snapshot.events.length === 0 && <li>Nothing has happened yet.</li>}
        </ul>
      </div>

      <div
        className="fg-log fg-runs"
        hidden={!replayOpen}
        role="dialog"
        aria-label="Past runs"
      >
        <header>
          <strong>Past runs</strong>
          <button
            type="button"
            className="fg-close"
            aria-label="Close past runs"
            onClick={() => setReplayOpen(false)}
          >
            <Icon name="close" size={14} />
          </button>
        </header>
        <p className="fg-detail">
          Play a finished run back through the graph, exactly as it happened.
        </p>
        {film.error && (
          <p className="fg-detail" role="alert">
            {film.error}
          </p>
        )}
        <ul className="fg-runs-list">
          {film.past.map((group) => (
            <li key={group.id}>
              <button
                type="button"
                disabled={Boolean(film.loading)}
                aria-current={
                  film.replay?.group.id === group.id ? "true" : undefined
                }
                onClick={() => {
                  void film.open(group);
                  setReplayOpen(false);
                }}
              >
                <strong>{group.title}</strong>
                <span>
                  {film.loading === group.id
                    ? "Loading…"
                    : `${statusLabels[group.state] || group.state} · ${time(group.createdAt)}`}
                </span>
              </button>
            </li>
          ))}
          {film.past.length === 0 && (
            <li className="fg-detail">No finished runs yet.</li>
          )}
        </ul>
      </div>

      {film.replay && film.step && (
        <div className="fg-transport" role="group" aria-label="Replay controls">
          <button
            type="button"
            className="fg-chip"
            onClick={film.toggle}
            aria-label={film.playing ? "Pause replay" : "Play replay"}
          >
            {film.playing ? "Pause" : "Play"}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(0, film.total - 1)}
            value={film.index}
            aria-label="Replay position"
            onChange={(event) => film.seek(Number(event.target.value))}
          />
          <div className="fg-transport-text">
            <strong>{film.step.caption}</strong>
            <span>
              {film.replay.replay.title} · {time(new Date(film.step.at))} ·{" "}
              {film.index + 1}/{film.total}
            </span>
          </div>
          <button type="button" className="fg-chip" onClick={film.close}>
            Back to live
          </button>
        </div>
      )}

      {nodes.length < 2 && !film.replay && (
        <p className="fg-empty">
          No machines have joined yet. Pair one on the Workers tab and it
          appears here.
        </p>
      )}
      <p className="fg-hint" aria-hidden="true">
        Drag to orbit · scroll to zoom · click a node
      </p>
    </div>
  );
}

/**
 * The view is an extra, so a fault in it must never take the dashboard down with it.
 * Anything thrown while rendering the scene is contained here.
 */
class Boundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed)
      return (
        <div className="fg-fallback" role="status">
          <Icon name="warning" size={16} />
          The topology view stopped. The rest of the workspace is unaffected —
          reload to try it again.
        </div>
      );
    return this.props.children;
  }
}

export function FleetNetwork({
  snapshot,
  active,
}: {
  snapshot: Snapshot;
  active: boolean;
}) {
  // Mounted only while its tab is open: no canvas, no animation frame, and no
  // listeners exist anywhere else in the app.
  if (!active) return null;
  return (
    <Boundary>
      <Scene snapshot={snapshot} />
    </Boundary>
  );
}
