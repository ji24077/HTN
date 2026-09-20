import { Icon } from "./Icon";
import type { Destination } from "../hooks/useWorkspaceNavigation";
import "./Guide.css";

type Stage = {
  name: string;
  line: string;
  why: string;
};

// The order is the point: a bad plan fails at the probe in seconds, and the
// validator runs before anyone sees a result.
const stages: Stage[] = [
  {
    name: "Plan",
    line: "The coordinator reads your files and picks runtime, dependencies, and worker.",
    why: "You do not choose the machine. If the source does not say what it needs, the agent asks you rather than guessing.",
  },
  {
    name: "Probe",
    line: "A short trial run on the chosen machine.",
    why: "A plan that does not hold fails here in seconds instead of after the full job has been metered.",
  },
  {
    name: "Run",
    line: "A worker claims the oldest compatible task under a database lock.",
    why: "The lock is why two machines can never take the same task, even when several are free at once.",
  },
  {
    name: "Validate",
    line: "The job's validator checks the output.",
    why: "A run that finishes but fails validation is a failed job, not a completed one. You never see an unchecked result.",
  },
  {
    name: "Meter",
    line: "The attempt is priced in CAD from cores, RAM, and elapsed time.",
    why: "Crossing the run's cap cancels the job. The cap is soft: attempts already in flight finish first.",
  },
];

type Page = {
  view: Destination;
  icon: "jobs" | "workers" | "assistant" | "activity" | "link" | "chip";
  line: string;
  can: string[];
  stops: string;
};

const pages: Page[] = [
  {
    view: "Jobs",
    icon: "jobs",
    line: "Every submission, with its status, task progress, and the worker holding it.",
    can: [
      "Open a row for the result, the files it wrote, and its metrics",
      "Read execution attempts and the supervisor's findings under Details",
      "Filter by state, or search by job, ID, or worker",
    ],
    stops:
      "One program on one worker. Python and PyTorch only, in process isolation rather than a sandbox.",
  },
  {
    view: "Workers",
    icon: "workers",
    line: "One card per machine, showing what it last reported about itself.",
    can: [
      "Create a one-time invite to pair a desktop or iOS agent",
      "Check last heartbeat and lease state under connection diagnostics",
    ],
    stops:
      "A GPU chip appears only after the worker ran a real tensor op on it. Policy allows compatible GPU work; it does not install GPU software for you.",
  },
  {
    view: "Assistant",
    icon: "assistant",
    line: "Chat with tools attached to your fleet.",
    can: [
      "Ask which workers are free, or what a task is doing",
      "Submit supported workloads, inspect or cancel tasks",
      "Change a run's spending cap",
    ],
    stops:
      "Eight tool calls and 90 seconds per turn, then it answers with what it has. No uploads, no shell, and it cannot mark a job finished — the server decides that.",
  },
  {
    view: "Activity",
    icon: "activity",
    line: "A direct read of the audit table: queueing, assignment, starts, results.",
    can: [
      "Trace exactly what the server decided, in order",
      "Find the assignment behind a retry or a cancellation",
    ],
    stops:
      "It records decisions, not program output. Execution logs live with the job.",
  },
  {
    view: "Topology",
    icon: "link",
    line: "The fleet drawn as a graph: control plane at the centre, workers dialing in.",
    can: [
      "See which workers hold leases right now",
      "Watch assignments move as jobs are scheduled",
    ],
    stops:
      "Every worker connects outbound, so the picture never shows an inbound port. There is nothing to open.",
  },
  {
    view: "Experiments",
    icon: "chip",
    line: "Recorded GPU migrations off the RTX 4090 baseline.",
    can: [
      "Open a run for its latency ratio, verdict, and changed answers",
      "Compare a saved result with /compare",
    ],
    stops:
      "Backed by the separate GPUShare service, in recorded mode by default. A speedup is accepted only when all 300 cases return the value they returned before.",
  },
];

const gates = [
  {
    name: "The GPU proves itself",
    body: "A worker runs a real tensor op on CUDA or Apple MPS before it may advertise the device. An explicit GPU request never falls back to CPU.",
  },
  {
    name: "The device signs the result",
    body: "Paired desktop and iOS agents sign every result with an Ed25519 key generated on the device. The private key never leaves it.",
  },
  {
    name: "The answers stay identical",
    body: "A speedup is accepted only when all 300 cases return exactly the value they returned before. Seeded trials are re-checked on a second worker against fresh seeds.",
  },
];

function Why({ children }: { children: string }) {
  return (
    <details className="guide-why">
      <summary aria-label="Why it works this way">
        <span aria-hidden="true">?</span>
      </summary>
      <p>{children}</p>
    </details>
  );
}

export function Guide({
  onNavigate,
}: {
  onNavigate: (view: Destination) => void;
}) {
  return (
    <div className="guide">
      <section className="guide-block">
        <div className="guide-block-head">
          <h3>How a job moves</h3>
          <p className="panel-description">
            Every submission walks the same five stages. Open a stage for why it
            sits where it does.
          </p>
        </div>
        <ol className="guide-stages">
          {stages.map((stage, index) => (
            <li key={stage.name}>
              <span className="guide-step">{index + 1}</span>
              <div>
                <h4>
                  {stage.name}
                  <Why>{stage.why}</Why>
                </h4>
                <p>{stage.line}</p>
              </div>
            </li>
          ))}
        </ol>
        <p className="guide-foot">
          A worker heartbeats every 5 seconds against a 45-second lease. Miss it
          and the task returns to the queue — a late result from that worker is
          refused.
        </p>
      </section>

      <section className="guide-block">
        <div className="guide-block-head">
          <h3>What each page is for</h3>
          <p className="panel-description">
            Open one to read what it does and where it stops, then jump straight
            to it.
          </p>
        </div>
        <div className="guide-pages">
          {pages.map((page) => (
            <details className="guide-page" key={page.view}>
              <summary>
                <span className="guide-page-icon">
                  <Icon name={page.icon} size={16} />
                </span>
                <span className="guide-page-name">
                  {page.view}
                  <small>{page.line}</small>
                </span>
                <Icon name="chevron" size={14} style={{ flex: "none" }} />
              </summary>
              <div className="guide-page-body">
                <div>
                  <div className="section-label">What you can do</div>
                  <ul>
                    {page.can.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <div className="section-label">Where it stops</div>
                  <p>{page.stops}</p>
                </div>
                <button
                  type="button"
                  className="outline-btn"
                  onClick={() => onNavigate(page.view)}
                >
                  Open {page.view}
                  <Icon name="arrow" size={14} />
                </button>
              </div>
            </details>
          ))}
        </div>
      </section>

      <section className="guide-block">
        <div className="guide-block-head">
          <h3>Before a result counts</h3>
          <p className="panel-description">
            Three checks, each covering something the previous one does not.
          </p>
        </div>
        <div className="guide-gates">
          {gates.map((gate, index) => (
            <div className="guide-gate" key={gate.name}>
              <span className="guide-gate-n">{`Gate 0${index + 1}`}</span>
              <h4>{gate.name}</h4>
              <p>{gate.body}</p>
            </div>
          ))}
        </div>
        <p className="guide-foot">
          Speed is never the deciding number. The fastest rejected migration on
          record ran 4.58× faster and still failed, because 13 of 300 answers
          changed.
        </p>
      </section>
    </div>
  );
}
