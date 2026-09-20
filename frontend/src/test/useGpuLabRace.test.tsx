import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, test, vi } from "vitest";
import { lab as request, type LabJob } from "../api/gpulab";
import { useGpuLab } from "../hooks/useGpuLab";

vi.mock("../api/gpulab", async (original) => ({
  ...(await original<typeof import("../api/gpulab")>()),
  lab: vi.fn(),
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (cause: Error) => void;
  const promise = new Promise<T>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

const submittedJob: LabJob = {
  id: "new-job",
  kind: "train-and-evaluate",
  status: "queued",
};
const completedJob: LabJob = {
  id: "previous-job",
  kind: "generate-data",
  status: "complete",
};
const experiment = { data: { train: 1800, heldout: 200 } };

beforeEach(() => {
  vi.mocked(request).mockReset();
});

test("a slow refresh preserves an intervening submitted job and finishes connecting", async () => {
  const pods = deferred<{ pods: [] }>();
  vi.mocked(request).mockImplementation(
    async <T,>(path: string): Promise<T> => {
      if (path === "/api/state")
        return { experiment, demo_read_only: false } as T;
      if (path === "/api/jobs") return { jobs: [completedJob] } as T;
      if (path === "/api/models")
        return { models: [], serving: { running: false } } as T;
      if (path === "/api/evidence")
        throw new Error("Saved evidence unavailable.");
      if (path === "/api/pods") return (await pods.promise) as T;
      if (path === "/api/jobs/train") return submittedJob as T;
      throw new Error(`Unexpected request: ${path}`);
    },
  );
  const { result } = renderHook(() => useGpuLab(false, vi.fn()));
  let refreshing!: Promise<void>;
  act(() => {
    refreshing = result.current.refresh();
  });
  await waitFor(() => expect(request).toHaveBeenCalledWith("/api/pods"));
  await act(async () => {
    await result.current.start("/api/jobs/train", {});
  });
  expect(result.current.jobs).toEqual([submittedJob]);
  await act(async () => {
    pods.resolve({ pods: [] });
    await refreshing;
  });
  expect(result.current.status).toBe("live");
  expect(result.current.jobs).toEqual([submittedJob, completedJob]);
  expect(result.current.evidenceError).toBe("Saved evidence unavailable.");
  expect(result.current.readOnly).toBe(false);
});

test("a fresh overlapping refresh wins even when the older snapshot finishes last", async () => {
  const oldPods = deferred<{ pods: [] }>();
  let generation = 0;
  let podCalls = 0;
  const latestJob = { ...submittedJob, status: "running", progress: 35 };
  vi.mocked(request).mockImplementation(
    async <T,>(path: string): Promise<T> => {
      if (path === "/api/state") {
        generation += 1;
        return { experiment, demo_read_only: generation === 2 } as T;
      }
      if (path === "/api/jobs")
        return { jobs: generation === 1 ? [completedJob] : [latestJob] } as T;
      if (path === "/api/models")
        return { models: [], serving: { running: false } } as T;
      if (path === "/api/evidence")
        throw new Error(
          generation === 1 ? "Old evidence error." : "New evidence error.",
        );
      if (path === "/api/pods") {
        podCalls += 1;
        return (podCalls === 1 ? await oldPods.promise : { pods: [] }) as T;
      }
      throw new Error(`Unexpected request: ${path}`);
    },
  );
  const { result } = renderHook(() => useGpuLab(false, vi.fn()));
  let older!: Promise<void>;
  act(() => {
    older = result.current.refresh();
  });
  await waitFor(() => expect(podCalls).toBe(1));
  await act(async () => {
    await result.current.refresh();
  });
  expect(result.current.jobs).toEqual([latestJob]);
  expect(result.current.readOnly).toBe(true);
  await act(async () => {
    oldPods.resolve({ pods: [] });
    await older;
  });
  expect(result.current.jobs).toEqual([latestJob]);
  expect(result.current.status).toBe("live");
  expect(result.current.readOnly).toBe(true);
  expect(result.current.evidenceError).toBe("New evidence error.");
});

test("a stale failed refresh cannot mark a newer successful connection offline", async () => {
  const oldState = deferred<unknown>();
  let stateCalls = 0;
  vi.mocked(request).mockImplementation(
    async <T,>(path: string): Promise<T> => {
      if (path === "/api/state") {
        stateCalls += 1;
        return (
          stateCalls === 1
            ? await oldState.promise
            : { experiment, demo_read_only: true }
        ) as T;
      }
      if (path === "/api/jobs") return { jobs: [completedJob] } as T;
      if (path === "/api/models")
        return { models: [], serving: { running: false } } as T;
      if (path === "/api/evidence") throw new Error("No evidence.");
      if (path === "/api/pods") return { pods: [] } as T;
      throw new Error(`Unexpected request: ${path}`);
    },
  );
  const { result } = renderHook(() => useGpuLab(false, vi.fn()));
  let older!: Promise<void>;
  act(() => {
    older = result.current.refresh();
  });
  await act(async () => {
    await result.current.refresh();
  });
  await act(async () => {
    oldState.reject(new Error("Old connection failed."));
    await older;
  });
  expect(result.current.status).toBe("live");
  expect(result.current.error).toBe("");
  expect(result.current.readOnly).toBe(true);
});

test("a later refresh can update a retained job without creating duplicates", async () => {
  const pods = deferred<{ pods: [] }>();
  let refreshCount = 0;
  const finished = { ...submittedJob, status: "complete", progress: 100 };
  vi.mocked(request).mockImplementation(
    async <T,>(path: string): Promise<T> => {
      if (path === "/api/state") {
        refreshCount += 1;
        return { experiment } as T;
      }
      if (path === "/api/jobs")
        return { jobs: refreshCount === 1 ? [] : [finished] } as T;
      if (path === "/api/models")
        return { models: [], serving: { running: false } } as T;
      if (path === "/api/evidence") throw new Error("No evidence.");
      if (path === "/api/pods")
        return (refreshCount === 1 ? await pods.promise : { pods: [] }) as T;
      if (path === "/api/jobs/train") return submittedJob as T;
      throw new Error(`Unexpected request: ${path}`);
    },
  );
  const { result } = renderHook(() => useGpuLab(false, vi.fn()));
  let older!: Promise<void>;
  act(() => {
    older = result.current.refresh();
  });
  await waitFor(() => expect(request).toHaveBeenCalledWith("/api/pods"));
  await act(async () => {
    await result.current.start("/api/jobs/train", {});
  });
  await act(async () => {
    pods.resolve({ pods: [] });
    await older;
  });
  await act(async () => {
    await result.current.refresh();
  });
  expect(result.current.jobs).toEqual([finished]);
});
