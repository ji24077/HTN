import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import { useWorkspaceNavigation } from "../hooks/useWorkspaceNavigation";
import type { Task } from "../api/types";

const getTask = vi.hoisted(() => vi.fn());
vi.mock("../api/client", () => ({ getTask }));

const task = (id: string) => ({ spec: { id } }) as Task;
function deferred() {
  let resolve!: (task: Task) => void;
  const promise = new Promise<Task>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function visit(path: string) {
  window.history.pushState(null, "", path);
  window.dispatchEvent(new HashChangeEvent("hashchange"));
}

beforeEach(() => {
  window.history.replaceState(null, "", "/");
  getTask.mockReset();
});

it("restores an encoded job link and its Files tab while showing loading", async () => {
  window.history.replaceState(null, "", "#/jobs/job%2Fone/files");
  const pending = deferred();
  getTask.mockReturnValue(pending.promise);
  const { result } = renderHook(useWorkspaceNavigation);
  expect(result.current.jobView).toBe("Files");
  expect(result.current.detail).toBeNull();
  expect(result.current.jobRequest).toEqual({
    error: undefined,
    retryable: true,
  });
  await act(async () => pending.resolve(task("job/one")));
  expect(result.current.detail?.spec.id).toBe("job/one");
  expect(result.current.jobRequest).toBeNull();
});

it("keeps a failed link retryable and rejects mismatched job responses", async () => {
  window.history.replaceState(null, "", "#/jobs/one/details");
  getTask
    .mockResolvedValueOnce(task("wrong-job"))
    .mockResolvedValueOnce(task("one"));
  const { result } = renderHook(useWorkspaceNavigation);
  await waitFor(() =>
    expect(result.current.jobRequest?.error).toContain("couldn’t load"),
  );
  expect(result.current.detail).toBeNull();
  act(() => result.current.retryJob());
  await waitFor(() => expect(result.current.detail?.spec.id).toBe("one"));
  expect(result.current.jobView).toBe("Details");
  expect(getTask).toHaveBeenCalledTimes(2);
});

it("does not reopen a job when its response arrives after leaving", async () => {
  window.history.replaceState(null, "", "#/jobs/one");
  const pending = deferred();
  getTask.mockReturnValue(pending.promise);
  const { result } = renderHook(useWorkspaceNavigation);
  const signal = getTask.mock.calls[0][1] as AbortSignal;
  act(() => result.current.navigate("Workers"));
  expect(signal.aborted).toBe(true);
  await act(async () => pending.resolve(task("one")));
  expect(result.current.view).toBe("Workers");
  expect(result.current.detail).toBeNull();
  expect(result.current.jobRequest).toBeNull();
});

it("preserves tab history and opens a known task without an extra route fetch", async () => {
  const { result } = renderHook(useWorkspaceNavigation);
  act(() => result.current.openDetail(task("one")));
  act(() => result.current.changeJobView("Files"));
  expect(window.location.hash).toBe("#/jobs/one/files");
  act(() => result.current.changeJobView("Activity"));
  expect(result.current.jobView).toBe("Activity");
  act(() => window.history.back());
  await waitFor(() => expect(result.current.jobView).toBe("Files"));
  expect(result.current.detail?.spec.id).toBe("one");
  expect(getTask).not.toHaveBeenCalled();
});

it("hides the previous job immediately while another job link loads", async () => {
  const pending = deferred();
  getTask.mockReturnValue(pending.promise);
  const { result } = renderHook(useWorkspaceNavigation);
  act(() => result.current.openDetail(task("one")));
  act(() => visit("#/jobs/two"));
  expect(result.current.detail).toBeNull();
  expect(result.current.jobRequest).not.toBeNull();
  await act(async () => pending.resolve(task("two")));
  expect(result.current.detail?.spec.id).toBe("two");
});

it("explains malformed links without sending a request and allows returning to jobs", () => {
  window.history.replaceState(null, "", "#/jobs/%broken");
  const { result } = renderHook(useWorkspaceNavigation);
  expect(result.current.jobRequest?.error).toContain("invalid");
  expect(result.current.jobRequest?.retryable).toBe(false);
  expect(getTask).not.toHaveBeenCalled();
  act(() => result.current.closeDetail());
  expect(window.location.hash).toBe("#/jobs");
  expect(result.current.jobRequest).toBeNull();
});
