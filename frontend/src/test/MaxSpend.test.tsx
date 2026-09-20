import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { TaskComposer } from "../components/TaskComposer";
import { parseMaxSpend } from "../components/MaxSpendField";
import { formatMoney } from "../lib/format";
import { submitTasks, uploadSimulation } from "../api/client";

afterEach(() => vi.unstubAllGlobals());

it.each(["", "2.50", "0"])("submits an optional CAD cap: %j", async (value) => {
  const submit = vi.fn().mockResolvedValue(undefined);
  render(
    <TaskComposer
      selected=""
      onSelect={() => {}}
      workers={[]}
      busy={false}
      onSubmit={submit}
    />,
  );
  const input = screen.getByLabelText(/Max spend \(CAD\)/);
  expect(input).toHaveAttribute("type", "text");
  expect(input).toHaveAccessibleName("Max spend (CAD) Optional");
  if (value) await userEvent.type(input, value);
  await userEvent.click(screen.getByRole("button", { name: "Send task" }));
  await waitFor(() => expect(submit).toHaveBeenCalledOnce());
  expect(submit.mock.calls[0][1]).toBe(value || undefined);
});

it("rejects invalid spend before creating a task", async () => {
  const submit = vi.fn();
  render(
    <TaskComposer
      selected=""
      onSelect={() => {}}
      workers={[]}
      busy={false}
      onSubmit={submit}
    />,
  );
  await userEvent.type(screen.getByLabelText(/Max spend \(CAD\)/), "-2");
  await userEvent.click(screen.getByRole("button", { name: "Send task" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Max spend must be a CAD amount",
  );
  expect(submit).not.toHaveBeenCalled();
});

it("preserves decimal precision and rejects malformed or excessive amounts", () => {
  expect(parseMaxSpend("  .000001  ")).toBe("0.000001");
  expect(parseMaxSpend("   ")).toBeUndefined();
  for (const value of [
    "NaN",
    "Infinity",
    "1e3",
    "1,2",
    "0.0000001",
    "1000000000.01",
  ])
    expect(() => parseMaxSpend(value)).toThrow();
});

it("sends CAD caps in both APIs and omits blank caps", async () => {
  const fetcher = vi.fn().mockImplementation(async () => new Response("{}"));
  vi.stubGlobal("fetch", fetcher);
  await submitTasks([], "0");
  await submitTasks([]);
  await uploadSimulation([], "example", "request", "2.50");
  await uploadSimulation([], "example", "request");
  const bodies = fetcher.mock.calls.map((call) => JSON.parse(call[1].body));
  expect(bodies[0].usage_cap).toBe("0");
  expect(bodies[1]).not.toHaveProperty("usage_cap");
  expect(bodies[2].usage_cap).toBe("2.50");
  expect(bodies[3]).not.toHaveProperty("usage_cap");
});

it("shows unavailable costs safely without disguising valid zero or tiny costs", () => {
  for (const value of [undefined, null, "", " ", "NaN", "Infinity"])
    expect(formatMoney(value)).toBe("—");
  expect(formatMoney("0")).toBe("CA$0.00");
  expect(formatMoney("-1")).toBe("-CA$1.00");
  expect(formatMoney("0.000001")).toBe("<CA$0.0001");
  expect(formatMoney("1.25")).toBe("CA$1.25");
});
