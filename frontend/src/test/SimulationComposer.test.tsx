import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { SimulationComposer } from "../components/SimulationComposer";
const upload = vi.hoisted(() => vi.fn());
vi.mock("../api/client", () => ({ uploadSimulation: upload }));

it("uploads the original files and run description, retaining the submission identity on retry", async () => {
  const user = userEvent.setup();
  const created = vi.fn();
  upload
    .mockReset()
    .mockRejectedValueOnce(new Error("Connection interrupted"))
    .mockResolvedValueOnce({ spec: { id: "sim-1" } });
  render(<SimulationComposer onCreated={created} />);
  const file = new File(["print('hello')"], "simulation.py", {
    type: "text/x-python",
  });
  await user.upload(screen.getByLabelText("Project files"), file);
  await user.type(
    screen.getByLabelText("What would you like to do?"),
    "Run 24 trials",
  );
  await user.click(screen.getByRole("button", { name: "Submit project" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Connection interrupted",
  );
  await user.click(screen.getByRole("button", { name: "Submit project" }));
  await waitFor(() => expect(created).toHaveBeenCalledOnce());
  expect(upload.mock.calls[0][0][0]).toBe(file);
  expect(upload.mock.calls[0][1]).toBe("Run 24 trials");
  expect(upload.mock.calls[0][2]).toBe(upload.mock.calls[1][2]);
  expect(upload.mock.calls[0][3]).toBeUndefined();
});

it("retains a CAD cap on retry and uses a new submission when the cap changes", async () => {
  const user = userEvent.setup();
  upload.mockReset().mockRejectedValue(new Error("Connection interrupted"));
  render(<SimulationComposer onCreated={vi.fn()} />);
  await user.upload(
    screen.getByLabelText("Project files"),
    new File(["pass"], "main.py"),
  );
  await user.type(
    screen.getByLabelText("What would you like to do?"),
    "Example",
  );
  await user.type(screen.getByLabelText(/Max spend \(CAD\)/), "2.50");
  const send = screen.getByRole("button", { name: "Submit project" });
  await user.click(send);
  await screen.findByRole("alert");
  await user.click(send);
  await screen.findByRole("alert");
  expect(upload.mock.calls[0][3]).toBe("2.50");
  expect(upload.mock.calls[1][3]).toBe("2.50");
  expect(upload.mock.calls[1][2]).toBe(upload.mock.calls[0][2]);
  await user.clear(screen.getByLabelText(/Max spend \(CAD\)/));
  await user.type(screen.getByLabelText(/Max spend \(CAD\)/), "0");
  await user.click(send);
  await screen.findByRole("alert");
  expect(upload.mock.calls[2][3]).toBe("0");
  expect(upload.mock.calls[2][2]).not.toBe(upload.mock.calls[1][2]);
});

it("rejects colliding filenames before submission", async () => {
  const user = userEvent.setup();
  render(<SimulationComposer onCreated={vi.fn()} />);
  await user.upload(screen.getByLabelText("Project files"), [
    new File(["a"], "main.py"),
    new File(["b"], "main.py"),
  ]);
  expect(screen.getByRole("alert")).toHaveTextContent("unique names");
  expect(screen.getByRole("button", { name: "Submit project" })).toBeDisabled();
});

it("lets the agent choose hosting and hardware from the request", async () => {
  const user = userEvent.setup();
  upload.mockReset().mockResolvedValue({ spec: { id: "sim-1" } });
  render(<SimulationComposer onCreated={vi.fn()} />);
  expect(screen.queryByLabelText("Run as")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Runtime")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Workload")).not.toBeInTheDocument();
  expect(screen.queryByText("Service settings")).not.toBeInTheDocument();
  await user.upload(
    screen.getByLabelText("Project files"),
    new File(["serve()"], "server.py"),
  );
  await user.type(
    screen.getByLabelText("What would you like to do?"),
    "Host my model for one hour",
  );
  await user.type(screen.getByLabelText(/Max spend \(CAD\)/), "2.50");
  await user.click(screen.getByRole("button", { name: "Submit project" }));
  await waitFor(() => expect(upload).toHaveBeenCalledOnce());
  expect(upload.mock.calls[0]).toHaveLength(4);
  expect(upload.mock.calls[0][1]).toBe("Host my model for one hour");
  expect(upload.mock.calls[0][3]).toBe("2.50");
});
