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
  await user.upload(screen.getByLabelText("Simulation files"), file);
  await user.type(screen.getByLabelText("Describe your run"), "Run 24 trials");
  await user.click(screen.getByRole("button", { name: "Submit simulation" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Connection interrupted",
  );
  await user.click(screen.getByRole("button", { name: "Submit simulation" }));
  await waitFor(() => expect(created).toHaveBeenCalledOnce());
  expect(upload.mock.calls[0][0][0]).toBe(file);
  expect(upload.mock.calls[0][1]).toBe("Run 24 trials");
  expect(upload.mock.calls[0][2]).toBe(upload.mock.calls[1][2]);
});

it("rejects colliding filenames before submission", async () => {
  const user = userEvent.setup();
  render(<SimulationComposer onCreated={vi.fn()} />);
  await user.upload(screen.getByLabelText("Simulation files"), [
    new File(["a"], "main.py"),
    new File(["b"], "main.py"),
  ]);
  expect(screen.getByRole("alert")).toHaveTextContent("unique names");
  expect(
    screen.getByRole("button", { name: "Submit simulation" }),
  ).toBeDisabled();
});
