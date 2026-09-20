import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { JobOutputs } from "../components/JobOutputs";

const api = vi.hoisted(() => ({
  jobOutputs: vi.fn(),
  downloadJobOutput: vi.fn(),
}));
vi.mock("../api/client", () => api);

it("downloads generated files and the final result with the job scope", async () => {
  api.jobOutputs.mockResolvedValue({
    files: [
      { id: "output-1", name: "images/frame.png", size: 200000, attempt: 2 },
    ],
  });
  api.downloadJobOutput.mockReset().mockResolvedValue(undefined);
  render(<JobOutputs jobId="job-1" phase="completed" />);
  const user = userEvent.setup();
  await user.click(
    await screen.findByRole("button", { name: "Download images/frame.png" }),
  );
  expect(api.downloadJobOutput).toHaveBeenCalledWith(
    "job-1",
    "images/frame.png",
    "output-1",
  );
  await user.click(
    screen.getByRole("button", { name: "Download raw result JSON" }),
  );
  expect(api.downloadJobOutput).toHaveBeenCalledWith(
    "job-1",
    "result.json",
    undefined,
  );
});

it("shows download failure and allows a retry", async () => {
  api.jobOutputs.mockResolvedValue({ files: [] });
  api.downloadJobOutput
    .mockReset()
    .mockRejectedValueOnce(new Error("Download failed"));
  render(<JobOutputs jobId="job-2" phase="completed" />);
  await userEvent
    .setup()
    .click(screen.getByRole("button", { name: "Download raw result JSON" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("Download failed");
  expect(
    screen.getByRole("button", { name: "Download raw result JSON" }),
  ).toBeEnabled();
});

it("distinguishes a terminal empty result from loading and retries a listing failure", async () => {
  api.jobOutputs
    .mockRejectedValueOnce(new Error("network"))
    .mockResolvedValue({ files: [] });
  render(<JobOutputs jobId="job-empty" phase="failed" />);
  expect(screen.getByRole("status")).toHaveTextContent("Loading output files");
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not load output files",
  );
  expect(screen.queryByText(/No file artifacts/)).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Retry files" }));
  expect(
    await screen.findByText(/No file artifacts were produced/),
  ).toBeVisible();
  expect(screen.queryByText(/will appear/)).not.toBeInTheDocument();
});
