import { afterEach, expect, it, vi } from "vitest";
import { downloadJobOutput, previewJobOutput } from "../api/client";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("checks authorization with HEAD and streams via the browser download manager", async () => {
  const fetcher = vi
    .fn()
    .mockResolvedValue(new Response(null, { status: 200 }));
  vi.stubGlobal("fetch", fetcher);
  const clicks: { href: string; name: string }[] = [];
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (
    this: HTMLAnchorElement,
  ) {
    clicks.push({ href: this.getAttribute("href")!, name: this.download });
  });
  await downloadJobOutput("job", "models/checkpoint.bin", "file");
  expect(fetcher).toHaveBeenCalledExactlyOnceWith("/v1/jobs/job/outputs/file", {
    method: "HEAD",
    credentials: "same-origin",
  });
  expect(clicks).toEqual([
    { href: "/v1/jobs/job/outputs/file", name: "checkpoint.bin" },
  ]);
});

it("reports unavailable files without starting a download", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response(null, { status: 404 })),
  );
  const click = vi
    .spyOn(HTMLAnchorElement.prototype, "click")
    .mockImplementation(() => {});
  await expect(
    downloadJobOutput("job", "missing.bin", "missing"),
  ).rejects.toThrow("Could not download");
  expect(click).not.toHaveBeenCalled();
});

it("bounds previews even when a server sends a file larger than the listed size", async () => {
  const cancel = vi.fn();
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new Uint8Array(2 * 1024 * 1024 + 1));
          },
          cancel,
        }),
      ),
    ),
  );
  await expect(previewJobOutput("job", "oversized")).rejects.toThrow(
    "too large to preview",
  );
  expect(cancel).toHaveBeenCalled();
});
