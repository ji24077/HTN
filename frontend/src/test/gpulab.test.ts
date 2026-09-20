import { expect, test } from "vitest";
import { completionNote, migrationAction, verification } from "../lib/gpulab";
import type { LabJob, Pod } from "../api/gpulab";

const job: LabJob = {
  id: "candidate",
  kind: "optimize-inference-speed",
  status: "complete",
};

const accepted = {
  status: "ok",
  policy: "strict_output_preservation",
  tolerance: 0,
  evaluation: { dataset_sha256: "frozen-suite-hash" },
  output_preservation_verified: true,
};

test.each([
  { status: "ok" },
  { ...accepted, tolerance: 0.02 },
  { ...accepted, tolerance: undefined },
  { ...accepted, policy: "unknown" },
  { ...accepted, evaluation: undefined },
  { ...accepted, output_preservation_verified: false },
  { ...accepted, output_preservation_verified: undefined },
])(
  "a passing status without current acceptance evidence stays unvalidated: %j",
  (validation) => {
    const result = verification({ ...job, result: { validation } });
    expect(result.label).toBe("Not validated");
    expect(result.detail).toContain("Historical or incomplete evaluation");
  },
);

test("accepts complete strict evidence and distinct training score policy", () => {
  expect(verification({ ...job, result: { validation: accepted } }).label).toBe(
    "Passed",
  );
  expect(
    verification({
      ...job,
      result: {
        validation: {
          ...accepted,
          policy: "aggregate_no_regression",
          output_preservation_verified: false,
        },
      },
    }).label,
  ).toBe("Passed");
});

test("a successful process and speedup cannot manufacture a quality pass", () => {
  expect(verification({ ...job, result: { speedup: 9 } }).label).toBe(
    "Not validated",
  );
  expect(
    completionNote({
      ...job,
      result: {
        speedup: 9,
        validation: { status: "regressed", detail: "Two outputs changed." },
      },
    }),
  ).toContain("Rejected. Two outputs changed.");
});

test("pending jobs cannot pass and unknown validation values fail closed", () => {
  expect(
    verification({
      ...job,
      status: "running",
      result: { validation: { status: "ok" } },
    }).label,
  ).toBe("Not validated");
  expect(
    verification({ ...job, result: { validation: { status: "new_status" } } })
      .label,
  ).toBe("Not validated");
});

test("unsupported AMD pairs and identical NVIDIA chips are not migration actions", () => {
  const pod: Pod = {
    id: "one",
    name: "one",
    gpu: "A5000",
    vendor: "nvidia",
    status: "running",
    cost_per_hour: 1,
  };
  expect(migrationAction(pod, { ...pod, id: "two" })).toBeNull();
  expect(
    migrationAction(
      { ...pod, vendor: "amd" },
      { ...pod, id: "two", vendor: "amd" },
    ),
  ).toBeNull();
  expect(migrationAction(pod, { ...pod, id: "two", gpu: "L40S" })?.name).toBe(
    "migrate-nextgen",
  );
});
