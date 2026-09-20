import type { SimulationStatus } from "../api/client";

export function AnalysisAgents({
  analysis,
}: {
  analysis: SimulationStatus["analysis"];
}) {
  if (!analysis?.children?.length) return null;
  return (
    <section aria-label="Analysis agents">
      <h3>Analysis agents</h3>
      <p>{analysis.rationale}</p>
      <p className="muted">
        Independent reports for the coordinator to review.
      </p>
      {analysis.children.map((child) => (
        <details key={child.id}>
          <summary>
            {child.role} · {child.status.replaceAll("_", " ")}
          </summary>
          <p>{child.question}</p>
          {child.error && <p>{child.error}</p>}
          {child.report && (
            <>
              <p>{child.report.summary}</p>
              {child.report.evidence.length > 0 && (
                <ul aria-label="Evidence">
                  {child.report.evidence.map((item, index) => (
                    <li key={index}>{item}</li>
                  ))}
                </ul>
              )}
              {child.report.questions.length > 0 && (
                <>
                  <h4>Open questions</h4>
                  <ul>
                    {child.report.questions.map((item, index) => (
                      <li key={index}>{item}</li>
                    ))}
                  </ul>
                </>
              )}
            </>
          )}
        </details>
      ))}
    </section>
  );
}
