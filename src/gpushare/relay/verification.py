from __future__ import annotations

from .models import QualityMetrics, VerificationReport


class QualityVerifier:
    """One quality gate shared by training, inference, and migration agents."""

    def verify(
        self,
        reference: QualityMetrics,
        candidate: QualityMetrics,
        *,
        tolerance: float,
        candidate_applied: bool = False,
    ) -> VerificationReport:
        reasons: list[str] = []
        comparable = bool(
            reference.evaluation_set_hash
            and candidate.evaluation_set_hash
            and reference.evaluation_set_hash == candidate.evaluation_set_hash
        )
        if not comparable:
            reasons.append("reference and candidate must use the same held-out evaluation set")
        if candidate.json_validity < reference.json_validity - tolerance:
            reasons.append("JSON validity regressed beyond the allowed threshold")
        if candidate.exact_match < reference.exact_match - tolerance:
            reasons.append("exact match regressed beyond the allowed threshold")
        if (
            reference.hallucination_rate is not None
            and candidate.hallucination_rate is not None
            and candidate.hallucination_rate > reference.hallucination_rate + tolerance
        ):
            reasons.append("hallucination rate regressed beyond the safety threshold")
        if (
            reference.omission_rate is not None
            and candidate.omission_rate is not None
            and candidate.omission_rate > reference.omission_rate + tolerance
        ):
            reasons.append("omission rate regressed beyond the quality threshold")
        if not candidate.safety_passed:
            reasons.append("candidate failed safety checks")
        status = "reject" if reasons else "pass"
        return VerificationReport(
            status=status,
            reasons=reasons or ["same evaluation set; quality and safety gates passed"],
            reference=reference,
            candidate=candidate,
            comparable=comparable,
            rollback_required=status == "reject" and candidate_applied,
        )
