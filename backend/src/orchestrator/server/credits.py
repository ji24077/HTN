"""Account-specific prototype credits; no payment processor or paid top-ups."""

from decimal import Decimal
from uuid import UUID


def account_id(request):
    claims = getattr(getattr(request, "state", None), "auth_claims", None)
    return UUID(claims["sub"]) if claims else None


async def account_credit(conn, account):
    if account is None:
        return None
    row = await conn.fetchrow(
        """SELECT (SELECT COALESCE(sum(amount_cad),0) FROM account_credits WHERE account_id=$1) AS credited,
            (SELECT COALESCE(sum(t.cost_cad),0) FROM supervised_jobs j
                JOIN usage_job_totals t ON t.job_id=j.id WHERE j.billing_account_id=$1)
            + (SELECT COALESCE(sum(u.estimated_cost_cad),0) FROM supervised_jobs j
                JOIN usage_record_totals u ON u.job_id=j.id
                WHERE j.billing_account_id=$1 AND u.ended_at IS NULL) AS spent""",
        account,
    )
    return {
        "currency": "CAD",
        "credited": str(row["credited"]),
        "spent": str(row["spent"]),
        "balance": str(row["credited"] - row["spent"]),
    }


async def grant_credit(store, account, amount, receipt, *, reason):
    """Operator-only, idempotent credit entry. Intentionally not an HTTP/chat tool."""
    from .db.store import Conflict

    amount = Decimal(str(amount))
    if not amount.is_finite() or amount <= 0 or amount > 1_000_000_000 or amount % Decimal("0.01"):
        raise ValueError("Credit must be positive CAD cents, up to 1,000,000,000")
    if not reason.strip():
        raise ValueError("A credit reason is required")
    async with store.change() as (conn, _):
        old = await conn.fetchrow("SELECT * FROM account_credits WHERE id=$1", receipt)
        if old:
            if (
                old["account_id"] != account
                or old["amount_cad"] != amount
                or old["reason"] != reason
            ):
                raise Conflict("Credit receipt already used with different details")
        else:
            await conn.execute(
                "INSERT INTO account_credits(id,account_id,amount_cad,reason) VALUES($1,$2,$3,$4)",
                receipt,
                account,
                amount,
                reason,
            )
        return await account_credit(conn, account)
