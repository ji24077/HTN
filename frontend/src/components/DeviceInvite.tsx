import { useRef, useState } from "react";
import { createDeviceInvite } from "../api/client";

export function DeviceInvite() {
  const [invite, setInvite] = useState<{ url: string; expires: string } | null>(
    null,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const pending = useRef(false);

  async function create() {
    if (pending.current) return;
    pending.current = true;
    setBusy(true);
    setError("");
    setCopied(false);
    setInvite(null);
    try {
      const result = await createDeviceInvite();
      const url = new URL("/join", result.server);
      url.searchParams.set("code", result.code);
      setInvite({
        url: url.href,
        expires: new Date(
          Date.now() + result.expires_in * 1000,
        ).toLocaleTimeString(),
      });
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Could not create an invite.",
      );
    } finally {
      pending.current = false;
      setBusy(false);
    }
  }

  return (
    <section className="device-invite" aria-label="Connect a device">
      <div>
        <h2>Connect a device</h2>
        <p>Pair a desktop worker or the iOS app with your fleet.</p>
      </div>
      <button
        className="outline-btn"
        disabled={busy}
        onClick={() => void create()}
      >
        {busy ? "Creating invite…" : "Create device invite"}
      </button>
      {error && <p role="alert">{error}</p>}
      {invite && (
        <div className="invite-result">
          <label className="field-label" htmlFor="device-invite-link">
            Paste this link in the worker app
          </label>
          <input
            id="device-invite-link"
            type="text"
            readOnly
            value={invite.url}
            onFocus={(event) => event.currentTarget.select()}
          />
          <button
            className="outline-btn"
            onClick={async () => {
              try {
                await navigator.clipboard.writeText(invite.url);
                setCopied(true);
              } catch {
                setError("Select and copy the invite link above.");
              }
            }}
          >
            {copied ? "Copied" : "Copy invite"}
          </button>
          <p>
            Works once. Expires at {invite.expires}. Keep this link private.
          </p>
        </div>
      )}
    </section>
  );
}
