import { useState } from "react";
import { login } from "../api/client";

export function Login({ onLogin }: { onLogin: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  return (
    <main className="login-page">
      <form
        className="composer login-card"
        onSubmit={async (event) => {
          event.preventDefault();
          if (busy) return;
          setBusy(true);
          setError("");
          try {
            await login(email, password);
            setPassword("");
            onLogin();
          } catch (cause) {
            setError(cause instanceof Error ? cause.message : "Sign in failed");
          } finally {
            setBusy(false);
          }
        }}
      >
        <h1>Sign in to dispatch.</h1>
        <p>Sign in with your fleet account.</p>
        <label className="field-label" htmlFor="email">
          Email
        </label>
        <input
          id="email"
          type="email"
          autoComplete="username"
          required
          value={email}
          onChange={(event) => setEmail(event.target.value)}
        />
        <label className="field-label" htmlFor="password">
          Password
        </label>
        <input
          id="password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(event) => setPassword(event.target.value)}
        />
        {error && <p role="alert">{error}</p>}
        <button className="submit" disabled={busy} type="submit">
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </main>
  );
}
