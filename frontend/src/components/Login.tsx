import { useState } from "react";
import {
  login,
  sendPasswordReset,
  signUp,
  updatePassword,
} from "../api/client";

type Mode = "login" | "signup" | "forgot" | "recovery";

export function Login({
  onLogin,
  recovery = false,
  initialError = "",
}: {
  onLogin: () => void;
  recovery?: boolean;
  initialError?: string;
}) {
  const [mode, setMode] = useState<Mode>(recovery ? "recovery" : "login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(initialError);
  const [message, setMessage] = useState("");
  const newPassword = mode === "signup" || mode === "recovery";
  const title = {
    login: "Sign in to dispatch.",
    signup: "Create your account.",
    forgot: "Reset your password.",
    recovery: "Choose a new password.",
  }[mode];
  const action = {
    login: "Sign in",
    signup: "Create account",
    forgot: "Send reset link",
    recovery: "Update password",
  }[mode];

  function switchMode(next: Mode) {
    setMode(next);
    setPassword("");
    setConfirmation("");
    setError("");
    setMessage("");
  }

  return (
    <main className="login-page">
      <form
        className="composer login-card"
        aria-label={title}
        aria-busy={busy}
        onSubmit={async (event) => {
          event.preventDefault();
          if (busy) return;
          setError("");
          setMessage("");
          if (newPassword && password !== confirmation) {
            setError("Passwords do not match.");
            return;
          }
          setBusy(true);
          try {
            if (mode === "forgot") {
              await sendPasswordReset(email.trim());
              setMessage(
                "If an account exists for this email, you’ll receive a password reset link shortly.",
              );
            } else if (mode === "recovery") {
              await updatePassword(password);
              switchMode("login");
              setMessage("Password updated. Sign in with your new password.");
            } else if (mode === "signup") {
              const signedIn = await signUp(email.trim(), password);
              setPassword("");
              setConfirmation("");
              if (signedIn) onLogin();
              else
                setMessage(
                  "Check your email to confirm your account, then sign in. A fleet administrator must grant access before you can use the dashboard.",
                );
            } else {
              await login(email.trim(), password);
              setPassword("");
              onLogin();
            }
          } catch (cause) {
            setError(
              cause instanceof Error
                ? cause.message
                : "Something went wrong. Please try again.",
            );
          } finally {
            setBusy(false);
          }
        }}
      >
        <div className="brand">
          <span className="brandmark">↗</span> dispatch
          <span className="brand-dot">.</span>
        </div>
        <h1>{title}</h1>
        <p>
          {mode === "signup"
            ? "Create an account to request access to your fleet."
            : mode === "forgot"
              ? "We’ll email you a link to choose a new password."
              : mode === "recovery"
                ? "Enter a new password for your account."
                : "Sign in with your fleet account."}
        </p>
        {mode !== "recovery" && (
          <>
            <label className="field-label" htmlFor="email">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="username"
              required
              disabled={busy}
              value={email}
              onChange={(event) => setEmail(event.target.value)}
            />
          </>
        )}
        {mode !== "forgot" && (
          <>
            <label className="field-label" htmlFor="password">
              {mode === "recovery" ? "New password" : "Password"}
            </label>
            <input
              id="password"
              type="password"
              autoComplete={newPassword ? "new-password" : "current-password"}
              required
              minLength={newPassword ? 8 : undefined}
              disabled={busy}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
            />
          </>
        )}
        {newPassword && (
          <>
            <p className="auth-hint">Use at least 8 characters.</p>
            <label className="field-label" htmlFor="confirmation">
              Confirm password
            </label>
            <input
              id="confirmation"
              type="password"
              autoComplete="new-password"
              required
              disabled={busy}
              value={confirmation}
              onChange={(event) => setConfirmation(event.target.value)}
            />
          </>
        )}
        {error && <p role="alert">{error}</p>}
        {message && (
          <p className="auth-message" role="status">
            {message}
          </p>
        )}
        <button className="submit" disabled={busy} type="submit">
          {busy ? "Please wait…" : action}
        </button>
        {mode !== "recovery" && (
          <div className="auth-actions">
            {mode === "login" ? (
              <>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => switchMode("forgot")}
                >
                  Forgot password?
                </button>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => switchMode("signup")}
                >
                  Create account
                </button>
              </>
            ) : (
              <button
                type="button"
                disabled={busy}
                onClick={() => switchMode("login")}
              >
                Back to sign in
              </button>
            )}
          </div>
        )}
      </form>
    </main>
  );
}
