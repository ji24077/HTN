import { useId, useState } from "react";

export function parseMaxSpend(value: string): string | undefined {
  const amount = value.trim();
  if (!amount) return undefined;
  if (
    !/^(?:\d+(?:\.\d{0,6})?|\.\d{1,6})$/.test(amount) ||
    Number(amount) > 1_000_000_000
  )
    throw new Error(
      "Max spend must be a CAD amount from 0 to 1,000,000,000 with up to 6 decimal places.",
    );
  return amount.startsWith(".") ? `0${amount}` : amount;
}

export function MaxSpendField({
  value,
  onChange,
  disabled,
}: {
  value: string;
  onChange: (value: string) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const [touched, setTouched] = useState(false);
  let error = "";
  if (touched) {
    try {
      parseMaxSpend(value);
    } catch {
      error = "Enter a valid CAD amount with up to 6 decimal places.";
    }
  }
  return (
    <>
      <label className="field-label" htmlFor={id}>
        Max spend (CAD) <span className="max-spend-optional">Optional</span>
      </label>
      <input
        id={id}
        type="text"
        inputMode="decimal"
        autoComplete="off"
        placeholder="No limit"
        maxLength={24}
        value={value}
        disabled={disabled}
        aria-invalid={!!error}
        onBlur={() => setTouched(true)}
        aria-describedby={`${id}-help${error ? ` ${id}-error` : ""}`}
        onChange={(event) => onChange(event.target.value)}
      />
      {error && (
        <p id={`${id}-error`} className="field-error">
          {error}
        </p>
      )}
      <p id={`${id}-help`} className="failover-help">
        Leave blank for no limit.
      </p>
    </>
  );
}
