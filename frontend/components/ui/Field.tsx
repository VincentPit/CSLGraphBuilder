'use client';

import { useId, type InputHTMLAttributes, type ReactNode, type TextareaHTMLAttributes } from 'react';

export function FieldLabel({
  htmlFor,
  hint,
  children,
}: {
  htmlFor?: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label htmlFor={htmlFor} className="field-label">
      {children}
      {hint && (
        <span
          className="ml-2 normal-case tracking-normal font-semibold"
          style={{ color: 'var(--text-muted)' }}
        >
          {hint}
        </span>
      )}
    </label>
  );
}

type BaseProps = {
  label?: string;
  hint?: string;
  error?: string;
};

export function Field({
  label,
  hint,
  error,
  id,
  className,
  ...rest
}: BaseProps & InputHTMLAttributes<HTMLInputElement>) {
  const generated = useId();
  const inputId = id ?? generated;
  return (
    <div className={className}>
      {label && <FieldLabel htmlFor={inputId} hint={hint}>{label}</FieldLabel>}
      <input
        id={inputId}
        className="input"
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${inputId}-error` : undefined}
        {...rest}
      />
      {error && (
        <p
          id={`${inputId}-error`}
          className="mt-1.5 text-[12px] font-semibold"
          style={{ color: 'var(--danger)' }}
        >
          {error}
        </p>
      )}
    </div>
  );
}

export function TextareaField({
  label,
  hint,
  error,
  id,
  className,
  ...rest
}: BaseProps & TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const generated = useId();
  const inputId = id ?? generated;
  return (
    <div className={className}>
      {label && <FieldLabel htmlFor={inputId} hint={hint}>{label}</FieldLabel>}
      <textarea
        id={inputId}
        className="input"
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? `${inputId}-error` : undefined}
        {...rest}
      />
      {error && (
        <p
          id={`${inputId}-error`}
          className="mt-1.5 text-[12px] font-semibold"
          style={{ color: 'var(--danger)' }}
        >
          {error}
        </p>
      )}
    </div>
  );
}
