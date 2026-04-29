'use client';

import { useEffect, useRef, type ReactNode } from 'react';

/**
 * Lightweight dialog primitive. Two flavours:
 *
 *  - **modal** (default): traps focus, blocks pointer events on the rest
 *    of the page via a backdrop, closes on Escape or backdrop click.
 *
 *  - **non-modal**: closes on Escape only, doesn't trap focus, doesn't
 *    render a backdrop. Use for popovers anchored to canvas elements
 *    (graph inspector, dropdowns).
 *
 * In both cases the dialog's own root receives focus on open and
 * restores focus to the previously-focused element on close. Pass
 * `labelledBy` (an element id) so screen readers announce the title.
 */
export function Dialog({
  open,
  onClose,
  modal = true,
  labelledBy,
  className,
  backdropClassName,
  children,
}: {
  open: boolean;
  onClose: () => void;
  modal?: boolean;
  labelledBy?: string;
  className?: string;
  /** Layout class for the modal backdrop wrapper. Defaults to centering. */
  backdropClassName?: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const restoreRef = useRef<HTMLElement | null>(null);

  // Capture the focused element on open, restore it on close.
  useEffect(() => {
    if (!open) return;
    restoreRef.current = (document.activeElement as HTMLElement) ?? null;
    // Defer one tick so child content can mount.
    const id = window.setTimeout(() => ref.current?.focus(), 0);
    return () => {
      window.clearTimeout(id);
      restoreRef.current?.focus?.();
    };
  }, [open]);

  // Escape closes; for modal dialogs, Tab traps focus inside.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
        return;
      }
      if (modal && e.key === 'Tab' && ref.current) {
        const focusables = ref.current.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])',
        );
        if (focusables.length === 0) {
          e.preventDefault();
          return;
        }
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        const active = document.activeElement as HTMLElement | null;
        if (e.shiftKey && active === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && active === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, modal, onClose]);

  if (!open) return null;

  const dialog = (
    <div
      ref={ref}
      role="dialog"
      aria-modal={modal ? 'true' : 'false'}
      aria-labelledby={labelledBy}
      tabIndex={-1}
      className={className}
      onKeyDown={(e) => {
        // Stop Escape from bubbling further once handled above.
        if (e.key === 'Escape') e.stopPropagation();
      }}
    >
      {children}
    </div>
  );

  if (!modal) return dialog;

  const layoutClass = backdropClassName ?? 'flex items-center justify-center p-4';
  return (
    <div
      className={`fixed inset-0 z-50 ${layoutClass}`}
      style={{ background: 'rgba(15, 11, 9, 0.45)', backdropFilter: 'blur(4px)' }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      {dialog}
    </div>
  );
}
