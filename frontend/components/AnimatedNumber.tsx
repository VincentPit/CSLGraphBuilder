'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * Animates a number from its previous value to ``value`` over ``duration`` ms
 * using an ease-out cubic curve. Re-fires whenever the value changes, so the
 * dashboard counters tick up smoothly on refetch.
 */
export default function AnimatedNumber({
  value,
  duration = 700,
  decimals = 0,
  className,
  format,
}: {
  value: number;
  duration?: number;
  decimals?: number;
  className?: string;
  format?: (v: number) => string;
}) {
  const [display, setDisplay] = useState(value);
  const fromRef = useRef(value);
  const startRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    fromRef.current = display;
    startRef.current = null;

    const tick = (ts: number) => {
      if (startRef.current === null) startRef.current = ts;
      const elapsed = ts - startRef.current;
      const t = Math.min(1, elapsed / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const next = fromRef.current + (value - fromRef.current) * eased;
      setDisplay(next);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, duration]);

  const rounded = decimals > 0
    ? Number(display.toFixed(decimals))
    : Math.round(display);

  return (
    <span className={className}>
      {format ? format(rounded) : rounded.toLocaleString()}
    </span>
  );
}
