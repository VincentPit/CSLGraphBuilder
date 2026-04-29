'use client';

import Link from 'next/link';
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

type Variant = 'primary' | 'success' | 'danger' | 'ghost' | 'icon';

const VARIANT_CLASS: Record<Variant, string> = {
  primary: 'btn-primary',
  success: 'btn-success',
  danger:  'btn-danger',
  ghost:   'btn-ghost',
  icon:    'btn-icon',
};

type CommonProps = {
  variant?: Variant;
  className?: string;
  children?: ReactNode;
};

type AsButton = CommonProps & ButtonHTMLAttributes<HTMLButtonElement> & {
  href?: undefined;
};

type AsLink = CommonProps & {
  href: string;
  'aria-label'?: string;
  title?: string;
  onClick?: () => void;
};

export type ButtonProps = AsButton | AsLink;

function classes(variant: Variant, extra?: string) {
  return [VARIANT_CLASS[variant], extra].filter(Boolean).join(' ');
}

function isLink(p: ButtonProps): p is AsLink {
  return typeof (p as AsLink).href === 'string';
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  props,
  ref,
) {
  const variant = props.variant ?? 'ghost';
  if (isLink(props)) {
    const { href, className, children, ...rest } = props;
    return (
      <Link href={href} className={classes(variant, className)} {...rest}>
        {children}
      </Link>
    );
  }
  const { variant: _v, className, children, type = 'button', ...rest } = props;
  return (
    <button
      ref={ref}
      type={type}
      className={classes(variant, className)}
      {...rest}
    >
      {children}
    </button>
  );
});
