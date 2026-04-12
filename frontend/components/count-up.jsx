"use client";

import { useEffect, useRef, useState } from "react";

function formatValue(current, { decimals = 0, suffix = "" }) {
  return `${current.toFixed(decimals)}${suffix}`;
}

export function CountUp({
  target,
  decimals = 0,
  suffix = "",
  duration = 1500,
  className = "",
}) {
  const ref = useRef(null);
  const [displayValue, setDisplayValue] = useState(() =>
    typeof window !== "undefined" && !("IntersectionObserver" in window)
      ? formatValue(target, { decimals, suffix })
      : formatValue(0, { decimals, suffix })
  );

  useEffect(() => {
    const node = ref.current;
    if (!node) {
      return undefined;
    }

    if (!("IntersectionObserver" in window)) {
      return undefined;
    }

    let animationFrame = 0;
    let started = false;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting || started) {
            return;
          }

          started = true;
          observer.unobserve(entry.target);

          const startTime = performance.now();

          function step(now) {
            const progress = Math.min((now - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const value = target * eased;

            setDisplayValue(formatValue(value, { decimals, suffix }));

            if (progress < 1) {
              animationFrame = requestAnimationFrame(step);
            } else {
              setDisplayValue(formatValue(target, { decimals, suffix }));
            }
          }

          animationFrame = requestAnimationFrame(step);
        });
      },
      { threshold: 0.5 }
    );

    observer.observe(node);

    return () => {
      observer.disconnect();
      cancelAnimationFrame(animationFrame);
    };
  }, [decimals, duration, suffix, target]);

  return (
    <span ref={ref} className={className}>
      {displayValue}
    </span>
  );
}
