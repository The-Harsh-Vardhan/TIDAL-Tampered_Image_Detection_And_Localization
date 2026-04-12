"use client";

import { useEffect, useRef, useState } from "react";

export function FadeIn({
  as: Component = "div",
  children,
  className = "",
  delayMs = 0,
  ...props
}) {
  const ref = useRef(null);
  const [isVisible, setIsVisible] = useState(() =>
    typeof window !== "undefined" && !("IntersectionObserver" in window)
  );

  useEffect(() => {
    const node = ref.current;
    if (!node) {
      return undefined;
    }

    if (!("IntersectionObserver" in window)) {
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible(true);
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.15,
        rootMargin: "0px 0px -40px 0px",
      }
    );

    observer.observe(node);

    return () => observer.disconnect();
  }, []);

  return (
    <Component
      ref={ref}
      className={`fade-in ${isVisible ? "visible" : ""} ${className}`.trim()}
      style={{
        transitionDelay: `${delayMs}ms`,
      }}
      {...props}
    >
      {children}
    </Component>
  );
}
