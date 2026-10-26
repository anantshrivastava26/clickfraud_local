import type { SVGProps } from "react";

type P = SVGProps<SVGSVGElement> & { size?: number };

function Svg({ size = 16, children, ...rest }: P) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      {...rest}
    >
      {children}
    </svg>
  );
}

export const Pulse = (p: P) => (
  <Svg {...p}>
    <path d="M2 9.5 5.5 3l3 6.5L10 7h4" />
  </Svg>
);
export const Layers = (p: P) => (
  <Svg {...p}>
    <path d="M8 2 1.5 5.5 8 9l6.5-3.5L8 2Z" />
    <path d="m1.5 8.5 6.5 3.5 6.5-3.5" />
  </Svg>
);
export const Scan = (p: P) => (
  <Svg {...p}>
    <path d="M2 5V3.5A1.5 1.5 0 0 1 3.5 2H5M11 2h1.5A1.5 1.5 0 0 1 14 3.5V5M14 11v1.5a1.5 1.5 0 0 1-1.5 1.5H11M5 14H3.5A1.5 1.5 0 0 1 2 12.5V11M2 8h12" />
  </Svg>
);
export const Upload = (p: P) => (
  <Svg {...p}>
    <path d="M8 10.5V2.5M5 5.5l3-3 3 3M2.5 10v2a1.5 1.5 0 0 0 1.5 1.5h8a1.5 1.5 0 0 0 1.5-1.5v-2" />
  </Svg>
);
export const Download = (p: P) => (
  <Svg {...p}>
    <path d="M8 2.5v8M5 7.5l3 3 3-3M2.5 10v2a1.5 1.5 0 0 0 1.5 1.5h8a1.5 1.5 0 0 0 1.5-1.5v-2" />
  </Svg>
);
export const FileIcon = (p: P) => (
  <Svg {...p}>
    <path d="M9 1.5H4.5A1.5 1.5 0 0 0 3 3v10a1.5 1.5 0 0 0 1.5 1.5h7A1.5 1.5 0 0 0 13 13V5.5l-4-4Z" />
    <path d="M9 1.5v4h4" />
  </Svg>
);
export const Search = (p: P) => (
  <Svg {...p} size={p.size ?? 14}>
    <circle cx="7" cy="7" r="4.5" />
    <path d="m10.5 10.5 3 3" />
  </Svg>
);
export const Close = (p: P) => (
  <Svg {...p}>
    <path d="m4 4 8 8M12 4l-8 8" />
  </Svg>
);
export const Info = (p: P) => (
  <Svg {...p}>
    <circle cx="8" cy="8" r="6.5" />
    <path d="M8 7.5v3.5M8 5h.01" />
  </Svg>
);
export const Sort = ({ dir, ...p }: P & { dir?: "asc" | "desc" | null }) => (
  <Svg {...p} size={12} strokeWidth={1.6} style={{ opacity: dir ? 1 : 0.35 }}>
    {dir === "asc" ? <path d="m4 10 4-4 4 4" /> : <path d="m4 6 4 4 4-4" />}
  </Svg>
);

/* Verdict glyphs: shape carries the meaning alongside colour. */
export const BotGlyph = (p: P) => (
  <Svg {...p} size={p.size ?? 12} strokeWidth={1.8}>
    <path d="M5.2 1.5h5.6l3.7 3.7v5.6l-3.7 3.7H5.2l-3.7-3.7V5.2z" fill="currentColor" stroke="none" />
    <path d="M8 4.8v3.6M8 11h.01" stroke="var(--surface)" />
  </Svg>
);
export const ReviewGlyph = (p: P) => (
  <Svg {...p} size={p.size ?? 12} strokeWidth={1.8}>
    <path d="M8 1.6 15 14H1L8 1.6Z" fill="currentColor" stroke="none" />
    <path d="M8 6.2v3.4M8 11.8h.01" stroke="var(--surface)" />
  </Svg>
);
export const HumanGlyph = (p: P) => (
  <Svg {...p} size={p.size ?? 12} strokeWidth={1.8}>
    <circle cx="8" cy="8" r="7" fill="currentColor" stroke="none" />
    <path d="m5 8.2 2 2 4-4.2" stroke="var(--surface)" />
  </Svg>
);
