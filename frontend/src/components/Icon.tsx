import type { CSSProperties } from "react";
const paths = {
  jobs: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
  workers: "M4 3h16v7H4zM4 14h16v7H4zM7 6.5h5M7 17.5h5M16 6.5h1M16 17.5h1",
  activity: "M3 12h4l3-8 4 16 3-8h4",
  assistant: "M9 3h6M12 3v3M4 6h16v13H4zM8 11h.01M16 11h.01M8 15h8",
  chip: "M7 7h10v10H7zM10 10h4v4h-4zM9 3v4M15 3v4M9 17v4M15 17v4M3 9h4M3 15h4M17 9h4M17 15h4",
  plus: "M12 5v14M5 12h14",
  search: "M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  arrow: "M5 12h14M13 6l6 6-6 6",
  close: "M6 6l12 12M6 18L18 6",
  check: "M5 12l4 4L19 6",
  clock: "M12 8v5l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  warning: "M12 3L2 21h20L12 3zM12 9v5M12 17h.01",
  download: "M12 3v12M7 10l5 5 5-5M4 16v5h16v-5",
  chevron: "M9 5l7 7-7 7",
  link: "M10 13a5 5 0 0 0 7 0l3-3a5 5 0 0 0-7-7l-2 2M14 11a5 5 0 0 0-7 0l-3 3a5 5 0 0 0 7 7l2-2",
};
export function Icon({
  name,
  size = 18,
  style,
}: {
  name: keyof typeof paths;
  size?: number;
  style?: CSSProperties;
}) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      style={style}
    >
      <path d={paths[name]} />
    </svg>
  );
}
