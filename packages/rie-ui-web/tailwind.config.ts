import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Coverage 3-state palette. Names map to CoverageState enum values
        // from rie-contracts. Logic stays data-driven (no pillar/indicator
        // names ever hardcoded in component code).
        coverage: {
          evidence: "#16a34a", // green-600  evidence_found
          absent: "#d97706",   // amber-600  no_evidence_in_searched_corpus
          insufficient: "#6b7280", // gray-500 insufficient_coverage
        },
        gate: {
          pass: "#16a34a",
          fail: "#dc2626",
          flag: "#d97706",
        },
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
