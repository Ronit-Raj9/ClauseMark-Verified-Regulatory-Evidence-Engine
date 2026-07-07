import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#F7F6F3",
        surface: "#FFFFFF",
        ink: "#111111",
        muted: "#787774",
        line: "#EAEAEA",
        coverage: {
          evidence: {
            bg: "#EDF3EC",
            text: "#346538",
            border: "#C5D9C4",
            solid: "#346538",
          },
          absent: {
            bg: "#FBF3DB",
            text: "#956400",
            border: "#E8D9A8",
            solid: "#956400",
          },
          insufficient: {
            bg: "#F1F1EF",
            text: "#787774",
            border: "#DDDDD8",
            solid: "#787774",
          },
        },
        gate: {
          pass: { bg: "#EDF3EC", text: "#346538", solid: "#346538" },
          fail: { bg: "#FDEBEC", text: "#9F2F2D", solid: "#9F2F2D" },
          flag: { bg: "#FBF3DB", text: "#956400", solid: "#956400" },
        },
        layer: {
          advisory: { bg: "#E1F3FE", text: "#1F6C9F" },
        },
      },
      fontFamily: {
        sans: ["DM Sans", "ui-sans-serif", "system-ui", "sans-serif"],
        serif: ["Newsreader", "Georgia", "serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(17, 17, 17, 0.04), 0 8px 24px rgba(17, 17, 17, 0.04)",
        float: "0 2px 8px rgba(17, 17, 17, 0.06)",
      },
      borderRadius: {
        xl: "12px",
        "2xl": "16px",
      },
      animation: {
        "fade-up": "fadeUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both",
      },
      keyframes: {
        fadeUp: {
          "0%": { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
