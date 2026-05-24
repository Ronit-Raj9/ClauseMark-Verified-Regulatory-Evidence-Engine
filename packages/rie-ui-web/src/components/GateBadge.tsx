import type { GateResultDTO } from "@/types";

// One pill per gate result. Colour reflects pass/fail; tooltip carries detail.
// Names come straight from the backend — no Python-side gate names are
// hard-coded in TypeScript.

interface Props {
  gate: GateResultDTO;
}

export default function GateBadge({ gate }: Props) {
  const tone = gate.passed ? "bg-gate-pass" : "bg-gate-fail";
  return (
    <span
      title={gate.detail || (gate.passed ? "passed" : "failed")}
      className={[
        "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-md text-[11px] font-mono text-white",
        tone,
      ].join(" ")}
    >
      <span className="font-semibold">{gate.gate}</span>
      <span className="opacity-90">{gate.passed ? "PASS" : "FAIL"}</span>
      {gate.score != null && <span className="opacity-90">{gate.score.toFixed(2)}</span>}
    </span>
  );
}
