import type { GateResultDTO } from "@/types";

interface Props {
  gate: GateResultDTO;
}

export default function GateBadge({ gate }: Props) {
  const tone = gate.passed
    ? "bg-gate-pass-bg text-gate-pass-text border-gate-pass-bg"
    : "bg-gate-fail-bg text-gate-fail-text border-gate-fail-bg";

  return (
    <span
      title={gate.detail || (gate.passed ? "passed" : "failed")}
      className={[
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-[11px]",
        tone,
      ].join(" ")}
    >
      <span className="font-semibold">{gate.gate}</span>
      <span className="opacity-80">{gate.passed ? "PASS" : "FAIL"}</span>
      {gate.score != null && <span className="opacity-80">{gate.score.toFixed(2)}</span>}
    </span>
  );
}
