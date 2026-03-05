import { useState } from "react";

const COLORS = {
  bg: "#0a0e1a",
  card: "#111827",
  border: "#1e2d45",
  embed: "#0ea5e9",
  concat: "#8b5cf6",
  linear: "#3b82f6",
  bn: "#f59e0b",
  relu: "#10b981",
  dropout: "#ef4444",
  output: "#f97316",
  text: "#e2e8f0",
  subtext: "#64748b",
  accent: "#38bdf8",
};

const embeddings = [
  { label: "Cat₁", input: 3, output: 4 },
  { label: "Cat₂", input: 7, output: 8 },
  { label: "Cat₃", input: 2, output: 3 },
  { label: "Cat₄", input: 3, output: 4 },
  { label: "Cat₅", input: 5, output: 6 },
  { label: "Cat₆", input: 3, output: 4 },
  { label: "Cat₇", input: 3, output: 3 },
];

const embeddingSum = embeddings.reduce((s, e) => s + e.output, 0); // 32
const continuousFeatures = 44 - embeddingSum; // 12

const denseBlocks = [
  { linear: "44 → 256", bn: "256", out: 256 },
  { linear: "256 → 128", bn: "128", out: 128 },
  { linear: "128 → 64", bn: "64", out: 64 },
];

function Tooltip({ text }) {
  return (
    <div style={{
      position: "absolute", bottom: "calc(100% + 6px)", left: "50%",
      transform: "translateX(-50%)", background: "#1e293b", color: "#e2e8f0",
      padding: "4px 8px", borderRadius: 4, fontSize: 11, whiteSpace: "nowrap",
      border: "1px solid #334155", zIndex: 100, pointerEvents: "none",
    }}>
      {text}
      <div style={{
        position: "absolute", top: "100%", left: "50%", transform: "translateX(-50%)",
        borderLeft: "5px solid transparent", borderRight: "5px solid transparent",
        borderTop: "5px solid #1e293b",
      }} />
    </div>
  );
}

function Chip({ label, color, tooltip, small }) {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        position: "relative", display: "inline-flex", alignItems: "center",
        gap: 4, padding: small ? "2px 7px" : "4px 10px",
        borderRadius: 4, fontSize: small ? 10 : 11, fontFamily: "'JetBrains Mono', monospace",
        fontWeight: 600, color: "#fff", cursor: "default",
        background: color + "22", border: `1px solid ${color}66`,
        boxShadow: hovered ? `0 0 8px ${color}66` : "none",
        transition: "box-shadow 0.2s",
      }}
    >
      <div style={{ width: 6, height: 6, borderRadius: "50%", background: color, flexShrink: 0 }} />
      {label}
      {hovered && tooltip && <Tooltip text={tooltip} />}
    </div>
  );
}

function Arrow({ label }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2, margin: "2px 0" }}>
      <div style={{ width: 1, height: 14, background: "#334155" }} />
      {label && <span style={{ fontSize: 9, color: COLORS.subtext, fontFamily: "monospace" }}>{label}</span>}
      <div style={{
        width: 0, height: 0,
        borderLeft: "4px solid transparent", borderRight: "4px solid transparent",
        borderTop: "5px solid #334155",
      }} />
    </div>
  );
}

function DenseBlock({ block, idx }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{
      border: `1px solid ${COLORS.border}`, borderRadius: 10,
      background: COLORS.card, overflow: "hidden",
      boxShadow: "0 4px 20px #00000044",
      transition: "box-shadow 0.2s",
    }}>
      <div
        onClick={() => setExpanded(v => !v)}
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: "10px 16px", cursor: "pointer", userSelect: "none",
          background: "#0f172a",
        }}
      >
        <span style={{ color: COLORS.accent, fontFamily: "monospace", fontSize: 12, fontWeight: 700 }}>
          Dense Block {idx + 1}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ color: COLORS.subtext, fontFamily: "monospace", fontSize: 11 }}>{block.linear}</span>
          <span style={{ color: COLORS.subtext, fontSize: 11 }}>{expanded ? "▲" : "▼"}</span>
        </div>
      </div>
      {expanded && (
        <div style={{ padding: "12px 16px", display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
          <Chip label={`Linear  ${block.linear}`} color={COLORS.linear} tooltip="Affine transformation: y = Wx + b" />
          <Arrow />
          <Chip label={`BatchNorm1d(${block.bn})`} color={COLORS.bn} tooltip="Normalizes activations per batch" />
          <Arrow />
          <Chip label="LeakyReLU(α=0.01)" color={COLORS.relu} tooltip="f(x) = x if x>0, else 0.01x" />
          <Arrow />
          <Chip label="Dropout(p=0.15)" color={COLORS.dropout} tooltip="15% neurons dropped during training" />
        </div>
      )}
      {!expanded && (
        <div style={{ padding: "8px 16px", display: "flex", flexWrap: "wrap", gap: 6 }}>
          <Chip label="Linear" color={COLORS.linear} small />
          <Chip label="BN" color={COLORS.bn} small tooltip="BatchNorm" />
          <Chip label="LeakyReLU" color={COLORS.relu} small />
          <Chip label="Drop 15%" color={COLORS.dropout} small />
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <div style={{
      minHeight: "100vh", background: COLORS.bg, padding: "32px 16px",
      fontFamily: "'Segoe UI', sans-serif", color: COLORS.text,
      display: "flex", flexDirection: "column", alignItems: "center",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0a0e1a; }
        ::-webkit-scrollbar-thumb { background: #1e2d45; border-radius: 3px; }
      `}</style>

      {/* Title */}
      <div style={{ textAlign: "center", marginBottom: 28 }}>
        <h1 style={{
          fontFamily: "'Syne', sans-serif", fontSize: 26, fontWeight: 800,
          color: "#fff", margin: 0, letterSpacing: "-0.5px",
        }}>StudentScoreNet</h1>
        <p style={{ color: COLORS.subtext, fontSize: 12, margin: "6px 0 0", fontFamily: "monospace" }}>
          Embedding + MLP Architecture  ·  Input: 44d  ·  Output: 1 (regression)
        </p>
      </div>

      <div style={{ width: "100%", maxWidth: 480, display: "flex", flexDirection: "column", gap: 0 }}>

        {/* ── Embedding Layer ── */}
        <div style={{
          border: `1px solid ${COLORS.embed}44`, borderRadius: 10,
          background: COLORS.card, padding: 16,
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <span style={{ fontFamily: "'Syne', sans-serif", fontSize: 13, fontWeight: 700, color: COLORS.embed }}>
              ModuleList — Embeddings
            </span>
            <span style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.subtext }}>→ 32d total</span>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {embeddings.map((e, i) => (
              <div key={i} style={{
                flex: "1 1 80px", minWidth: 80, background: "#0a1628",
                border: `1px solid ${COLORS.embed}33`, borderRadius: 7, padding: "8px 10px",
                display: "flex", flexDirection: "column", gap: 3, alignItems: "center",
              }}>
                <span style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.subtext }}>{e.label}</span>
                <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ fontFamily: "monospace", fontSize: 11, color: "#94a3b8" }}>{e.input}</span>
                  <span style={{ color: COLORS.embed, fontSize: 11 }}>→</span>
                  <span style={{ fontFamily: "monospace", fontSize: 11, color: COLORS.embed, fontWeight: 700 }}>{e.output}</span>
                </div>
              </div>
            ))}
            {/* Continuous features */}
            <div style={{
              flex: "1 1 80px", minWidth: 80, background: "#0a1628",
              border: `1px dashed #334155`, borderRadius: 7, padding: "8px 10px",
              display: "flex", flexDirection: "column", gap: 3, alignItems: "center",
            }}>
              <span style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.subtext }}>Numeric</span>
              <span style={{ fontFamily: "monospace", fontSize: 11, color: "#94a3b8", fontWeight: 700 }}>{continuousFeatures}d</span>
            </div>
          </div>
        </div>

        {/* Concat arrow */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", margin: "2px 0" }}>
          <div style={{ width: 1, height: 10, background: "#334155" }} />
          <div style={{
            padding: "4px 14px", background: COLORS.concat + "22",
            border: `1px solid ${COLORS.concat}66`, borderRadius: 20,
            fontFamily: "monospace", fontSize: 11, color: COLORS.concat, fontWeight: 700,
          }}>concat → 44d</div>
          <div style={{ width: 1, height: 10, background: "#334155" }} />
          <div style={{ width: 0, height: 0, borderLeft: "5px solid transparent", borderRight: "5px solid transparent", borderTop: "6px solid #334155" }} />
        </div>

        {/* ── Dense Blocks ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 12, fontWeight: 700, color: "#94a3b8", marginBottom: 2, paddingLeft: 4 }}>
            Sequential — Dense Stack  <span style={{ color: COLORS.subtext, fontWeight: 400 }}>(click to expand)</span>
          </div>
          {denseBlocks.map((block, idx) => (
            <div key={idx} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0 }}>
              <DenseBlock block={block} idx={idx} />
              {idx < denseBlocks.length - 1 && <Arrow />}
            </div>
          ))}
        </div>

        {/* Output arrow */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", margin: "2px 0" }}>
          <div style={{ width: 1, height: 10, background: "#334155" }} />
          <div style={{ width: 0, height: 0, borderLeft: "5px solid transparent", borderRight: "5px solid transparent", borderTop: "6px solid #334155" }} />
        </div>

        {/* Output */}
        <div style={{
          border: `1px solid ${COLORS.output}55`, borderRadius: 10,
          background: COLORS.card, padding: "14px 16px",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}>
          <div>
            <span style={{ fontFamily: "'Syne', sans-serif", fontSize: 13, fontWeight: 700, color: COLORS.output }}>Output Layer</span>
            <div style={{ fontFamily: "monospace", fontSize: 11, color: COLORS.subtext, marginTop: 3 }}>
              Linear(64 → 1)  ·  No activation
            </div>
          </div>
          <div style={{
            width: 38, height: 38, borderRadius: "50%", border: `2px solid ${COLORS.output}`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontFamily: "monospace", fontSize: 13, fontWeight: 700, color: COLORS.output,
            background: COLORS.output + "15",
          }}>ŷ</div>
        </div>

        {/* Legend */}
        <div style={{
          marginTop: 20, borderRadius: 10, border: `1px solid ${COLORS.border}`,
          background: COLORS.card, padding: "12px 16px",
        }}>
          <div style={{ fontFamily: "monospace", fontSize: 10, color: COLORS.subtext, marginBottom: 8, textTransform: "uppercase", letterSpacing: 1 }}>Legend</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {[
              { color: COLORS.embed, label: "Embedding" },
              { color: COLORS.concat, label: "Concat" },
              { color: COLORS.linear, label: "Linear" },
              { color: COLORS.bn, label: "BatchNorm" },
              { color: COLORS.relu, label: "LeakyReLU" },
              { color: COLORS.dropout, label: "Dropout" },
              { color: COLORS.output, label: "Output" },
            ].map(({ color, label }) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, fontFamily: "monospace", color: "#94a3b8" }}>
                <div style={{ width: 8, height: 8, borderRadius: 2, background: color }} />
                {label}
              </div>
            ))}
          </div>
        </div>

        {/* Param summary */}
        <div style={{
          marginTop: 10, borderRadius: 10, border: `1px solid ${COLORS.border}`,
          background: "#0f172a", padding: "12px 16px",
          display: "flex", justifyContent: "space-around",
        }}>
          {[
            { label: "Hidden Layers", value: "3" },
            { label: "Max Width", value: "256" },
            { label: "Dropout", value: "15%" },
            { label: "Regulariz.", value: "BN + Drop" },
          ].map(({ label, value }) => (
            <div key={label} style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 14, fontWeight: 700, color: COLORS.accent }}>{value}</div>
              <div style={{ fontFamily: "monospace", fontSize: 9, color: COLORS.subtext, marginTop: 2 }}>{label}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
