const FLAG_COLORS: Record<string, string> = {
  high: "high", critical: "high", abnormal: "high", urgent: "high",
  low: "low",
  normal: "normal", stable: "normal", resolved: "normal",
};

const FLAG_FIELD_NAMES = new Set([
  "flag",
  "interpretation",
  "priority",
  "risk",
  "risk_level",
  "severity",
  "status",
]);

function isFlagValue(v: unknown): v is string {
  return typeof v === "string" && v.toLowerCase() in FLAG_COLORS;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function formatLabel(key: string): string {
  return key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function compactObj(obj: Record<string, unknown>): string {
  return Object.entries(obj)
    .map(([k, v]) =>
      typeof v === "object" && v !== null
        ? `${formatLabel(k)}: ${JSON.stringify(v)}`
        : `${formatLabel(k)}: ${v ?? "\u2014"}`
    )
    .join(", ");
}

function RenderValue({ value, fieldKey }: { value: unknown; fieldKey?: string }) {
  if (value === null || value === undefined)
    return <span className="fv-null" aria-label="Not provided">&mdash;</span>;

  if (fieldKey && FLAG_FIELD_NAMES.has(fieldKey.toLowerCase()) && isFlagValue(value))
    return (
      <span className={`fv-badge fv-badge-${FLAG_COLORS[value.toLowerCase()]}`}>
        {value}
      </span>
    );

  if (typeof value === "boolean")
    return <span className="fv-primitive">{value ? "Yes" : "No"}</span>;

  if (typeof value === "number" || typeof value === "string")
    return <span className="fv-primitive">{String(value)}</span>;

  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="fv-null">None found</span>;
    if (value.every((item) => !isRecord(item)))
      return <span className="fv-primitive">{value.map(String).join(", ")}</span>;
    return (
      <ul className="fv-compact-list">
        {value.map((item, i) => (
          <li key={i}>
            {isRecord(item) ? compactObj(item) : <RenderValue value={item} fieldKey={fieldKey} />}
          </li>
        ))}
      </ul>
    );
  }

  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return (
      <dl className="fv-kv fv-kv-nested">
        {Object.entries(obj).map(([k, v]) => (
          <div key={k} className="fv-kv-row">
            <dt>{formatLabel(k)}</dt>
            <dd><RenderValue value={v} fieldKey={k} /></dd>
          </div>
        ))}
      </dl>
    );
  }

  return <span className="fv-primitive">{String(value)}</span>;
}

function tryParseJSON(v: unknown): unknown {
  if (typeof v !== "string") return v;
  let s = v.trim();
  s = s.replace(/^```(?:json)?\s*\n?/, "").replace(/\n?```\s*$/, "").trim();
  if ((s.startsWith("{") && s.endsWith("}")) ||
      (s.startsWith("[") && s.endsWith("]"))) {
    try { return JSON.parse(s); } catch { return v; }
  }
  return v;
}

export default function FormattedView({ data: rawData }: { data: unknown }) {
  const data = tryParseJSON(rawData);

  if (data === null || data === undefined)
    return <span className="fv-null">No data</span>;

  if (typeof data !== "object") {
    if (typeof data === "string" && data.length > 40) {
      return <pre className="fv-fallback">{data}</pre>;
    }
    return <RenderValue value={data} />;
  }

  if (Array.isArray(data)) {
    if (data.length === 0)
      return <span className="fv-null">Empty list</span>;

    if (data.every(isRecord)) {
      const keys = Array.from(
        new Set(data.flatMap((row) => Object.keys(row)))
      );
      return (
        <div className="fv-table-wrap">
          <table>
            <caption className="sr-only">Extracted structured data</caption>
            <thead>
              <tr>
                {keys.map((k) => (
                  <th key={k}>{formatLabel(k)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr key={i}>
                  {keys.map((k) => (
                    <td key={k}>
                      <RenderValue value={row[k]} fieldKey={k} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    return (
      <ul className="fv-compact-list">
        {data.map((item, index) => (
          <li key={index}><RenderValue value={item} /></li>
        ))}
      </ul>
    );
  }

  const obj = data as Record<string, unknown>;
  return (
    <>
      {Object.entries(obj).map(([key, value]) => {
        const isSection =
          Array.isArray(value) &&
          value.length > 0 &&
          value.every(isRecord);

        return (
          <div key={key} className="fv-card">
            <div className="fv-card-title">{formatLabel(key)}</div>
            <div className="fv-card-body">
              {isSection ? (
                <FormattedView data={value} />
              ) : typeof value === "object" && value !== null && !Array.isArray(value) ? (
                <dl className="fv-kv">
                  {Object.entries(value as Record<string, unknown>).map(([k, v]) => (
                    <div key={k} className="fv-kv-row">
                      <dt>{formatLabel(k)}</dt>
                      <dd><RenderValue value={v} fieldKey={k} /></dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <RenderValue value={value} fieldKey={key} />
              )}
            </div>
          </div>
        );
      })}
    </>
  );
}
