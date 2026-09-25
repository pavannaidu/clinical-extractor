import {
  useEffect,
  useId,
  useRef,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";

interface Props {
  onClose: () => void;
}

const RATE_LIMITS_URL =
  "https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/limits#rate-limits-by-model";

export default function HelpDrawer({ onClose }: Props) {
  const titleId = useId();
  const dialogRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    closeButtonRef.current?.focus();
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("keydown", handleEscape);
      previouslyFocused?.focus();
    };
  }, [onClose]);

  const handleDialogKeyDown = (event: ReactKeyboardEvent<HTMLElement>) => {
    if (event.key !== "Tab") return;
    const focusable = Array.from(
      dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), summary, [href], [tabindex]:not([tabindex="-1"])'
      ) ?? []
    );
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return (
    <div className="settings-drawer-overlay" onClick={onClose}>
      <aside
        ref={dialogRef}
        className="settings-drawer help-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={handleDialogKeyDown}
      >
        <div className="pm-header">
          <div>
            <h2 id={titleId}>Guide &amp; Definitions</h2>
          </div>
          <button
            ref={closeButtonRef}
            className="pm-close"
            onClick={onClose}
            aria-label="Close help"
          >
            &times;
          </button>
        </div>

        <div className="pm-body help-body">
          <section className="help-intro" aria-labelledby="help-quick-start">
            <span className="settings-kicker">Quick start</span>
            <h3 id="help-quick-start">Choose the test that answers your question</h3>
            <div className="help-mode-grid">
              <div>
                <strong>Burst</strong>
                <p>Find saturation quickly and verify 429 retry behavior.</p>
              </div>
              <div>
                <strong>Target rate</strong>
                <p>Measure sustained behavior at a planned production traffic level.</p>
              </div>
            </div>
          </section>

          <section className="help-section" aria-labelledby="help-load-testing">
            <h3 id="help-load-testing">Load testing</h3>

            <details open>
              <summary>What is the difference between Burst and Target rate?</summary>
              <p>
                Burst starts requests immediately up to the worker limit. ITPM and OTPM are
                references only. Target rate schedules new arrivals by wall clock for the chosen
                duration; the token targets determine how frequently requests start.
              </p>
            </details>

            <details>
              <summary>What are ITPM, OTPM, and cumulative output tokens?</summary>
              <p>
                ITPM is input tokens processed in a trailing minute. OTPM is output tokens produced
                in a trailing minute. Cumulative output is the total generated across the whole run,
                so it always grows and is not itself a rate.
              </p>
            </details>

            <details>
              <summary>How is the Target rate arrival interval calculated?</summary>
              <p>
                The app estimates requests per minute from both axes: target ITPM divided by input
                tokens per request, and target OTPM divided by expected output tokens. It uses the
                lower request rate, so the stricter token axis controls arrivals.
              </p>
            </details>

            <details>
              <summary>How is output size estimated?</summary>
              <p>
                Output length is unknown before generation, so the app reuses the median from the
                latest matching workload. For a new endpoint or payload, Calibrate sends one request
                and saves the measured output size. Manual override remains available in Advanced.
                Max output tokens is still the hard cap for each response.
              </p>
            </details>

            <details>
              <summary>What do 100%, 125%, and 150% load mean?</summary>
              <p>
                They scale the wall-clock arrival rate relative to the user-entered targets. Use 100%
                for boundary behavior and 125% or 150% when you intentionally want to provoke
                throttling. These percentages do not change published workspace limits.
              </p>
            </details>

            <details>
              <summary>How are retries handled?</summary>
              <p>
                Retriable 429 and output-validation failures use exponential backoff with jitter.
                Backing-off requests release their concurrency slot, then compete with new traffic
                when they retry. Attempt counts and any reported token usage are retained.
              </p>
            </details>

            <details>
              <summary>How should I read the charts?</summary>
              <p>
                The solid line is observed cumulative output; the dashed line is the user-entered
                OTPM reference projected over wall time. Request markers show first-try success,
                retried success, and failure on the same time axis. Use p95/p99 latency and rolling
                token rates for sustained behavior rather than relying only on the final total.
              </p>
            </details>

            <details>
              <summary>Are the displayed rate limits authoritative?</summary>
              <p>
                No. Limits can vary by model, region, and workspace configuration. Verify them before
                testing in the official{" "}
                <a href={RATE_LIMITS_URL} target="_blank" rel="noreferrer">
                  Databricks FMAPI limits documentation
                </a>
                . The app never changes targets simply because another model is selected.
              </p>
            </details>
          </section>

          <section className="help-section" aria-labelledby="help-models">
            <h3 id="help-models">Models and safe operation</h3>

            <details>
              <summary>Why does a vLLM model need to be made ready?</summary>
              <p>
                Scale-to-zero endpoints may need a cold start. Open Settings and use Make ready to
                send a tiny synthetic request, then wait for the endpoint to report ready before a
                comparison or load test.
              </p>
            </details>

            <details>
              <summary>How do I choose which models appear?</summary>
              <p>
                Open Settings, select the workspace Model Serving endpoints you want, and save. The
                selection is persisted for your user and controls both the Playground and Load Testing
                dropdowns.
              </p>
            </details>

            <details>
              <summary>How should I run a test safely?</summary>
              <p>
                Start small, verify the endpoint and payload, then increase duration or load. Load tests
                consume serving capacity and may incur cost. Avoid sensitive clinical text; the built-in
                sample is synthetic and cache-busted for measurement.
              </p>
            </details>
          </section>
        </div>
      </aside>
    </div>
  );
}
