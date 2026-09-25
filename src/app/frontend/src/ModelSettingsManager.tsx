import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import {
  fetchModelSettings,
  fetchWorkspaceEndpoints,
  updateModelSettings,
  warmWorkspaceEndpoint,
} from "./api";
import type { WorkspaceEndpoint } from "./types";

interface Props {
  onClose: () => void;
  onModelsChanged: () => Promise<void>;
}

export default function ModelSettingsManager({
  onClose,
  onModelsChanged,
}: Props) {
  const [endpoints, setEndpoints] = useState<WorkspaceEndpoint[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [storage, setStorage] = useState<"saved" | "memory" | "default">("default");
  const [wakeStatus, setWakeStatus] = useState<Record<string, "waking" | "error">>({});
  const [wakeMessage, setWakeMessage] = useState<Record<string, string>>({});
  const mountedRef = useRef(true);
  const titleId = useId();
  const dialogRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    closeButtonRef.current?.focus();
    return () => previouslyFocused?.focus();
  }, []);

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !saving) onClose();
    };
    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [onClose, saving]);

  const handleDialogKeyDown = (event: ReactKeyboardEvent<HTMLElement>) => {
    if (event.key !== "Tab") return;
    const focusable = Array.from(
      dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'
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

  useEffect(() => {
    let cancelled = false;
    Promise.all([fetchWorkspaceEndpoints(), fetchModelSettings()])
      .then(([eps, settings]) => {
        if (cancelled) return;
        setEndpoints(eps);
        const availableNames = new Set(eps.map((entry) => entry.name));
        setSelected(
          new Set(
            settings.endpoints
              .map((entry) => entry.name)
              .filter((name) => availableNames.has(name))
          )
        );
        setStorage(settings.source);
        setError("");
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const msg =
          e && typeof e === "object" && "message" in e
            ? (e as { message: string }).message
            : "Failed to load workspace endpoints";
        setError(msg);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return endpoints;
    return endpoints.filter((e) => e.name.toLowerCase().includes(q));
  }, [endpoints, search]);

  const toggle = (name: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  const handleWake = async (endpoint: WorkspaceEndpoint) => {
    setWakeStatus((previous) => ({ ...previous, [endpoint.name]: "waking" }));
    setWakeMessage((previous) => ({
      ...previous,
      [endpoint.name]: endpoint.ready
        ? "Checking the runtime with a tiny synthetic request…"
        : "Sending a tiny synthetic wake request…",
    }));
    try {
      const response = await warmWorkspaceEndpoint(endpoint.name);
      if (!mountedRef.current) return;
      if (response.status === "ready") {
        setEndpoints((previous) =>
          previous.map((item) =>
            item.name === endpoint.name ? { ...item, ready: true } : item
          )
        );
        setWakeStatus((previous) => {
          const next = { ...previous };
          delete next[endpoint.name];
          return next;
        });
        setWakeMessage((previous) => ({
          ...previous,
          [endpoint.name]: response.message,
        }));
        return;
      }

      setWakeMessage((previous) => ({
        ...previous,
        [endpoint.name]: response.message,
      }));
      for (let attempt = 0; attempt < 12; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 5000));
        if (!mountedRef.current) return;
        const refreshed = await fetchWorkspaceEndpoints();
        if (!mountedRef.current) return;
        setEndpoints(refreshed);
        if (refreshed.some((item) => item.name === endpoint.name && item.ready)) {
          setWakeStatus((previous) => {
            const next = { ...previous };
            delete next[endpoint.name];
            return next;
          });
          setWakeMessage((previous) => ({
            ...previous,
            [endpoint.name]: "Endpoint is ready.",
          }));
          return;
        }
      }
      throw new Error("The endpoint is still starting. Try again in a moment.");
    } catch (e: unknown) {
      if (!mountedRef.current) return;
      const message =
        e && typeof e === "object" && "message" in e
          ? (e as { message: string }).message
          : "The endpoint could not be woken.";
      setWakeStatus((previous) => ({ ...previous, [endpoint.name]: "error" }));
      setWakeMessage((previous) => ({ ...previous, [endpoint.name]: message }));
    }
  };

  const handleSave = async () => {
    if (selected.size === 0) return;
    setSaving(true);
    setError("");
    try {
      // Preserve the order endpoints appear in the list for a stable dropdown.
      const selectedEndpoints = endpoints
        .filter((e) => selected.has(e.name))
        .map(({ name, model_type, serving_runtime }) => ({
          name,
          model_type,
          serving_runtime,
        }));
      const settings = await updateModelSettings(selectedEndpoints);
      setStorage(settings.source);
      await onModelsChanged();
      onClose();
    } catch (e: unknown) {
      const msg =
        e && typeof e === "object" && "message" in e
          ? (e as { message: string }).message
          : "Failed to save model selection";
      setError(msg);
    } finally {
      setSaving(false);
    }
  };

  const selectedCount = selected.size;

  return (
    <div
      className="settings-drawer-overlay"
      onClick={() => {
        if (!saving) onClose();
      }}
    >
      <aside
        ref={dialogRef}
        className="settings-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleDialogKeyDown}
      >
        <div className="pm-header">
          <div>
            <span className="settings-kicker">Workspace preferences</span>
            <h2 id={titleId}>Settings</h2>
          </div>
          <button
            ref={closeButtonRef}
            className="pm-close"
            onClick={onClose}
            aria-label="Close model settings"
            disabled={saving}
          >
            &times;
          </button>
        </div>

        <div className="pm-body">
          <h3 className="settings-section-title">Models</h3>
          <p className="pm-preset-desc">
            Choose which Model Serving endpoints appear in the model dropdowns.
            FMAPI choices use the short service IDs shown under system.ai
            Services. Cold vLLM endpoints can be woken here with a tiny
            synthetic request; no clinical text is sent.
          </p>

          <div className="pm-settings-notice" role="status" aria-live="polite">
            {storage === "memory"
              ? "Lakebase is temporarily unavailable. Changes will last until the app restarts."
              : storage === "saved"
                ? "Your selection is saved to Lakebase for your account and follows you across browsers."
                : "Using the curated starter set. Save any change to persist it to Lakebase for your account."}
          </div>

          {error && <div className="sb-parse-error" role="alert">{error}</div>}

          {!loading && endpoints.length > 0 && (
            <div className="settings-filter-row">
              <input
                id="model-settings-search"
                name="model-settings-search"
                className="pm-search"
                placeholder="Search Model Serving endpoints..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                aria-label="Search Model Serving endpoints"
              />
              <button
                type="button"
                className="settings-text-button"
                onClick={() =>
                  setSelected(
                    new Set(
                      endpoints
                        .filter((entry) => entry.ready)
                        .map((entry) => entry.name)
                    )
                  )
                }
              >
                Select ready
              </button>
              <button
                type="button"
                className="settings-text-button"
                onClick={() => setSelected(new Set())}
              >
                Clear
              </button>
            </div>
          )}

          {loading ? (
            <div className="pm-preset-desc" role="status">
              Loading workspace chat endpoints...
            </div>
          ) : filtered.length === 0 ? (
            <div className="pm-preset-desc">
              {endpoints.length === 0
                ? "No chat endpoints are visible in this workspace."
                : "No endpoints match your search."}
            </div>
          ) : (
            filtered.map((ep) => (
              <div key={ep.name} className="pm-preset-row pm-model-row">
                <input
                  id={`model-setting-${ep.name}`}
                  name="selected-models"
                  type="checkbox"
                  checked={selected.has(ep.name)}
                  onChange={() => toggle(ep.name)}
                  aria-label={`Include ${ep.name}`}
                />
                <label
                  className="pm-preset-info pm-model-label"
                  htmlFor={`model-setting-${ep.name}`}
                >
                  <span className="pm-preset-name">{ep.name}</span>
                  <div className="pm-model-tags">
                    <span
                      className={`pm-badge-type ${
                        ep.model_type === "custom" ? "is-custom" : "is-fmapi"
                      }`}
                    >
                      {ep.model_type === "custom" ? "Custom" : "FMAPI"}
                    </span>
                    <span
                      className={`pm-badge-state ${ep.ready ? "is-ready" : "is-down"}`}
                    >
                      {ep.ready ? "Ready" : "Not ready"}
                    </span>
                    {ep.serving_runtime === "vllm_beta" && (
                      <span className="pm-badge-type">vLLM Beta</span>
                    )}
                  </div>
                  {wakeMessage[ep.name] && (
                    <span
                      className={`pm-wake-message ${wakeStatus[ep.name] === "error" ? "is-error" : ""}`}
                      role="status"
                    >
                      {wakeMessage[ep.name]}
                    </span>
                  )}
                </label>
                {ep.model_type === "custom" &&
                  ep.serving_runtime === "vllm_beta" && (
                    <button
                      type="button"
                      className="pm-wake-button"
                      onClick={() => handleWake(ep)}
                      disabled={wakeStatus[ep.name] === "waking"}
                      aria-label={`${ep.ready ? "Ping" : "Wake"} ${ep.name}`}
                    >
                      {wakeStatus[ep.name] === "waking"
                        ? ep.ready
                          ? "Checking…"
                          : "Waking…"
                        : ep.ready
                          ? "Ping"
                          : "Wake"}
                    </button>
                  )}
              </div>
            ))
          )}
        </div>

        <div className="pm-footer pm-model-footer">
          <span className="pm-preset-desc">
            {selectedCount > 0 ? `${selectedCount} selected` : "Select at least one model"}
          </span>
          <div className="pm-create-actions">
            <button className="pm-btn" onClick={onClose} disabled={saving}>
              Cancel
            </button>
            <button
              className="pm-btn-primary"
              onClick={handleSave}
              disabled={saving || loading || selectedCount === 0}
            >
              {saving ? "Saving..." : "Save"}
            </button>
          </div>
        </div>
      </aside>
    </div>
  );
}
