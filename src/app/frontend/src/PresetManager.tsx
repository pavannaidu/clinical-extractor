import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { createPreset, deletePreset } from "./api";
import type { Preset } from "./types";

interface Props {
  presets: Preset[];
  currentSchema: string;
  currentPrompt: string;
  currentSystemPrompt: string;
  onClose: () => void;
  onPresetsChanged: () => void;
}

export default function PresetManager({
  presets,
  currentSchema,
  currentPrompt,
  currentSystemPrompt,
  onClose,
  onPresetsChanged,
}: Props) {
  const [showCreate, setShowCreate] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [error, setError] = useState("");
  const titleId = useId();
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const mutationPending = saving || deleting !== null;

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    closeButtonRef.current?.focus();
    return () => previouslyFocused?.focus();
  }, []);

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !mutationPending) onClose();
    };
    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [mutationPending, onClose]);

  const handleDialogKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
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

  const handleCreate = async () => {
    if (!name.trim()) return;
    setSaving(true);
    setError("");
    try {
      await createPreset({
        name: name.trim(),
        description: description.trim(),
        schema_json: currentSchema,
        example_prompt: currentPrompt,
        system_prompt: currentSystemPrompt,
      });
      setName("");
      setDescription("");
      setShowCreate(false);
      onPresetsChanged();
    } catch (e: unknown) {
      const msg = e && typeof e === "object" && "message" in e ? (e as { message: string }).message : "Failed to save";
      setError(msg);
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (id: string) => {
    setDeleting(id);
    setError("");
    try {
      await deletePreset(id);
      onPresetsChanged();
    } catch (e: unknown) {
      const msg = e && typeof e === "object" && "message" in e ? (e as { message: string }).message : "Failed to delete";
      setError(msg);
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div
      className="preset-manager-overlay"
      onClick={() => {
        if (!mutationPending) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className="preset-manager"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleDialogKeyDown}
      >
        <div className="pm-header">
          <h2 id={titleId}>Manage Presets</h2>
          <button
            ref={closeButtonRef}
            className="pm-close"
            onClick={onClose}
            aria-label="Close preset manager"
            disabled={mutationPending}
          >
            &times;
          </button>
        </div>

        <div className="pm-body">
          {error && <div className="sb-parse-error" role="alert">{error}</div>}
          {presets.map((p) => (
            <div key={p.id} className="pm-preset-row">
              <div className="pm-preset-info">
                <span className="pm-preset-name">{p.name}</span>
                {p.is_builtin && <span className="pm-badge-builtin">Built-in</span>}
                {p.description && (
                  <div className="pm-preset-desc">{p.description}</div>
                )}
                {p.system_prompt && (
                  <div className="pm-preset-desc">Includes tailored system instructions</div>
                )}
              </div>
              <div className="pm-actions">
                {!p.is_builtin && (
                  <button
                    className="pm-btn pm-btn-danger"
                    onClick={() => handleDelete(p.id)}
                    disabled={deleting === p.id}
                    aria-label={`Delete preset ${p.name}`}
                  >
                    {deleting === p.id ? "..." : "Delete"}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>

        <div className="pm-footer">
          {!showCreate ? (
            <button className="pm-btn" onClick={() => setShowCreate(true)}>
              + Save Current as Preset
            </button>
          ) : (
            <div className="pm-create-form">
              <input
                id="preset-name"
                name="preset-name"
                placeholder="Preset name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoFocus
                aria-label="Preset name"
              />
              <input
                id="preset-description"
                name="preset-description"
                placeholder="Description (optional)"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                aria-label="Preset description"
              />
              <div className="pm-create-actions">
                <button className="pm-btn" onClick={() => setShowCreate(false)}>
                  Cancel
                </button>
                <button
                  className="pm-btn-primary"
                  onClick={handleCreate}
                  disabled={!name.trim() || saving}
                >
                  {saving ? "Saving..." : "Save"}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
