import { useState, useEffect, useRef, useCallback } from "react";
import { createPortal } from "react-dom";
import "./SchemaBuilder.css";

type FieldType = "string" | "number" | "integer" | "boolean" | "object" | "array";
type ItemType = "string" | "number" | "integer" | "boolean" | "object";

interface FieldDef {
  id: string;
  name: string;
  description: string;
  type: FieldType;
  required: boolean;
  nullable: boolean;
  enumValues: string[];
  children: FieldDef[];
  itemType: ItemType;
  itemNullable: boolean;
  itemDescription: string;
  itemChildren: FieldDef[];
  schemaTemplate: Record<string, unknown>;
  itemSchemaTemplate: Record<string, unknown>;
}

interface VisualSchemaAssessment {
  reason: string | null;
  hasPreservedConstraints: boolean;
}

const FIELD_TYPES: FieldType[] = [
  "string",
  "number",
  "integer",
  "boolean",
  "object",
  "array",
];
const ITEM_TYPES: ItemType[] = ["string", "number", "integer", "boolean", "object"];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasOwn(value: Record<string, unknown>, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(value, key);
}

let _counter = 0;
function newField(): FieldDef {
  const n = _counter++;
  return {
    id: "f_" + n,
    name: "field_" + n,
    description: "",
    type: "string",
    required: false,
    nullable: false,
    enumValues: [],
    children: [],
    itemType: "string",
    itemNullable: false,
    itemDescription: "",
    itemChildren: [],
    schemaTemplate: {},
    itemSchemaTemplate: {},
  };
}

function getBaseType(typeValue: unknown): FieldType | ItemType {
  if (Array.isArray(typeValue)) {
    const concreteType = typeValue.find(
      (value): value is FieldType | ItemType =>
        typeof value === "string" && value !== "null"
    );
    return (concreteType as FieldType | ItemType) ?? "string";
  }
  return (typeValue as FieldType | ItemType) ?? "string";
}

function isNullableType(typeValue: unknown): boolean {
  return Array.isArray(typeValue) && typeValue.includes("null");
}

function toSchemaType(fieldType: FieldType | ItemType, nullable: boolean): string | string[] {
  return nullable ? [fieldType, "null"] : fieldType;
}

function parseSchemaToFields(schema: Record<string, unknown>, depth = 0): FieldDef[] {
  const props = (schema.properties ?? {}) as Record<string, Record<string, unknown>>;
  const req = Array.isArray(schema.required) ? (schema.required as string[]) : [];
  return Object.entries(props).map(([key, val]) => {
    const field = newField();
    field.name = key;
    field.description = typeof val.description === "string" ? val.description : "";
    field.type = getBaseType(val.type) as FieldType;
    field.required = req.includes(key);
    field.nullable = isNullableType(val.type);
    field.schemaTemplate = val;
    if (field.type === "string" && Array.isArray(val.enum)) {
      field.enumValues = val.enum.filter((entry): entry is string => typeof entry === "string");
    }
    if (field.type === "object" && val.properties && depth < 2) {
      field.children = parseSchemaToFields(val as Record<string, unknown>, depth + 1);
    }
    if (field.type === "array" && val.items) {
      const items = val.items as Record<string, unknown>;
      field.itemType = getBaseType(items.type) as ItemType;
      field.itemNullable = isNullableType(items.type);
      field.itemDescription = typeof items.description === "string" ? items.description : "";
      field.itemSchemaTemplate = items;
      if (field.itemType === "object" && items.properties && depth < 2) {
        field.itemChildren = parseSchemaToFields(items as Record<string, unknown>, depth + 1);
      }
    }
    return field;
  });
}

function fieldsToSchema(
  fields: FieldDef[],
  template: Record<string, unknown> = {},
): Record<string, unknown> {
  const properties: Record<string, unknown> = {};
  const required: string[] = [];
  for (const f of fields) {
    if (!f.name.trim()) continue;
    let prop: Record<string, unknown> = {
      ...f.schemaTemplate,
      type: toSchemaType(f.type, f.nullable),
    };
    if (f.description.trim()) {
      prop.description = f.description;
    } else {
      delete prop.description;
    }
    if (f.type === "string" && f.enumValues.length > 0) {
      prop.enum = f.nullable ? [...f.enumValues, null] : [...f.enumValues];
    } else if (f.type === "string") {
      delete prop.enum;
    }
    if (f.type === "object") {
      prop = {
        ...fieldsToSchema(f.children, prop),
        type: toSchemaType(f.type, f.nullable),
      };
      if (Object.keys(f.schemaTemplate).length === 0) {
        prop.additionalProperties = false;
      }
    }
    if (f.type === "array") {
      let items: Record<string, unknown> = {
        ...f.itemSchemaTemplate,
        type: toSchemaType(f.itemType, f.itemNullable),
      };
      if (f.itemDescription.trim()) {
        items.description = f.itemDescription;
      } else {
        delete items.description;
      }
      if (f.itemType === "object") {
        items = {
          ...fieldsToSchema(f.itemChildren, items),
          type: toSchemaType(f.itemType, f.itemNullable),
        };
        if (Object.keys(f.itemSchemaTemplate).length === 0) {
          items.additionalProperties = false;
        }
      }
      prop.items = items;
    }
    properties[f.name] = prop;
    if (f.required) required.push(f.name);
  }

  const schema: Record<string, unknown> = {
    ...template,
    type: hasOwn(template, "type") ? template.type : "object",
    properties,
  };
  if (required.length > 0) {
    schema.required = required;
  } else {
    delete schema.required;
  }
  if (Object.keys(template).length === 0) {
    schema.additionalProperties = false;
  }
  return schema;
}

function readVisualType<T extends string>(
  typeValue: unknown,
  allowedTypes: readonly T[],
): { type: T; nullable: boolean } | null {
  if (typeof typeValue === "string") {
    return allowedTypes.includes(typeValue as T)
      ? { type: typeValue as T, nullable: false }
      : null;
  }
  if (!Array.isArray(typeValue)) return null;

  const uniqueTypes = new Set(typeValue);
  const concreteTypes = typeValue.filter((value) => value !== "null");
  if (
    uniqueTypes.size !== typeValue.length ||
    typeValue.length !== 2 ||
    !uniqueTypes.has("null") ||
    concreteTypes.length !== 1 ||
    typeof concreteTypes[0] !== "string" ||
    !allowedTypes.includes(concreteTypes[0] as T)
  ) {
    return null;
  }
  return { type: concreteTypes[0] as T, nullable: true };
}

function assessObjectSchema(
  schema: Record<string, unknown>,
  fieldDepth: number,
  path: string,
): VisualSchemaAssessment {
  const allowedObjectKeys = new Set(["type", "description", "properties", "required"]);
  let hasPreservedConstraints = Object.keys(schema).some((key) => !allowedObjectKeys.has(key));
  if (schema.description !== undefined && typeof schema.description !== "string") {
    return {
      reason: `${path}.description must be a string`,
      hasPreservedConstraints,
    };
  }
  const propertiesValue = schema.properties;
  if (propertiesValue !== undefined && !isRecord(propertiesValue)) {
    return {
      reason: `${path}.properties must be an object`,
      hasPreservedConstraints,
    };
  }
  const properties = propertiesValue ?? {};
  const propertyNames = new Set(Object.keys(properties));

  if (schema.required !== undefined) {
    if (
      !Array.isArray(schema.required) ||
      schema.required.some((name) => typeof name !== "string")
    ) {
      return {
        reason: `${path}.required must be an array of property names`,
        hasPreservedConstraints,
      };
    }
    const requiredNames = schema.required as string[];
    if (
      new Set(requiredNames).size !== requiredNames.length ||
      requiredNames.some((name) => !propertyNames.has(name))
    ) {
      return {
        reason: `${path}.required contains duplicate or unknown property names`,
        hasPreservedConstraints,
      };
    }
  }

  for (const [name, value] of Object.entries(properties)) {
    const fieldPath = `${path}.properties.${name}`;
    if (!isRecord(value)) {
      return {
        reason: `${fieldPath} must be an object schema`,
        hasPreservedConstraints,
      };
    }
    const typeInfo = readVisualType(value.type, FIELD_TYPES);
    if (!typeInfo) {
      return {
        reason: `${fieldPath}.type is not a single supported type with optional null`,
        hasPreservedConstraints,
      };
    }

    if (value.description !== undefined && typeof value.description !== "string") {
      return {
        reason: `${fieldPath}.description must be a string`,
        hasPreservedConstraints,
      };
    }
    const visibleKeys = new Set(["type", "description"]);
    if (typeInfo.type === "string" && value.enum !== undefined) {
      visibleKeys.add("enum");
      const enumValues = Array.isArray(value.enum) ? value.enum : [];
      const enumIncludesNull = enumValues.includes(null);
      if (
        !Array.isArray(value.enum) ||
        value.enum.length === 0 ||
        value.enum.some((entry) => typeof entry !== "string" && entry !== null) ||
        !value.enum.some((entry) => typeof entry === "string") ||
        new Set(value.enum).size !== value.enum.length ||
        enumIncludesNull !== typeInfo.nullable
      ) {
        return {
          reason: `${fieldPath}.enum cannot be represented without changing its null semantics`,
          hasPreservedConstraints,
        };
      }
    }

    if (typeInfo.type === "object") {
      if (fieldDepth >= 2) {
        return {
          reason: `${fieldPath} exceeds Visual mode's supported nesting depth`,
          hasPreservedConstraints,
        };
      }
      visibleKeys.add("properties");
      visibleKeys.add("required");
      const nested = assessObjectSchema(value, fieldDepth + 1, fieldPath);
      if (nested.reason) return nested;
      hasPreservedConstraints ||= nested.hasPreservedConstraints;
    }

    if (typeInfo.type === "array") {
      if (fieldDepth >= 2) {
        return {
          reason: `${fieldPath} exceeds Visual mode's supported nesting depth`,
          hasPreservedConstraints,
        };
      }
      visibleKeys.add("items");
      if (!isRecord(value.items)) {
        return {
          reason: `${fieldPath}.items must be a single object schema`,
          hasPreservedConstraints,
        };
      }
      const itemTypeInfo = readVisualType(value.items.type, ITEM_TYPES);
      if (!itemTypeInfo) {
        return {
          reason: `${fieldPath}.items.type is not a supported item type with optional null`,
          hasPreservedConstraints,
        };
      }
      if (
        value.items.description !== undefined &&
        typeof value.items.description !== "string"
      ) {
        return {
          reason: `${fieldPath}.items.description must be a string`,
          hasPreservedConstraints,
        };
      }
      if (itemTypeInfo.type === "object") {
        if (fieldDepth >= 1) {
          return {
            reason: `${fieldPath}.items exceeds Visual mode's supported nesting depth`,
            hasPreservedConstraints,
          };
        }
        const nestedItems = assessObjectSchema(
          value.items,
          fieldDepth + 1,
          `${fieldPath}.items`,
        );
        if (nestedItems.reason) return nestedItems;
        hasPreservedConstraints ||= nestedItems.hasPreservedConstraints;
      } else {
        hasPreservedConstraints ||= Object.keys(value.items).some(
          (key) => key !== "type" && key !== "description"
        );
      }
    }

    hasPreservedConstraints ||= Object.keys(value).some((key) => !visibleKeys.has(key));
  }

  return { reason: null, hasPreservedConstraints };
}

function assessVisualSchema(value: unknown): VisualSchemaAssessment {
  if (!isRecord(value)) {
    return { reason: "the schema root must be an object", hasPreservedConstraints: false };
  }
  const rootType = readVisualType(value.type, ["object"] as const);
  if (!rootType || rootType.nullable) {
    return {
      reason: "the schema root type must be object",
      hasPreservedConstraints: false,
    };
  }
  return assessObjectSchema(value, 0, "schema");
}

function findFieldNameIssue(fields: FieldDef[], path = "schema"): string | null {
  const names = new Set<string>();
  for (const field of fields) {
    if (!field.name.trim()) return `${path} contains an empty property name`;
    if (names.has(field.name)) return `${path} contains duplicate property name "${field.name}"`;
    names.add(field.name);

    if (field.type === "object") {
      const nestedIssue = findFieldNameIssue(field.children, `${path}.${field.name}`);
      if (nestedIssue) return nestedIssue;
    }
    if (field.type === "array" && field.itemType === "object") {
      const itemIssue = findFieldNameIssue(field.itemChildren, `${path}.${field.name}.items`);
      if (itemIssue) return itemIssue;
    }
  }
  return null;
}

type FieldCollection = "children" | "itemChildren";

interface AddTarget {
  parentId: string | null;
  collection: FieldCollection | "root";
  depth: number;
}

function cloneField(field: FieldDef): FieldDef {
  return {
    ...field,
    enumValues: [...field.enumValues],
    children: field.children.map(cloneField),
    itemChildren: field.itemChildren.map(cloneField),
    schemaTemplate: { ...field.schemaTemplate },
    itemSchemaTemplate: { ...field.itemSchemaTemplate },
  };
}

function updateFieldById(
  fields: FieldDef[],
  fieldId: string,
  update: (field: FieldDef) => FieldDef,
): FieldDef[] {
  return fields.map((field) => {
    if (field.id === fieldId) return update(field);
    return {
      ...field,
      children: updateFieldById(field.children, fieldId, update),
      itemChildren: updateFieldById(field.itemChildren, fieldId, update),
    };
  });
}

function removeFieldById(fields: FieldDef[], fieldId: string): FieldDef[] {
  return fields
    .filter((field) => field.id !== fieldId)
    .map((field) => ({
      ...field,
      children: removeFieldById(field.children, fieldId),
      itemChildren: removeFieldById(field.itemChildren, fieldId),
    }));
}

function typeLabel(field: FieldDef): string {
  const nullable = field.nullable ? " | null" : "";
  if (field.type !== "array") return `${field.type}${nullable}`;
  const itemType = field.itemType === "object" ? "obj" : field.itemType;
  const itemNullable = field.itemNullable ? " | null" : "";
  return `array[${itemType}${itemNullable}]${nullable}`;
}

function EditIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 20h4.2L19 9.2 14.8 5 4 15.8V20Zm2-3.4 8.8-8.8 1.4 1.4L7.4 18H6v-1.4ZM17.8 3.4a2 2 0 0 1 2.8 2.8l-1.2 1.2-4.2-4.2 1.2-1.2 1.4 1.4Z" />
    </svg>
  );
}

function DeleteIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M8 4V2h8v2h5v2h-2l-1 16H6L5 6H3V4h5Zm2 0h4V3h-4v1ZM7 6l.9 14h8.2L17 6H7Zm3 2h2v10h-2V8Zm4 0h2v10h-2V8Z" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="m6.4 5 5.6 5.6L17.6 5 19 6.4 13.4 12l5.6 5.6-1.4 1.4-5.6-5.6L6.4 19 5 17.6l5.6-5.6L5 6.4 6.4 5Z" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6V5Z" />
    </svg>
  );
}

function ModalShell({
  titleId,
  descriptionId,
  alert = false,
  onClose,
  children,
}: {
  titleId: string;
  descriptionId?: string;
  alert?: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const modalRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const preferredFocus = modalRef.current?.querySelector<HTMLElement>("[data-autofocus]");
    const focusable = preferredFocus ?? modalRef.current?.querySelector<HTMLElement>(
      "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled])",
    );
    focusable?.focus();

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !modalRef.current) return;
      const controls = Array.from(
        modalRef.current.querySelectorAll<HTMLElement>(
          "button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled])",
        ),
      );
      if (controls.length === 0) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [onClose]);

  return createPortal(
    <div
      className="sb-modal-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={modalRef}
        className={`sb-modal ${alert ? "sb-confirm-modal" : ""}`}
        role={alert ? "alertdialog" : "dialog"}
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
      >
        {children}
      </div>
    </div>,
    document.body,
  );
}

function EnumEditor({
  values,
  onChange,
  fieldId,
}: {
  values: string[];
  onChange: (values: string[]) => void;
  fieldId: string;
}) {
  const [input, setInput] = useState("");
  const inputId = `schema-field-${fieldId}-enum-value`;
  const add = () => {
    const value = input.trim();
    if (value && !values.includes(value)) onChange([...values, value]);
    setInput("");
  };

  return (
    <div className="sb-modal-enum">
      <label htmlFor={inputId}>Allowed values</label>
      <div className="sb-enum-row">
        {values.map((value) => (
          <span key={value} className="sb-enum-chip">
            {value}
            <button
              type="button"
              onClick={() => onChange(values.filter((entry) => entry !== value))}
              aria-label={`Remove allowed value ${value}`}
            >
              &times;
            </button>
          </span>
        ))}
        <input
          id={inputId}
          name={inputId}
          className="sb-enum-input"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              add();
            }
          }}
          onBlur={add}
          placeholder="Add a value…"
        />
      </div>
    </div>
  );
}

function FieldEditorModal({
  mode,
  initialField,
  depth,
  onCancel,
  onSave,
}: {
  mode: "add" | "edit";
  initialField: FieldDef;
  depth: number;
  onCancel: () => void;
  onSave: (field: FieldDef) => boolean;
}) {
  const [draft, setDraft] = useState(() => cloneField(initialField));
  const [nameError, setNameError] = useState("");
  const titleId = `schema-field-modal-${draft.id}-title`;
  const descriptionId = `schema-field-modal-${draft.id}-description`;
  const typeOptions: FieldType[] = depth >= 2
    ? ["string", "number", "integer", "boolean"]
    : FIELD_TYPES;
  const itemTypeOptions: ItemType[] = depth >= 1
    ? ["string", "number", "integer", "boolean"]
    : ITEM_TYPES;
  const update = (partial: Partial<FieldDef>) => setDraft((current) => ({ ...current, ...partial }));

  const save = () => {
    if (!draft.name.trim()) {
      setNameError("Enter a field name.");
      return;
    }
    const nextDraft = { ...draft, name: draft.name.trim() };
    if (!onSave(nextDraft)) {
      setNameError("Field names must be unique within the same object.");
    }
  };

  return (
    <ModalShell
      titleId={titleId}
      descriptionId={descriptionId}
      onClose={onCancel}
    >
      <div className="sb-modal-header">
        <div>
          <h2 id={titleId}>{mode === "edit" ? "Edit field" : "Add field"}</h2>
          <p id={descriptionId}>
            {mode === "edit"
              ? <>Editing <code>{initialField.name}</code> field</>
              : "Define the field that should appear in the structured output."}
          </p>
        </div>
        <button
          type="button"
          className="sb-modal-close"
          onClick={onCancel}
          aria-label="Close field editor"
        >
          <CloseIcon />
        </button>
      </div>

      <form
        className="sb-modal-form"
        onSubmit={(event) => {
          event.preventDefault();
          save();
        }}
      >
        <label className="sb-form-field" htmlFor={`schema-field-${draft.id}-name`}>
          <span>Field name</span>
          <input
            id={`schema-field-${draft.id}-name`}
            name={`schema-field-${draft.id}-name`}
            value={draft.name}
            onChange={(event) => {
              update({ name: event.target.value });
              setNameError("");
            }}
            placeholder="field_name"
            autoComplete="off"
            data-autofocus
            aria-invalid={Boolean(nameError)}
            aria-describedby={nameError ? `schema-field-${draft.id}-name-error` : undefined}
          />
          {nameError && (
            <small id={`schema-field-${draft.id}-name-error`} className="sb-form-error">
              {nameError}
            </small>
          )}
        </label>

        <label className="sb-form-field" htmlFor={`schema-field-${draft.id}-type`}>
          <span>Field type</span>
          <select
            id={`schema-field-${draft.id}-type`}
            name={`schema-field-${draft.id}-type`}
            value={draft.type}
            onChange={(event) => {
              const type = event.target.value as FieldType;
              update({
                type,
                enumValues: type === "string" ? draft.enumValues : [],
                children: type === "object" ? draft.children : [],
                itemType: type === "array" ? draft.itemType : "string",
                itemChildren: type === "array" ? draft.itemChildren : [],
              });
            }}
          >
            {typeOptions.map((type) => <option key={type} value={type}>{type}</option>)}
          </select>
        </label>

        {draft.type === "array" && (
          <label className="sb-form-field" htmlFor={`schema-field-${draft.id}-item-type`}>
            <span>Type of array items</span>
            <select
              id={`schema-field-${draft.id}-item-type`}
              name={`schema-field-${draft.id}-item-type`}
              value={draft.itemType}
              onChange={(event) => {
                const itemType = event.target.value as ItemType;
                update({
                  itemType,
                  itemChildren: itemType === "object" ? draft.itemChildren : [],
                });
              }}
            >
              {itemTypeOptions.map((type) => <option key={type} value={type}>{type}</option>)}
            </select>
          </label>
        )}

        <label className="sb-form-field" htmlFor={`schema-field-${draft.id}-description`}>
          <span>Description</span>
          <textarea
            id={`schema-field-${draft.id}-description`}
            name={`schema-field-${draft.id}-description`}
            value={draft.description}
            onChange={(event) => update({ description: event.target.value })}
            placeholder="Describe what this field should contain…"
            rows={4}
          />
        </label>

        {draft.type === "array" && (
          <label className="sb-form-field" htmlFor={`schema-field-${draft.id}-item-description`}>
            <span>Array item description</span>
            <textarea
              id={`schema-field-${draft.id}-item-description`}
              name={`schema-field-${draft.id}-item-description`}
              value={draft.itemDescription}
              onChange={(event) => update({ itemDescription: event.target.value })}
              placeholder="Describe each array item…"
              rows={3}
            />
          </label>
        )}

        {draft.type === "string" && (
          <EnumEditor
            values={draft.enumValues}
            onChange={(enumValues) => update({ enumValues })}
            fieldId={draft.id}
          />
        )}

        <div className="sb-modal-options" aria-label="Field options">
          <label>
            <input
              type="checkbox"
              checked={draft.required}
              onChange={(event) => update({ required: event.target.checked })}
            />
            <span>
              <strong>Required</strong>
              Property must appear in the output.
            </span>
          </label>
          <label>
            <input
              type="checkbox"
              checked={draft.nullable}
              onChange={(event) => update({ nullable: event.target.checked })}
            />
            <span>
              <strong>Nullable</strong>
              Field may explicitly return null.
            </span>
          </label>
          {draft.type === "array" && (
            <label>
              <input
                type="checkbox"
                checked={draft.itemNullable}
                onChange={(event) => update({ itemNullable: event.target.checked })}
              />
              <span>
                <strong>Nullable items</strong>
                Individual array entries may be null.
              </span>
            </label>
          )}
        </div>

        <div className="sb-modal-footer">
          <button type="button" className="sb-button sb-button-secondary" onClick={onCancel}>
            Cancel
          </button>
          <button type="submit" className="sb-button sb-button-primary">
            {mode === "edit" ? "Save changes" : "Add field"}
          </button>
        </div>
      </form>
    </ModalShell>
  );
}

function DeleteFieldModal({
  field,
  onCancel,
  onConfirm,
}: {
  field: FieldDef;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  const titleId = `delete-field-${field.id}-title`;
  const descriptionId = `delete-field-${field.id}-description`;
  return (
    <ModalShell
      titleId={titleId}
      descriptionId={descriptionId}
      alert
      onClose={onCancel}
    >
      <div className="sb-modal-header">
        <div>
          <h2 id={titleId}>Delete field?</h2>
          <p id={descriptionId}>
            <code>{field.name}</code> and any nested fields will be removed from the output schema.
          </p>
        </div>
      </div>
      <div className="sb-modal-footer">
        <button type="button" className="sb-button sb-button-secondary" onClick={onCancel}>
          Cancel
        </button>
        <button type="button" className="sb-button sb-button-danger" onClick={onConfirm}>
          Delete field
        </button>
      </div>
    </ModalShell>
  );
}

function AddFieldButton({ onClick, label = "Add new field", disabled }: {
  onClick: () => void;
  label?: string;
  disabled: boolean;
}) {
  return (
    <button type="button" className="sb-add-btn" onClick={onClick} disabled={disabled}>
      <PlusIcon />
      {label}
    </button>
  );
}

function FieldTree({
  fields,
  depth,
  disabled,
  onEdit,
  onDelete,
  onAdd,
}: {
  fields: FieldDef[];
  depth: number;
  disabled: boolean;
  onEdit: (field: FieldDef, depth: number) => void;
  onDelete: (field: FieldDef) => void;
  onAdd: (target: AddTarget) => void;
}) {
  return (
    <div className={`sb-field-list ${depth > 0 ? "is-nested" : ""}`}>
      {fields.map((field) => {
        const childCollection: FieldCollection | null = field.type === "object"
          ? "children"
          : field.type === "array" && field.itemType === "object"
            ? "itemChildren"
            : null;
        const childFields = childCollection ? field[childCollection] : [];
        return (
          <div className="sb-field-block" key={field.id}>
            <div className="sb-field-card">
              <div className="sb-field-content">
                <div className="sb-field-heading">
                  <code>{field.name}</code>
                  <span className="sb-type-badge">{typeLabel(field)}</span>
                  {field.required && <span className="sb-required-badge">required</span>}
                </div>
                <p className={field.description.trim() ? "" : "is-empty"}>
                  {field.description.trim() || "No description provided"}
                </p>
              </div>
              <div className="sb-field-actions" aria-label={`Actions for ${field.name}`}>
                <button
                  type="button"
                  onClick={() => onEdit(field, depth)}
                  aria-label={`Edit field ${field.name}`}
                  title={`Edit ${field.name}`}
                  disabled={disabled}
                >
                  <EditIcon />
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(field)}
                  aria-label={`Delete field ${field.name}`}
                  title={`Delete ${field.name}`}
                  disabled={disabled}
                >
                  <DeleteIcon />
                </button>
              </div>
            </div>

            {childCollection && (
              <div className="sb-children">
                <FieldTree
                  fields={childFields}
                  depth={depth + 1}
                  disabled={disabled}
                  onEdit={onEdit}
                  onDelete={onDelete}
                  onAdd={onAdd}
                />
                <AddFieldButton
                  label={childFields.length > 0 ? "Add nested field" : "Add first nested field"}
                  onClick={() => onAdd({
                    parentId: field.id,
                    collection: childCollection,
                    depth: depth + 1,
                  })}
                  disabled={disabled}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function SchemaBuilder({
  schemaText,
  onChange,
  disabled = false,
}: {
  schemaText: string;
  onChange: (s: string) => void;
  disabled?: boolean;
}) {
  const [fields, setFields] = useState<FieldDef[]>([]);
  const [schemaDescription, setSchemaDescription] = useState("");
  const [parseError, setParseError] = useState("");
  const [readOnlyReason, setReadOnlyReason] = useState("");
  const [editError, setEditError] = useState("");
  const [editingField, setEditingField] = useState<{ field: FieldDef; depth: number } | null>(null);
  const [addTarget, setAddTarget] = useState<AddTarget | null>(null);
  const [deletingField, setDeletingField] = useState<FieldDef | null>(null);
  const lastSerialized = useRef("");
  const schemaTemplate = useRef<Record<string, unknown>>({});

  useEffect(() => {
    if (schemaText === lastSerialized.current) return;
    try {
      const parsedSchema: unknown = JSON.parse(schemaText);
      const assessment = assessVisualSchema(parsedSchema);
      schemaTemplate.current = isRecord(parsedSchema) ? parsedSchema : {};
      setSchemaDescription(
        isRecord(parsedSchema) && typeof parsedSchema.description === "string"
          ? parsedSchema.description
          : ""
      );
      setReadOnlyReason(assessment.reason ?? "");
      setFields(
        assessment.reason || !isRecord(parsedSchema)
          ? []
          : parseSchemaToFields(parsedSchema),
      );
      setParseError("");
      setEditError("");
    } catch {
      setFields([]);
      setSchemaDescription("");
      setParseError("Could not parse the JSON schema. Use JSON mode to fix it.");
      setReadOnlyReason("");
      setEditError("");
    }
  }, [schemaText]);

  const emitChange = useCallback(
    (newFields: FieldDef[], nextSchemaDescription = schemaDescription) => {
      const fieldIssue = findFieldNameIssue(newFields);
      if (fieldIssue) {
        setEditError(`${fieldIssue}. The JSON schema has not been changed.`);
        return false;
      }
      setFields(newFields);
      setEditError("");
      const nextSchema = fieldsToSchema(newFields, schemaTemplate.current);
      if (nextSchemaDescription.trim()) {
        nextSchema.description = nextSchemaDescription;
      } else {
        delete nextSchema.description;
      }
      const json = JSON.stringify(nextSchema, null, 2);
      lastSerialized.current = json;
      onChange(json);
      return true;
    },
    [onChange, schemaDescription]
  );

  const visualUnavailable = Boolean(parseError || readOnlyReason);

  return (
    <div className="sb-container" aria-disabled={disabled || visualUnavailable}>
      {parseError && <div className="sb-parse-error" role="alert">{parseError}</div>}
      {readOnlyReason && (
        <div className="sb-parse-error" role="alert">
          Visual editing is unavailable because {readOnlyReason}. The schema has not been
          changed; use JSON mode to edit it without losing information.
        </div>
      )}
      {editError && <div className="sb-parse-error" role="alert">{editError}</div>}
      {!visualUnavailable && (
        <>
          {fields.length === 0 && (
            <div className="sb-empty-state">
              <strong>No fields defined</strong>
              <span>Add a field to describe the structured output.</span>
            </div>
          )}
          <FieldTree
            fields={fields}
            depth={0}
            disabled={disabled}
            onEdit={(field, depth) => setEditingField({ field, depth })}
            onDelete={setDeletingField}
            onAdd={setAddTarget}
          />
          <AddFieldButton
            onClick={() => setAddTarget({ parentId: null, collection: "root", depth: 0 })}
            disabled={disabled}
          />
        </>
      )}

      {editingField && (
        <FieldEditorModal
          mode="edit"
          initialField={editingField.field}
          depth={editingField.depth}
          onCancel={() => setEditingField(null)}
          onSave={(updatedField) => {
            const saved = emitChange(
              updateFieldById(fields, editingField.field.id, () => updatedField),
            );
            if (saved) setEditingField(null);
            return saved;
          }}
        />
      )}

      {addTarget && (
        <FieldEditorModal
          mode="add"
          initialField={newField()}
          depth={addTarget.depth}
          onCancel={() => setAddTarget(null)}
          onSave={(field) => {
            let nextFields: FieldDef[];
            if (addTarget.collection === "root") {
              nextFields = [...fields, field];
            } else {
              const collection = addTarget.collection;
              nextFields = updateFieldById(fields, addTarget.parentId as string, (parent) => ({
                ...parent,
                [collection]: [...parent[collection], field],
              }));
            }
            const saved = emitChange(nextFields);
            if (saved) setAddTarget(null);
            return saved;
          }}
        />
      )}

      {deletingField && (
        <DeleteFieldModal
          field={deletingField}
          onCancel={() => setDeletingField(null)}
          onConfirm={() => {
            emitChange(removeFieldById(fields, deletingField.id));
            setDeletingField(null);
          }}
        />
      )}
    </div>
  );
}
