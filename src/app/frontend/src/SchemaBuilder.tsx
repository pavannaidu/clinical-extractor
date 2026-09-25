import { useState, useEffect, useRef, useCallback } from "react";
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

function EnumEditor({
  values,
  onChange,
  fieldId,
  fieldLabel,
  fieldName,
  disabled,
}: {
  values: string[];
  onChange: (v: string[]) => void;
  fieldId: string;
  fieldLabel: string;
  fieldName: string;
  disabled: boolean;
}) {
  const [input, setInput] = useState("");
  const inputId = `schema-field-${fieldId}-enum-value`;
  const add = () => {
    const v = input.trim();
    if (v && !values.includes(v)) onChange([...values, v]);
    setInput("");
  };
  return (
    <div className="sb-enum-row">
      {values.map((v) => (
        <span key={v} className="sb-enum-chip">
          {v}
          <button
            type="button"
            onClick={() => onChange(values.filter((x) => x !== v))}
            aria-label={`Remove enum value ${v} from ${fieldLabel}: ${fieldName || "unnamed field"}`}
            disabled={disabled}
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
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), add())}
        placeholder="Add enum..."
        aria-label={`Add enum value for ${fieldLabel}: ${fieldName || "unnamed field"}`}
        disabled={disabled}
      />
    </div>
  );
}

function DescriptionEditor({
  id,
  value,
  onChange,
  label,
  actionLabel,
  placeholder,
  disabled,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
  label: string;
  actionLabel: string;
  placeholder: string;
  disabled: boolean;
}) {
  const [expanded, setExpanded] = useState(Boolean(value.trim()));

  useEffect(() => {
    if (value.trim()) setExpanded(true);
  }, [value]);

  return (
    <div className={`sb-description-editor ${expanded ? "is-expanded" : ""}`}>
      <button
        type="button"
        className="sb-description-toggle"
        aria-expanded={expanded}
        aria-controls={id}
        onClick={() => setExpanded((current) => !current)}
        disabled={disabled}
      >
        {value.trim() ? actionLabel : `+ Add ${actionLabel.toLowerCase()}`}
      </button>
      {expanded && (
        <textarea
          id={id}
          name={id}
          className="sb-description-input"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          aria-label={label}
          disabled={disabled}
          rows={2}
        />
      )}
    </div>
  );
}

function FieldRow({
  field,
  depth,
  fieldLabel,
  disabled,
  onUpdate,
  onRemove,
}: {
  field: FieldDef;
  depth: number;
  fieldLabel: string;
  disabled: boolean;
  onUpdate: (f: FieldDef) => void;
  onRemove: () => void;
}) {
  const typeOptions: FieldType[] =
    depth >= 2
      ? ["string", "number", "integer", "boolean"]
      : ["string", "number", "integer", "boolean", "object", "array"];

  const itemTypeOptions: ItemType[] =
    depth >= 1
      ? ["string", "number", "integer", "boolean"]
      : ["string", "number", "integer", "boolean", "object"];

  const update = (partial: Partial<FieldDef>) => onUpdate({ ...field, ...partial });
  const controlId = (suffix: string) => `schema-field-${field.id}-${suffix}`;
  const requiredId = controlId("required");
  const nullableId = controlId("nullable");
  const nameId = controlId("name");
  const typeId = controlId("type");
  const itemTypeId = controlId("item-type");
  const itemNullableId = controlId("item-nullable");
  const descriptionId = controlId("description");
  const itemDescriptionId = controlId("item-description");

  return (
    <div className="sb-field-block">
      <div className="sb-field-row">
        <input
          id={nameId}
          name={nameId}
          className="sb-name-input"
          value={field.name}
          onChange={(e) => update({ name: e.target.value })}
          placeholder="field_name"
          aria-label={`${fieldLabel} name`}
          disabled={disabled}
        />
        <select
          id={typeId}
          name={typeId}
          className="sb-type-select"
          value={field.type}
          aria-label={`${fieldLabel} type: ${field.name || "unnamed field"}`}
          disabled={disabled}
          onChange={(e) => {
            const t = e.target.value as FieldType;
            update({
              type: t,
              enumValues: t === "string" ? field.enumValues : [],
              children: t === "object" ? field.children : [],
              itemType: t === "array" ? field.itemType : "string",
              itemChildren: t === "array" ? field.itemChildren : [],
            });
          }}
        >
          {typeOptions.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <button
          type="button"
          className="sb-remove-btn"
          onClick={onRemove}
          title="Remove field"
          aria-label={`Remove ${fieldLabel}: ${field.name || "unnamed field"}`}
          disabled={disabled}
        >
          &times;
        </button>
      </div>
      <div className="sb-field-details">
        <div className="sb-field-flags">
          <label
            className="sb-field-flag"
            htmlFor={requiredId}
            title="Require this property to be present in the output object"
          >
            <input
              id={requiredId}
              name={requiredId}
              type="checkbox"
              checked={field.required}
              onChange={(e) => update({ required: e.target.checked })}
              aria-label={`${fieldLabel} required: ${field.name || "unnamed field"}`}
              disabled={disabled}
            />
            Required
          </label>
          <label
            className="sb-field-flag"
            htmlFor={nullableId}
            title="Allow this property's value to be null"
          >
            <input
              id={nullableId}
              name={nullableId}
              type="checkbox"
              checked={field.nullable}
              onChange={(e) => update({ nullable: e.target.checked })}
              aria-label={`${fieldLabel} nullable: ${field.name || "unnamed field"}`}
              disabled={disabled}
            />
            Nullable
          </label>
        </div>
        <DescriptionEditor
          id={descriptionId}
          value={field.description}
          onChange={(description) => update({ description })}
          label={`${fieldLabel} description`}
          actionLabel="Description"
          placeholder="Describe what this field should contain…"
          disabled={disabled}
        />
      </div>

      {field.type === "string" && (
        <div className="sb-sub-section">
          <EnumEditor
            values={field.enumValues}
            onChange={(enumValues) => update({ enumValues })}
            fieldId={field.id}
            fieldLabel={fieldLabel}
            fieldName={field.name}
            disabled={disabled}
          />
        </div>
      )}

      {field.type === "object" && (
        <div className="sb-nested-block">
          {field.children.map((child, i) => (
            <FieldRow
              key={child.id}
              field={child}
              depth={depth + 1}
              fieldLabel={`${fieldLabel} sub-field ${i + 1}`}
              disabled={disabled}
              onUpdate={(updated) => {
                const next = [...field.children];
                next[i] = updated;
                update({ children: next });
              }}
              onRemove={() => update({ children: field.children.filter((_, j) => j !== i) })}
            />
          ))}
          <button
            type="button"
            className="sb-add-btn"
            onClick={() => update({ children: [...field.children, newField()] })}
            disabled={disabled}
          >
            + Add Sub-field
          </button>
        </div>
      )}

      {field.type === "array" && (
        <div className="sb-nested-block">
          <div className="sb-item-row">
            <label className="sb-item-label" htmlFor={itemTypeId}>Item type:</label>
            <select
              id={itemTypeId}
              name={itemTypeId}
              className="sb-type-select"
              value={field.itemType}
              aria-label={`${fieldLabel} item type: ${field.name || "unnamed field"}`}
              disabled={disabled}
              onChange={(e) => {
                const it = e.target.value as ItemType;
                update({
                  itemType: it,
                  itemChildren: it === "object" ? field.itemChildren : [],
                });
              }}
            >
              {itemTypeOptions.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
            <label
              className="sb-field-flag"
              htmlFor={itemNullableId}
              title="Allow individual array entries to be null"
            >
              <input
                id={itemNullableId}
                name={itemNullableId}
                type="checkbox"
                checked={field.itemNullable}
                onChange={(e) => update({ itemNullable: e.target.checked })}
                aria-label={`${fieldLabel} nullable item: ${field.name || "unnamed field"}`}
                disabled={disabled}
              />
              Nullable item
            </label>
          </div>
          <DescriptionEditor
            id={itemDescriptionId}
            value={field.itemDescription}
            onChange={(itemDescription) => update({ itemDescription })}
            label={`${fieldLabel} item description`}
            actionLabel="Item description"
            placeholder="Describe each array item…"
            disabled={disabled}
          />
          {field.itemType === "object" && (
            <div className="sb-nested-block">
              {field.itemChildren.map((child, i) => (
                <FieldRow
                  key={child.id}
                  field={child}
                  depth={depth + 1}
                  fieldLabel={`${fieldLabel} item field ${i + 1}`}
                  disabled={disabled}
                  onUpdate={(updated) => {
                    const next = [...field.itemChildren];
                    next[i] = updated;
                    update({ itemChildren: next });
                  }}
                  onRemove={() =>
                    update({ itemChildren: field.itemChildren.filter((_, j) => j !== i) })
                  }
                />
              ))}
              <button
                type="button"
                className="sb-add-btn"
                onClick={() => update({ itemChildren: [...field.itemChildren, newField()] })}
                disabled={disabled}
              >
                + Add Item Field
              </button>
            </div>
          )}
        </div>
      )}
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
  const [hasPreservedConstraints, setHasPreservedConstraints] = useState(false);
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
      setHasPreservedConstraints(assessment.hasPreservedConstraints);
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
      setHasPreservedConstraints(false);
    }
  }, [schemaText]);

  const emitChange = useCallback(
    (newFields: FieldDef[], nextSchemaDescription = schemaDescription) => {
      setFields(newFields);
      const fieldIssue = findFieldNameIssue(newFields);
      if (fieldIssue) {
        setEditError(`${fieldIssue}. The JSON schema has not been changed.`);
        return;
      }
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
    },
    [onChange, schemaDescription]
  );

  const visualUnavailable = Boolean(parseError || readOnlyReason);

  return (
    <div className="sb-container" aria-disabled={disabled || visualUnavailable}>
      <div className="sb-semantics-note" role="note">
        <strong>Required</strong> controls property presence. <strong>Nullable</strong> controls
        whether the value type accepts <code>null</code>.
      </div>
      {!visualUnavailable && (
        <DescriptionEditor
          id="schema-root-description"
          value={schemaDescription}
          onChange={(description) => {
            setSchemaDescription(description);
            emitChange(fields, description);
          }}
          label="Schema description"
          actionLabel="Schema description"
          placeholder="Describe the overall output contract…"
          disabled={disabled}
        />
      )}
      {parseError && <div className="sb-parse-error" role="alert">{parseError}</div>}
      {readOnlyReason && (
        <div className="sb-parse-error" role="alert">
          Visual editing is unavailable because {readOnlyReason}. The schema has not been
          changed; use JSON mode to edit it without losing information.
        </div>
      )}
      {editError && <div className="sb-parse-error" role="alert">{editError}</div>}
      {!visualUnavailable && hasPreservedConstraints && (
        <div className="sb-preserved-note" role="note">
          Constraints not shown here are preserved unchanged. Use JSON mode to edit them.
        </div>
      )}
      {!visualUnavailable && (
        <>
          {fields.map((field, i) => (
            <FieldRow
              key={field.id}
              field={field}
              depth={0}
              fieldLabel={`Field ${i + 1}`}
              disabled={disabled}
              onUpdate={(updated) => {
                const next = [...fields];
                next[i] = updated;
                emitChange(next);
              }}
              onRemove={() => emitChange(fields.filter((_, j) => j !== i))}
            />
          ))}
          <button
            type="button"
            className="sb-add-btn"
            onClick={() => emitChange([...fields, newField()])}
            disabled={disabled}
          >
            + Add Field
          </button>
        </>
      )}
    </div>
  );
}
