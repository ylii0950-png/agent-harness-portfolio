"use strict";

async function reference() {
  const section = document.getElementById("endpoints");
  try {
    const response = await fetch("/openapi.json");
    if (!response.ok) throw new Error();
    const spec = await response.json();
    for (const [path, methods] of Object.entries(spec.paths)) {
      for (const [method, operation] of Object.entries(methods)) {
        const endpoint = document.createElement("details");
        const title = document.createElement("summary");
        title.textContent = `${method.toUpperCase()} ${path} · ${operation.summary}`;
        const schema = document.createElement("pre");
        schema.textContent = JSON.stringify(operation, null, 2);
        endpoint.append(title, schema);
        section.append(endpoint);
      }
    }
    const schemas = document.createElement("pre");
    schemas.textContent = JSON.stringify(spec.components.schemas, null, 2);
    section.append(schemas);
  } catch {
    section.append("OpenAPI schema unavailable.");
  }
}
reference();
