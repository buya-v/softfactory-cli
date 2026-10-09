import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { Factory } from "./Factory.js";

const factory: any = new Factory("/tmp", "", { strategy: "local" });

describe("test writer scope", () => {
  it("allows a test writer that touches only test files", () => {
    const diff = [
      "diff --git a/src/a.test.ts b/src/a.test.ts",
      "--- a/src/a.test.ts",
      "+++ b/src/a.test.ts",
      "@@ -0,0 +1 @@",
      '+it("works", () => expect(true).toBe(true));',
      "",
    ].join("\n");
    assert.equal(
      factory.testWriterScopeViolation({ agent_role: "test_writer" }, diff),
      null,
    );
  });

  it("names the source file a test writer changed", () => {
    const diff = [
      "diff --git a/src/a.test.ts b/src/a.test.ts",
      "--- a/src/a.test.ts",
      "+++ b/src/a.test.ts",
      "@@ -0,0 +1 @@",
      '+it("works", () => expect(true).toBe(true));',
      "diff --git a/src/a.ts b/src/a.ts",
      "--- a/src/a.ts",
      "+++ b/src/a.ts",
      "@@ -1,1 +1,1 @@",
      "-export const a = 1;",
      "+export const a = 2;",
      "",
    ].join("\n");
    const message = factory.testWriterScopeViolation(
      { agent_role: "test_writer" },
      diff,
    );
    assert.ok(typeof message === "string" && message.length > 0);
    assert.match(message, /src\/a\.ts/);
  });

  it("returns null for a coder task touching source", () => {
    const diff = [
      "diff --git a/src/a.ts b/src/a.ts",
      "--- a/src/a.ts",
      "+++ b/src/a.ts",
      "@@ -1,1 +1,1 @@",
      "-export const a = 1;",
      "+export const a = 2;",
      "",
    ].join("\n");
    assert.equal(
      factory.testWriterScopeViolation({ agent_role: "coder" }, diff),
      null,
    );
  });
});
