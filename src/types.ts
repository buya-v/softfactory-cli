import { z } from 'zod';

export const TaskSchema = z.object({
  id: z.string().regex(/^[A-Za-z0-9][A-Za-z0-9._-]*$/, 'Task IDs must be safe path components'),
  title: z.string(),
  description: z.string(),
  agent_role: z.enum(['coder', 'test_writer', 'docs', 'devops', 'architect', 'builder']),
  phase: z.number(),
  dependencies: z.array(z.string()).default([]),
  blockedBy: z.array(z.string()).default([]),
  files_hint: z.array(z.string()).default([]),
  creates_files: z.array(z.string()).default([]),
  acceptance_criteria: z.array(z.string()).default([]),
  verification: z.array(z.string()).default([]),
  /**
   * Who performs the task. Only "agent" tasks reach a worker. "orchestrator"
   * is host or live-system work (secrets, deploys, host config, pushes) and
   * "user" is a decision; both pause the run until --mark-done.
   */
  executor: z.enum(['agent', 'orchestrator', 'user']).default('agent'),
  /** Touches a path in .softfactory/protected-paths.txt: never the light tier, always the blind review. */
  protected: z.boolean().default(false),
  /** Questions the blind reviewer will verify; the worker answers each with file:line. */
  review_questions: z.array(z.string()).default([]),
  /** Relative path of the acceptance evidence log for the latest attempt. */
  evidence: z.string().optional(),
  /** What the acceptance command did against the base branch at planning time. */
  acceptance_baseline: z.object({ exit_code: z.number().nullable(), summary: z.string() }).optional(),
  status: z.enum(['pending', 'active', 'done', 'failed', 'rejected', 'blocked', 'manual']).default('pending'),
  worktree_branch: z.string().optional(),
  attempts: z.number().default(0),
  escalation_level: z.number().default(0),
  error_history: z.array(z.string()).default([]),
  last_feedback: z.string().nullish(),
  model: z.string().optional(),
  provider: z.string().optional(),
  worker_type: z.enum(['local', 'cloud', 'auto']).default('auto'),
  production_targets: z.array(z.string()).default([]),
  test_targets: z.array(z.string()).default([]),
  behavior_tags: z.array(z.string()).default([]),
  verification_command: z.string().optional(),
  complexity: z.enum(['simple', 'moderate', 'complex']).optional(),
  allocation_reason: z.string().optional(),
  /** A pointer into a shared spec, such as "docs/api.md §3". */
  spec: z.string().optional(),
  /** Things the worker must not touch, even if they seem related. */
  out_of_scope: z.array(z.string()).default([]),
  /** Invariants and hints the planner does not want to lose in the task description. */
  notes_to_executor: z.array(z.string()).default([]),
  replan_reason: z.string().optional(),
  /** How the last merge decision was reached: a model reviewed the diff, or only mechanical gates ran. */
  inspection: z.enum(['model', 'mechanical-only', 'blind-review']).optional(),
  inspector_model: z.string().optional(),
  /** One mechanical review fix per task is allowed without charging an attempt; this records that it was used. */
  micro_fix_used: z.boolean().default(false),
  /** The worker's own summary of the attempt that passed, for the handoff index. */
  handoff_summary: z.string().optional(),
});

export type Task = z.infer<typeof TaskSchema>;

/**
 * Bumped when factory.json gains a field an older softfactory would misread.
 * 2: executor routing, protected tasks, acceptance evidence, manual/paused.
 */
export const CURRENT_SCHEMA_VERSION = 2;

export const FactoryStateSchema = z.object({
  schema_version: z.number().int().positive().default(1),
  feature_id: z.string().default(''),
  feature: z.string(),
  /** The user's requirement, verbatim, so every attempt (and every resume) can hand it to the worker. */
  requirement: z.string().default(''),
  /** A spec, value table or rename map shared by every task in this run, written once. */
  brief: z.string().default(''),
  /** Exports with no call site when the run started (file#name); the final gate refuses any new one. */
  wiring_baseline: z.array(z.string()).optional(),
  status: z.enum(['planning', 'planned', 'active', 'paused', 'completed', 'failed']),
  strategy: z.enum(['hybrid', 'local', 'cloud']).default('hybrid'),
  cloud_provider: z.enum(['agy', 'openai', 'openai-api', 'anthropic', 'deepseek', 'qwen', 'openai-compat', 'claude-code', 'gemini-cli', 'muse']).default('agy'),
  planner_provider: z.enum(['openai', 'agy', 'local', 'deepseek', 'anthropic', 'openai-api', 'qwen', 'openai-compat', 'claude-code', 'gemini-cli', 'muse']).default('openai'),
  created_at: z.string(),
  last_updated: z.string(),
  state_revision: z.number().int().nonnegative().default(0),
  runtime: z.object({
    strategy: z.enum(['hybrid', 'local', 'cloud']),
    local_model: z.string().optional(),
    cloud_provider: z.string().optional(),
    cloud_model: z.string().optional(),
    frontier_model: z.string().optional(),
  }).optional(),
  tasks: z.array(TaskSchema),
  memory_patterns: z.array(z.string()).default([]),
  /** Tokens the run has consumed across every attempt, where backends report them. */
  usage: z.object({ input_tokens: z.number().default(0), output_tokens: z.number().default(0) }).default({ input_tokens: 0, output_tokens: 0 }),
  /** Tokens consumed per model, where backends report them. Optional so old state files still parse. */
  usage_by_model: z.record(z.object({ input: z.number(), output: z.number() })).optional(),
  /** The outcome of the run's UAT gate: passed, failed, skipped (--no-uat) or none (no UAT defined). Optional for old state files. */
  uat_result: z.enum(['passed', 'failed', 'skipped', 'none']).optional(),
  /** Where the requirement came from, when it was not typed on the command line. */
  source: z.object({ kind: z.literal('issue'), number: z.number(), url: z.string(), title: z.string() }).optional(),
  /** PR mode: passing tasks merge here instead of the base branch, and the branch becomes a pull request. */
  integration_branch: z.string().optional(),
  base_branch: z.string().optional(),
  pull_request: z.object({ number: z.number(), url: z.string(), branch: z.string() }).optional(),
});

/**
 * The schema supplies `brief` a default, but state objects are also assembled
 * by hand before the schema runs, so the in-memory type keeps it optional.
 */
export type FactoryState = Omit<z.infer<typeof FactoryStateSchema>, 'brief'> & { brief?: string };

export const LessonSchema = z.object({
  id: z.string(), lesson: z.string().min(12).max(500), tags: z.array(z.string()).max(8), scope: z.enum(['project', 'language', 'path']).default('project'), confidence: z.number().min(0).max(1),
  evidence: z.object({ taskId: z.string(), failure: z.string().max(500), resolution: z.string().max(500) }),
  /** The gate or command that enforces the lesson. Softhouse: a rule nobody checks is recorded again and recurs. */
  check: z.string().max(240).optional(),
  createdAt: z.string(), lastUsedAt: z.string().optional(), useCount: z.number().default(0), status: z.enum(['active', 'superseded', 'rejected', 'retired']).default('active'),
});
export type Lesson = z.infer<typeof LessonSchema>;

export const FeatureHistoryEntrySchema = z.object({
  feature_id: z.string(),
  feature: z.string(),
  status: z.string(),
  created_at: z.string(),
  completed_at: z.string().optional(),
  task_count: z.number(),
  tasks_done: z.number(),
});

export type FeatureHistoryEntry = z.infer<typeof FeatureHistoryEntrySchema>;

export const HandoffSchema = z.object({
  task_id: z.string(),
  status: z.enum(['success', 'failure']),
  files_changed: z.array(z.string()),
  test_summary: z.string().optional(),
  unresolved_issues: z.string().optional(),
  summary: z.string(),
  /** One answer per review question, each citing file:line. */
  review_answers: z.array(z.string()).optional(),
  /**
   * Material claims with where they can be checked: "path:line", "path:start-end",
   * "command: <what ran>" or "UNVERIFIED". A path:line that does not exist is a
   * fabricated source and rejects the task (Softhouse's honesty tags).
   */
  claims: z.array(z.object({ text: z.string(), source: z.string() })).optional(),
  /** Parts of the task the worker itself reports as unfinished; a non-empty list of acceptance work rejects the task. */
  not_done: z.array(z.string()).optional(),
});

export type Handoff = z.infer<typeof HandoffSchema>;

export interface TaskResult {
  task: Task;
  branchName: string;
}

export type FactoryEvent =
  | { type: 'task:start'; task: Task; worker: string; model: string }
  | { type: 'task:done'; task: Task }
  | { type: 'task:fail'; task: Task; feedback?: string }
  | { type: 'verifier:start'; taskId: string }
  | { type: 'verifier:done'; taskId: string; passed: boolean; testSummary: string; output: string }
  | { type: 'state:update'; state: FactoryState }
  | { type: 'worker:output'; taskId: string; line: string }
  | { type: 'phase:start'; phase: number }
  | { type: 'task:auto-replan'; task_id: string; reason: string; replacements: string[] }
  | { type: 'task:auto-replan-failed'; task_id: string; reason: string; error: string }
  | { type: 'task:manual'; task: Task }
  | { type: 'build:complete'; status: 'completed' | 'failed' | 'planned' | 'paused'; feature: string };
