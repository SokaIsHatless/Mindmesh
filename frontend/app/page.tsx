"use client";

import { useEffect, useState, type KeyboardEvent } from "react";

type StepType =
  | "plan"
  | "check"
  | "no_tool"
  | "writing"
  | "testing"
  | "pass"
  | "fail"
  | "answer";

type Step = {
  type: StepType;
  label: string;
  detail?: string;
};

/* =========================================================
   MOCK AGENT LOGIC
   ---------------------------------------------------------
   This is the ONLY place containing mock agent behavior.

   INTEGRATION POINT (later):
   Replace mockRun(task) in runAgent() with a real backend call
   that returns Step[] in this same shape — ideally a one-line change.
   ========================================================= */
function mockRun(task: string): Step[] {
  const input = task.trim();

  // Scenario 2 — plain arithmetic (e.g. "2+5"): answer directly, no tool.
  const arithmeticPattern =
    /^\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?$/;

  if (arithmeticPattern.test(input)) {
    try {
      // Mock/demo only — evaluate simple arithmetic for the answer label.
      // eslint-disable-next-line no-new-func
      const result = Function(`"use strict"; return (${input})`)();

      return [
        {
          type: "plan",
          label: "This is trivial — I can answer directly",
        },
        {
          type: "answer",
          label: `Answer: ${result}`,
        },
      ];
    } catch {
      // Fall through to tool-writing scenario if eval fails.
    }
  }

  // Scenario 1 — described calculation (e.g. train speed): write a tool.
  return [
    {
      type: "plan",
      label: "Planning: this needs a speed calculation",
    },
    {
      type: "check",
      label: "Checking toolbox for a matching tool",
    },
    {
      type: "no_tool",
      label: "No tool found — writing a new one",
    },
    {
      type: "writing",
      label: "Writing a Python tool",
      detail:
        "def speed(distance_km, time_hr): return distance_km / time_hr",
    },
    {
      type: "testing",
      label: "Testing in sandbox against known values",
    },
    {
      type: "pass",
      label: "Test passed: speed(120, 2) == 60",
    },
    {
      type: "answer",
      label: "Answer: 60 km/h",
    },
  ];
}

/* =========================================================
   VISUAL STYLES FOR EACH STEP TYPE
   ========================================================= */
const stepStyles: Record<
  StepType,
  {
    dot: string;
    text: string;
    badge: string;
    card?: string;
  }
> = {
  plan: {
    dot: "bg-blue-400",
    text: "text-blue-100",
    badge: "PLAN",
  },
  check: {
    dot: "bg-yellow-400",
    text: "text-yellow-100",
    badge: "CHECK",
  },
  no_tool: {
    dot: "bg-orange-400",
    text: "text-orange-100",
    badge: "NO TOOL",
  },
  writing: {
    dot: "bg-purple-400 animate-pulse",
    text: "text-purple-100",
    badge: "WRITING",
  },
  testing: {
    dot: "bg-cyan-400 animate-pulse",
    text: "text-cyan-100",
    badge: "TESTING",
  },
  pass: {
    dot: "bg-green-400",
    text: "text-green-200",
    badge: "PASS",
    card: "border-green-500/20 bg-green-500/[0.04]",
  },
  fail: {
    dot: "bg-red-400",
    text: "text-red-200",
    badge: "FAIL",
    card: "border-red-500/20 bg-red-500/[0.04]",
  },
  answer: {
    dot: "bg-white shadow-[0_0_12px_rgba(255,255,255,0.55)]",
    text: "text-white font-semibold",
    badge: "ANSWER",
    card: "border-white/15 bg-white/[0.06]",
  },
};

const STREAM_MS = 600;

export default function Home() {
  const [task, setTask] = useState("");
  const [allSteps, setAllSteps] = useState<Step[]>([]);
  const [visibleSteps, setVisibleSteps] = useState<Step[]>([]);
  const [isRunning, setIsRunning] = useState(false);

  const runAgent = async () => {
    const trimmedTask = task.trim();
    if (!trimmedTask || isRunning) return;

    setAllSteps([]);
    setVisibleSteps([]);
    setIsRunning(true);

    try {
      const res = await fetch("http://localhost:8000/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: trimmedTask }),
      });
      if (!res.ok) throw new Error("Backend returned " + res.status);
      const data = await res.json();
      const generatedSteps: Step[] = data.trace;

      // Show the first step immediately, then stream the rest.
      setAllSteps(generatedSteps);
      setVisibleSteps(generatedSteps.slice(0, 1));
      setIsRunning(generatedSteps.length > 1);
    } catch {
      const errorStep: Step = {
        type: "fail",
        label: "Could not reach the backend — is it running on port 8000?",
      };
      setAllSteps([errorStep]);
      setVisibleSteps([errorStep]);
      setIsRunning(false);
    }
  };

  // Stream remaining steps one at a time (~600ms apart).
  useEffect(() => {
    if (!isRunning || allSteps.length === 0) return;

    if (visibleSteps.length >= allSteps.length) {
      setIsRunning(false);
      return;
    }

    const timer = window.setTimeout(() => {
      setVisibleSteps((current) => {
        if (current.length >= allSteps.length) return current;
        const nextStep = allSteps[current.length];
        return nextStep ? [...current, nextStep] : current;
      });
    }, STREAM_MS);

    return () => window.clearTimeout(timer);
  }, [allSteps, visibleSteps, isRunning]);

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") runAgent();
  };

  return (
    <main className="min-h-screen bg-[#080b12] text-white">
      <div className="mx-auto flex min-h-screen w-full max-w-4xl flex-col px-5 py-10 sm:px-8 sm:py-16">
        {/* Header */}
        <header className="mb-10">
          <div className="mb-3 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-blue-400/20 bg-blue-500/10">
              <div className="h-3 w-3 rounded-full bg-blue-400 shadow-[0_0_15px_rgba(96,165,250,0.8)]" />
            </div>
            <span className="text-sm font-medium tracking-widest text-blue-400">
              OFFLINE AI AGENT
            </span>
          </div>

          <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">
            Edge Smart Agent
          </h1>

          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400 sm:text-base">
            Watch the agent plan, reason, test tools, and produce an answer —
            completely locally.
          </p>
        </header>

        {/* Task input */}
        <section className="mb-8">
          <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-3 shadow-2xl shadow-black/20">
            <div className="flex flex-col gap-3 sm:flex-row">
              <input
                type="text"
                value={task}
                onChange={(event) => setTask(event.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your task here..."
                disabled={isRunning}
                className="min-w-0 flex-1 rounded-xl border border-white/10 bg-black/20 px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500 transition focus:border-blue-400/50 focus:ring-2 focus:ring-blue-400/10 disabled:cursor-not-allowed disabled:opacity-60"
              />

              <button
                type="button"
                onClick={runAgent}
                disabled={!task.trim() || isRunning}
                className="rounded-xl bg-blue-500 px-7 py-3 text-sm font-semibold text-white transition hover:bg-blue-400 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-500"
              >
                {isRunning ? "Running..." : "Run"}
              </button>
            </div>
          </div>

          <p className="mt-2 px-1 text-xs text-slate-600">
            Try <span className="text-slate-500">2+5</span> for a direct answer,
            or{" "}
            <span className="text-slate-500">
              train speed from distance and time
            </span>{" "}
            to see the tool-building trace.
          </p>
        </section>

        {/* Agent Trace */}
        <section className="overflow-hidden rounded-2xl border border-white/10 bg-[#0d111a] shadow-2xl shadow-black/30">
          <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
            <div>
              <h2 className="text-sm font-semibold text-white">Agent Trace</h2>
              <p className="mt-1 text-xs text-slate-500">
                {isRunning
                  ? "Agent is working..."
                  : visibleSteps.length > 0
                    ? "Run complete"
                    : "Waiting for a task"}
              </p>
            </div>

            {isRunning && (
              <div className="flex items-center gap-2 text-xs text-blue-400">
                <span className="h-2 w-2 animate-pulse rounded-full bg-blue-400" />
                Streaming
              </div>
            )}
          </div>

          <div className="min-h-[360px] p-5">
            {visibleSteps.length === 0 ? (
              <div className="flex min-h-[310px] flex-col items-center justify-center text-center">
                <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.03]">
                  <div className="h-2.5 w-2.5 rounded-full bg-slate-600" />
                </div>
                <p className="text-sm text-slate-500">No reasoning trace yet</p>
                <p className="mt-1 text-xs text-slate-700">
                  Enter a task above and press Run
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {visibleSteps.map((step, index) => {
                  if (!step) return null;
                  const style = stepStyles[step.type];
                  if (!style) return null;

                  return (
                    <div
                      key={`${step.type}-${index}`}
                      className={`trace-step rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 ${style.card ?? ""}`}
                    >
                      <div className="flex items-start gap-3">
                        <div className="relative flex shrink-0 items-center justify-center pt-1">
                          <span className={`h-2.5 w-2.5 rounded-full ${style.dot}`} />
                          {index < visibleSteps.length - 1 && (
                            <span className="absolute left-1/2 top-5 h-[calc(100%+1rem)] w-px -translate-x-1/2 bg-white/10" />
                          )}
                        </div>

                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className={`text-sm font-medium ${style.text}`}>
                              {step.label}
                            </span>
                            <span className="rounded-md border border-white/5 bg-white/[0.03] px-1.5 py-0.5 text-[9px] font-semibold tracking-wider text-slate-600">
                              {style.badge}
                            </span>
                          </div>

                          {step.detail && (
                            <pre className="mt-3 overflow-x-auto rounded-lg border border-white/10 bg-black/40 p-3 font-mono text-xs leading-6 text-slate-300">
                              <code>{step.detail}</code>
                            </pre>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}

                {isRunning && (
                  <div className="flex items-center gap-3 px-1 py-2 text-xs text-slate-600">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-400" />
                    Processing next step...
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        <footer className="mt-6 flex items-center justify-between text-xs text-slate-700">
          <span>Edge Smart Agent</span>
          <span>Mock execution • No backend connected</span>
        </footer>
      </div>
    </main>
  );
}
