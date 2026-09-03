export default function Home() {
  return (
    <main className="min-h-screen p-10">
      <h1 className="text-3xl font-bold">
        EDGE-SMART
      </h1>

      <p className="mt-2 text-gray-500">
        Edge-Native Agentic Assistant
      </p>

      <div className="mt-8">
        <label htmlFor="task" className="block mb-2 font-medium">
          Enter your task:
        </label>

        <textarea
          id="task"
          placeholder="Enter your task..."
          className="w-full max-w-2xl rounded-lg border p-4"
          rows={4}
        />

        <button
          type="button"
          className="mt-4 rounded-lg bg-black px-6 py-3 text-white"
        >
          Run Task
        </button>
      </div>

      <section className="mt-10">
        <h2 className="text-xl font-semibold">
          Agent Trace
        </h2>

        <div
          className="mt-4 min-h-48 w-full max-w-2xl rounded-lg border p-5"
          aria-label="Agent Trace"
        />
      </section>
    </main>
  );
}
