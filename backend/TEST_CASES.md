# EDGE-SMART — Test Cases & Run Guide

How to run and verify the current hackathon build:

1. **Backend orchestrator** (real decision logic + toolbox, **mocked** tool factory)
2. **Frontend reasoning-trace UI** (**mocked** steps — **not** connected to the backend yet)

---

## Part A — How to run the backend

### A1. Prerequisites

- Python 3.10+ with `fastapi` and `uvicorn` installed:

```powershell
pip install fastapi uvicorn
```

### A2. Start the API

```powershell
cd D:\Mindmesh\backend
python -m uvicorn main:app --reload
```

You should see:

```
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

Leave this terminal open.

### A3. Where to send requests

| Method | URL | Purpose |
|--------|-----|---------|
| Browser | http://127.0.0.1:8000/docs | Swagger UI (easiest) |
| PowerShell | `POST http://127.0.0.1:8000/run` | Scripted tests |

**Do not** open `http://127.0.0.1:8000/` alone — there is no homepage (`GET /` → 404 is normal).

### A4. PowerShell request template

On Windows, prefer `Invoke-RestMethod` (plain `curl` is an alias and will fail):

```powershell
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/run `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"task": "YOUR_TASK_HERE"}' | ConvertTo-Json -Depth 6
```

Or with real curl:

```powershell
curl.exe -X POST http://127.0.0.1:8000/run `
  -H "Content-Type: application/json" `
  -d "{\"task\": \"YOUR_TASK_HERE\"}"
```

### A5. Swagger UI steps

1. Open http://127.0.0.1:8000/docs  
2. Click **POST /run** → **Try it out**  
3. Edit the body, e.g. `{"task": "2+5"}`  
4. Click **Execute**  
5. Check **Code** = `200` and the JSON **Response body**

### A6. Reset toolbox between reuse tests (optional)

If you want to re-test the factory path from a clean slate:

```powershell
cd D:\Mindmesh\backend
Set-Content -Path .\toolbox\manifest.json -Value "[]`n"
Remove-Item -Path .\toolbox\*.py -ErrorAction SilentlyContinue
```

---

## Part B — Backend test cases

Endpoint: `POST /run` with body `{"task": "<text>"}`.

Every successful response includes:

- `path` — `"trivial"` | `"reuse"` | `"factory"`
- `answer` — string or factory result object
- `trace` — list of reasoning steps (strings)

### B1. Trivial arithmetic (no tool)

| # | Task | Expected `path` | Expected `answer` | Must NOT happen |
|---|------|-----------------|-------------------|-----------------|
| B1.1 | `2+5` | `trivial` | `"7"` | No factory call; no new toolbox file |
| B1.2 | `10-3` | `trivial` | `"7"` | Same |
| B1.3 | `4*6` | `trivial` | `"24"` | Same |
| B1.4 | `8/2` | `trivial` | `"4"` | Same |
| B1.5 | `what's 2+5` | `trivial` | `"7"` | Arithmetic still detected in text |

**Commands:**

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/run -Method POST -ContentType "application/json" -Body '{"task": "2+5"}' | ConvertTo-Json -Depth 6
```

**Pass:** `path` is `trivial`, answer is correct, `trace` mentions answering directly.

---

### B2. Factory path (first time — create + save tool)

**Precondition:** no `speed` entry in `toolbox/manifest.json` (reset if needed — see A6).

| # | Task | Expected `path` | Side effects |
|---|------|-----------------|--------------|
| B2.1 | `train speed from distance and time` | `factory` | Creates `toolbox/speed.py` and adds `speed` to `manifest.json` |
| B2.2 | `compute speed from distance and time` | `factory` *(if speed not already saved)* | Same |

**Expected `answer` shape (mocked factory):**

```json
{
  "success": true,
  "tool_name": "speed",
  "code": "def speed(distance_km, time_hr): return distance_km / time_hr",
  "attempts": 1,
  "error": null
}
```

**Also check on disk:**

```powershell
Get-Content D:\Mindmesh\backend\toolbox\manifest.json
Get-Content D:\Mindmesh\backend\toolbox\speed.py
```

**Pass:** `path` is `factory`, mock dict returned, `speed.py` exists, manifest lists `speed`.

---

### B3. Reuse path (second time — load from toolbox)

**Precondition:** B2 already succeeded (`speed` in manifest + `speed.py` present).

| # | Task | Expected `path` | Expected `answer` |
|---|------|-----------------|-------------------|
| B3.1 | `train speed from distance and time` | `reuse` | `"60.0"` (sandbox: `speed(120, 2)`) |
| B3.2 | Same task again | `reuse` | `"60.0"` |

**Pass:** `path` is `reuse`, answer from sandbox, `trace` says it found the tool in the manifest (not calling factory).

---

### B4. HTTP / endpoint checks

| # | Action | Expected |
|---|--------|----------|
| B4.1 | `GET /` | `404` (OK — no homepage) |
| B4.2 | `GET /docs` | Swagger UI loads |
| B4.3 | `POST /run` with `{"task": "2+5"}` | `200` + JSON |
| B4.4 | `POST /run` with `{}` (missing `task`) | `422` validation error |

```powershell
# Missing task -> 422
try {
  Invoke-RestMethod -Uri http://127.0.0.1:8000/run -Method POST -ContentType "application/json" -Body '{}'
} catch {
  $_.Exception.Response.StatusCode.value__   # expect 422
}
```

---

### B5. Backend checklist

- [ ] Server starts with no import errors
- [ ] `2+5` → `trivial` / `7`
- [ ] First train-speed → `factory` + files written
- [ ] Second train-speed → `reuse` / `60.0`
- [ ] `/docs` works
- [ ] Invalid body → `422`
- [ ] Factory is still the **mock** (no real `tool_factory` import yet)

---

## Part C — How to run the frontend

### C1. Prerequisites

```powershell
cd D:\Mindmesh\frontend
npm install
```

### C2. Free port 3000 (if “port in use”)

```powershell
# Find and kill whatever owns port 3000
netstat -ano | findstr ":3000" | findstr "LISTENING"
# Note the PID in the last column, then:
taskkill /PID <PID> /F
```

Or:

```powershell
Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force }
```

### C3. Start the UI

```powershell
cd D:\Mindmesh\frontend
npm run dev
```

Open **exactly one** URL:

- http://localhost:3000

If Next says it moved to `3001`, stop the old process (C2) and restart so only **3000** is used.

### C4. Important: frontend is still mocked

The UI does **not** call the orchestrator yet.

- Clicking **Run** calls local `mockRun(task)` in `frontend/app/page.tsx`
- Backend can be **off** and the UI will still work
- Connecting UI → `POST /run` is a later integration step

---

## Part D — Frontend test cases (mock trace UI)

### D1. Tool-writing scenario (7 steps)

1. Open http://localhost:3000  
2. Type: `train speed from distance and time`  
3. Click **Run** (button is disabled until the input has text)

**Expect steps to appear one-by-one (~600ms), in order:**

| # | Type | Label / detail |
|---|------|----------------|
| 1 | `plan` | Planning: this needs a speed calculation |
| 2 | `check` | Checking toolbox for a matching tool |
| 3 | `no_tool` | No tool found — writing a new one |
| 4 | `writing` | Writing a Python tool + monospace code: `def speed(distance_km, time_hr): return distance_km / time_hr` |
| 5 | `testing` | Testing in sandbox against known values |
| 6 | `pass` | Test passed: speed(120, 2) == 60 |
| 7 | `answer` | Answer: 60 km/h |

**Pass:** all 7 stream in with fade-in; writing step shows a code block; header shows “Streaming” then “Run complete”.

---

### D2. Trivial arithmetic scenario (2 steps)

1. Clear / replace the input with: `2+5`  
2. Click **Run**

**Expect:**

| # | Type | Label |
|---|------|-------|
| 1 | `plan` | This is trivial — I can answer directly |
| 2 | `answer` | Answer: 7 |

**Pass:** only 2 steps; no writing / testing / code block.

---

### D3. UI behavior checks

| # | Action | Expected |
|---|--------|----------|
| D3.1 | Empty input, click Run | Button disabled — nothing happens |
| D3.2 | Type text, click Run | First step appears immediately; rest stream |
| D3.3 | Press Enter in the input | Same as clicking Run |
| D3.4 | Click Run while streaming | Ignored until the run finishes |

---

### D4. Frontend checklist

- [ ] `npm run dev` serves http://localhost:3000
- [ ] Train-speed shows 7 mocked steps + code snippet
- [ ] `2+5` shows 2-step direct answer
- [ ] Empty Run does nothing (disabled)
- [ ] Footer still says mock / no backend connected
- [ ] No network calls to `:8000` when clicking Run (DevTools → Network)

---

## Part E — Full local demo (both apps)

Use **two terminals**:

**Terminal 1 — backend**

```powershell
cd D:\Mindmesh\backend
python -m uvicorn main:app --reload
```

**Terminal 2 — frontend**

```powershell
cd D:\Mindmesh\frontend
npm run dev
```

Then:

| App | URL | What you’re testing |
|-----|-----|---------------------|
| Backend | http://127.0.0.1:8000/docs | Real orchestrator paths |
| Frontend | http://localhost:3000 | Mocked streaming trace UI |

They run **independently** until frontend↔orchestrator integration.

---

## Quick “done” summary

| Layer | Smoke test | Pass looks like |
|-------|------------|-----------------|
| Backend | `2+5` | `path: trivial`, answer `7` |
| Backend | train-speed (1st) | `path: factory`, `toolbox/speed.py` created |
| Backend | train-speed (2nd) | `path: reuse`, answer `60.0` |
| Frontend | train-speed | 7 mocked steps stream in |
| Frontend | `2+5` | 2 mocked steps, answer 7 |
