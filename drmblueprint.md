## **E: EXECUTABLE BLUEPRINT — DRM–AGI System Core (Revision E.10)**

### Project Origin

This system is designed as a foundation for **Dynamic Reflective Memory (DRM)** — an adaptive memory framework enabling emergent AGI-like behavior. It integrates reasoning, self-adjustment, and long-term continuity through structured memory and modular LLM workflows.

---

### 1. Architecture Overview

* **Execution Model**: Local-first hybrid architecture

  * App runtime: Python + PySide6 GUI
  * LLM layer: `litellm` abstraction
  * Providers: Azure OpenAI (primary), Ollama (secondary/local)
  * Memory layers:

    * Redis (Docker) — short-term session/working memory
    * ChromaDB — long-term semantic memory with concept-node graph support
  * External audit & self-adjustment loop with meta-controller
* **Goal**: Provide a memory substrate enabling reasoning continuity, context awareness, self-monitoring, and adaptation.

---

### 2. Core Components

* **Memory Context**: merges Redis (short-term) and ChromaDB (long-term) into a unified working space.
* **Adaptive Prompt Engine**: builds prompts dynamically based on context, history, and goals.
* **Task Executor**: handles LLM routing (fast/reasoning/local) and model selection logic.
* **Self-Adjusting Controller**: detects drift, reweights parameters, updates memory embeddings, and logs feedback cycles.
* **Review Layer**: hybrid validation after every task — automated + human review.

---

### 3. Memory Layers & Representation

| Layer               | Role                                    | Representation                                                                        |
| ------------------- | --------------------------------------- | ------------------------------------------------------------------------------------- |
| **Working Memory**  | Current session state, context, runtime | Redis key-value store with TTL, storing JSON objects                                  |
| **Episodic Memory** | Time-ordered experiences/events         | ChromaDB embeddings + timestamp + metadata + optional edges                           |
| **Semantic Memory** | Abstracted knowledge, concept nodes     | Graph structure: nodes (id, label, definition, sources, timestamp), edges (relations) |
| **Review Memory**   | Audit logs of tasks, corrections, drift | Embeddings + metadata, stored in ChromaDB + Redis for fast access                     |

---

### 4. Workflow Matrix – LLM Models

| Workflow      | LLM Model    | Use Case                              |
| ------------- | ------------ | ------------------------------------- |
| **fast**      | GPT-4.1      | Quick responses, lightweight tasks    |
| **reasoning** | GPT-5        | Logic-intensive, deep reasoning tasks |
| **local**     | Ollama (TBD) | Offline/local inference, autonomy     |

---

### 5. Operational Flow

1. **Initialization:** GUI/CLI launches → system verifies Redis, ChromaDB, LLM endpoints.
2. **User Context Definition:** User inputs or selects workflow (task). Active memory and previous context retrieved.
3. **Prompt Construction:** Adaptive prompt engine merges task + memory context using self-adjusting weights.
4. **Execution & Review:** LLM executes task → result validated via hybrid review.
5. **Memory Update:**

   * Working memory updates in Redis.
   * Episodic entry in ChromaDB.
   * If eligible, concept node created/expanded in Semantic Memory.
   * Review object stored in Review Memory.
6. **Self-Adjustment:** Controller monitors patterns; if drift detected, recalibrates weights, memory context, workflow priorities.
7. **User Feedback Loop:** Output returned to GUI/CLI; user can approve, correct, or refine.

---

### 6. User Interaction Model

* **Interface:** GUI (PySide6) or CLI.
* **Experience:** Seamless continuity — system recalls prior context and evolves.
* **Displayed to user:**

  * Task status (workflow, model used)
  * Memory state indicators (active vs archived, recent drift)
  * Review history and correction suggestions
* **Goal:** Make adaptive intelligence visible yet non-intrusive.

---

### 7. External Review Process

* Triggered **after every task**.
* Hybrid of:

  * **Automated Audit:** Secondary LLM checks logic, fact consistency, redundancy
  * **Human Review:** Optional validation, correction
* Review outcomes feed into Review Memory and can trigger drift detection.

---

### 8. Self-Adjustment & Drift Detection

* Monitoring metrics: output quality, latency, consistency, redundancy.
* When a **Performance Drift Event (PDE)** is detected:

  * Run meta-review agent → issues corrective action (adjust weights, attenuate memory, rollback).
  * Controller applies adjustments or escalates for human approval.
  * Maintenance of versioning and rollback support for memory changes.

---

### 9. Local Docker Components

| Container | Purpose                  | Port | Index         |
| --------- | ------------------------ | ---- | -------------- |
| Redis     | Live working memory      | 6379 | 5  |

---

### 10. System Goal

To implement **adaptive, memory-driven intelligence** through local components, enabling each reasoning cycle to reinforce learning via long-term memory, review feedback, and self-adjustment, moving toward a minimal AGI substrate.
