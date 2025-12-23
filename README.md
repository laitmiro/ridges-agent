# SWE Agent - Educational Software Engineering Agent

> An educational agentic system built to explore and demonstrate autonomous software engineering workflows through intelligent code understanding, generation, and iterative refinement.

## Overview

This SWE (Software Engineering) Agent is built as an **educational project** to explore agentic workflows for software engineering tasks. The goal is to experiment with and demonstrate concepts like autonomous problem-solving, multi-model inference, and intelligent tool use in a controlled learning environment.

### Key Capabilities

- **Autonomous Problem Solving** - Understands requirements, investigates codebases, and implements solutions
- **Multi-Model Inference** - Leverages multiple LLMs (GLM-4.6, Kimi-K2, Qwen3-Coder) with task-specific routing
- **Intelligent Tool Use** - Equipped with comprehensive tools for file operations, code analysis, and execution
- **Fault Tolerance** - Built-in retry logic, error recovery, and timeout handling
- **Context Management** - Automatic summarization to maintain context within token limits
- **Iterative Refinement** - Tests, debugs, and refines solutions until all requirements are met

---

## Architecture

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     SWE Agent System                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐      ┌──────────────┐      ┌─────────┐   │
│  │  Input Layer  │─────▶│ Agent Engine │─────▶│ Output  │   │
│  └───────────────┘      └──────────────┘      └─────────┘   │
│         │                      │                     │      │
│         │                      │                     │      │
│         ▼                      ▼                     ▼      │
│  Problem Statement      ┌─────────────┐         Git Patch   │
│  Repository Info        │ Multi-Model │         Test Results│
│  Test Cases             │  Inference  │         Logs        │
│                         └─────────────┘                     │
│                              │                              │
│                              ▼                              │
│                     ┌─────────────────┐                     │
│                     │   Tool System   │                     │
│                     └─────────────────┘                     │
│                              │                              │
│        ┌─────────────────────┼─────────────────────┐        │
│        │                     │                     │        │
│        ▼                     ▼                     ▼        │
│  ┌──────────┐        ┌─────────────┐      ┌──────────────┐  │
│  │File Ops  │        │Code Analysis│      │  Execution   │  │
│  └──────────┘        └─────────────┘      └──────────────┘  │
│  • read_file         • get_function_body  • run_bash_cmd    │
│  • create_file       • search             • run_code        │
│  • edit_file         • search_in_files    • think           │
│  • get_dir_tree                           • finish          │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. **Agent Engine** ([main.py:2122-2427](main.py))

The central orchestrator that:

- Manages conversation flow with system/user/tool messages
- Implements automatic context summarization
- Handles tool call execution and retry logic
- Detects and prevents infinite loops

#### 2. **Multi-Model Inference System** ([main.py:316-386](main.py))

Dynamic model selection based on task type:

- **REASONING Tasks** → Kimi-K2, Qwen3-Coder (low temp: 0.1-0.3)
- **THINKING Tasks** → GLM-4.6, Qwen3-Coder (medium temp: 0.3-0.7)
- **ANSWERING Tasks** → Qwen3-Coder, GLM-4.6 (low temp: 0.0-0.3)

#### 3. **Tool System** ([main.py:1658-2089](main.py))

Comprehensive toolkit with automatic documentation generation:

- File operations (read, write, edit, tree navigation)
- Code analysis (search, function extraction)
- Execution (bash commands, code running)
- Meta-tools (think, finish, initial solution generation)

#### 4. **Timeout Handler** ([main.py:2091-2120](main.py))

Ensures agent execution completes within configured time limits.

#### 5. **Error Management** ([main.py:124-162](main.py))

Unified error handling with 20+ error types and custom exception system.

---

## Workflow

### Agent Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT EXECUTION WORKFLOW                         │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │ START: Receive   │
    │ Problem Statement│
    └────────┬─────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ Phase 1: UNDERSTAND THE PROBLEM                │
    │ • Parse requirements and constraints           │
    │ • Identify edge cases and ambiguities          │
    │ • Understand success criteria                  │
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ Phase 2: INVESTIGATE CODEBASE                  │
    │ • Explore directory structure                  │
    │ • Search for relevant code patterns            │
    │ • Find working examples (as templates)         │
    │ • Identify utilities and helpers               │
    │ • Trace data flow and dependencies             │
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ Phase 3: GENERATE INITIAL SOLUTION (if new)    │
    │ • Create source code using codebase patterns   │
    │ • Generate comprehensive test cases            │
    │ • Establish baseline implementation            │
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ Phase 4: PLAN THE SOLUTION                     │
    │ • Break down into incremental steps            │
    │ • Identify files to modify                     │
    │ • Define validation strategy                   │
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ Phase 5: IMPLEMENT INCREMENTALLY               │
    │ • Make small, focused changes                  │
    │ • Test after each modification                 │
    │ • Validate against requirements                │
    └────────┬───────────────────────────────────────┘
             │
             ▼
         ┌───┴────┐
         │  Tests │
         │  Pass? │
         └───┬────┘
             │
        ┌────┼────┐
        │    │    │
       NO   YES   │
        │    │    │
        ▼    │    │
    ┌───────────────────────────────┐
    │ Phase 6: DEBUG & REFINE       │
    │ • Analyze test failures       │
    │ • Trace root causes           │
    │ • Apply fixes                 │
    │ • Re-run tests                │
    └───────┬───────────────────────┘
            │
            │ (loop until tests pass)
            │
            └─────────────┐
                          │
                          ▼
                ┌──────────────────────┐
                │ Phase 7: FINAL       │
                │ VALIDATION           │
                │ • Run all tests      │
                │ • Check edge cases   │
                │ • Generate git patch │
                └─────────┬────────────┘
                          │
                          ▼
                    ┌──────────┐
                    │  FINISH  │
                    └──────────┘
```

### Inference Loop with Context Management

```
┌─────────────────────────────────────────────────────────────┐
│             INFERENCE LOOP (Agent.execute)                  │
└─────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │  Start Loop     │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────┐
    │ Check Timeout               │
    │ (TimeoutHandler)            │
    └────────┬────────────────────┘
             │
             ▼
    ┌─────────────────────────────┐
    │ Context Summarization       │
    │ (if messages > threshold)   │
    │ • Compress old messages     │
    │ • Retain recent messages    │
    │ • Preserve system context   │
    └────────┬────────────────────┘
             │
             ▼
    ┌─────────────────────────────┐
    │ Multi-Model Inference       │
    │ • Select model by task type │
    │ • Set temperature range     │
    │ • Retry on failure          │
    └────────┬────────────────────┘
             │
             ▼
    ┌─────────────────────────────┐
    │ Parse Response              │
    │ • Extract content           │
    │ • Extract tool calls        │
    │ • Fault tolerance           │
    └────────┬────────────────────┘
             │
             ▼
    ┌─────────────────────────────┐
    │ Repeated Tool Call Check    │
    │ • Detect loops (max 2×)     │
    │ • Rebase if detected        │
    └────────┬────────────────────┘
             │
             ▼
    ┌─────────────────────────────┐
    │ Execute Tool Calls          │
    │ • Run each tool             │
    │ • Collect outputs           │
    │ • Handle errors             │
    └────────┬────────────────────┘
             │
             ▼
         ┌───┴────┐
         │ finish │──────┐
         │ called?│      │
         └───┬────┘      │
             │           │
            NO          YES
             │           │
             │           ▼
             │    ┌────────────┐
             │    │   RETURN   │
             │    └────────────┘
             │
             └──────┐
                    │
                    ▼
            ┌──────────────────┐
            │ Update Messages  │
            │ • Add response   │
            │ • Add tool output│
            └────────┬─────────┘
                     │
                     └──────────┐
                                │
                (Loop continues)│
                                │
                                ▼
```

### Tool Call Fault Tolerance

```
┌──────────────────────────────────────────────────────┐
│           TOOL CALL FAULT TOLERANCE                  │
└──────────────────────────────────────────────────────┘

  LLM Response
       │
       ▼
  ┌─────────────┐
  │ Has tool    │──NO──┐
  │ calls?      │      │
  └──┬──────────┘      │
     │                 │
    YES                │
     │                 ▼
     │         ┌────────────────────┐
     │         │ Tool Call Fault    │
     │         │ Tolerant Parser    │
     │         │ • Use secondary LLM│
     │         │ • Extract intent   │
     │         │ • Format tool calls│
     │         └─────────┬──────────┘
     │                   │
     │                   │
     └───────────────────┘
                         │
                         ▼
                 ┌───────────────┐
                 │ Execute Tools │
                 └───────────────┘
```

---

## Tool System

### Available Tools

| Category            | Tool                          | Description                             |
| ------------------- | ----------------------------- | --------------------------------------- |
| **File Operations** | `get_dir_tree`                | Hierarchical directory structure        |
|                     | `read_file`                   | Read file contents with line ranges     |
|                     | `create_file`                 | Create new files                        |
|                     | `edit_file`                   | Apply precise edits to existing files   |
| **Code Analysis**   | `search`                      | Search for files/directories by pattern |
|                     | `search_in_all_files_content` | Full-text search across codebase        |
|                     | `get_function_body`           | Extract function/class definitions      |
| **Execution**       | `run_bash_command`            | Execute shell commands                  |
|                     | `run_code`                    | Run Python/Node.js code with tests      |
| **Meta-Tools**      | `think`                       | Reasoning and planning                  |
|                     | `finish`                      | Complete task execution                 |
|                     | `generate_initial_solution`   | Bootstrap new implementations           |

### Tool Documentation System

All tools are **self-documenting** through introspection:

- Automatic parameter extraction from function signatures
- Type inference and JSON schema generation
- Docstring parsing for descriptions
- Required vs optional parameter detection

---

## Configuration

### Environment Variables

```bash
# Gateway URL for LLM inference
SANDBOX_PROXY_URL=http://localhost:1234

# Agent execution timeout (seconds)
AGENT_TIMEOUT=1500

# Evaluation run identifier
EVALUATION_RUN_ID=<uuid>
```

### Model Configuration

**Supported Models:**

- `zai-org/GLM-4.6-FP8` - General reasoning and thinking
- `moonshotai/Kimi-K2-Instruct` - Advanced reasoning
- `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` - Code-focused tasks

**Temperature Ranges by Task:**

- REASONING: 0.1 - 0.3 (precise, analytical)
- THINKING: 0.3 - 0.7 (creative, exploratory)
- ANSWERING: 0.0 - 0.3 (factual, deterministic)

---

## Usage

### Basic Execution

```python
from main import agent_main

# Define the problem
input_dict = {
    "problem_statement": """
    Implement a function `calculate_fibonacci(n)` that returns
    the nth Fibonacci number. Include comprehensive test coverage.
    """
}

# Execute agent
result = agent_main(
    input_dict=input_dict,
    repo_dir="/path/to/repository"
)
```

### Workflow Example

```python
# 1. Agent receives problem statement
# 2. Investigates repository structure
# 3. Searches for similar implementations
# 4. Generates initial solution and tests
# 5. Runs tests and identifies failures
# 6. Iteratively debugs and refines
# 7. Validates all edge cases
# 8. Returns git patch with changes
```

---

## Design Principles (Educational Focus)

### 1. **Error Handling Exploration**

- Comprehensive error handling with 20+ error types
- Automatic retry logic with exponential backoff
- Timeout protection to prevent runaway execution
- Graceful degradation on model failures

### 2. **Context Management Techniques**

- Automatic message summarization to manage token limits
- Configurable retention windows (summary_length, retain_length)
- Tool output truncation for large responses
- Efficient codebase exploration with targeted reads

### 3. **Iterative Refinement**

- Repeated tool call detection prevents infinite loops
- Rebase mechanism to backtrack from failures
- Incremental testing after each change
- Root cause analysis for systematic debugging

### 4. **Code Quality Standards**

- Type hints throughout codebase
- Comprehensive docstrings with argument descriptions
- Tree-sitter based code parsing for AST analysis
- Git-based change tracking and patch generation

### 5. **Observability & Logging**

- Structured logging with timestamps
- Tool call tracing with arguments and outputs
- Inference tracking (model, temperature, retries)
- Token usage monitoring

---

## Logging

### Log Structure

```
logs/
└── agent_YYYYMMDD_HHMMSS.log
```

### Log Events

- `[PROBLEM STATEMENT]` - Input task
- `[INFERENCE]` - Model selection and parameters
- `[TOOL CALL]` - Tool execution with arguments
- `[SUMMARIZING]` - Context compression events
- `[REPEATED]` - Loop detection warnings
- `[AGENT TIMEOUT]` - Execution time limits
- `[CUSTOM ERROR]` - Structured error details

---

## Error Handling

### Error Types

The system defines comprehensive error types for precise failure diagnosis:

| Category        | Error Types                                                                         |
| --------------- | ----------------------------------------------------------------------------------- |
| **File System** | FILE_NOT_FOUND, DIRECTORY_NOT_FOUND, PERMISSION_DENIED, NOT_A_FILE, NOT_A_DIRECTORY |
| **Search**      | TARGET_NOT_FOUND, INVALID_REGEX, INVALID_PARAMETER                                  |
| **Execution**   | COMMAND_TIMEOUT, COMMAND_NOT_FOUND, COMMAND_FAILED                                  |
| **I/O**         | READ_ERROR, WRITE_ERROR, ENCODING_ERROR                                             |
| **Network**     | TIMEOUT, CONNECTION_ERROR, HTTP_ERROR                                               |
| **Agent**       | AGENT_TIMEOUT, PLANNER_FAILED, TESTER_FAILED, CODER_FAILED                          |
| **Inference**   | INVALID_REQUEST, INVALID_JSON, INFINITE_INFERENCE                                   |

### Retry Strategy

```python
# Inference retry with model rotation
max_retries = 5
models = [Model1, Model2, Model3]
current_model = models[(attempt % len(models))]

# Agent-level retry for transient failures
max_retries = 3
# Rebase on repeated tool calls
# Full retry on CustomError exceptions
```

---

## Advanced Features

### 1. **Context Summarization**

When conversation length exceeds thresholds:

```
Messages: [1...base_length] + [summary] + [recent messages]
          ↑                    ↑          ↑
    System prompts      Compressed    Retain N most recent
                        history
```

### 2. **Tool Call Loop Detection**

Detects repeated sequences of tool calls:

```python
# Detects patterns like:
# [tool1, tool2] → [tool1, tool2]  (2x repeat)
# [toolA] → [toolA] → [toolA]      (3x repeat)

# Action: Rebase to before repetition started
```

### 3. **Multi-File Code Editing**

Edit tool supports:

- Line-based replacements
- Multi-line insertions
- Deletion of code blocks
- Preserves indentation and formatting

### 4. **AST-Based Code Analysis**

Uses tree-sitter for:

- Function/class extraction
- Precise code location
- Language-agnostic parsing
- Syntax-aware search

---

## Performance Characteristics

### Typical Execution Profile

| Phase            | Time Range      | Primary Activity                            |
| ---------------- | --------------- | ------------------------------------------- |
| Understanding    | 1-2 iterations  | Problem analysis, requirement clarification |
| Investigation    | 2-5 iterations  | Codebase exploration, pattern finding       |
| Initial Solution | 1-3 iterations  | Code generation, test creation              |
| Implementation   | 3-10 iterations | Incremental changes, testing                |
| Debugging        | 2-8 iterations  | Failure analysis, fixes                     |
| Validation       | 1-2 iterations  | Final testing, edge cases                   |

### Resource Usage

- **Context Window**: Dynamically managed via summarization
- **Token Efficiency**: ~60-80% compression via summarization
- **Tool Calls**: Average 15-30 per task
- **Model Switches**: 1-3 per phase based on task type
- **Timeout**: Configurable (default: 1450s)

---

## Contributing

This agent is designed as an **educational project** to explore agentic workflows. When contributing:

1. **Maintain clarity** - Add error handling and clear explanations for new features
2. **Preserve learning value** - Document design decisions and trade-offs
3. **Document thoroughly** - Update docstrings and README with educational context
4. **Test reasonably** - Validate core functionality
5. **Log appropriately** - Ensure observability for learning purposes

---

## License

See LICENSE file for details.

---

## Acknowledgments

Built with:

- Tree-sitter for code parsing
- Multiple state-of-the-art LLMs
- Production-tested error handling patterns
- Iterative refinement methodology

---

**This agent is built as an educational project to explore agentic workflows for software engineering tasks. It demonstrates concepts and techniques in autonomous problem-solving, not intended for production use.**
