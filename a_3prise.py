import os
import re
import subprocess
import inspect
import requests
import random
import logging
import json
import asyncio
from datetime import datetime
from enum import Enum
from json import JSONDecodeError
from typing import Any, Dict, Optional
from pathlib import Path
from uuid import uuid4


def setup_logger(name: str = "agent") -> logging.Logger:
    """
    Setup logger with timestamped file output and optional console output.
    Console output only appears when VERBOSE environment variable is set to 1.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"agent_execution_{timestamp}.log"

    # File handler - always writes to file
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    # file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


class Config:
    MODEL_GLM_46 = "zai-org/GLM-4.6-FP8"
    MODEL_KIMI_K2 = "moonshotai/Kimi-K2-Instruct"
    MODEL_DEEPSEEK_V3 = "deepseek-ai/DeepSeek-V3-0324"
    MODEL_QWEN3_CODER = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

    GATEWAY_URL = os.getenv("SANDBOX_PROXY_URL", "http://localhost:1234")
    AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1500")) - 50
    EVALUATION_RUN_ID = os.getenv("EVALUATION_RUN_ID", str(uuid4()))

    NOT_FOUND_LITERAL = "Not found"

    TEMPERATURE_BY_MODEL_TASK = {
        MODEL_GLM_46: {
            "planning": (0.3, 0.7),
            "testing": (0.4, 0.8),
            "coding": (0.0, 0.2),
            "summarizing": (0.0, 0.3),
            "evaluating": (0.0, 0.2),
        },
        MODEL_KIMI_K2: {
            "planning": (0.3, 0.7),
            "testing": (0.4, 0.8),
            "coding": (0.0, 0.2),
            "summarizing": (0.0, 0.3),
            "evaluating": (0.0, 0.2),
        },
        MODEL_DEEPSEEK_V3: {
            "planning": (0.3, 0.7),
            "testing": (0.4, 0.8),
            "coding": (0.0, 0.2),
            "summarizing": (0.0, 0.3),
            "evaluating": (0.0, 0.2),
        },
        MODEL_QWEN3_CODER: {
            "planning": (0.3, 0.7),
            "testing": (0.4, 0.8),
            "coding": (0.0, 0.2),
            "summarizing": (0.0, 0.3),
            "evaluating": (0.0, 0.2),
        },
    }

    MODEL_BY_TASK = {
        "planning": [
            MODEL_GLM_46,
            MODEL_GLM_46,
            MODEL_QWEN3_CODER,
            MODEL_KIMI_K2,
            MODEL_DEEPSEEK_V3,
        ],
        "testing": [
            MODEL_GLM_46,
            MODEL_GLM_46,
            MODEL_QWEN3_CODER,
            MODEL_KIMI_K2,
            MODEL_DEEPSEEK_V3,
        ],
        "coding": [
            MODEL_GLM_46,
            MODEL_GLM_46,
            MODEL_QWEN3_CODER,
            MODEL_KIMI_K2,
            MODEL_DEEPSEEK_V3,
        ],
        "summarizing": [
            MODEL_QWEN3_CODER,
            MODEL_QWEN3_CODER,
            MODEL_KIMI_K2,
            MODEL_GLM_46,
            MODEL_DEEPSEEK_V3,
        ],
        "evaluating": [
            MODEL_KIMI_K2,
            MODEL_KIMI_K2,
            MODEL_DEEPSEEK_V3,
            MODEL_QWEN3_CODER,
            MODEL_GLM_46,
        ],
    }

    logger = setup_logger()


# Error Types Enum
class ErrorType(Enum):
    FILE_NOT_FOUND = "file_not_found"
    DIRECTORY_NOT_FOUND = "directory_not_found"
    PERMISSION_DENIED = "permission_denied"
    NOT_A_FILE = "not_a_file"
    NOT_A_DIRECTORY = "not_a_directory"
    TARGET_NOT_FOUND = "target_not_found"
    INVALID_REGEX = "invalid_regex"
    INVALID_PARAMETER = "invalid_parameter"
    COMMAND_TIMEOUT = "command_timeout"
    COMMAND_NOT_FOUND = "command_not_found"
    COMMAND_FAILED = "command_failed"
    READ_ERROR = "read_error"
    WRITE_ERROR = "write_error"
    ENCODING_ERROR = "encoding_error"
    UNKNOWN_ERROR = "unknown_error"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    HTTP_ERROR = "http_error"
    INVALID_REQUEST = "invalid_request"
    INVALID_JSON = "invalid_json"
    INFINITE_INFERENCE = "infinite_inference"
    AGENT_TIMEOUT = "agent_timeout"
    PLANNER_FAILED = "planner_failed"
    TESTER_FAILED = "tester_failed"
    CODER_FAILED = "coder_failed"


# Custom Exception Class
class CustomError(Exception):
    """Unified exception class for all operations."""

    def __init__(self, error_type: ErrorType, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"CustomError {self.error_type} {self.message}"


class Utils:
    @classmethod
    def get_git_patch(cls) -> str:
        """
        Generates git diff patch containing all modifications in working directory
        Useful for capturing comprehensive change summary before finalization
        """
        try:
            command = f"""
            cp .gitignore .gitignore.backup 2>/dev/null || true
            echo 'src/agent.py' >> .gitignore
            echo 'src/agent_runner.py' >> .gitignore
            
            git add .

            git diff --cached > .patch.txt
            cat .patch.txt

            mv .gitignore.backup .gitignore 2>/dev/null || true
            """

            output = subprocess.run(
                ["bash", "-c", command], timeout=30, capture_output=True, text=True
            )

            # output = output.stdout.decode("utf-8") + '\n' + output.stderr.decode("utf-8")
            return output.stdout
        except Exception as e:
            return f"Error generating git patch: {e}"

    @classmethod
    def count_tokens(cls, messages: list | str) -> int:
        if isinstance(messages, list):
            text = " ".join(
                str(m.get("content", "") if isinstance(m, dict) else m)
                for m in messages
            )
        else:
            text = messages

        tokens = re.findall(r"\w+|[^\w\s]|\s+", text)
        count = 0

        for token in tokens:
            if token.isspace():
                continue
            elif len(token) == 1:
                count += 1
            else:
                count += max(1, (len(token) + 2) // 3)

        return count

    @classmethod
    def is_valid_directory(cls, name: str) -> bool:
        # Ignore hidden directories (starting with .)
        if name.startswith("."):
            return False

        ignore_directories = {
            ".git",
            "__pycache__",
            ".pytest_cache",
            "node_modules",
            ".tox",
            ".venv",
            "venv",
            ".mypy_cache",
            ".eggs",
            "build",
            "dist",
        }
        # Ignore directories in the ignore list
        if name in ignore_directories:
            return False

        # Ignore .egg-info directories
        if name.endswith(".egg-info"):
            return False

        return True

    @classmethod
    def is_valid_file(cls, name: str, only_code_relative: bool = False) -> bool:
        name = name.lower()

        if name in [".env.example", ".gitignore", ".dockerignore"]:
            return True
        # Ignore hidden files (starting with .)
        if name.startswith("."):
            return False

        # Ignore .egg-info files
        if name.endswith(".egg-info"):
            return False

        if only_code_relative:
            code_extensions = {
                # Python files
                ".py",
                ".pyi",
                ".pyx",
                # Node.js / JavaScript / TypeScript files
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
                ".mjs",
                ".cjs",
                # Configuration files
                ".json",
                ".yaml",
                ".yml",
                ".toml",
                ".ini",
                ".cfg",
                # Project files
                "Dockerfile",
                ".dockerignore",
                "requirements.txt",
                ".gitignore",
                ".env.example",
            }
            if not any(
                name.endswith(ext) if ext.startswith(".") else name == ext
                for ext in code_extensions
            ):
                return False

        return True

    @classmethod
    def inference(
        cls,
        messages: list,
        task_type: str,
        tool_mode: str = "auto",
        tools: list = [],
        max_retries: int = 5,
        model_indx: int = 0,
    ) -> dict:
        retries = 0
        url = f"{Config.GATEWAY_URL.rstrip('/')}/api/inference"
        headers = {"Content-Type": "application/json"}
        request_data = {
            "evaluation_run_id": Config.EVALUATION_RUN_ID,
            "messages": messages,
            "tool_mode": tool_mode,
            "tools": tools,
        }

        while True:
            try:
                err = None
                model = Config.MODEL_BY_TASK[task_type][
                    (model_indx + retries) % len(Config.MODEL_BY_TASK[task_type])
                ]
                model_temperature = Config.TEMPERATURE_BY_MODEL_TASK[model][task_type]
                temperature = random.uniform(model_temperature[0], model_temperature[1])
                request_data["model"] = model
                request_data["temperature"] = temperature

                response = requests.post(
                    url, json=request_data, timeout=(30, 150), headers=headers
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.Timeout:
                err = CustomError(ErrorType.TIMEOUT, f"model: {model}")
            except requests.exceptions.ConnectionError as e:
                err = CustomError(
                    ErrorType.CONNECTION_ERROR,
                    f"model: {model} error: {e}",
                )
            except requests.exceptions.HTTPError as e:
                err = CustomError(
                    ErrorType.HTTP_ERROR,
                    f"model: {model} status: {e.response.status_code} error: {e}",
                )
            except requests.exceptions.RequestException as e:
                err = CustomError(
                    ErrorType.INVALID_REQUEST,
                    f"model: {model} error: {e}",
                )
            except JSONDecodeError as e:
                err = CustomError(
                    ErrorType.INVALID_JSON,
                    f"model: {model} error: {e}",
                )
            except Exception as e:
                err = CustomError(ErrorType.UNKNOWN_ERROR, f"model: {model} error: {e}")
            finally:
                if err:
                    retries += 1
                    Config.logger.error(
                        f"[INFERENCE]: ({retries}/{max_retries}) \n\t{err}"
                    )
                else:
                    Config.logger.info(f"[INFERENCE]: {model}")
                if retries >= max_retries:
                    raise err

    @classmethod
    def _get_dir_tree(cls, path: str = "/", depth: int = 1) -> str:
        if not os.path.exists(path):
            raise CustomError(
                ErrorType.DIRECTORY_NOT_FOUND, f"Directory does not exist: {path}"
            )

        if not os.path.isdir(path):
            raise CustomError(
                ErrorType.NOT_A_DIRECTORY, f"Path is not a directory: {path}"
            )

        def tree(dir_path: str, prefix: str = "", current_depth: int = 0) -> list[str]:
            if current_depth > depth:
                return []
            try:
                items = sorted(os.listdir(dir_path))
                dirs = []
                files = []

                for item in items:
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path):
                        if cls.is_valid_directory(item):
                            dirs.append(item)
                    elif os.path.isfile(item_path):
                        if cls.is_valid_file(item):
                            files.append(item)

                entries = []
                for i, d in enumerate(dirs):
                    is_last = i == len(dirs) - 1 and not files
                    entries.append(f"{prefix}{'└── ' if is_last else '├── '}{d}/")
                    if current_depth < depth:
                        entries.extend(
                            tree(
                                os.path.join(dir_path, d),
                                prefix + ("    " if is_last else "│   "),
                                current_depth + 1,
                            )
                        )
                for i, f in enumerate(files):
                    entries.append(
                        f"{prefix}{'└── ' if i == len(files) - 1 else '├── '}{f}"
                    )
                return entries
            except PermissionError as e:
                raise CustomError(
                    ErrorType.PERMISSION_DENIED,
                    f"Permission denied accessing directory: {dir_path} ({str(e)})",
                )
            except Exception as e:
                raise CustomError(
                    ErrorType.READ_ERROR,
                    f"Error reading directory {dir_path}: {str(e)}",
                )

        lines = tree(path, "", 0)
        dir_count = sum(1 for l in lines if l.rstrip().endswith("/"))
        file_count = sum(
            1 for l in lines if not l.rstrip().endswith("/") and "[" not in l
        )
        return (
            f"Directory structure (depth={depth}):\n{path}/\n"
            + "\n".join(lines)
            + f"\n\n{dir_count}-dirs, {file_count}-files"
        )

    @classmethod
    def _read_file(cls, path: str, start_line: int = 1, limit: int = 500) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                total_lines = len(lines)
                start_idx = max(0, start_line - 1)
                end_idx = min(start_idx + limit, total_lines)
                result_lines = [
                    f"File: {path} (lines {start_line}-{end_idx-1})\n",
                    "```\n",
                ]
                result_lines.extend(lines[start_idx:end_idx])

                if end_idx < total_lines:
                    result_lines.append(f"\n```({total_lines - end_idx} more lines)")

                return "".join(result_lines)
        except FileNotFoundError:
            raise CustomError(ErrorType.FILE_NOT_FOUND, f"File not found: {path}")
        except PermissionError:
            raise CustomError(
                ErrorType.PERMISSION_DENIED,
                f"Permission denied while reading file: {path}",
            )
        except IsADirectoryError:
            raise CustomError(
                ErrorType.NOT_A_FILE, f"Path is a directory, not a file: {path}"
            )
        except UnicodeDecodeError as e:
            raise CustomError(
                ErrorType.ENCODING_ERROR,
                f"Unable to decode file {path} with UTF-8 encoding: {str(e)}",
            )
        except Exception as e:
            raise CustomError(
                ErrorType.READ_ERROR, f"Error reading file {path}: {str(e)}"
            )

    @classmethod
    def _search_in_file(
        cls,
        path: str,
        term: str,
        match_type: str = "exact",
        context_lines: int = 5,
        occurrences_limit: int = 20,
    ) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                source_lines = f.read().splitlines()
        except FileNotFoundError:
            raise CustomError(ErrorType.FILE_NOT_FOUND, f"File not found: {path}")
        except PermissionError:
            raise CustomError(
                ErrorType.PERMISSION_DENIED,
                f"Permission denied while reading file: {path}",
            )
        except Exception as e:
            raise CustomError(
                ErrorType.READ_ERROR, f"Error reading file {path}: {str(e)}"
            )

        # Determine if term is multi-line
        term_lines = term.splitlines()
        is_multiline = len(term_lines) > 1

        match_lines = []

        if match_type == "exact":
            if is_multiline:
                # Multi-line exact matching
                # Search for code parts that ends with first line of term,
                # rest lines matching, starts with last line of term
                first_line_term = term_lines[0]
                middle_lines_term = term_lines[1:-1] if len(term_lines) > 2 else []
                last_line_term = term_lines[-1]

                i = 0
                while i < len(source_lines):
                    # Check if current line ends with first line of term
                    if source_lines[i].rstrip().endswith(first_line_term.rstrip()):
                        # Check if we have enough lines ahead
                        if i + len(term_lines) - 1 < len(source_lines):
                            match = True

                            # Check middle lines for exact match
                            for j, middle_term in enumerate(middle_lines_term):
                                if (
                                    source_lines[i + 1 + j].strip()
                                    != middle_term.strip()
                                ):
                                    match = False
                                    break

                            # Check if last line starts with last line of term
                            if match and source_lines[
                                i + len(term_lines) - 1
                            ].lstrip().startswith(last_line_term.lstrip()):
                                match_lines.append(
                                    i + 1
                                )  # Store the starting line (1-based)
                                i += len(term_lines)
                                continue
                    i += 1
            else:
                # Single-line exact matching
                match_lines = [
                    idx + 1 for idx, line in enumerate(source_lines) if term in line
                ]

        elif match_type == "regex":
            try:
                if is_multiline:
                    # Multi-line regex matching
                    full_text = "\n".join(source_lines)
                    pattern = re.compile(term, re.MULTILINE | re.DOTALL)

                    for match in pattern.finditer(full_text):
                        # Find which line this match starts on
                        start_pos = match.start()
                        line_num = full_text[:start_pos].count("\n") + 1
                        match_lines.append(line_num)
                else:
                    # Single-line regex matching
                    pattern = re.compile(term)
                    match_lines = [
                        idx + 1
                        for idx, line in enumerate(source_lines)
                        if pattern.search(line)
                    ]
            except re.error as e:
                raise CustomError(
                    ErrorType.INVALID_REGEX,
                    f"Invalid regex pattern '{term}': {str(e)}",
                )
        else:
            raise CustomError(
                ErrorType.INVALID_PARAMETER,
                f"Invalid match_type '{match_type}'. Must be 'exact' or 'regex'",
            )

        if not match_lines:
            return Config.NOT_FOUND_LITERAL

        chunks: list[str] = [f"=== {path} ==="]
        last_line = -1

        for ln in match_lines[:occurrences_limit]:
            start_line = max(1, ln - context_lines)
            end_line = min(len(source_lines), ln + context_lines)

            if start_line - 1 > last_line:
                chunks.append(f"\n(lines {start_line}-{end_line}):")
            if end_line > last_line:
                chunks.append(
                    "\n".join(source_lines[max(start_line - 1, last_line) : end_line])
                )

            last_line = max(last_line, end_line)

        remaining = len(match_lines) - occurrences_limit
        if remaining > 0:
            chunks.append(f"\n...({remaining} more occurrences)")

        return "\n".join(chunks)

    @classmethod
    def _search_in_directory(
        cls,
        path: str,
        term: str,
        match_type: str = "exact",
        context_lines: int = 5,
        occurrences_limit: int = 20,
        files_limit: int = 10,
    ) -> str:
        """
        Search all Python and Node.js relevant files in a directory for the specified term.
        """
        if not os.path.exists(path):
            raise CustomError(
                ErrorType.DIRECTORY_NOT_FOUND, f"Directory does not exist: {path}"
            )

        if not os.path.isdir(path):
            raise CustomError(
                ErrorType.NOT_A_DIRECTORY, f"Path is not a directory: {path}"
            )

        try:
            results = []
            for root, dirs, files in os.walk(path):
                # Filter out invalid directories
                dirs[:] = [d for d in dirs if cls.is_valid_directory(d)]

                for file in files:
                    # Skip invalid files
                    if not cls.is_valid_file(file, True):
                        continue

                    file_path = os.path.join(root, file)
                    try:
                        match_result = cls._search_in_file(
                            file_path,
                            term,
                            match_type,
                            context_lines,
                            occurrences_limit,
                        )
                        if match_result != Config.NOT_FOUND_LITERAL:
                            results.append(match_result)
                        if len(results) >= files_limit:
                            results.append(
                                "\n=== Reached maximum number of files with occurrences. Refile your search term into more specific terms. ==="
                            )
                            break
                    except CustomError:
                        # File doesn't contain the term or couldn't be read
                        continue

                if len(results) >= files_limit:
                    break

            if not results:
                return Config.NOT_FOUND_LITERAL

            return "\n".join(results)
        except PermissionError as e:
            raise CustomError(
                ErrorType.PERMISSION_DENIED,
                f"Permission denied while accessing directory {path}: {str(e)}",
            )
        except CustomError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise CustomError(
                ErrorType.UNKNOWN_ERROR, f"Error searching directory {path}: {str(e)}"
            )

    @classmethod
    def _search(
        cls,
        path: str,
        term: str,
        match_type: str = "exact",
        context_lines: int = 5,
        occurrences_limit: int = 2,
        files_limit: int = 5,
    ) -> str:
        """
        Search for occurrences of a specified string within a file or all files in a directory.
        """
        # Check if path exists
        if not os.path.exists(path):
            # Determine the appropriate error type
            if os.path.isdir(path):
                raise CustomError(
                    ErrorType.DIRECTORY_NOT_FOUND, f"Directory does not exist: {path}"
                )
            else:
                raise CustomError(
                    ErrorType.FILE_NOT_FOUND, f"File does not exist: {path}"
                )

        # Determine if path is a file or directory
        if os.path.isfile(path):
            # Search in a single file (files_limit is ignored)
            return (
                f'Result of searching "{term}" with type of "{match_type}" match in file {path} is:\n'
                + cls._search_in_file(
                    path, term, match_type, context_lines, occurrences_limit
                )
            )
        elif os.path.isdir(path):
            # Search in a directory
            return (
                f'Result of searching "{term}" with type of "{match_type}" match in directory {path} is:\n'
                + cls._search_in_directory(
                    path,
                    term,
                    match_type,
                    context_lines,
                    occurrences_limit,
                    files_limit,
                )
            )
        else:
            raise CustomError(
                ErrorType.INVALID_PARAMETER,
                f"Path is neither a file nor a directory: {path}",
            )

    @classmethod
    def _edit_file(
        cls,
        path: str,
        edit_type: str,
        edit_content: str,
        target_type: str,
        target_content: str,
    ) -> str:
        """
        Apply an edit operation to a file, modifying every occurrence of a specified target string or pattern.
        """
        # Validate edit_type
        valid_edit_types = ["insert_before", "insert_after", "replace", "delete"]
        if edit_type not in valid_edit_types:
            raise CustomError(
                ErrorType.INVALID_PARAMETER,
                f"Invalid edit_type '{edit_type}'. Must be one of {valid_edit_types}",
            )

        # Validate target_type
        valid_target_types = ["exact", "regex"]
        if target_type not in valid_target_types:
            raise CustomError(
                ErrorType.INVALID_PARAMETER,
                f"Invalid target_type '{target_type}'. Must be one of {valid_target_types}",
            )

        # Read the file
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except FileNotFoundError:
            raise CustomError(ErrorType.FILE_NOT_FOUND, f"File not found: {path}")
        except PermissionError:
            raise CustomError(
                ErrorType.PERMISSION_DENIED,
                f"Permission denied while reading file: {path}",
            )
        except Exception as e:
            raise CustomError(
                ErrorType.READ_ERROR, f"Error reading file {path}: {str(e)}"
            )

        # Find all match locations first
        match_infos = []  # List of (start_pos, end_pos, matched_text)

        if target_type == "exact":
            occurrences = content.count(target_content)
            if occurrences == 0:
                raise CustomError(
                    ErrorType.TARGET_NOT_FOUND,
                    f"Target content not found in file: {path}",
                )

            pos = 0
            while True:
                pos = content.find(target_content, pos)
                if pos == -1:
                    break
                match_infos.append((pos, pos + len(target_content), target_content))
                pos += len(target_content)

        elif target_type == "regex":
            try:
                pattern = re.compile(target_content)
            except re.error as e:
                raise CustomError(
                    ErrorType.INVALID_REGEX,
                    f"Invalid regex pattern '{target_content}': {str(e)}",
                )

            matches = list(pattern.finditer(content))
            if len(matches) == 0:
                raise CustomError(
                    ErrorType.TARGET_NOT_FOUND,
                    f"Target pattern not found in file: {path}",
                )

            for match in matches:
                match_infos.append((match.start(), match.end(), match.group(0)))

        occurrences = len(match_infos)

        # Apply edits one by one from end to start to maintain correct positions
        # and collect snippets after each edit
        snippets = []
        context_lines = 3

        # Process matches in reverse order to maintain position accuracy
        for match_start, match_end, matched_text in reversed(match_infos):
            # Determine replacement text
            if edit_type == "insert_before":
                replacement = edit_content + matched_text
            elif edit_type == "insert_after":
                replacement = matched_text + edit_content
            elif edit_type == "replace":
                replacement = edit_content
            elif edit_type == "delete":
                replacement = ""

            # Apply the edit
            content = content[:match_start] + replacement + content[match_end:]

            # Calculate the line range affected by this edit
            # Find the starting line of the match
            start_line_num = content[:match_start].count("\n") + 1

            # Count lines in the replacement text
            replacement_line_count = replacement.count("\n")

            # Calculate how many lines the matched text had
            matched_line_count = matched_text.count("\n")

            # Get content lines
            content_lines = content.splitlines(keepends=True)

            # Calculate snippet range to show all affected lines plus context
            # Start: original match line minus context
            snippet_start = max(1, start_line_num - context_lines)

            # End: match start + replacement lines + context
            snippet_end = min(
                len(content_lines),
                start_line_num + replacement_line_count + context_lines,
            )

            # Extract snippet
            snippet_lines = content_lines[snippet_start - 1 : snippet_end]
            snippet = "".join(snippet_lines).rstrip()

            # Store snippet info (will reverse later to show in original order)
            snippets.append(
                {"start": snippet_start, "end": snippet_end, "content": snippet}
            )

        # Write the modified content back to the file
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
        except PermissionError:
            raise CustomError(
                ErrorType.PERMISSION_DENIED,
                f"Permission denied while writing to file: {path}",
            )
        except Exception as e:
            raise CustomError(
                ErrorType.WRITE_ERROR, f"Error writing to file {path}: {str(e)}"
            )

        # Generate result with snippets (reverse to show in original order)
        result_parts = [f"Updated {occurrences} parts in {path} file.\n"]

        for snippet_info in reversed(snippets):
            result_parts.append(
                f"\nFile: {path} (lines {snippet_info['start']}-{snippet_info['end']})"
            )
            result_parts.append(f"```\n{snippet_info['content']}\n```")

        return "\n".join(result_parts)

    @classmethod
    def _create_file(cls, path: str, content: str = "", overwrite: bool = True) -> str:
        """
        Create a new file at the specified path with the given content.
        """
        try:
            # Check if file already exists
            if os.path.exists(path):
                if not overwrite:
                    return f"File already exists: {path}. If you want to overwrite it, set the 'overwrite' argument to True."
                # If overwrite is True, continue with file creation (will overwrite)

            # Create parent directories if they don't exist
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                try:
                    os.makedirs(parent_dir, exist_ok=True)
                except PermissionError:
                    raise CustomError(
                        ErrorType.PERMISSION_DENIED,
                        f"Permission denied while creating directory: {parent_dir}",
                    )
                except Exception as e:
                    raise CustomError(
                        ErrorType.WRITE_ERROR,
                        f"Error creating directory {parent_dir}: {str(e)}",
                    )

            # Write content to the file
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except PermissionError:
                raise CustomError(
                    ErrorType.PERMISSION_DENIED,
                    f"Permission denied while creating file: {path}",
                )
            except Exception as e:
                raise CustomError(
                    ErrorType.WRITE_ERROR, f"Error writing to file {path}: {str(e)}"
                )

            # Generate preview of created file
            lines = content.splitlines()
            total_lines = len(lines)
            preview_limit = 10
            preview_lines = lines[:preview_limit]

            result_parts = [f"Successfully created file: {path}"]
            if total_lines > 0:
                result_parts.append(
                    f"\nFile preview (lines 1-{min(preview_limit, total_lines)}):"
                )
                result_parts.append("```")
                result_parts.extend(preview_lines)
                if total_lines > preview_limit:
                    result_parts.append(
                        f"```\n({total_lines - preview_limit} more lines)"
                    )
                else:
                    result_parts.append("```")
            else:
                result_parts.append("(empty file)")

            return "\n".join(result_parts)
        except CustomError:
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            raise CustomError(
                ErrorType.UNKNOWN_ERROR, f"Error creating file {path}: {str(e)}"
            )

    @classmethod
    def _run_bash_command(
        cls, command: str = "ls", cwd: str = None, timeout: int = 30
    ) -> str:
        """
        Execute a bash command and return its stdout.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )

            output_parts = [f'Result of running bash command "{command}" is:\n\n']
            if result.stdout:
                output_parts.append(result.stdout.rstrip())
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr.rstrip()}")
            if result.returncode != 0:
                output_parts.append(f"Return code: {result.returncode}")

            return "\n".join(output_parts) if output_parts else "(no output)"
        except subprocess.TimeoutExpired:
            raise CustomError(
                ErrorType.COMMAND_TIMEOUT,
                f"Command timed out after {timeout} seconds: {command}",
            )
        except FileNotFoundError:
            raise CustomError(
                ErrorType.COMMAND_NOT_FOUND,
                f"Command or working directory not found: {command}"
                + (f" (cwd: {cwd})" if cwd else ""),
            )
        except PermissionError:
            raise CustomError(
                ErrorType.PERMISSION_DENIED,
                f"Permission denied while executing command: {command}",
            )
        except Exception as e:
            raise CustomError(
                ErrorType.COMMAND_FAILED,
                f"Error executing command '{command}': {str(e)}",
            )


class Tools:
    ALL_TOOLS = {
        "get_dir_tree",
        "read_file",
        "create_file",
        "edit_file",
        "search",
        "run_bash_command",
    }

    def __init__(self):
        pass

    @classmethod
    def documentation(cls, available_tools: set = None):
        """
        Generate documentation for all tools in the Tools class.

        Arguments:
            available_tools: Optional set of tool names to generate documentation for.
                If None, generates documentation for all tools in ALL_TOOLS.

        Returns:
            List of documentation objects, one for each tool.
        """
        # Determine which tools to document
        tools_to_document = (
            available_tools if available_tools is not None else cls.ALL_TOOLS.copy()
        )

        # Type mapping from Python types to JSON schema types
        type_mapping = {
            bool: "boolean",
            int: "integer",
            float: "number",
            str: "string",
            list: "array",
            tuple: "array",
            set: "array",
            dict: "object",
        }

        documentation = []

        for tool_name in tools_to_document:
            # Skip if tool not in ALL_TOOLS
            if tool_name not in cls.ALL_TOOLS:
                continue

            # Get the method
            method = getattr(cls, tool_name, None)
            if method is None:
                continue

            # Get method signature
            sig = inspect.signature(method)

            # Get docstring
            docstring = inspect.getdoc(method)
            if not docstring:
                docstring = ""

            # Parse docstring to extract description and parameter descriptions
            lines = docstring.split("\n")
            description_lines = []
            param_descriptions = {}

            in_arguments_section = False
            current_param = None

            for line in lines:
                line_stripped = line.strip()

                # Check if we've entered the Arguments section
                if line_stripped.lower() in ["arguments:", "parameters:", "args:"]:
                    in_arguments_section = True
                    continue

                # If in arguments section, parse parameter descriptions
                if in_arguments_section:
                    # Check if this is a parameter line (starts with word followed by colon)
                    if (
                        line_stripped
                        and ":" in line_stripped
                        and not line_stripped.startswith(" ")
                    ):
                        param_match = line_stripped.split(":", 1)
                        if len(param_match) == 2:
                            current_param = param_match[0].strip()
                            param_descriptions[current_param] = param_match[1].strip()
                    # Continuation of previous parameter description
                    elif current_param and line_stripped:
                        param_descriptions[current_param] += " " + line_stripped
                else:
                    # Collect description lines (before Arguments section)
                    if line_stripped:
                        description_lines.append(line_stripped)

            # Join description lines
            description = " ".join(description_lines)

            # Build parameters list
            parameters = []
            for param_name, param in sig.parameters.items():
                # Skip 'self' and 'cls' parameters
                if param_name in ["self", "cls"]:
                    continue

                # Determine JSON type
                json_type = "string"  # default
                if param.annotation != inspect.Parameter.empty:
                    # Get the base type (handle Optional and Union types)
                    param_type = param.annotation

                    # Handle string annotations
                    if isinstance(param_type, str):
                        type_str = param_type.lower()
                        if "int" in type_str:
                            json_type = "integer"
                        elif "float" in type_str:
                            json_type = "number"
                        elif "bool" in type_str:
                            json_type = "boolean"
                        elif (
                            "list" in type_str
                            or "tuple" in type_str
                            or "set" in type_str
                        ):
                            json_type = "array"
                        elif "dict" in type_str:
                            json_type = "object"
                        else:
                            json_type = "string"
                    else:
                        # Direct type mapping
                        json_type = type_mapping.get(param_type, "string")

                # Determine if required (no default value)
                required = param.default == inspect.Parameter.empty

                # Get parameter description from docstring
                param_desc = param_descriptions.get(param_name, "")

                parameters.append(
                    {
                        "type": json_type,
                        "name": param_name,
                        "description": param_desc,
                        "required": required,
                    }
                )

            # Add tool documentation
            documentation.append(
                {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters,
                }
            )

        return documentation

    @classmethod
    def get_dir_tree(cls, path: str = "/", depth: int = 0) -> str:
        """
        Generate a hierarchical representation of the specified directory, recursively enumerating subdirectories and files up to the given depth.

        Arguments:
            path: Absolute path to the root directory from which to build the tree. Defaults to "/".
            depth: Maximum traversal depth. A depth of 0 lists only immediate children. Defaults to 0.
        """
        try:
            return Utils._get_dir_tree(path, depth)
        except CustomError as e:
            return f"Error [{e.error_type.value}]: {e.message}"

    @classmethod
    def read_file(cls, path: str, start_line: int = 1, limit: int = 500) -> str:
        """
        Read a portion of a file's contents.
        Recommended to read minimal portion that is only relevant.
        Avoid reading the same code parts repeatedly.

        Arguments:
            path: Absolute path to the target file.
            start_line: The 1-based line number at which reading begins. Defaults to 1.
            limit: Maximum number of lines to return starting from start_line. Defaults to 500.
        """
        try:
            return Utils._read_file(path, start_line, limit)
        except CustomError as e:
            return f"Error [{e.error_type.value}]: {e.message}"

    @classmethod
    def create_file(cls, path: str, content: str = "", overwrite: bool = True) -> str:
        """
        Create a new file at the specified path with the given content. If the parent directories do not exist,
        they will be created automatically.

        Arguments:
            path: Absolute path where the new file should be created.
            content: The content to write to the new file. Defaults to an empty string (creating an empty file).
            overwrite: If True, overwrite the file if it already exists. If False, return a message indicating the file exists. Defaults to True.
        """
        try:
            return Utils._create_file(path, content, overwrite)
        except CustomError as e:
            return f"Error [{e.error_type.value}]: {e.message}"

    @classmethod
    def edit_file(
        cls,
        path: str,
        edit_type: str,
        edit_content: str,
        target_type: str,
        target_content: str,
    ) -> str:
        """
        Apply an edit operation to a file by modifying every occurrence of a specified target string or pattern.

        This function scans the file, identifies all matches for `target_content`, and applies the specified edit action
        to each occurrence. Supported actions include inserting text before or after the match, replacing the match, or
        deleting the match entirely.

        The function returns:
        - A success message in the format:
                "Successfully edited {number_of_occurrences} parts in {path}."
        - An error message if the edit operation fails (e.g., invalid path, unreadable file, or invalid pattern).

        Important behavior:
        - Edits should be applied based on the current state of the file. Avoid referencing or targeting text that was
            already modified earlier and always make sure to use latest version of content to be targeting.
        - If no occurrences of the target are found on the first pass, the function should re-read the file and attempt
            the match again to ensure the latest content is used.

        Arguments:
            path: Absolute path to the file to edit.
            edit_type: The type of edit to perform — one of: "insert_before", "insert_after", "replace", or "delete".
                Note: "insert_before" and "insert_after" do NOT add extra newlines. They insert the provided content directly before or after the target content without introducing additional line breaks.
            edit_content: Content used for the edit operation. Ignored when edit_type is "delete".
            target_type: Matching strategy, either "exact" for literal matching or "regex" for pattern matching.
            target_content: The text or pattern to search for within the file.
        """
        try:
            return Utils._edit_file(
                path, edit_type, edit_content, target_type, target_content
            )
        except CustomError as e:
            return f"Error [{e.error_type.value}]: {e.message}"

    @classmethod
    def search(
        cls,
        path: str,
        term: str,
        match_type: str = "exact",
        context_lines: int = 5,
        occurrences_limit: int = 2,
        files_limit: int = 5,
    ) -> str:
        """
        Search for occurrences of a specified string within a file or all files in a directory.
        Behavior:
            - If `path` points to a file: search only within that file.
            - If `path` points to a directory: recursively search all files under that directory.
            - The `files_limit` parameter is ignored when searching a single file.

        Arguments:
            path: Absolute path to a file or directory to search.
            term: The search term, which may be single-line or multi-line.
            match_type: "exact" for literal matching, or "regex" for regex pattern matching.
            context_lines: Number of surrounding context lines to include before and after each match. Defaults to 5.
            occurrences_limit: Maximum number of occurrences to return per file. Defaults to 2.
            files_limit: Maximum number of files to return when searching a directory. Ignored when `path` is a file.
        """
        try:
            return Utils._search(
                path, term, match_type, context_lines, occurrences_limit, files_limit
            )
        except CustomError as e:
            return f"Error [{e.error_type.value}]: {e.message}"

    @classmethod
    def search_in_file(
        cls,
        path: str,
        term: str,
        match_type: str = "exact",
        context_lines: int = 5,
        occurrences_limit: int = 3,
    ) -> str:
        """
        Search the given file for all occurrences of the specified string, returning the matches in sequential order and optionally including surrounding context.

        Arguments:
            path: Absolute path to the file to search.
            term: The search term, which may be single-line or multi-line.
            match_type: "exact" for literal matching, or "regex" for regex pattern matching.
            context_lines: Number of surrounding context lines to include before and after each matched region. Defaults to 5.
            occurrences_limit: Maximum number of occurrences to return. Defaults to 3.
        """
        try:
            return Utils._search_in_file(
                path, term, match_type, context_lines, occurrences_limit
            )
        except CustomError as e:
            return f"Error [{e.error_type.value}]: {e.message}"

    @classmethod
    def search_in_directory(
        cls,
        path: str,
        term: str,
        match_type: str = "exact",
        context_lines: int = 5,
        occurrences_limit: int = 2,
        files_limit: int = 10,
    ) -> str:
        """
        Search all files within the specified directory for the given string and return all occurrences found.

        Arguments:
            path: Absolute path to the directory to search.
            term: String to search for, which may span multiple lines.
            match_type: "exact" for literal matching, or "regex" for regex pattern matching.
            context_lines: Number of context lines to include before and after each match. Defaults to 5.
            occurrences_limit: Maximum number of occurrences to return per file. Defaults to 2.
            files_limit: Maximum number of files with occurrences to return. Defaults to 10.
        """
        try:
            return Utils._search_in_directory(
                path, term, match_type, context_lines, occurrences_limit, files_limit
            )
        except CustomError as e:
            return f"Error [{e.error_type.value}]: {e.message}"

    @classmethod
    def run_bash_command(cls, command: str = "ls") -> str:
        """
        Execute a Linux bash command and return the command's stdout.
        The bash environment includes python, pytest, pip, node, jest, npm.

        Arguments:
            command: The bash command string to run. Defaults to "ls".
        """
        try:
            return Utils._run_bash_command(command)
        except CustomError as e:
            return f"Error [{e.error_type.value}]: {e.message}"


class TimeoutHandler:
    """Handler for managing execution timeouts with graceful interruption."""

    def __init__(self, timeout_seconds: int):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.is_timed_out = False

    def start(self):
        """Start the timeout timer."""
        self.start_time = datetime.now()
        self.is_timed_out = False

    def check(self) -> bool:
        """Check if timeout has been exceeded. Returns True if timed out."""
        if self.start_time is None:
            return False
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > self.timeout_seconds:
            self.is_timed_out = True
            return True
        return False

    def remaining_time(self) -> float:
        """Get remaining time in seconds."""
        if self.start_time is None:
            return self.timeout_seconds
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return max(0, self.timeout_seconds - elapsed)


class Agent:
    def __init__(
        self,
        system_inst: str,
        user_inst: str,
        eval_inst: str,
        task_type: str,
        repo_dir: str,
        summary_length: int = 15,
        retain_length: int = 15,
        evaluate_result: bool = True,
        timeout_handler: Optional[TimeoutHandler] = None,
    ):
        self.messages = [
            {"role": "system", "content": system_inst},
            {"role": "user", "content": user_inst},
            {
                "role": "system",
                "content": f"Your working directory is [{repo_dir}]. Always use the SAME string for the path to your working directory.",
            },
        ]
        self.base_length = 3
        self.system_inst = system_inst
        self.user_inst = user_inst
        self.eval_inst = eval_inst
        self.task_type = task_type
        self.summarization = ""
        self.summary_length = summary_length
        self.retain_length = retain_length
        self.evaluate_result = evaluate_result
        self.result: Optional[str] = None
        self.tool_calls: list = []
        self.tool_outputs: list = []
        self.timeout_handler = timeout_handler

    def _messages_with_simmarization(self):
        if not self.summarization:
            return self.messages

        return (
            self.messages[: self.base_length]
            + [{"role": "user", "content": self.summarization}]
            + self.messages[self.base_length :]
        )

    def _summarize(self):
        if len(self.messages) - self.retain_length < self.summary_length:
            return

        Config.logger.info(
            f"[SUMMARIZING({self.task_type})]: {self.summary_length} messages..."
        )

        messages_to_summarize = self.messages[
            self.base_length : self.base_length + self.summary_length
        ]
        compact_content = (
            f"""
This is previous summarization.
{self.summarization}\n\n
"""
            if self.summarization
            else ""
        )
        compact_content += f"""
These are tools used in the steps below.
{Tools.documentation()}\n\n
"""

        messages_content = ""
        for msg in messages_to_summarize:
            messages_content += f"[{msg['role']}]: {msg.get('content', '')}\n"

        compact_content += f"""
This is inference history.
{messages_content}
"""

        # Create summarization using inference
        messages = [
            {
                "role": "system",
                "content": Prompts.SUMMARIZING_INSTRUCTION.format(self.user_inst),
            },
            {"role": "user", "content": compact_content},
        ]
        response = Utils.inference(
            messages,
            "summarizing",
            "none",
        )
        self.summarization = response["content"]

        Config.logger.info(
            f"[SUMMARIZED({self.task_type})]: {Utils.count_tokens(messages_content)} -> {Utils.count_tokens(self.summarization)} tokens \n{self.summarization}"
        )
        self.messages = (
            self.messages[: self.base_length]
            + self.messages[self.base_length + self.summary_length :]
        )

    def _is_same_tool_calls(self, i: int, j: int) -> bool:
        return (
            self.tool_calls[i]["names"] == self.tool_calls[j]["names"]
            and self.tool_calls[i]["args"] == self.tool_calls[j]["args"]
        )

    def _rebase_to_tool_call(self, i: int):
        idx = self.tool_calls[i]["index"]
        self.messages = self.messages[: idx + 1]
        self.tool_calls = self.tool_calls[: i + 1]

    def _check_and_rebase_repeated_tool_call(self) -> bool:
        max_repeat = 2
        for repeat in range(1, max_repeat + 1):
            if len(self.tool_calls) < repeat * 2:
                break
            repeated = True
            for i in range(repeat):
                if not self._is_same_tool_calls(-2 * repeat + i, -repeat + i):
                    repeated = False
                    break
            if repeated:
                Config.logger.warning(f"[REPEATED({self.task_type})]: {repeat} tools")
                for i in range(repeat):
                    Config.logger.warning(
                        f"\t[TOOL CALL {i+1}({self.task_type})]: {json.dumps(self.tool_calls[-repeat+i])}\n"
                    )
                self._rebase_to_tool_call(-2 * repeat)
                return True
        return False

    def _eval_result(self, result_like) -> bool:
        return self.eval_inst in result_like

    def _tool_call_fault_tolerant(self, content):
        try:
            response = Utils.inference(
                [
                    {"role": "system", "content": Prompts.TOOL_CALL_FAULT_TOLERANT},
                    {"role": "user", "content": content},
                ],
                "summarizing",
                "none",
            )
            Config.logger.info(f"[TOOL TOLERANT]: {json.dumps(response)}")
            if "content" in response and response["content"]:
                return response
            if "tool_calls" not in response or not response["tool_calls"]:
                return None
            for tool_call in response["tool_calls"]:
                if "name" not in tool_call:
                    return None
                if not tool_call["arguments"]:
                    continue
                for tool_arg in tool_call["arguments"]:
                    if "name" not in tool_arg or "value" not in tool_arg:
                        return None
            return response
        except Exception as e:
            Config.logger.error(f"[TOOL TOLERANT]: {e}")
            return None

    def execute(self):
        tool_mode = "auto"
        tools = Tools.documentation()
        max_retries = 5
        retries = 0

        while True:
            if retries >= max_retries:
                Config.logger.error(
                    f"[INFERENCE MAX RETRY({self.task_type})]: {max_retries}. Returning {self.result[:30]}..."
                )
                return self.result

            with open(f"../logs/{self.task_type}.txt", "w") as f:
                f.write(json.dumps(self.messages, indent=2))

            # Check for timeout before each iteration
            if self.timeout_handler and self.timeout_handler.check():
                Config.logger.error(
                    f"[AGENT TIMEOUT({self.task_type})]: Agent execution exceeded {self.timeout_handler.timeout_seconds}s timeout"
                )
                raise CustomError(
                    ErrorType.AGENT_TIMEOUT,
                    f"Agent execution timed out after {self.timeout_handler.timeout_seconds} seconds",
                )

            Config.logger.info("=" * 60)

            response = Utils.inference(
                self._messages_with_simmarization(),
                self.task_type,
                tool_mode,
                tools,
            )
            response["content"] = (
                str(response["content"]).replace("</think>", "").strip()
            )

            if not response["tool_calls"] and not response["content"]:
                retries += 1
                Config.logger.error(
                    f"[NO INFERENCE RESULT({self.task_type})]: ({retries}/{max_retries})"
                )
                continue

            if not response["tool_calls"]:
                if self._eval_result(response["content"]):
                    self.result = response["content"]
                    return self.result
                else:
                    Config.logger.warning(
                        f"[NON RESULT({self.task_type})]: Evaluated as non-result text.\n{response["content"]}"
                    )
                    response = self._tool_call_fault_tolerant(response["content"])
                    if not response:
                        continue
                    response["content"] = response.get("content", "")

            Config.logger.info(
                f"[INFERENCE CONTENT({self.task_type})]: {response["content"]}"
            )

            if response["content"]:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response["content"],
                    }
                )

            if not response["tool_calls"]:
                continue

            tool_names = [tool["name"] for tool in response["tool_calls"]]
            tool_args = [
                {arg["name"]: arg["value"] for arg in tool["arguments"]}
                for tool in response["tool_calls"]
            ]
            self.tool_calls.append(
                {"index": len(self.messages), "names": tool_names, "args": tool_args}
            )
            tool_call_outputs = []

            if self._check_and_rebase_repeated_tool_call():
                if retries < max_retries:
                    retries += 1
                    Config.logger.warning(
                        f"[REBASE REPEATED({self.task_type})]: ({retries}/{max_retries})"
                    )
                    continue

            for tool_call in response["tool_calls"]:
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                kwargs = (
                    {arg["name"]: arg["value"] for arg in tool_args}
                    if tool_args
                    else ({})
                )
                tool_output = "No response"

                if hasattr(Tools, tool_name) and callable(getattr(Tools, tool_name)):
                    tool = getattr(Tools, tool_name)
                    try:
                        tool_output = str(tool(**kwargs))
                        tool_call_outputs.append(
                            {
                                "role": "tool",
                                "name": tool_name,
                                "content": tool_output,
                            }
                        )
                    except Exception as e:
                        tool_call_outputs.append(
                            {
                                "role": "tool",
                                "name": tool_name,
                                "content": f"Failed to run {tool_name} tool with argument {json.dumps(kwargs)}. Error: {e}",
                            }
                        )
                else:
                    tool_call_outputs.append(
                        {
                            "role": "tool",
                            "name": tool_name,
                            "content": f"{tool_name} tool doesn't exist",
                        }
                    )

                tool_output_preview = tool_output.split("\n")[:5]
                Config.logger.info(
                    f"[TOOL CALL({self.task_type})]: {tool_name}({json.dumps(kwargs, indent=4)}) \n\t[ANSWER]: {"\n".join(tool_output_preview)}"
                )

            self.tool_outputs.append(tool_call_outputs)
            self.messages.extend(tool_call_outputs)
            self._summarize()


class Prompts:
    PLANNING_INSTRUCTION = """
    Role:
        You are a senior software engineer responsible for writing detailed plan to implement a software engineering task. You are only allowed to make plan, NOT for actual implementation.

    Environment:
        - No internet access.
        - Supported languages: Python, Node.js (JavaScript/TypeScript).
        - Task types: implement feature, fix bug(s), refactor, or design new module.
        - Constraint: Do NOT write or change the repository code. Produce a planning and review artifact only.
        - You will receive:
            1. Problem statement to solve.
            2. The repository contents or relevant files as context.

    Goals:
        For each assigned task, produce a plan that an LLM can follow to implement or fix the code with accurate and minimal changes.
        Do NOT attempt to reproduce the error; it is already known, and the plan is intended for an LLM, NOT a human.

    Required output format (JSON-like / Markdown):
        Always mention that, "Planning work is fully completed."

        1. Summary
            - 1-2 sentence description of the task in your own words.
            - Key deliverables.

        2. Preconditions & Inputs
            - Files, functions, dependencies, environment needed.
            - All relevant code snippets the context is aware of. You MUST designate the exact line numbers for the code snippets included.
            File: path/to/file1.py (lines 120-140)
            ```
            # THE EXACT, FULL CODE HERE
            ```

            File: path/to/file2.py (lines 200-250)
            ```
            # THE EXACT, FULL CODE HERE
            ```
            - If any input is missing, list the exact file paths, function names, or data needed.

        3. High-level plan step-by-step
            - Short numbered list of major phases.

        4. Detailed subtask breakdown
            For each subtask provide:
            - ID and title  
            (e.g., S1: Add validation for X)
            - Objective  
            (one concise sentence)
            - Inputs
                * Provide a directory tree showing only the relevant files.
                * Include all relevant **actual code snippets** (NOT placeholders).  
                Each snippet must be preceded by its file path and line numbers. Example:
                    File: path/to/file.py (lines 120-148)
                    ```
                    # INSERT THE EXACT, FULL CODE HERE
                    ```
                The snippet must contain the real code, not a summary, not ellipses, not just file paths.
            - Outputs
                * File paths and exact line numbers where code must be added, modified, or removed.
                * Include updated or reference code snippets, each with file path and line numbers, following the required structure.
            - Step-by-step actions (numbered)
                * Each step must be concrete and immediately executable.  
                (e.g., "Add parameter parsing in `path/to/file.js` between lines 45-78".)
            - Assumptions, risks, and edge cases
            - Complexity: low / medium / high

    Inference rules:
        - Always make tool calls unless you are finishing with final output.
        - Do NOT invoke the same tool consecutively with identical arguments.
        - Do NOT request information already available in the context.
        - Batch as many tool calls as possible in a single step.
    """

    CODING_INSTRUCTION = """
    Role:
        You are a senior software engineer responsible for autonomously implementing code changes exactly according to the provided step-by-step plan.

    Environment:
        - No internet access.
        - Supported languages: Python, Node.js (JavaScript/TypeScript).
        - You will receive:
            1. A detailed task plan with subtasks, required files, and code snippet references.
            2. The repository contents or relevant files as context.

    Responsibilities:
        - Implement ONLY what is specified in the plan.
        - Apply code changes using diff/patch tool calls (or editor tools).
        - Run tests frequently using the available testing tools. If tests fail, analyze failures and revise your patch.
        - Write additional tests if needed to capture important behaviors or edge cases.
        - Ensure all tests pass before finalizing.
        - If repository code differs from plan snippets, trust the repository and report the mismatch.
        - Keep diffs minimal, targeted, and high quality.
        - Write clean, safe, maintainable production-level code.

    Rules:
        - You MUST NOT ask the user any questions. Tool calls are allowed, but natural-language questions are strictly forbidden.
        - Never output speculative or hypothetical code.
        - Follow the plan exactly and completely.
        - When a next step exists, proceed immediately.
        - If a step seems ambiguous, request missing artifacts via tool call (never via question).
        - Do NOT request information already available in the context.
        - Do NOT invoke the same tool consecutively with identical arguments.
        - Batch as many tool calls as possible in a single step.

    Deliverables:
        - Fully implemented code changes.
        - All relevant tests passed.
        - No commentary outside what the plan expects.
        - No questions. No confirmations.
        - End your response with a brief confirmation explicitly including "Coding work is fully completed.".
    """

    TESTING_INSTRUCTION = """
    Role:
        You are a senior software engineer responsible for writing tests that evaluate the correctness of a specific task implementation.

    Environment:
        - No internet access.
        - Supported languages: Python, Node.js (JavaScript/TypeScript).
        - You will receive:
            1. A task description.
            2. A detailed implementation plan.
            3. The repository contents for all relevant files.
        - Do NOT work on the task itself. Only work on writing tests to evaluate the implementation once completed.
        - Task types include: implementing a feature, fixing bug(s), refactoring, or designing a new module.
        - Tasks are not implemented. Just generate test scripts, not trying to run them.

    Goals:
        You write a SINGLE test file with MULTIPLE test cases that will validate the code after the task is implemented.
        You can refer to existing tests which are relevant while writing tests.
        Make sure to generate accurate, thorough, efficient, and edge-case-complete tests.

    Rules:
        - Always make tool calls unless you are finishing with final output.
        - Do NOT invoke the same tool consecutively with identical arguments.
        - Do NOT request information already provided in the context.
        - Batch as many tool calls as possible into a single step.
        - Generate thorough, efficient, and edge-case-complete tests.
        - When adding tests, prefer using built-in test frameworks (e.g., `unittest`, `node:test`, `node:assert`) if possible.
        - For existing tests, refer to ONLY the ones which are relevant to provided task. Do NOT care about not relevant tests.
        - If testing the task result is too complex and tricky, include instruction to type check at minimal in the output.

    Required output format (JSON-like / Markdown):
        Always mention that, "Testing work is fully completed."

        1. Newly added test cases with explanation.
        2. Bash command to run the test cases.
    """

    SUMMARIZING_INSTRUCTION = """
    You are a summarization expert.
    You will be given an interaction history between the LLM (assistant role) and tool calls (tool role). This history represents part of an SWE agent's reasoning process while working on a specific software engineering task.

    Your job is to summarize this reasoning history by removing any content that is not useful for understanding progress toward solving the task.

    Output Format:
    1. Relevant directory/file structures  
    Provide only the directory and file structures that are pertinent to the task.

    2. Relevant code snippets  
    Include all code snippets from the history that matter for solving the task.  
    Each snippet must be preceded by its file path and line range.

    3. What the assistant was doing and why  
    Provide a concise semantic summary describing the assistant's actions and the purpose behind them.

    Task the SWE agent is working on:
    {}
    """

    EVALUATE_PLANNER_RESULT = "Planning work is fully completed"

    EVALUATE_CODER_RESULT = "Coding work is fully completed"

    EVALUATE_TESTER_RESULT = "Testing work is fully completed"

    TOOL_CALL_FAULT_TOLERANT = """
    You will evaluate an LLM-generated message to determine whether it contains an incomplete or malformed XML tool call.
    The message may or may not relate to tool calling. Your task is to analyze the text, detect whether any tool call is present, and—if so—extract and normalize it into a corrected JSON representation.

    Return format (return only the JSON object, with no additional explanation):
    ```JSON
    {
        "content": string,    // The reasoning or narrative text outside of the XML tool call, cleaned and de-duplicated
        "tool_calls": [       // Extracted and normalized tool call data
            {
                "name": string,       // Tool name
                "arguments": [        // Tool arguments list
                    {
                        "name": string,   // Argument name
                        "value": any      // Argument value
                    }
                ]
            }
        ]
    }
    ```

    Additional rules:
        - Detect and correct anomalies in the "content" field, such as repeated reasoning segments; ensure the content is clean and appears only once.
        - If no tool call is detected, the "tool_calls" field must be an empty array.
        - Field names must match the specified schema exactly.
    """


async def _execute_agent_logic_async(
    input_dict: Dict[str, Any],
    repo_dir: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """
    Execute agent logic with timeout handling and retry logic.
    Returns a result dictionary indicating success or failure.
    """
    max_retries = 3

    async def execute_async(retries: int = 0) -> Dict[str, Any]:
        try:
            # Set up timeout handler
            timeout_handler = TimeoutHandler(timeout_seconds)
            timeout_handler.start()

            problem_statement = input_dict["problem_statement"]
            Config.logger.info(f"[PROBLEM STATEMENT]: {problem_statement[:200]}...")

            # Create planner and tester agents with timeout handler
            planner = Agent(
                Prompts.PLANNING_INSTRUCTION,
                problem_statement,
                Prompts.EVALUATE_PLANNER_RESULT,
                "planning",
                repo_dir,
                timeout_handler=timeout_handler,
            )

            # tester = Agent(
            #     Prompts.TESTING_INSTRUCTION,
            #     problem_statement,
            #     Prompts.EVALUATE_TESTER_RESULT,
            #     "testing",
            #     repo_dir,
            #     timeout_handler=timeout_handler,
            # )

            # # Execute planner and tester in parallel
            # Config.logger.info(
            #     "[PARALLEL EXECUTION]: Starting planner and tester in parallel"
            # )
            # planner_task = asyncio.get_event_loop().run_in_executor(
            #     None, planner.execute
            # )
            # tester_task = asyncio.get_event_loop().run_in_executor(None, tester.execute)

            # await asyncio.gather(planner_task, tester_task)
            # Config.logger.info("[PARALLEL EXECUTION]: Planner and tester completed")

            # if not planner.result:
            #     Config.logger.error(
            #         f"[PLANNER FAILURE]: Planner agent failed. Last response: {planner.messages[-1]['content'] if planner.messages else 'No messages'}"
            #     )
            #     raise CustomError(
            #         ErrorType.PLANNER_FAILED,
            #         f"Planner agent failed. Last response: {planner.messages[-1]['content'] if planner.messages else 'No messages'}",
            #     )

            planner.execute()
            Config.logger.info(f"[PLANNER RESULT]: {planner.result}")

            # if not tester.result:
            #     Config.logger.error(
            #         f"[TESTER FAILURE]: Tester agent failed. Last response: {tester.messages[-1]['content'] if tester.messages else 'No messages'}"
            #     )
            #     raise CustomError(
            #         ErrorType.TESTER_FAILED,
            #         "Tester agent failed. Last response: {tester.messages[-1]['content'] if tester.messages else 'No messages'}",
            #     )

            # Config.logger.info(f"[TESTER RESULT]: {tester.result}")

            # Create coder agent with same timeout handler
            coder = Agent(
                Prompts.CODING_INSTRUCTION,
                f"This is task plan:\n{planner.result}\n",
                Prompts.EVALUATE_CODER_RESULT,
                "coding",
                repo_dir,
                timeout_handler=timeout_handler,
            )
            coder.execute()
            Config.logger.info(f"[FINISHED RESULT]: {coder.result}")

            return {"success": True, "result": coder.result}
        except CustomError as e:
            if e.error_type == ErrorType.AGENT_TIMEOUT:
                Config.logger.error(f"[TIMEOUT ERROR]: {e.message}")
                return {"success": False, "error": "timeout", "message": str(e)}
            else:
                Config.logger.error(f"[CUSTOM ERROR]: {e}")
                if retries < max_retries:
                    Config.logger.warning(
                        f"[RETRYING]: Attempt {retries + 1}/{max_retries}"
                    )
                    return await execute_async(retries + 1)
                else:
                    return {
                        "success": False,
                        "error": str(e.error_type.value),
                        "message": str(e),
                    }
        except Exception as e:
            Config.logger.error(f"[UNEXPECTED ERROR]: {e}")
            if retries < max_retries:
                Config.logger.warning(
                    f"[RETRYING]: Attempt {retries + 1}/{max_retries}"
                )
                return await execute_async(retries + 1)
            else:
                return {"success": False, "error": "exception", "message": str(e)}

    # Start execution with retry logic
    return await execute_async()


def _execute_agent_logic(
    input_dict: Dict[str, Any],
    repo_dir: str,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """
    Synchronous wrapper for async agent logic execution.
    """
    return asyncio.run(
        _execute_agent_logic_async(input_dict, repo_dir, timeout_seconds)
    )


def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo") -> str:
    """
    Main entry point for agent execution with timeout handling.
    Timeout is managed by the TimeoutHandler within the agent execution.
    """
    repo_dir = os.path.abspath(repo_dir)
    os.chdir(repo_dir)

    if repo_dir not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + ":" + repo_dir
    if Path(repo_dir + "/lib").exists() and repo_dir + "/lib" not in os.environ.get(
        "PYTHONPATH", ""
    ):
        os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"] + ":" + repo_dir + "/lib"

    # Initialize git repository if needed
    if not os.path.exists(".git"):
        subprocess.run(["git", "init"], check=True)
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", repo_dir]
        )
        subprocess.run(
            ["git", "config", "--global", "user.email", "agent@sandbox.local"],
            check=True,
        )
        subprocess.run(
            ["git", "config", "--global", "user.name", "sandbox_agent"], check=True
        )
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            check=False,
            capture_output=True,
            text=True,
        )
    else:
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", repo_dir]
        )

    Config.logger.info(
        f"[STARTING] Agent execution with {Config.AGENT_TIMEOUT}s timeout"
    )

    # Execute agent logic directly (timeout handled internally via TimeoutHandler)
    execution_result = _execute_agent_logic(input_dict, repo_dir, Config.AGENT_TIMEOUT)

    if not execution_result.get("success"):
        Config.logger.error(
            f"[EXECUTION FAILED] Error: {execution_result.get('error')} - {execution_result.get('message', '')}"
        )

    return Utils.get_git_patch()
