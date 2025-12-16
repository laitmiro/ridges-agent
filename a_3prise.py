import os
import re
import subprocess
import inspect
import requests
import random
import logging
import json
import asyncio
import traceback
from datetime import datetime
from enum import Enum
from json import JSONDecodeError
from typing import Any, Dict, Optional, List
from pathlib import Path
from uuid import uuid4
from tree_sitter import Parser
from tree_sitter_language_pack import get_language


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
    MODEL_QWEN3_CODER = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

    GATEWAY_URL = os.getenv("SANDBOX_PROXY_URL", "http://localhost:1234")
    AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "1500")) - 50
    EVALUATION_RUN_ID = os.getenv("EVALUATION_RUN_ID", str(uuid4()))

    NOT_FOUND_LITERAL = "Not found"

    class TaskType(Enum):
        REASONING = "reasoning"
        THINKING = "thinking"
        ANSWERING = "answering"

    TEMPERATURE_BY_MODEL_TASK = {
        MODEL_GLM_46: {
            TaskType.REASONING: (0.3, 0.7),
            TaskType.THINKING: (0.0, 0.2),
            TaskType.ANSWERING: (0.0, 0.3),
        },
        MODEL_KIMI_K2: {
            TaskType.REASONING: (0.3, 0.7),
            TaskType.THINKING: (0.0, 0.2),
            TaskType.ANSWERING: (0.0, 0.3),
        },
        MODEL_QWEN3_CODER: {
            TaskType.REASONING: (0.3, 0.7),
            TaskType.THINKING: (0.0, 0.2),
            TaskType.ANSWERING: (0.0, 0.3),
        },
    }

    MODEL_BY_TASK = {
        TaskType.REASONING: [
            MODEL_KIMI_K2,
            MODEL_KIMI_K2,
            MODEL_QWEN3_CODER,
        ],
        TaskType.THINKING: [
            MODEL_GLM_46,
            MODEL_GLM_46,
            MODEL_QWEN3_CODER,
        ],
        TaskType.ANSWERING: [
            MODEL_QWEN3_CODER,
            MODEL_QWEN3_CODER,
            MODEL_GLM_46,
        ],
    }

    VERSION_COMPATIBILITY_FIX = """
    import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
    collections.Mapping = collections.abc.Mapping;
    collections.MutableMapping = collections.abc.MutableMapping;
    collections.MutableSet = collections.abc.MutableSet;
    collections.Sequence = collections.abc.Sequence;
    collections.Callable = collections.abc.Callable;
    collections.Iterable = collections.abc.Iterable;
    collections.Iterator = collections.abc.Iterator;
    urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
    pytest.RemovedInPytest4Warning = DeprecationWarning;
    _pytest.pytester.Testdir = _pytest.pytester.Pytester;
    numpy.PINF = numpy.inf;
    numpy.unicode_ = numpy.str_;
    numpy.bytes_ = numpy.bytes_;
    numpy.float_ = numpy.float64;
    numpy.string_ = numpy.bytes_;
    numpy.NaN = numpy.nan;
    """

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
    _parsers = {}
    _language_cache = {}

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
    def _get_parser(cls, language: str):
        if Parser is None or get_language is None:
            return None
        if language not in cls._parsers:
            try:
                lang_obj = get_language(language)
                if lang_obj is None:
                    return None
                parser = Parser(lang_obj)
                cls._parsers[language] = parser
            except Exception as e:
                Config.logger.warning(f"Error creating parser for {language}: {e}")
                return None
        return cls._parsers[language]

    @classmethod
    def inference(
        cls,
        messages: list,
        task_type: Config.TaskType,
        tool_mode: str = "auto",
        tools: list = [],
        max_retries: int = 5,
        model_indx: int = 0,
    ) -> dict:
        retries = 0
        model = ""
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
                traceback.print_exc()
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
    def _search_in_all_files(cls, grep_search_command: str) -> str:
        result_prefix = f"Result of running {grep_search_command}\n"
        cmd = grep_search_command.lstrip()
        if not cmd.startswith("grep"):
            return f"{result_prefix} Error: Invalid command. Expected a grep command but got: '{grep_search_command}'"
        try:
            result = subprocess.run(
                ["bash", "-c", grep_search_command],
                capture_output=True,
                text=True,
                timeout=45,
            )
        except Exception as e:
            return f"{result_prefix} Error: Failed to execute grep command: {e}"
        if result.returncode > 1:
            error_msg = result.stderr.strip() or "Unknown error"
            return f"{result_prefix} Error: Grep command failed with return code {result.returncode}: {error_msg}"
        output = result.stdout

        if not output.strip():
            return f"{result_prefix} No matches found for pattern in codebase."
        if Utils.count_tokens(output) > 3000:
            return f"{result_prefix} Search results are too long. Please refine your search term into more specific terms."
        return output

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

    @classmethod
    def _run_code(
        cls,
        content: str,
        file_path: str,
        run_command: list[str],
    ) -> str:
        result_prefix = f"Result of running {" ".join(run_command)}\n"

        if file_path.endswith((".py", ".pyw", ".pyx", ".pyi", ".pxd", ".pxi", ".pyz")):
            content = Config.VERSION_COMPATIBILITY_FIX + "\n\n" + content
        cls._create_file(file_path, content)
        try:
            result = subprocess.run(
                run_command, capture_output=True, text=True, check=False, timeout=60
            )
            if result.returncode != 0:
                return f"{result_prefix} Error running code: {result.stderr}"
            return f"{result_prefix} {result.stdout}\n"
        except Exception as e:
            return f"{result_prefix} Error: {e}"

    @classmethod
    def _check_language(cls, source: str, file_path: str | None = None) -> str | None:
        if (
            file_path
            and not os.path.exists(file_path)
            or not source
            or not source.strip()
        ):
            return None
        if file_path:
            file_path = os.path.abspath(file_path) if file_path else None
            if file_path and file_path in cls._language_cache:
                return cls._language_cache[file_path]
        stripped_source = source.strip()
        sample = (
            stripped_source
            if len(stripped_source) <= 1000
            else f"{stripped_source[:500]}\n\n... [middle content omitted] ...\n\n{stripped_source[-500:]}"
        )

        detected_language = cls.inference(
            [
                {"role": "system", "content": Prompts.DETECT_PROGRAMMING_LANGUAGE},
                {
                    "role": "user",
                    "content": f"Here is code snippet\n```\n{sample}\n```",
                },
            ],
            Config.TaskType.ANSWERING,
            "none",
        )

        if file_path:
            cls._language_cache[file_path] = detected_language
        return detected_language

    @classmethod
    def _is_identifier_node(self, node) -> bool:
        return "identifier" in node.type.lower()

    @classmethod
    def _classify_node_type(self, node) -> tuple[str, int | None]:
        node_type_str = node.type.lower()
        if "function" in node_type_str or "method" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("function", i)
            return ("function", None)
        elif "class" in node_type_str:
            for i, child in enumerate(node.children):
                if self._is_identifier_node(child):
                    return ("class", i)
            return ("class", None)
        return ("other", None)

    @classmethod
    def _find_specific_function(
        cls,
        node,
        source_lines: list[str],
        target_qualified: str,
        target_simple: str,
        class_name: str = "",
        parent_node=None,
    ) -> dict | None:
        if not node.children:
            return None
        node_type, name_child_index = cls._classify_node_type(node)
        if node_type == "class":
            name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    name = (
                        line[name_start[1] : name_end[1]].strip()
                        if name_start[0] == name_end[0]
                        else line[name_start[1] :].strip()
                    )
            if not name and parent_node:
                for child in parent_node.children:
                    if cls._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = (
                                line[name_start[1] : name_end[1]].strip()
                                if name_start[0] == name_end[0]
                                else line[name_start[1] :].strip()
                            )
                            if name:
                                break
            if name:
                new_class_name = f"{class_name}.{name}" if class_name else name
                for child in node.children:
                    result = cls._find_specific_function(
                        child,
                        source_lines,
                        target_qualified,
                        target_simple,
                        new_class_name,
                        node,
                    )
                    if result is not None:
                        return result

        elif node_type == "function":
            name = internal_name = None
            if name_child_index is not None and name_child_index < len(node.children):
                name_child = node.children[name_child_index]
                name_start, name_end = name_child.start_point, name_child.end_point
                if name_start[0] < len(source_lines):
                    line = source_lines[name_start[0]]
                    internal_name = (
                        line[name_start[1] : name_end[1]].strip()
                        if name_start[0] == name_end[0]
                        else line[name_start[1] :].strip()
                    )
            if parent_node:
                for child in parent_node.children:
                    if cls._is_identifier_node(child) and child != node:
                        name_start, name_end = child.start_point, child.end_point
                        if name_start[0] < len(source_lines):
                            line = source_lines[name_start[0]]
                            name = (
                                line[name_start[1] : name_end[1]].strip()
                                if name_start[0] == name_end[0]
                                else line[name_start[1] :].strip()
                            )
                            if name:
                                break
            if not name:
                name = internal_name
            if name:
                qualified_name = f"{class_name}.{name}" if class_name else name
                is_qualified_target = "." in target_qualified
                is_match = qualified_name == target_qualified or (
                    not is_qualified_target and name == target_simple
                )
                if is_match:
                    at_start = node.start_point[0]
                    for i in range(at_start - 1, -1, -1):
                        if source_lines[i].strip().startswith("@"):
                            at_start = i
                        elif source_lines[i].strip():
                            break
                    return {
                        "start_line": at_start + 1,
                        "end_line": node.end_point[0] + 1,
                    }
            for child in node.children:
                result = cls._find_specific_function(
                    child,
                    source_lines,
                    target_qualified,
                    target_simple,
                    class_name,
                    node,
                )
                if result is not None:
                    return result
        for child in node.children:
            result = cls._find_specific_function(
                child, source_lines, target_qualified, target_simple, class_name, node
            )
            if result is not None:
                return result
        return None

    @classmethod
    def _get_function_body(
        cls, file_path: str, function_name: str, add_line_numbers: bool = False
    ) -> str:
        if not function_name:
            return "Error: Provide function name explicitly"
        if not os.path.exists(file_path):
            return f"Error: {file_path} file doesn't exist"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            return f"Error reading file {file_path}: {e}"

        if not source or Parser is None:
            return ""
        try:
            source_bytes, source_lines = bytes(source, "utf8"), source.splitlines()
            language = cls._check_language(source, file_path=file_path)
            if not language:
                return ""
            parser = cls._get_parser(language)
            if parser is None:
                return ""
            tree = parser.parse(source_bytes)
            target_qualified, target_simple = (
                function_name,
                function_name.split(".")[-1],
            )
            func_info = cls._find_specific_function(
                tree.root_node, source_lines, target_qualified, target_simple
            )
            if func_info is None:
                return ""
            start_idx, end_idx = func_info["start_line"] - 1, func_info["end_line"] - 1
            if 0 <= start_idx < len(source_lines) and 0 <= end_idx < len(source_lines):
                body_lines = source_lines[start_idx : end_idx + 1]
                return (
                    "\n".join(
                        f"{start_idx + i + 1}| {line}"
                        for i, line in enumerate(body_lines)
                    )
                    if add_line_numbers
                    else "\n".join(body_lines)
                )
        except Exception as e:
            Config.logger.warning(
                f"Error finding function {function_name} in {file_path}: {e}"
            )
        return ""


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

    def search_in_all_files_content(cls, grep_search_command: str) -> str:
        """
        Performs grep search across all files in the codebase.

        Arguments:
            grep_search_command: grep search command to locate (e.g., "grep <your grep command>").
        """
        return Utils._search_in_all_files(grep_search_command)

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

    @classmethod
    def run_code(cls, content: str, file_path: str, run_command: list[str]) -> str:
        """
        Runs any code. You can use this tool directly to run any test code or bug reproduction code.
        Saves the code at the given file_path and then runs it.
        Do not use this tool to create test or files to reproduce the error unless user has specifically asked you to create test files as part of problem statement.

        Arguments:
            content: text code to write in file
            file_path: path of the file to save the code in. This file should always be in the current working directory.
            run_command: command to run the file (i.e., ["python", "file.py"] or ["node", "file.js"] etc)
        """
        return Utils._run_code(
            content,
            file_path,
            run_command,
        )

    @classmethod
    def get_function_body(cls, file_path: str, function_name: str) -> str:
        """
        Retrieves the complete body of a function from a file, including decorators.
        Arguments:
            file_path: filesystem path to target file.
            function_name: name of the function to retrieve (supports both qualified names like "ClassName.method_name" and simple names like "method_name").
        Returns:
            The complete function body including decorators, or empty string if function not found.
        """
        return Utils._get_function_body(file_path, function_name, add_line_numbers=True)

    @classmethod
    def think(cls, thought: str) -> str:
        """
        Use the tool to think about something.
        It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.
        For example, if you explore the repo and discover the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.
        Alternatively, if you receive some test results, call this tool to brainstorm ways to fix the failing tests.

        Arguments:
            thought: Your thoughts.
        """
        return "ok"

    @classmethod
    def finish(cls):
        """
        Signals completion of the current workflow execution
        """
        return "finish"


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

        Config.logger.info(f"[SUMMARIZING]: {self.summary_length} messages...")

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
            Config.TaskType.ANSWERING,
            "none",
        )
        self.summarization = response["content"]

        Config.logger.info(
            f"[SUMMARIZED]: {Utils.count_tokens(messages_content)} -> {Utils.count_tokens(self.summarization)} tokens \n{self.summarization}"
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
                Config.logger.warning(f"[REPEATED]: {repeat} tools")
                for i in range(repeat):
                    Config.logger.warning(
                        f"\t[TOOL CALL {i+1}]: {json.dumps(self.tool_calls[-repeat+i])}\n"
                    )
                self._rebase_to_tool_call(-2 * repeat)
                return True
        return False

    def _tool_call_fault_tolerant(self, content):
        try:
            response = Utils.inference(
                [
                    {"role": "system", "content": Prompts.TOOL_CALL_FAULT_TOLERANT},
                    {"role": "user", "content": content},
                ],
                Config.TaskType.ANSWERING,
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
            response["content"] = response.get("content", "")
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
                Config.logger.error(f"[INFERENCE MAX RETRY]: {max_retries}.")
                return

            # Check for timeout before each iteration
            if self.timeout_handler and self.timeout_handler.check():
                Config.logger.error(
                    f"[AGENT TIMEOUT]: Agent execution exceeded {self.timeout_handler.timeout_seconds}s timeout"
                )
                raise CustomError(
                    ErrorType.AGENT_TIMEOUT,
                    f"Agent execution timed out after {self.timeout_handler.timeout_seconds} seconds",
                )

            Config.logger.info("=" * 60)

            response = Utils.inference(
                self._messages_with_simmarization(),
                Config.TaskType.REASONING,
                tool_mode,
                tools,
            )
            response["content"] = (
                str(response["content"]).replace("</think>", "").strip()
            )

            if not response["tool_calls"] and not response["content"]:
                retries += 1
                Config.logger.error(f"[NO INFERENCE RESULT]: ({retries}/{max_retries})")
                continue

            if not response["tool_calls"]:
                Config.logger.warning(f"[NO TOOL CALLS]: {response["content"]}")
                response = self._tool_call_fault_tolerant(response["content"])
                if not response:
                    Config.logger.error(f"[NO INFERENCE RESULT] {response}")
                    continue

            Config.logger.info(f"[INFERENCE CONTENT]: {response["content"]}")

            if response["content"]:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response["content"],
                    }
                )

            if not response["tool_calls"]:
                continue

            tool_name_list = [tool["name"] for tool in response["tool_calls"]]
            tool_args_list = [
                {arg["name"]: arg["value"] for arg in tool["arguments"]}
                for tool in response["tool_calls"]
            ]
            self.tool_calls.append(
                {
                    "index": len(self.messages),
                    "names": tool_name_list,
                    "args": tool_args_list,
                }
            )
            tool_call_outputs = []

            if self._check_and_rebase_repeated_tool_call():
                if retries < max_retries:
                    retries += 1
                    Config.logger.warning(
                        f"[REBASE REPEATED]: ({retries}/{max_retries})"
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

                if tool_name == "finish":
                    return

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
                    f"[TOOL CALL]: {tool_name}({json.dumps(kwargs, indent=4)}) \n\t[ANSWER]: {"\n".join(tool_output_preview)}"
                )

            self.tool_outputs.append(tool_call_outputs)
            self.messages.extend(tool_call_outputs)
            self._summarize()


class Prompts:
    REASONING_INSTRUCTION = """
# Software Engineering Agent - Universal Task Prompt

## Role and Context

You are a senior software engineering agent capable of implementing, modifying, and debugging code. You will be provided with a task that requires solving a software engineering problem.

**CRITICAL CONTEXT:**
- The problem statement is the authoritative source of truth
- Both source code AND test files may contain errors or be missing
- You have everything needed in the repository - no internet connection required
- There are hidden tests that must pass beyond the visible test cases
- Initial solutions and test cases may be auto-generated and contain errors

## High-Level Problem Solving Strategy

1. **Understand the problem deeply** - Read carefully and identify requirements, ambiguities, and potential misunderstandings
2. **Investigate the codebase** - Explore relevant files, search for patterns, and gather context
3. **Generate initial solution and tests (if needed)** - For new implementations, create initial code and test files first
4. **Identify edge cases and root causes** - Think critically about boundaries, errors, and core issues
5. **Develop a clear plan** - Break down the solution into manageable, incremental steps
6. **Implement changes incrementally** - Make small, testable modifications
7. **Test frequently** - Verify correctness after each change
8. **Debug as needed** - Use systematic debugging techniques to isolate issues
9. **Iterate until perfect** - Continue refining until all tests pass and requirements are met
10. **Final validation** - Reflect comprehensively and test edge cases rigorously

## Detailed Workflow

### 1. Deeply Understand the Problem

- Carefully read the problem statement and requirements
- Identify what behavior is expected and what success looks like
- Identify potential misunderstanding points and ambiguous requirements
- Think critically about what might be causing any discrepancies
- Consider whether this is a symptom or the root cause

**Common Analysis Points:**
- Boundary conditions (empty inputs, null values, max/min sizes)
- Invalid inputs and error handling requirements
- Ambiguous requirements that could be misinterpreted
- Common pitfalls and corner cases
- Complex scenarios that might not be obvious
- Type coercion, case sensitivity, off-by-one errors
- Default parameter behavior vs. explicit values

### 2. Codebase Investigation

**CRITICAL: Find working examples first, then identify what needs to be done**

- Search for key terms from the problem throughout the codebase
- Find similar functionality that WORKS correctly - this is your template
- Study how working code accomplishes what you need
- Review existing similar functionality for patterns and conventions
- Understand the project structure and where code should live
- Identify existing utilities, helpers, or base classes to leverage
- Check for coding standards, naming conventions, and architectural patterns
- Review any automatically generated code or scaffolding (both source and tests)
- Look beyond surface symptoms - search in domains, helpers, utilities, base classes
- Trace to where mechanisms are actually DEFINED, not just where they're called
- Find the ROOT files where functionality is implemented

**Trace Data Flow:**
- Start with working feature's final output, trace backwards to find generator
- If something isn't working, start with the problematic output and trace backwards to find what's missing
- Compare the paths: where do they diverge?
- Don't stop at the first file - keep tracing back to where behavior originates
- Compare working vs non-working code: what's different? Missing calls? Missing imports?

**General Investigation:**
- Read and understand relevant code snippets
- Validate and update your understanding continuously as you gather context
- Search broadly: domain logic, helpers, utilities, base classes, configurations
- Look for patterns that might need similar treatment in multiple locations

### 3. Generate Initial Solution and Tests (When Implementing New Features)

**IMPORTANT: When implementing new functionality, prefer to generate an initial solution and tests FIRST, then iterate to fix them.**

This approach allows you to:
- Establish a baseline implementation quickly
- Create test cases that validate the requirements
- Iterate and refine both code and tests together
- Catch misunderstandings early through test failures

**Initial Generation Process:**
1. **Generate source code** - Create the initial implementation based on requirements and patterns from the codebase
2. **Generate test files** - Write comprehensive test cases covering:
- Basic functionality
- Edge cases identified in step 1
- Error handling scenarios
- Integration with existing code
3. **Run initial tests** - Execute the tests to establish a baseline
4. **Iterate to fix** - Use the workflow below to refine both source and tests until all tests pass

**What to include in initial generation:**
- Follow existing project structure and conventions
- Use similar patterns found in working code
- Include placeholder logic that can be refined
- Write tests that validate requirements, not implementation details
- Cover edge cases from the beginning

### 4. Root Cause Identification & Edge Case Analysis

**Root Cause Verification:**

Before implementing any changes, verify you understand the root cause:

**Trace the COMPLETE data flow:**
1. Find similar WORKING feature
2. Trace working feature through all stages from start to final output
3. Trace problematic feature through all stages from start to final output
4. Find EXACT point where paths diverge

**Compare at EACH stage:**
- What does working code do that the current code doesn't?
- What functions are called? What imports exist?
- Where does the behavior differ?
- Keep tracing backwards until you find the root cause

**Find root, not symptoms:**
- Don't patch surface symptoms - find the missing or different mechanism
- Trace all the way back to where the behavior originates
- The fix location may be far from where symptoms appear
- Compare: How does working feature accomplish the task? How does current feature differ?

**Search comprehensively:**
- Is this pattern missing in multiple places? Search the whole repository
- Are there similar files/classes that need the same changes?
- Fix all instances, not just one example

**Edge Case Analysis:**

**Always identify and handle edge cases BEFORE implementing main logic:**
- Empty/null inputs
- Boundary values (zero, negative, maximum, minimum)
- Invalid inputs and error handling
- Special conditions and corner cases
- Overflow/underflow conditions for numeric operations
- String edge cases (empty strings, special characters, unicode)
- Collection edge cases (empty arrays, single elements, duplicates)

**Watch for common misunderstandings:**
- Ambiguous requirements - clarify assumptions explicitly
- Off-by-one errors in loops and indexing
- Case sensitivity in string comparisons
- Type coercion and implicit conversions
- Default parameter behavior vs. explicit values
- Null vs. undefined vs. empty string/array

### 5. Exception and Error Handling

- Identify opportunities to implement exception handling
- Ensure error handling doesn't introduce new bugs or break existing functionality
- Maintain backward compatibility
- Validate at system boundaries (user input, external APIs)
- Don't add error handling for scenarios that can't happen
- Trust internal code and framework guarantees
- Provide clear, actionable error messages
- Handle error cases gracefully and meaningfully

### 6. Develop a Detailed Plan

**Use the think tool to outline your approach:**
- Break down the solution into specific, simple, verifiable steps
- Make each step small and incremental
- Identify which files need to be modified or created
- Specify the order of implementation
- Plan for testing at each stage

**Planning considerations:**
- List components/functions to implement or modify
- Define data structures and interfaces
- Plan test cases for each feature
- Consider integration points
- List files that need modification
- Specify exact changes needed at each location
- Plan verification steps
- Consider similar issues or needs elsewhere

### 7. Making Code Changes

**General Principles:**
- **Before editing, ALWAYS read the relevant file contents** to ensure complete context
- Make small, testable, incremental changes
- Keep changes minimal and focused - don't refactor unrelated code
- Use the appropriate edit tool and ensure patches apply correctly
- If a patch fails, re-read the file and try again

**Implementation Guidelines:**
- Start with core functionality, then add edge case handling
- Follow existing patterns and conventions in the codebase
- Write clean, readable, maintainable code
- **Copy patterns from working code - make minimal focused changes**
- **Use the EXACT same pattern as working code**: same functions, imports, structure
- Fix root cause, not symptoms
- **Search for similar locations**: Is this pattern needed elsewhere?
- Fix all instances if it's systemic
- Trace back to where mechanisms are defined, not just where they're called
- The fix location is often far from where the problem first appears

**Avoid Over-Engineering:**
- Don't add features beyond what was asked
- Implement what's needed, not hypothetical features
- Don't add unnecessary abstractions, utilities, or helpers
- Only add comments where logic isn't self-evident
- Don't add docstrings, type annotations, or comments to unchanged code
- Don't add error handling for scenarios that can't happen
- Don't create helpers for one-time operations
- Don't design for hypothetical future requirements
- Three similar lines is better than premature abstraction
- Simple solutions don't need extra configurability

**Backward Compatibility:**
- Code must be backward compatible unless explicitly stated otherwise
- Don't use compatibility hacks like renaming to `_var`, re-exporting removed types, or `// removed` comments
- If something is unused, delete it completely

### 8. Test File Management

**Test Principles:**
- Fix test files if they conflict with the problem statement using the edit tool
- Test cases should validate requirements, not implementation details
- Add test cases for all identified edge cases
- Test exception cases explicitly
- If tests fail repeatedly and contradict the problem statement, fix the test
- Use the edit tool for test file modifications
- Ensure tests validate the correct behavior

**General Testing Principles:**
- Don't create new test files unless absolutely necessary
- Always check both expected output in problem statement AND in test cases
- Write tests for edge cases you identified
- Ensure tests are deterministic and repeatable
- When generating initial solutions, create comprehensive test files
- Tests should cover basic functionality, edge cases, and error handling

### 9. Testing and Verification

**Test frequently using provided tools:**
- Run tests after EACH change to verify correctness
- Don't batch multiple changes before testing
- Use the appropriate testing tool rather than shell commands directly
- Test edge cases explicitly - don't assume they're covered

**Critical Testing Areas:**
- Empty inputs and null values
- Boundary values (zero, negative, max/min)
- Invalid inputs
- Corner cases and special conditions
- All identified edge cases from step 4
- Exception handling paths

**If tests fail:**
- Analyze the failure carefully
- Determine if it's the code or the test that's wrong
- Revise your approach based on the failure
- Re-run tests to verify the fix

**Failing to test edge cases rigorously is the NUMBER ONE failure mode**

### 10. Debugging (When Needed)

**Systematic Debugging:**
- **Fix root cause, not symptoms**
- **Search broadly across the repository**
- Make changes only with high confidence they solve the problem
- Don't just patch calling code - trace to where mechanism is defined
- The fix location is often far from where problem is noticed
- Use print statements, logs, or temporary code to inspect state
- Add descriptive messages to understand what's happening
- Revisit assumptions if unexpected behavior occurs
- Look for similar patterns that might need the same fix
- Debug for as long as needed to identify the root cause
- Use systematic elimination of possibilities

### 11. Iteration

**Continue refining until:**
- All visible tests pass
- All edge cases are handled
- Requirements match the problem statement exactly
- Code is clean, readable, and maintainable
- No backward compatibility is broken
- Changes are exhaustive across the codebase

**Don't stop prematurely:**
- Hidden tests must also pass
- Edge cases must be verified
- The solution must be robust and comprehensive
- NEVER GIVE UP without solving the problem completely
- Feature implements all requirements
- Edge cases are properly handled
- Tests cover all functionality
- Code is maintainable and follows best practices
- Root cause is fixed, not just symptoms
- Similar issues elsewhere are also addressed
- Working features remain working
- Changes are minimal and focused

### 12. Final Reflection and Validation

**Comprehensive reflection:**
- Think carefully about the original intent and problem statement
- Consider potential scenarios not covered by existing tests
- Write additional tests that would validate correctness
- Run new tests and ensure they pass
- Be aware of hidden tests that must also pass

**Final checklist:**
- Requirements are fully met
- All edge cases are handled
- All tests pass (visible and your additional ones)
- Code follows project conventions
- No backward compatibility broken
- Changes are exhaustive and don't break other functionality
- Solution is robust and comprehensive

## Critical Requirements

**Universal Requirements:**
- **Backward compatibility**: Code must be backward compatible unless explicitly stated otherwise
- **Exhaustive changes**: Thoroughly check entire codebase to ensure changes don't break functionality
- **File creation**: Don't create new files unless absolutely necessary (exception: initial solution generation)
- **Output validation**: Check both expected output in problem statement AND in test cases
- **No internet**: If dependencies are missing, don't try to solve it - you have no internet access
- **Complete the task**: Never claim a task is too large or that you lack time
- **Think before acting**: Plan extensively before function calls and reflect on outcomes
- **Use tools appropriately**: Choose the right tool for the job and use it correctly
- **Find and fix root causes**: Don't just address symptoms - trace to the source
- **Search comprehensively**: Fix similar issues throughout codebase
- **Copy working patterns**: Leverage what already works with exact same patterns
- **Handle all edge cases**: Before considering task complete, test edge cases explicitly
- **Generate then iterate**: For new features, generate initial solution and tests first, then fix

## Step Efficiency

You have a limited step budget:
- **Target**: 5-15 steps for straightforward tasks
- **Maximum**: 30 steps for complex tasks

**Efficiency guidelines:**
- Prioritize simpler, faster solutions
- Make forward progress with each step
- Test frequently to catch issues early
- Don't over-investigate - implement once you understand
- Balance thoroughness with efficiency
- Use think tool to plan before acting
- Generate initial solutions quickly, then iterate

## Tool Usage Guidelines

**Choose the right tool:**
- Use file reading tools for understanding context
- Use search tools to find patterns and similar code
- Use edit tools for modifying existing files
- Use write tools for generating initial solutions and new files
- Use test execution tools for verification
- Use think tool for planning and reflection

**Tool best practices:**
- Read files before editing them
- Use exact values provided by the user
- Don't make up values for optional parameters
- If patches fail, re-read and retry
- Use search to find all occurrences before making changes

## Core Principles Summary

1. **Problem statement is the source of truth**
2. **Generate initial solutions first for new features** - then iterate to perfection
3. **Test rigorously and frequently** - this is the #1 failure prevention
4. **Handle edge cases explicitly** - don't assume they're covered
5. **Fix root causes, not symptoms** - trace to the source
6. **Make incremental changes** - small, testable steps
7. **Copy working patterns** - leverage what already works
8. **Search comprehensively** - fix all instances of systemic issues
9. **Be thorough and systematic** - follow the workflow step by step
10. **Think before acting** - plan and reflect extensively
11. **Never give up** - iterate until the solution is perfect

## Remember

- Your thinking should be thorough - it's fine if it's very long
- Think step by step before and after each action
- When you say you'll make a tool call, ACTUALLY make it
- The problem can definitely be solved without the internet
- Take your time and think through every step
- Check your solution rigorously and watch for boundary cases
- Your solution must be perfect - if not, keep iterating
- Hidden tests exist and must pass
- Don't assume the task is complete when visible tests pass
- Continue refining until confident the solution is robust and comprehensive
- For new implementations, generate initial solution and tests first, then iterate
- Both source code AND test files can contain errors - fix both as needed

**GO SOLVE THE PROBLEM!**
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

    DETECT_PROGRAMMING_LANGUAGE = """
Your task is to detect programming language of provided code snippet.
Analyze the code and determine which programming language it is written in.
Return ONLY the language name in lowercase.
If you cannot determine the language, return "unknown".

Return ONLY the language name in **lowercase**, no other text or explanation.
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
        error = None
        try:
            # Set up timeout handler
            timeout_handler = TimeoutHandler(timeout_seconds)
            timeout_handler.start()

            problem_statement = input_dict["problem_statement"]
            Config.logger.info(f"[PROBLEM STATEMENT]: {problem_statement[:200]}...")

            # Create planner and tester agents with timeout handler
            agent = Agent(
                Prompts.REASONING_INSTRUCTION,
                problem_statement,
                repo_dir,
                timeout_handler=timeout_handler,
            )

            agent.execute()
            Config.logger.info(f"[AGENT RESULT]:")
            return {"success": True}
        except CustomError as e:
            if e.error_type == ErrorType.AGENT_TIMEOUT:
                Config.logger.error(f"[TIMEOUT ERROR]: {e.message}")
                return {"success": False, "error": "timeout", "message": str(e)}
            else:
                Config.logger.error(f"[CUSTOM ERROR]: {e}")
                error = f"CustomError {str(e.error_type.value)} - {str(e)}"
        except Exception as e:
            Config.logger.error(f"[UNEXPECTED ERROR]: {e}")
            error = f"UnknownError {str(e)}"
        finally:
            if error:
                if retries < max_retries:
                    Config.logger.warning(
                        f"[RETRYING]: Attempt {retries + 1}/{max_retries}"
                    )
                    return await execute_async(retries + 1)
                else:
                    return {
                        "success": False,
                        "error": error,
                    }

    # Start execution with retry logic
    return await execute_async()


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
    execution_result = asyncio.run(
        _execute_agent_logic_async(input_dict, repo_dir, Config.AGENT_TIMEOUT)
    )

    if not execution_result.get("success"):
        Config.logger.error(
            f"[EXECUTION FAILED] Error: {execution_result.get('error')}"
        )

    return Utils.get_git_patch()
