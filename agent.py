import os
import json
import argparse
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from tools import (
    clone_repository,
    set_workspace,
    get_workspace,
    get_results_directory_name,
    set_tasks_folder,
    get_tasks_folder,
    list_files,
    read_file,
    write_file,
    run_command,
    git_prepare_branch,
    exclude_report_from_git,
    git_status,
    git_diff,
    git_commit,
    git_push,
)


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEYS = list(
    dict.fromkeys(
        key
        for key in (
            os.getenv("OPENROUTER_API_KEY_1"),
            os.getenv("OPENROUTER_API_KEY_2"),
            os.getenv("OPENROUTER_API_KEY_3"),
            os.getenv("OPENROUTER_API_KEY"),
        )
        if key and key.strip()
    )
)

MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "openrouter/free",
)

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_AGENT_TURNS = 40
TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_REQUESTS = 0


def load_skills() -> str:
    skills_path = BASE_DIR / "skills" / "coding_rules.md"

    if not skills_path.is_file():
        raise FileNotFoundError(
            f"Coding rules not found: {skills_path}"
        )

    return skills_path.read_text(encoding="utf-8")

def emit_stage(stage_id: str):
    print(f"STUDYFLOW_STAGE:{stage_id}", flush=True)

def emit_artifact_status(artifact: str, status: str, path=None):
    event = {
        "artifact": artifact,
        "status": status,
    }
    if path is not None:
        event["path"] = str(path)
    print(
        "STUDYFLOW_ARTIFACT:" + json.dumps(event, ensure_ascii=False),
        flush=True,
    )

def call_model(messages, tools):
    global TOTAL_PROMPT_TOKENS
    global TOTAL_COMPLETION_TOKENS
    global TOTAL_REQUESTS
    if not API_KEYS:
        raise RuntimeError(
            "No OpenRouter API keys found. "
            "Add OPENROUTER_API_KEY_1, "
            "OPENROUTER_API_KEY_2 and "
            "OPENROUTER_API_KEY_3 to .env."
        )

    errors = []

    for index, api_key in enumerate(API_KEYS, start=1):
        print(
            f"OpenRouter: trying API key {index}/{len(API_KEYS)}"
        )

        try:
            response = requests.post(
                API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "tools": tools,
                    "reasoning": {
                        "enabled": True
                    },
                },
                timeout=120,
            )

        except requests.RequestException as error:
            errors.append(
                f"Key {index}: connection error ({type(error).__name__})"
            )
            print(
                f"OpenRouter: connection failed for key {index}; "
                "trying the next key."
            )
            continue

        try:
            data = response.json()
        except ValueError:
            data = {}

        if response.status_code == 200:
            if (
                not isinstance(data.get("choices"), list)
                or not data["choices"]
            ):
                raise RuntimeError(
                    "OpenRouter returned no choices:\n"
                    f"{json.dumps(data, indent=2, ensure_ascii=False)}"
                )

            choice = data["choices"][0]
            message = choice.get("message") or {}

            if not message.get("content") and not message.get("tool_calls"):
                print(
                    "OpenRouter returned an empty message:\n"
                    + json.dumps(
                        {
                            "model": data.get("model"),
                            "finish_reason": choice.get("finish_reason"),
                            "usage": data.get("usage"),
                            "message": message,
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                )

            usage = data.get("usage") or {}

            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            TOTAL_PROMPT_TOKENS += prompt_tokens
            TOTAL_COMPLETION_TOKENS += completion_tokens
            TOTAL_REQUESTS += 1

            print(
                f"Tokens: input={prompt_tokens}, "
                f"output={completion_tokens}, "
                f"total={prompt_tokens + completion_tokens}"
            )

            print(
                f"Session total: input={TOTAL_PROMPT_TOKENS}, "
                f"output={TOTAL_COMPLETION_TOKENS}, "
                f"requests={TOTAL_REQUESTS}"
            )

            return data

        error_message = data.get("error", data)

        if not isinstance(error_message, dict):
            error_message = {"message": str(error_message)}

        status = response.status_code
        message = str(error_message.get("message", ""))
        metadata = error_message.get("metadata") or {}

        retryable = (
            status == 429
            or status in (500, 502, 503, 504)
            or (
                status == 400
                and (
                    "Provider returned error" in message
                    or bool(metadata.get("provider_name"))
                )
            )
        )

        errors.append(
            f"Key {index}: HTTP {status} — {message}"
        )

        if retryable and index < len(API_KEYS):
            print(
                f"OpenRouter returned HTTP {status} for key {index}. "
                "Trying the next API key."
            )
            continue

        if not retryable:
            raise RuntimeError(
                f"OpenRouter error {status}:\n"
                f"{json.dumps(error_message, indent=2, ensure_ascii=False)}"
            )

    raise RuntimeError(
        "All OpenRouter API keys failed.\n"
        + "\n".join(errors)
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in the current repository.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the current repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Create or replace a file in the current repository. "
                "Use the tasks folder for assignment files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    },
                    "content": {
                        "type": "string"
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the repository. "
                "Use relative paths. Never use cd or absolute paths. "
                "Do not run git commands."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string"
                    }
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_tool(name, arguments):
    if name == "list_files":
        return "\n".join(list_files())

    if name == "read_file":
        return read_file(arguments["path"])

    if name == "write_file":
        return write_file(
            arguments["path"],
            arguments["content"],
        )

    if name == "run_command":
        return run_command(arguments["command"])

    raise ValueError(f"Unknown tool: {name}")


def run_agent(
    assignment: str,
    tasks_folder: str,
    generate_report: bool,
    student_profile: dict[str, str],
):
    skills = load_skills()
    folder_instructions = (
        f"""
RESULTS FOLDER:

All solution files for this assignment MUST be placed inside the existing
folder named `{tasks_folder}/`. Do not write solution files elsewhere unless
the assignment explicitly requires a file at a specific repository path.
"""
        if tasks_folder
        else """
RESULTS FOLDER:

No results folder is enabled for this subject. Do not create a task/results
directory. Implement files at the repository paths required by the assignment.
"""
    )
    report_instructions = (
        f"""
REPORT:

- A report is enabled. Create `report.txt` in `{tasks_folder}/` when a results
  folder is enabled; otherwise create `report.txt` at the repository root.
- The host excludes report.txt from Git.
- Do not overwrite an existing report.
"""
        if generate_report
        else """
REPORT:

- Report generation is disabled. Do not create report.txt or another report.
"""
    )

    system_prompt = f"""
You are an autonomous coding agent.

You are working inside the root directory of a cloned Git repository.

Your task is to complete the assignment given by the user.

CODING RULES:

{skills}

IMPORTANT ENVIRONMENT RULES:

- You are already inside the repository root.
- NEVER use cd.
- NEVER use absolute paths.
- NEVER access files outside the current repository.
- NEVER access /mnt or another external project directory.
- Use relative paths only.
- Do not run git commands.
- Git is controlled by the host program.
- Do not modify unrelated files.

{folder_instructions}

ASSIGNMENT RULES:

- Follow the assignment EXACTLY as written.
- Read the entire assignment before implementing anything.
- Identify every task and every requirement.
- Do not improve, reinterpret, simplify, redesign, or correct the assignment.
- Do not replace requested behavior with your preferred implementation.
- If the assignment asks for unusual, unreachable, incorrect, or intentionally problematic code, write it exactly as requested.
- If the assignment explicitly asks for code after return, write that code after return.
- If the assignment explicitly asks to demonstrate an error, demonstrate that error.
- Do not remove required code because Python considers it unreachable.
- The assignment requirements have priority over your personal coding preferences.

COMMENTS:

- Do not write comments in source code unless the assignment explicitly requires them.
- Do not add explanatory comments or TODO comments.

STUDENT DATA:

Use only the profile data explicitly supplied by StudyFlow.
Never invent, assume, or reuse personal or student data from another task.
Do not include profile data in source comments, commit messages, or logs unless
the assignment explicitly requires it in the resulting application.

STUDENT PROFILE:

First name: {student_profile["first_name"] or "не вказано"}
Last name: {student_profile["last_name"] or "не вказано"}
Student group: {student_profile["student_group"] or "не вказано"}

REPORT REQUIREMENTS:

- For programming practical reports, the first line must follow this format:
  Практична №{{номер}} | {{Прізвище}} {{Ім'я}} | {{Група}} | Програмування
- Determine the practical number from the assignment if it is provided.
- Never invent missing personal information or a practical number.
- Use "не вказано" for missing profile fields.
- Follow the user's report template when one is supplied.

IMPLEMENTATION:

- Use exact function names required by the assignment.
- Use exact file names required by the assignment.
- Write simple student-level code.
- Do not add unnecessary libraries.
- Do not create unnecessary files.
- Do not modify unrelated files.
- Do not hardcode output instead of implementing functionality.
- Do not hide errors.
- Do not remove required examples.

{report_instructions}

TESTING:

- Run every relevant program after implementation.
- Check stdout and stderr.
- Exit code 0 does not automatically mean the test passed.
- TEST RESULT: FAILED means the task is not finished.
- If output contains an unexpected error, investigate and fix it.
- If the assignment explicitly requires an error, preserve it.
- If an interactive program needs input, provide enough input.
- Compare actual output with the assignment requirements.
- Perform final verification of all required tasks.
- Do not finish while an unexpected relevant test failure remains.

ASSIGNMENT:

{assignment}
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    verification_started = False

    empty_response_retries = 0

    for turn in range(1, MAX_AGENT_TURNS + 1):
        print(f"\n--- AI turn {turn}/{MAX_AGENT_TURNS} ---")

        data = call_model(messages, TOOLS)
        message = data["choices"][0]["message"]

        messages.append(message)

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            content = message.get("content") or ""

            if content:
                print("\nAGENT:")
                print(content)

            if message.get("refusal"):
                raise RuntimeError(
                    f"The model refused the assignment: "
                    f"{message['refusal']}"
                )

            if not content.strip():
                finish_reason = data["choices"][0].get("finish_reason")
            
                print(
                    f"WARNING: Empty model response. "
                    f"Finish reason: {finish_reason}"
                )
            
                if finish_reason == "length":
                    raise RuntimeError(
                        "The model response exceeded its output token limit."
                    )
            
                if finish_reason == "content_filter":
                    raise RuntimeError(
                        "The model response was blocked by a content filter."
                    )
            
                empty_response_retries += 1
            
                if empty_response_retries <= 2:
                    print(
                        f"Retrying empty response "
                        f"({empty_response_retries}/2)..."
                    )
            
                    messages.pop()
            
                    messages.append({
                        "role": "user",
                        "content": (
                            "Your previous response was empty. "
                            "Continue working on the original assignment. "
                            "Use the available tools if necessary. "
                            "Do not repeat completed work unnecessarily. "
                            "If the task is complete, provide your final response."
                        ),
                    })
            
                    continue
                
                raise RuntimeError(
                    "The model returned empty responses repeatedly. "
                    f"Last finish reason: {finish_reason}. "
                    "Check the configured OpenRouter model and API response."
                )
            
            empty_response_retries = 0

            if not verification_started:
                verification_started = True

                print("\n--- Independent output verification ---")

                messages.append(
                    {
                        "role": "user",
                        "content": """
Perform an independent final verification of the assignment.

1. Re-read the original assignment.
2. Inspect all relevant files.
3. Compare the implementation with every requirement.
4. Run relevant tests and inspect their results.
5. Fix failures and rerun the affected tests.
6. Do not claim a test passed without evidence.
7. Preserve behavior explicitly required by the assignment.

Your FINAL response must begin with exactly one marker:

STUDYFLOW_VERIFICATION: PASSED

or

STUDYFLOW_VERIFICATION: FAILED

Use PASSED only if all requirements have been checked,
all relevant tests passed, and no unexpected failures remain.
Otherwise use FAILED.

After the marker, briefly list the checks and their results.
"""
                    }
                )

                continue

            verification_result = content.strip()

            print("\n--- VERIFIER RAW RESPONSE ---")
            print(repr(content))
            print("--- END VERIFIER RESPONSE ---\n")

            if verification_result.startswith(
                "STUDYFLOW_VERIFICATION: FAILED"
            ):
                raise RuntimeError(
                    "Output verification failed.\n"
                    f"Verifier response:\n{content}"
                )

            if not verification_result.startswith(
                "STUDYFLOW_VERIFICATION: PASSED"
            ):
                raise RuntimeError(
                    "The verifier did not return a valid result marker.\n"
                    f"Verifier response:\n{content}"
                )

            print("\nOUTPUT VERIFICATION: PASSED")
            print(content)
            return

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            name = function.get("name")
            raw_arguments = function.get("arguments", "{}")

            print(f"\nTOOL: {name}")

            try:
                arguments = json.loads(raw_arguments)

                if not isinstance(arguments, dict):
                    raise ValueError(
                        "Tool arguments must be a JSON object."
                    )

                result = execute_tool(name, arguments)
                result_text = str(result)

            except Exception as error:
                result_text = f"TOOL ERROR: {error}"
                print(result_text)
            else:
                print(result_text)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result_text,
                }
            )

    raise RuntimeError(
        f"The agent exceeded {MAX_AGENT_TURNS} turns."
    )


def load_assignment(path: str) -> str:
    assignment_path = Path(path).expanduser()

    if not assignment_path.is_absolute():
        assignment_path = (
            Path.cwd() / "workspace" / assignment_path
        ).resolve()

    if not assignment_path.exists():
        raise FileNotFoundError(
            f"Assignment file not found: {assignment_path}"
        )

    if not assignment_path.is_file():
        raise ValueError(
            f"Assignment path is not a file: {assignment_path}"
        )

    return assignment_path.read_text(encoding="utf-8")


def main():
    print("=== Student Agent ===")

    parser = argparse.ArgumentParser(
        description="StudyFlow student coding agent."
    )

    parser.add_argument("--assignment-file")
    parser.add_argument("--assignment")
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--branch", default="")
    parser.add_argument("--commit-message", required=True)

    args = parser.parse_args()

    student_profile = {
        "first_name": os.environ.get(
            "STUDYFLOW_STUDENT_FIRST_NAME", ""
        ).strip(),
        "last_name": os.environ.get(
            "STUDYFLOW_STUDENT_LAST_NAME", ""
        ).strip(),
        "student_group": os.environ.get(
            "STUDYFLOW_STUDENT_GROUP", ""
        ).strip(),
    }

    if args.assignment is None and args.assignment_file is None:
        parser.error(
            "Provide either --assignment or --assignment-file."
        )

    assignment = (
        args.assignment
        if args.assignment is not None
        else load_assignment(args.assignment_file)
    )

    if not assignment.strip():
        raise ValueError("Assignment cannot be empty.")

    emit_stage("clone")
    print("\n--- Cloning repository ---")

    repo_path = clone_repository(args.repo_url)

    print("Repository cloned into the task workspace.")

    set_workspace(repo_path)

    emit_stage("prepare")
    
    requested_branch = args.branch.strip()

    branch_mode = os.environ.get(
        "STUDYFLOW_BRANCH_MODE",
        "",
    ).strip()
    
    if branch_mode not in {"named", "current"}:
        branch_mode = "named" if requested_branch else "current"
    
    if os.environ.get("STUDYFLOW_REQUIRE_CLEAN_TREE") != "true":
        raise RuntimeError(
            "StudyFlow clean-tree safety check is not enabled."
        )
    
    configured_target = os.environ.get(
        "STUDYFLOW_PUSH_TARGET",
        "",
    ).strip()
    
    if requested_branch and configured_target != requested_branch:
        raise RuntimeError(
            "The configured GitHub push target does not match "
            "the requested branch."
        )
    
    actual_branch = git_prepare_branch(
        requested_branch,
        branch_mode,
    )
    
    if configured_target and configured_target != actual_branch:
        raise RuntimeError(
            "The configured GitHub push target does not match "
            "the actual target branch."
        )
    
    os.environ["STUDYFLOW_PUSH_TARGET"] = actual_branch
    
    args.branch = actual_branch

    if actual_branch != args.branch:
        raise RuntimeError(
            "The actual checked-out branch does not match the requested target."
        )

    generate_archive = (
        os.environ.get("STUDYFLOW_GENERATE_ARCHIVE") == "true"
    )
    generate_report_requested = (
        os.environ.get("STUDYFLOW_GENERATE_REPORT") == "true"
    )
    output_mode = os.environ.get(
        "STUDYFLOW_OUTPUT_DIRECTORY_MODE",
        "disabled",
    )

    tasks_folder = ""
    tasks_path = None
    if generate_archive:
        if output_mode not in ("exclusive", "reuse"):
            raise RuntimeError(
                "StudyFlow did not authorize a safe, exclusive results folder."
            )

        try:
            tasks_folder = get_results_directory_name(args.branch)
            supplied_folder = os.environ.get("STUDYFLOW_OUTPUT_DIRECTORY", "")

            if supplied_folder != tasks_folder:
                raise RuntimeError(
                    "The results folder does not match the target Git branch."
                )

            tasks_path = set_tasks_folder(
                tasks_folder,
                allow_existing=output_mode == "reuse",
            )
            emit_artifact_status("archive", "ready", tasks_path)
        except (OSError, RuntimeError, ValueError) as error:
            tasks_folder = ""
            tasks_path = None
            emit_artifact_status("archive", "failed")
            print(
                "WARNING: Results folder could not be prepared; "
                f"continuing without it ({type(error).__name__})."
            )
    else:
        if output_mode != "disabled":
            raise RuntimeError(
                "StudyFlow unexpectedly enabled a results folder."
            )

    generate_report = generate_report_requested
    report_path = (
        tasks_path / "report.txt"
        if tasks_path is not None
        else get_workspace() / "report.txt"
    )
    if generate_report_requested:
        if report_path.exists() or report_path.is_symlink():
            emit_artifact_status("report", "failed")
            generate_report = False
        else:
            try:
                relative_report_path = report_path.relative_to(
                    get_workspace()
                ).as_posix()
                exclude_report_from_git(relative_report_path)
            except (OSError, ValueError, RuntimeError):
                emit_artifact_status("report", "failed")
                generate_report = False

    if generate_report and (report_path.exists() or report_path.is_symlink()):
        if generate_report_requested:
            emit_artifact_status("report", "failed")
        generate_report = False

    if tasks_path is None and tasks_folder:
        raise RuntimeError("The results folder was not safely prepared.")

    print("\n--- Repository prepared ---")
    print(f"Checked-out branch: {actual_branch}")
    if tasks_path:
        print(f"Results folder: {tasks_path.name}")
    else:
        print("Results folder: disabled")
    print("\n--- Assignment loaded ---")
    print(f"Assignment length: {len(assignment)} characters")

    emit_stage("ai")
    print("\n--- Starting AI agent ---")

    try:
        run_agent(
            assignment,
            tasks_folder,
            generate_report,
            student_profile,
        )
    except Exception:
        if generate_report_requested:
            if report_path.is_file() and not report_path.is_symlink() and report_path.stat().st_size > 0:
                emit_artifact_status("report", "ready", report_path)
            elif generate_report:
                emit_artifact_status("report", "failed")
        if generate_archive and tasks_path is not None:
            has_result_files = any(
                path.is_file() and not path.is_symlink()
                for path in tasks_path.rglob("*")
            )
            if not has_result_files:
                emit_artifact_status("archive", "failed")
        raise

    if generate_report_requested and generate_report:
        if (
            report_path.is_file()
            and not report_path.is_symlink()
            and report_path.stat().st_size > 0
        ):
            emit_artifact_status("report", "ready", report_path)
        else:
            emit_artifact_status("report", "failed")

    if generate_archive and tasks_path is not None:
        has_result_files = any(
            path.is_file() and not path.is_symlink()
            for path in tasks_path.rglob("*")
        )
        if not has_result_files:
            emit_artifact_status("archive", "failed")
        elif output_mode == "reuse":
            emit_artifact_status("archive", "ready", tasks_path)

    emit_stage("git_check")
    print("\n--- Checking changes ---")

    status = git_status()

    print(status if status else "No changes.")

    if not status:
        raise RuntimeError(
            "The agent completed without making repository changes. "
            "No commit was created or pushed."
        )

    print("\n--- Git diff ---")

    diff = git_diff()
    print(diff)

    emit_stage("commit")
    print("\n--- Commit ---")

    commit_result = git_commit(args.commit_message)
    print(commit_result)

    if commit_result == "No changes to commit.":
        print("\nNothing to push.")
        return

    emit_stage("push")
    print("\n--- Push ---")

    push_result, pushed_commit = git_push(args.branch)
    print(push_result)
    print(f"STUDYFLOW_PUSHED:{pushed_commit}", flush=True)

    print("\n=== DONE ===")
    print(f"\nRepository:\n{get_workspace()}")
    print(
        "\nResults folder:\n"
        f"{get_tasks_folder().name if get_tasks_folder() else 'disabled'}"
    )
    print(f"\nBranch:\n{args.branch}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAgent interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(
            f"\nFATAL ERROR: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
