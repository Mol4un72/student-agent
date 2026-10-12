from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import base64
import json
import os
import re
import subprocess


# Directory for isolated repository workspaces.
PROJECTS_DIR = (
    Path.home() / "StudyFlow" / "workspaces"
).resolve()

PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

WORKSPACE: Optional[Path] = None
TASKS_FOLDER: Optional[Path] = None


# --------------------------------------------------
# GitHub authentication
# --------------------------------------------------

def get_git_auth_env() -> dict:
    """
    Return an environment with GitHub App token authentication.

    The token is passed through Git configuration,
    not embedded in the repository URL.
    """
    token = os.environ.get("GITHUB_TOKEN")

    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set.")

    credentials = base64.b64encode(
        f"x-access-token:{token}".encode("utf-8")
    ).decode("ascii")

    env = os.environ.copy()

    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = (
        "http.https://github.com/.extraheader"
    )
    env["GIT_CONFIG_VALUE_0"] = (
        f"AUTHORIZATION: basic {credentials}"
    )

    return env


# --------------------------------------------------
# Workspace management
# --------------------------------------------------

def set_workspace(path: Union[Path, str]) -> None:
    global WORKSPACE

    target = Path(path).resolve()

    if target.parent != PROJECTS_DIR:
        raise ValueError(
            f"Workspace must be directly inside {PROJECTS_DIR}."
        )

    if not target.exists() or not target.is_dir():
        raise FileNotFoundError(
            f"Workspace does not exist: {target}"
        )

    WORKSPACE = target


def get_workspace() -> Path:
    if WORKSPACE is None:
        raise RuntimeError("Workspace is not set.")

    return WORKSPACE


def set_tasks_folder(
    folder_name: Optional[str],
    allow_existing: bool = False,
) -> Optional[Path]:
    global TASKS_FOLDER

    workspace = get_workspace()

    if folder_name is None:
        TASKS_FOLDER = None
        return None

    folder_name = folder_name.strip()

    if (
        not folder_name
        or folder_name in (".", "..")
        or Path(folder_name).name != folder_name
        or "/" in folder_name
        or "\\" in folder_name
    ):
        raise ValueError(
            "Results folder must be a single safe directory name."
        )

    candidate = workspace / folder_name

    if candidate.is_symlink():
        raise FileExistsError(
            "The results folder for this branch already exists. "
            "Nothing was overwritten; choose another branch or move the existing results."
        )

    if candidate.exists():
        if allow_existing and candidate.is_dir():
            TASKS_FOLDER = candidate.resolve()
            if TASKS_FOLDER.parent != workspace:
                raise ValueError("Tasks folder must be inside the repository.")
            return TASKS_FOLDER

        raise FileExistsError(
            "The results folder for this branch already exists. "
            "Nothing was overwritten; choose another branch or move the existing results."
        )

    target = candidate.resolve()

    if target.parent != workspace:
        raise ValueError(
            "Tasks folder must be inside the repository."
        )

    candidate.mkdir(exist_ok=False)

    TASKS_FOLDER = target

    return target


def get_tasks_folder() -> Optional[Path]:
    return TASKS_FOLDER


def get_results_directory_name(branch: str) -> str:
    check = subprocess.run(
        ["git", "check-ref-format", "--branch", branch],
        cwd=get_workspace(),
        capture_output=True,
        text=True,
    )

    if check.returncode != 0:
        raise ValueError(
            "The target branch name is invalid for Git."
        )

    encoded = base64.urlsafe_b64encode(
        branch.encode("utf-8")
    ).decode("ascii").rstrip("=")

    try:
        decoded = base64.urlsafe_b64decode(
            encoded + "=" * (-len(encoded) % 4)
        ).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError(
            "The target branch cannot be mapped safely to a results folder."
        ) from error

    folder_name = f"branch-{encoded}"

    if (
        decoded != branch
        or Path(folder_name).name != folder_name
        or len(os.fsencode(folder_name)) > 255
    ):
        raise ValueError(
            "The target branch cannot be mapped safely to a results folder."
        )

    return folder_name


# --------------------------------------------------
# Repository cloning
# --------------------------------------------------

def clone_repository(repo_url: str) -> Path:
    repo_url = repo_url.strip()

    if not repo_url:
        raise ValueError(
            "Repository URL cannot be empty."
        )

    if not repo_url.startswith("https://github.com/"):
        raise ValueError(
            "Only HTTPS GitHub repository URLs are allowed."
        )

    repo_path = repo_url.removeprefix(
        "https://github.com/"
    ).strip("/")

    parts = repo_path.split("/")

    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "Invalid GitHub repository URL."
        )

    repo_name = parts[1]

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    if not repo_name:
        raise ValueError(
            "Cannot determine repository name."
        )

    # Each task receives its own repository copy.
    target = (
        PROJECTS_DIR
        / f"{repo_name}-{uuid4().hex[:8]}"
    ).resolve()

    if target.parent != PROJECTS_DIR:
        raise ValueError(
            "Invalid repository path."
        )

    result = subprocess.run(
        ["git", "clone", repo_url, str(target)],
        cwd=PROJECTS_DIR,
        capture_output=True,
        text=True,
        env=get_git_auth_env(),
    )

    if result.returncode != 0:
        output = (
            result.stdout + "\n" + result.stderr
        ).strip()

        raise RuntimeError(
            f"git clone failed:\n{output}"
        )

    return target


# --------------------------------------------------
# Git configuration and branch management
# --------------------------------------------------

def exclude_report_from_git(report_relative_path: str = "report.txt") -> None:
    workspace = get_workspace()
    git_dir = workspace / ".git"

    if not git_dir.exists():
        raise FileNotFoundError(
            ".git directory not found."
        )

    exclude_file = git_dir / "info" / "exclude"

    exclude_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = ""

    if exclude_file.exists():
        existing = exclude_file.read_text(
            encoding="utf-8"
        )

    report_relative_path = report_relative_path.replace("\\", "/").lstrip("/")

    if report_relative_path not in existing.splitlines():
        with exclude_file.open(
            "a",
            encoding="utf-8",
        ) as file:
            if existing and not existing.endswith("\n"):
                file.write("\n")

            file.write(f"{report_relative_path}\n")


def get_remote_branch_sha(branch: str) -> Optional[str]:
    remote = subprocess.run(
        [
            "git",
            "ls-remote",
            "--heads",
            "origin",
            f"refs/heads/{branch}",
        ],
        cwd=get_workspace(),
        capture_output=True,
        text=True,
        env=get_git_auth_env(),
        timeout=30,
    )

    if remote.returncode != 0:
        raise RuntimeError(
            "Could not verify repository access and branch state on GitHub."
        )

    expected_ref = f"refs/heads/{branch}"
    return next(
        (
            line.split("\t", 1)[0]
            for line in remote.stdout.splitlines()
            if "\t" in line and line.split("\t", 1)[1] == expected_ref
        ),
        None,
    )

def git_prepare_branch(
    branch: str,
    branch_mode: str,
) -> str:
    branch = branch.strip()

    workspace = get_workspace()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if status.returncode != 0:
        raise RuntimeError(
            "Could not inspect the repository working tree."
        )

    if status.stdout.strip():
        raise RuntimeError(
            "The repository has uncommitted changes. "
            "Branch operation stopped without changing them."
        )

    current = subprocess.run(
        [
            "git",
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if current.returncode != 0 or not current.stdout.strip():
        raise RuntimeError(
            "Could not determine the current Git branch."
        )

    current_branch = current.stdout.strip()

    if branch_mode == "current":
        remote_sha = get_remote_branch_sha(current_branch)

        if remote_sha is None:
            raise RuntimeError(
                f"The current branch '{current_branch}' "
                "does not exist on GitHub."
            )

        local_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
        )

        if (
            local_sha.returncode != 0
            or local_sha.stdout.strip() != remote_sha
        ):
            raise RuntimeError(
                "The current branch differs from GitHub. "
                "No branch or existing commit was changed."
            )

        return current_branch

    if branch_mode != "named":
        raise ValueError("Unknown branch mode.")

    if not branch:
        raise ValueError("The new branch name cannot be empty.")

    check = subprocess.run(
        [
            "git",
            "check-ref-format",
            "--branch",
            branch,
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if check.returncode != 0:
        raise ValueError(
            "The target branch name is invalid for Git."
        )

    local_ref = subprocess.run(
        [
            "git",
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if local_ref.returncode == 0:
        raise RuntimeError(
            f"The local branch '{branch}' already exists. "
            "Choose another branch name."
        )

    if local_ref.returncode not in (0, 1):
        raise RuntimeError(
            "Could not verify the local branch state."
        )

    remote_sha = get_remote_branch_sha(branch)

    if remote_sha is not None:
        raise RuntimeError(
            f"The GitHub branch '{branch}' already exists. "
            "Choose another branch name."
        )

    result = subprocess.run(
        [
            "git",
            "checkout",
            "-b",
            branch,
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        output = (
            result.stdout + "\n" + result.stderr
        ).strip()

        raise RuntimeError(
            f"Could not create branch '{branch}':\n{output}"
        )

    checked_out = subprocess.run(
        [
            "git",
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if (
        checked_out.returncode != 0
        or checked_out.stdout.strip() != branch
    ):
        raise RuntimeError(
            "Git did not confirm the newly created branch."
        )

    return branch

# --------------------------------------------------
# File access
# --------------------------------------------------

def safe_path(path: str) -> Path:
    workspace = get_workspace()
    target = (workspace / path).resolve()

    if target != workspace and workspace not in target.parents:
        raise ValueError(
            "Access outside repository is not allowed."
        )

    git_directory = (workspace / ".git").resolve()
    if target == git_directory or git_directory in target.parents:
        raise ValueError(
            "Repository Git metadata is controlled by the host program."
        )

    return target


def list_files() -> list[str]:
    workspace = get_workspace()
    files = []

    for path in workspace.rglob("*"):
        if ".git" in path.parts:
            continue

        if path.is_file():
            files.append(
                str(path.relative_to(workspace))
            )

    return files


def read_file(path: str) -> str:
    target = safe_path(path)

    if not target.exists():
        raise FileNotFoundError(path)

    if not target.is_file():
        raise ValueError(
            f"Not a file: {path}"
        )

    return target.read_text(
        encoding="utf-8"
    )


def write_file(path: str, content: str) -> str:
    target = safe_path(path)

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.write_text(
        content,
        encoding="utf-8",
    )

    return f"File written: {path}"


# --------------------------------------------------
# Command execution
# --------------------------------------------------

def run_command(command: str) -> str:
    workspace = get_workspace()

    if re.search(r"\bgit(?:\s|$)", command, flags=re.IGNORECASE):
        return "Git commands are controlled by the host program and cannot be run here."

    # Do not expose the GitHub token to commands
    # executed by the AI agent.
    command_env = os.environ.copy()
    for variable in (
        "GITHUB_TOKEN",
        "STUDYFLOW_SENSITIVE_VALUES",
        "STUDYFLOW_PERSONAL_VALUES",
        "OPENROUTER_API_KEY",
        "OPENROUTER_API_KEY_1",
        "OPENROUTER_API_KEY_2",
        "OPENROUTER_API_KEY_3",
    ):
        command_env.pop(variable, None)

    result = subprocess.run(
        command,
        shell=True,
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
        env=command_env,
    )

    output = result.stdout

    if result.stderr:
        output += "\nSTDERR:\n"
        output += result.stderr

    output += f"\nEXIT CODE: {result.returncode}"

    error_markers = [
        "Traceback (most recent call last):",
        "SyntaxError:",
        "TypeError:",
        "NameError:",
        "ValueError:",
        "AttributeError:",
        "KeyError:",
        "IndexError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "AssertionError:",
        "EOFError:",
    ]

    detected_errors = [
        marker
        for marker in error_markers
        if marker in output
    ]

    if detected_errors:
        output += (
            "\n\nTEST RESULT: FAILED\n"
            "Detected errors: "
            + ", ".join(detected_errors)
        )
    elif result.returncode != 0:
        output += "\n\nTEST RESULT: FAILED"
    else:
        output += "\n\nTEST RESULT: PASSED"

    return redact_sensitive_values(output)


# --------------------------------------------------
# Git status and diff
# --------------------------------------------------

def git_status() -> str:
    workspace = get_workspace()

    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        output = (
            result.stdout + "\n" + result.stderr
        ).strip()

        raise RuntimeError(
            f"git status failed:\n{output}"
        )

    return result.stdout.strip()


def get_sensitive_values(environment_variable: str) -> list[str]:
    raw_values = os.environ.get(environment_variable, "[]")

    try:
        values = json.loads(raw_values)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "StudyFlow supplied invalid personal-data safeguards."
        ) from error

    if not isinstance(values, list) or not all(
        isinstance(value, str) for value in values
    ):
        raise RuntimeError(
            "StudyFlow supplied invalid personal-data safeguards."
        )

    return [value for value in values if value]


def redact_sensitive_values(text: str) -> str:
    for value in sorted(
        get_sensitive_values("STUDYFLOW_SENSITIVE_VALUES"),
        key=len,
        reverse=True,
    ):
        text = re.sub(
            re.escape(value),
            "[redacted]",
            text,
            flags=re.IGNORECASE,
        )

    return text


def git_diff() -> str:
    workspace = get_workspace()

    result = subprocess.run(
        ["git", "diff"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        output = (
            result.stdout + "\n" + result.stderr
        ).strip()

        raise RuntimeError(
            f"git diff failed:\n{output}"
        )

    return redact_sensitive_values(result.stdout)


# --------------------------------------------------
# Git commit and push
# --------------------------------------------------

def git_commit(message: str) -> str:
    workspace = get_workspace()
    message = message.strip()
    expected_branch = os.environ.get("STUDYFLOW_PUSH_TARGET", "").strip()

    if not message:
        raise ValueError(
            "Commit message cannot be empty."
        )

    current = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if (
        current.returncode != 0
        or not expected_branch
        or current.stdout.strip() != expected_branch
    ):
        raise RuntimeError(
            "The checked-out branch does not match the configured push target. "
            "Nothing was committed."
        )

    personal_values = get_sensitive_values(
        "STUDYFLOW_PERSONAL_VALUES"
    )

    if any(value.casefold() in message.casefold() for value in personal_values):
        raise RuntimeError(
            "The commit message contains personal profile data. "
            "Nothing was committed."
        )

    add_result = subprocess.run(
        ["git", "add", "--all"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if add_result.returncode != 0:
        output = (
            add_result.stdout + "\n" + add_result.stderr
        ).strip()

        raise RuntimeError(
            f"git add failed:\n{output}"
        )

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if status.returncode != 0:
        raise RuntimeError(
            f"git status failed:\n{status.stderr.strip()}"
        )

    if not status.stdout.strip():
        return "No changes to commit."

    diff_result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--no-ext-diff",
            "--unified=0",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if diff_result.returncode != 0:
        raise RuntimeError(
            "Could not inspect staged changes. They remain local and uncommitted."
        )

    staged_changes = diff_result.stdout

    if any(
        value.casefold() in staged_changes.casefold()
        for value in personal_values
    ):
        raise RuntimeError(
            "Staged changes contain personal profile data. "
            "They remain local and were not committed or pushed."
        )

    secret_patterns = (
        r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}\b",
        r"\bsk-[A-Za-z0-9_-]{20,}\b",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"(?i)\b(?:OPENROUTER_API_KEY|GITHUB_TOKEN|OPENAI_API_KEY)\s*[:=]\s*['\"]?[A-Za-z0-9_-]{12,}",
    )

    if any(
        re.search(pattern, staged_changes + "\n" + message)
        for pattern in secret_patterns
    ):
        raise RuntimeError(
            "Staged changes appear to contain a secret. "
            "They remain local and were not committed or pushed."
        )

    # Configure identity locally for this repository only.
    identity_settings = [
        ("user.name", "StudyFlow Agent"),
        ("user.email", "studyflow-agent@users.noreply.github.com"),
    ]

    for key, value in identity_settings:
        identity_result = subprocess.run(
            ["git", "config", "--local", key, value],
            cwd=workspace,
            capture_output=True,
            text=True,
        )

        if identity_result.returncode != 0:
            raise RuntimeError(
                f"Failed to configure local Git setting {key}: "
                f"{identity_result.stderr.strip()}"
            )

    result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    output = (
        result.stdout + "\n" + result.stderr
    ).strip()

    if result.returncode != 0:
        raise RuntimeError(
            f"git commit failed:\n{output}"
        )

    return output


def git_push(branch: str) -> tuple[str, str]:
    workspace = get_workspace()
    branch = branch.strip()

    if not branch:
        raise ValueError(
            "Branch name cannot be empty."
        )

    current = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if current.returncode != 0 or current.stdout.strip() != branch:
        raise RuntimeError(
            "The current branch does not match the requested push target. "
            "No push was attempted."
        )

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError(
            "The working tree is not clean after commit. "
            "No push was attempted."
        )

    local_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    if local_sha.returncode != 0:
        raise RuntimeError(
            "Не вдалося визначити локальний commit перед push. "
            "Зміни залишилися в робочому середовищі агента."
        )
    commit_sha = local_sha.stdout.strip()

    result = subprocess.run(
        [
            "git",
            "push",
            "--porcelain",
            "origin",
            f"HEAD:refs/heads/{branch}",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=get_git_auth_env(),
    )

    output = (
        result.stdout + "\n" + result.stderr
    ).strip()

    if result.returncode != 0:
        raise RuntimeError(
            "GitHub не підтвердив push. Локальний commit "
            f"{commit_sha} та зміни збережено в робочому середовищі агента; "
            "до GitHub їх не опубліковано.\n"
            f"{output}"
        )

    remote = subprocess.run(
        [
            "git",
            "ls-remote",
            "--heads",
            "origin",
            f"refs/heads/{branch}",
        ],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=get_git_auth_env(),
        timeout=30,
    )

    if remote.returncode != 0:
        raise RuntimeError(
            "Push було надіслано, але кінцевий стан гілки не вдалося перевірити. "
            f"Локальний commit {commit_sha} збережено; результат не підтверджено."
        )

    remote_sha = next(
        (
            line.split("\t", 1)[0]
            for line in remote.stdout.splitlines()
            if line.endswith(f"\trefs/heads/{branch}")
        ),
        None,
    )

    if remote_sha != commit_sha:
        raise RuntimeError(
            "GitHub branch не вказує на commit завдання після push. "
            f"Локальний commit {commit_sha} збережено; результат не підтверджено."
        )

    return output, commit_sha