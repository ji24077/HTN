"""Stop only the isolated Relay fleet demo database; never initialize or delete it."""

from pathlib import Path
import os
import subprocess
import sys


def database_path(workspace: Path) -> Path:
    workspace = workspace.resolve(strict=True)
    isolated = workspace / ".relay-fleet-local"
    expected = isolated / ".demo" / "postgres"
    resolved = expected.resolve()
    # Reject junctions/symlinks that redirect the explicitly named demo folder.
    if resolved != expected or not resolved.is_relative_to(isolated):
        raise RuntimeError("Refusing to stop a database outside the isolated demo path.")
    return resolved


def matching_process(database: Path, binary: Path):
    """Return the verified postmaster, or None when it has already exited."""
    import psutil

    pid_file = database / "postmaster.pid"
    if not pid_file.exists():
        return None
    if pid_file.resolve() != pid_file:
        raise RuntimeError("Refusing a redirected database process file.")
    lines = pid_file.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise RuntimeError("Cannot verify the local database process file.")
    try:
        pid, started = int(lines[0]), int(lines[2])
    except ValueError:
        raise RuntimeError("Cannot verify the local database process identity.") from None
    if pid <= 0 or Path(lines[1]).resolve() != database:
        raise RuntimeError("Database process file does not match the isolated demo.")
    try:
        process = psutil.Process(pid)
        if not process.is_running():
            return None
        # A stale PID must never cause a different process to be stopped.
        if abs(process.create_time() - started) > 2:
            return None
        if Path(process.exe()).resolve() != binary.resolve():
            raise RuntimeError("Database executable does not match bundled PostgreSQL.")
        arguments = process.cmdline()
        actual_data = None
        for index, argument in enumerate(arguments):
            if argument in {"-D", "--pgdata"} and index + 1 < len(arguments):
                actual_data = arguments[index + 1]
                break
            if argument.startswith("--pgdata="):
                actual_data = argument.split("=", 1)[1]
                break
        if actual_data is None or Path(actual_data).resolve() != database:
            raise RuntimeError("Running database does not use the isolated demo directory.")
        return process
    except psutil.NoSuchProcess:
        return None
    except psutil.AccessDenied:
        raise RuntimeError("Cannot verify the local database process; nothing was stopped.") from None


def stop_database(workspace: Path) -> None:
    database = database_path(workspace)
    if not database.exists() or not (database / "postmaster.pid").exists():
        print("Relay fleet database is already stopped.")
        return
    if not (database / "PG_VERSION").is_file():
        raise RuntimeError("The isolated demo directory is not a PostgreSQL database.")

    # Importing the binary path does not construct a server. In particular, do
    # not call pgserver.get_server(), which would start a stopped database.
    from pgserver._commands import POSTGRES_BIN_PATH

    suffix = ".exe" if os.name == "nt" else ""
    postgres = POSTGRES_BIN_PATH / f"postgres{suffix}"
    pg_ctl = POSTGRES_BIN_PATH / f"pg_ctl{suffix}"
    process = matching_process(database, postgres)
    if process is None:
        print("Relay fleet database is already stopped.")
        return
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    result = subprocess.run(
        [str(pg_ctl), "-D", str(database), "-w", "-t", "20", "-m", "fast", "stop"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=25,
        creationflags=flags,
        check=False,
    )
    # The owning demo server may have completed cleanup concurrently.
    if result.returncode != 0 and process.is_running():
        raise RuntimeError("PostgreSQL did not stop cleanly; database files were preserved.")
    print("Relay fleet database stopped; database files preserved.")


def main() -> int:
    try:
        stop_database(Path(__file__).resolve().parent)
    except (OSError, RuntimeError, ImportError, subprocess.SubprocessError):
        # Do not expose connection details, raw command output, or environment.
        print("Could not safely stop the local fleet database. Its files were preserved.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
