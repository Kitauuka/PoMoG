#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


P2RANK_VERSION = "2.5.1"
P2RANK_URL = f"https://github.com/rdk/p2rank/releases/download/{P2RANK_VERSION}/p2rank_{P2RANK_VERSION}.tar.gz"


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("+", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _conda_env_exists(env_name: str) -> bool:
    proc = subprocess.run(["conda", "env", "list"], check=True, text=True, stdout=subprocess.PIPE)
    needle = f"/envs/{env_name}"
    return any(line.split() and (line.split()[0] == env_name or needle in line) for line in proc.stdout.splitlines())


def _download(url: str, path: Path, *, dry_run: bool = False) -> None:
    print(f"+ download {url} -> {path}", flush=True)
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def _install_fpocket(env_name: str, *, dry_run: bool = False) -> None:
    local_fpocket = Path("external_tools/fpocket-4.2.3/bin/fpocket").resolve()
    if local_fpocket.exists():
        print(f"[setup] local fpocket already installed: {local_fpocket}")
        return
    packages = ["openjdk=17", "fpocket=4.2.3"]
    if not _conda_env_exists(env_name):
        _run(["conda", "create", "-n", env_name, "-y", "-c", "conda-forge", *packages], dry_run=dry_run)
    else:
        _run(["conda", "install", "-n", env_name, "-y", "-c", "conda-forge", *packages], dry_run=dry_run)


def _install_p2rank(tool_dir: Path, *, dry_run: bool = False) -> None:
    target = tool_dir / f"p2rank_{P2RANK_VERSION}"
    prank = target / "prank"
    if prank.exists():
        print(f"[setup] P2Rank already installed: {prank}")
        return

    archive = tool_dir / "downloads" / f"p2rank_{P2RANK_VERSION}.tar.gz"
    _download(P2RANK_URL, archive, dry_run=dry_run)
    print(f"+ extract {archive} -> {tool_dir}", flush=True)
    if dry_run:
        return
    tool_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(tool_dir)
    prank.chmod(prank.stat().st_mode | 0o111)


def _verify(env_name: str, tool_dir: Path, *, dry_run: bool = False) -> None:
    p2rank_home = tool_dir / f"p2rank_{P2RANK_VERSION}"
    local_fpocket = tool_dir / "fpocket-4.2.3" / "bin" / "fpocket"
    local_java = tool_dir / "java17" / "bin" / "java"
    fpocket_cmd = [str(local_fpocket), "-h"] if local_fpocket.exists() else ["conda", "run", "-n", env_name, "fpocket", "--help"]
    java_cmd = [str(local_java), "-version"] if local_java.exists() else ["conda", "run", "-n", env_name, "java", "-version"]
    if local_java.exists():
        p2rank_cmd = [
            "/usr/bin/env",
            f"JAVA_HOME={local_java.parent.parent}",
            f"PATH={local_java.parent}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            str(p2rank_home / "prank"),
            "-version",
        ]
    else:
        p2rank_cmd = [str(p2rank_home / "prank"), "-version"]
    commands = [fpocket_cmd, java_cmd, p2rank_cmd]
    for cmd in commands:
        try:
            _run(cmd, dry_run=dry_run)
        except subprocess.CalledProcessError as exc:
            print(f"[setup] verify failed for {' '.join(cmd)}: {exc}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install or download external pocket tools for PoMoG.")
    parser.add_argument("--env-name", default=os.environ.get("POMOG_POCKET_TOOL_ENV", "pomog-pocket-tools"))
    parser.add_argument("--tool-dir", default="external_tools")
    parser.add_argument("--skip-fpocket", action="store_true")
    parser.add_argument("--skip-p2rank", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tool_dir = Path(args.tool_dir).resolve()
    if not args.skip_fpocket:
        _install_fpocket(args.env_name, dry_run=args.dry_run)
    if not args.skip_p2rank:
        _install_p2rank(tool_dir, dry_run=args.dry_run)
    _verify(args.env_name, tool_dir, dry_run=args.dry_run)

    print("[setup] PoMoG pocket tool setup complete")
    print(f"[setup] P2Rank prank: {tool_dir / f'p2rank_{P2RANK_VERSION}' / 'prank'}")


if __name__ == "__main__":
    main()
