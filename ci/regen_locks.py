"""Regenerate the hash-locked CI requirement files.

    python ci/regen_locks.py

Writes ci/requirements.txt (CI toolchain), ci/requirements-oev.txt
(dev+app closure for tests.yml) and ci/requirements-docker.txt
(backbone+serve closure for the Dockerfile), each with sha256 hashes for
pip --require-hashes. Every release file of every pinned version is
hashed, so the locks install on any platform. torch is pinned to the CPU
build from download.pytorch.org (wheels hashed via pip download), so
installs of the oev/docker locks need torch preinstalled from the CPU
index first (pip then treats the == pin as satisfied). Version selection
respects every constraint in the dependency graph (e.g. gradio's
huggingface-hub<2.0) and every pin is deliberate: ruff 0.16.9 is the lint
gate, matplotlib 3.11.2 the chart pixel gate.
"""
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import tomllib
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

HERE = Path(__file__).parent
ROOT = HERE.parent
LINUX = {"python_version": "3.11.16", "python_full_version": "3.11.16",
         "sys_platform": "linux", "platform_system": "Linux", "os_name": "posix",
         "platform_machine": "x86_64", "implementation_name": "cpython",
         "platform_python_implementation": "CPython", "extra": ""}
WINDOWS = dict(LINUX, sys_platform="win32", platform_system="Windows", os_name="nt")
ENVS = [LINUX, WINDOWS]  # union: locks must install on CI (linux) and locally (win)
TOOLCHAIN = {"ruff": "0.16.9", "pip": "26.2.1", "setuptools": "84.0.0",
             "pip-audit": "2.10.1", "build": "1.6.1", "twine": "7.0.0",
             "matplotlib": "3.11.2", "numpy": "2.5.3", "pillow": "12.3.0",
             "huggingface_hub[cli]": "2.0.0"}


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "oev-ci-lock/1.0"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def canon(name):
    return name.lower().replace("_", "-")


def dep_edges(meta, extras):
    """Dependency specs of one package that apply on either platform."""
    envs = [dict(e, extra=x) for e in ENVS for x in (extras or [None])]
    out = []
    for rd in meta["info"].get("requires_dist") or []:
        try:
            dep = Requirement(rd)
        except Exception:  # noqa: BLE001, S112 - skip unparseable dep specs rather than fail the whole lock
            continue
        if dep.marker and not any(dep.marker.evaluate(e) for e in envs):
            continue
        out.append(rd)
    return out


def graph(root_specs):
    """BFS the dependency graph using each package's latest-version metadata.
    Returns {canonical name: [dependency requirement strings]}."""
    edges, stack = {}, [Requirement(s) for s in root_specs]
    seen = set()
    while stack:
        r = stack.pop()
        key = canon(r.name)
        if key in seen:
            continue
        seen.add(key)
        meta = get(f"https://pypi.org/pypi/{key}/json")
        edges[key] = dep_edges(meta, r.extras)
        for rd in edges[key]:
            stack.append(Requirement(rd))
    return edges


def constraints_of(root_specs, edges):
    """Union of every specifier pointing at each package, incl. root specs."""
    cons = defaultdict(list)
    for s in root_specs:
        r = Requirement(s)
        if r.specifier:
            cons[canon(r.name)].append(r.specifier)
    for specs in edges.values():
        for rd in specs:
            r = Requirement(rd)
            if r.specifier:
                cons[canon(r.name)].append(r.specifier)
    return cons


def pick_version(key, cons_list, meta):
    """Newest stable release satisfying all constraints."""
    spec = SpecifierSet(",".join(str(c) for c in cons_list))
    candidates = []
    for ver, files in meta["releases"].items():
        if not files:
            continue
        try:
            v = Version(ver)
        except InvalidVersion:
            continue
        if v.is_prerelease or all(f.get("yanked") for f in files):
            continue
        candidates.append(v)
    if cons_list:
        candidates = [v for v in candidates if v in spec]
    assert candidates, f"{key}: no stable release satisfies {spec}"
    return str(max(candidates))


def lock(root_specs):
    edges = graph(root_specs)
    cons = constraints_of(root_specs, edges)
    lines = []
    for key in sorted(edges):
        meta = get(f"https://pypi.org/pypi/{key}/json")
        ver = pick_version(key, cons.get(key, []), meta)
        hashes = "".join(f" --hash=sha256:{f['digests']['sha256']}"
                         for f in meta["releases"][ver]
                         if f.get("digests", {}).get("sha256"))
        assert hashes, f"{key}=={ver}: no hashed files"
        lines.append(f"{key}=={ver}{hashes}")
    return lines


def torch_cpu_lines():
    # CPU-build wheels for the platforms CI or a local venv installs on;
    # pip download fetches for the RUNNING interpreter unless told otherwise
    out = HERE / "_torchcpu"
    lines = []
    for platform_, abi in (("manylinux_2_28_x86_64", "cp311"), ("win_amd64", "cp311")):
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--quiet",
                        "-d", str(out), "--only-binary=:all:",
                        "--python-version", "311", "--platform", platform_,
                        "--abi", abi, "--index-url",
                        "https://download.pytorch.org/whl/cpu", "torch==2.14.0"],
                       check=True)
        wheel = next(out.glob("torch-*.whl"))
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        lines.append(f"torch==2.14.0 --hash=sha256:{digest}")
        wheel.unlink()
    subprocess.run(["rm", "-rf", str(out)], check=True)
    return lines


def write(path, lines, comment):
    path.write_text(f"# {comment}\n"
                    "# regen: python ci/regen_locks.py\n\n" + "\n".join(lines) + "\n",
                    encoding="utf-8")
    print(f"{path.name}: {len(lines)} packages")


def pyproject_deps(*extras):
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    deps = list(proj["dependencies"])
    for e in extras:
        deps += proj["optional-dependencies"][e]
    # torch is replaced by the CPU-build lines appended separately
    return [d for d in deps if canon(Requirement(d).name) != "torch"]


torch = torch_cpu_lines()

write(HERE / "requirements.txt", lock([f"{k}=={v}" if not k.endswith("]") else k
                                       for k, v in TOOLCHAIN.items()]),
      "hash-locked CI toolchain: lint, audit, build, chart gates.")

write(HERE / "requirements-oev.txt",
      lock(pyproject_deps("dev", "app")) + torch,
      "hash-locked closure of pyproject [dev,app]; torch pinned to the CPU "
      "build - preinstall it from --index-url download.pytorch.org/whl/cpu "
      "before requiring this file.")

write(HERE / "requirements-docker.txt",
      lock(pyproject_deps("backbone", "serve")) + torch,
      "hash-locked closure of pyproject [backbone,serve]; same torch rule "
      "as requirements-oev.txt.")
