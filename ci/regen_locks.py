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
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

HERE = Path(__file__).parent
ROOT = HERE.parent
LINUX = {"python_version": "3.11.16", "python_full_version": "3.11.16",
         "sys_platform": "linux", "platform_system": "Linux", "os_name": "posix",
         "platform_machine": "x86_64", "implementation_name": "cpython",
         "platform_python_implementation": "CPython", "extra": ""}
WINDOWS = dict(LINUX, sys_platform="win32", platform_system="Windows", os_name="nt")
ENVS = [LINUX, WINDOWS]  # union: locks must install on CI (linux) and locally (win)
PY_TARGET = "3.11.16"  # CI runs python 3.11; never pick versions it cannot install
TOOLCHAIN = {"ruff": "0.16.9", "pip": "26.2.1", "setuptools": "84.0.0",
             "pip-audit": "2.10.1", "build": "1.6.1", "twine": "7.0.0",
             "matplotlib": "3.11.2", "numpy": None, "pillow": "12.3.0",
             "huggingface_hub[cli]": "2.0.0"}
BUILD = {"build": "1.6.1", "installer": None, "setuptools": "84.0.0"}


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


def installable_on_target(ver, files):
    """A release counts only if its wheel Requires-Python admits the CI
    interpreter. sdists often omit requires_python, so only wheels are
    trusted here (falling back to all files for sdist-only packages)."""
    wheels = [f for f in files if f.get("packagetype") == "bdist_wheel"] or files
    for f in wheels:
        rp = f.get("requires_python")
        if not rp:
            return True
        try:
            if SpecifierSet(rp).contains(PY_TARGET):
                return True
        except InvalidSpecifier:
            # legacy star forms like >=3.6.* - pip tolerates them, so do we
            return True
    return False


def pick_version(key, cons_list, meta):
    """Newest stable, non-yanked release satisfying all constraints AND
    the target interpreter's Requires-Python."""
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
        if not installable_on_target(ver, files):
            continue
        candidates.append(v)
    if cons_list:
        candidates = [v for v in candidates if v in spec]
    assert candidates, f"{key}: no stable release for py{PY_TARGET} satisfies {spec}"
    return str(max(candidates))


def lock(root_specs):
    """Constraint-aware closure: dependencies are discovered from the
    metadata of the version actually selected, not from latest - older
    picks (e.g. huggingface-hub<2.0) declare dependencies the latest
    metadata no longer has, and the lock must include them."""
    cons = defaultdict(list)
    for s in root_specs:
        r = Requirement(s)
        if r.specifier:
            cons[canon(r.name)].append(r.specifier)
    picked, lines, stack = {}, [], [Requirement(s) for s in root_specs]
    while stack:
        r = stack.pop()
        key = canon(r.name)
        meta = get(f"https://pypi.org/pypi/{key}/json")
        ver = pick_version(key, cons.get(key, []), meta)
        if picked.get(key) == ver:
            continue
        picked[key] = ver
        hashes = "".join(f" --hash=sha256:{f['digests']['sha256']}"
                         for f in meta["releases"][ver]
                         if f.get("digests", {}).get("sha256"))
        assert hashes, f"{key}=={ver}: no hashed files"
        lines.append(f"{key}=={ver}{hashes}")
        # dependencies OF THE PICKED VERSION (per-version metadata)
        vm = get(f"https://pypi.org/pypi/{key}/{ver}/json")
        for rd in dep_edges(vm, r.extras):
            d = Requirement(rd)
            if d.specifier:
                cons[canon(d.name)].append(d.specifier)
            stack.append(d)
    return sorted(lines)


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


def spec_of(name, pin):
    return f"{name}=={pin}" if pin else name


torch = torch_cpu_lines()

write(HERE / "requirements.txt",
      lock([spec_of(k, v) for k, v in TOOLCHAIN.items()]),
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

write(HERE / "requirements-build.txt", lock(BUILD),
      "hash-locked build toolchain: wheel building and pip-free installs "
      "(python -m build --no-isolation, python -m installer).")
