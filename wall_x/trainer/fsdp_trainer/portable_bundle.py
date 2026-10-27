"""Portable, run-root-scoped assets for training checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml

BUNDLE_VERSION = 1


def validate_checkpoint_root(save_path: str) -> None:
    """Create *save_path* and prove that files can be atomically written there."""
    root = Path(save_path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    fd, probe = tempfile.mkstemp(prefix=".wallx-write-probe-", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(b"wall-x\n")
            stream.flush()
            os.fsync(stream.fileno())
        committed = str(probe) + ".ok"
        os.replace(probe, committed)
        os.unlink(committed)
    finally:
        if os.path.exists(probe):
            os.unlink(probe)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src: str, dst: Path) -> None:
    source = Path(src).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Portable checkpoint dependency not found: {source}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dst)


def _copy_tree(src: str, dst: Path) -> None:
    source = Path(src).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"Portable checkpoint dependency not found: {source}")
    shutil.copytree(source, dst)


def _is_dino_processor_checkpoint(path: str) -> bool:
    config_path = Path(path).expanduser() / "processor_config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    return (
        config.get("image_processor", {}).get("image_processor_type")
        == "PluggableImageProcessor"
    )


def _save_processor_assets(processor: Any, source: Optional[str], dst: Path) -> None:
    """Save a processor without serializing wall-x's nested DINO processor."""
    if source and _is_dino_processor_checkpoint(os.fspath(source)):
        # DinoVideoProcessor owns a DinoImageProcessor instance. Hugging Face's
        # generic processor serialization leaves that instance in to_dict(),
        # which then fails JSON encoding. Preserve the files used by wall-x's
        # DINO reconstruction protocol and serialize only the mutable tokenizer.
        dst.mkdir(parents=True)
        for filename in ("processor_config.json", "chat_template.jinja"):
            source_file = Path(source).expanduser() / filename
            if source_file.is_file():
                _copy_file(os.fspath(source_file), dst / filename)

        tokenizer = getattr(processor, "tokenizer", None)
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if not callable(save_tokenizer):
            raise TypeError(
                "A DINO processor must expose tokenizer.save_pretrained() "
                "to create portable checkpoint assets"
            )
        save_tokenizer(dst)
        return

    save_processor = getattr(processor, "save_pretrained", None)
    if callable(save_processor):
        save_processor(dst)
    elif source:
        _copy_tree(source, dst)


def _write_yaml(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, allow_unicode=True, sort_keys=False)


def _asset_inventory(root: Path) -> list[dict]:
    inventory = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name == "manifest.json":
            continue
        inventory.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return inventory


def _fsync_tree(root: Path) -> None:
    for path in (p for p in root.rglob("*") if p.is_file()):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    directory_fd = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _dependency_signature(config: dict, *, include_pretrained: bool = False) -> str:
    model = copy.deepcopy(config.get("model", {}))
    data = copy.deepcopy(config.get("data", {}))
    path_keys = (
        "config_path",
        "processor_path",
        "pretrained_path",
        "action_tokenizer_path",
        "action_tokenizer_checkpoint_path",
        "action_tokenizer_config_dir",
    )
    paths = {key: model.pop(key, None) for key in path_keys}
    data.pop("dataset_config_path", None)
    norm = data.get("normalizer_config") or {}
    norm_path = norm.pop("customized_action_statistic_dof", None)

    def file_size(path: Optional[str]) -> Optional[int]:
        try:
            return Path(str(path)).stat().st_size if path else None
        except OSError:
            return None

    def tree_shape(path: Optional[str]) -> Optional[list[int]]:
        try:
            root = Path(str(path))
            sizes = sorted(p.stat().st_size for p in root.rglob("*") if p.is_file())
            return sizes
        except OSError:
            return None

    selected = {
        "model_type": config.get("model_type"),
        "task": config.get("task"),
        "model": model,
        "data": data,
        "model_config_size": file_size(paths["config_path"]),
        "action_tokenizer_size": file_size(paths["action_tokenizer_checkpoint_path"]),
        "action_config_sizes": tree_shape(paths["action_tokenizer_config_dir"]),
        "normalizer_size": file_size(norm_path),
        "include_pretrained": include_pretrained,
        "pretrained_sizes": (
            tree_shape(paths["pretrained_path"]) if include_pretrained else None
        ),
    }
    payload = json.dumps(selected, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _portable_data_config(effective: Optional[dict], source: Optional[str]) -> dict:
    if effective is not None:
        value = copy.deepcopy(effective)
        dataset = value.get("dataset") if isinstance(value, dict) else None
        if isinstance(dataset, dict):
            includes = dataset.pop("sources_include", None) or []
            if isinstance(includes, str):
                includes = [includes]
            merged_sources = list(dataset.get("sources") or [])
            merged_tasks = {}
            base = Path(source).expanduser().resolve().parent if source else Path.cwd()
            for include in includes:
                include_path = Path(str(include)).expanduser()
                if not include_path.is_absolute():
                    include_path = base / include_path
                with include_path.open("r", encoding="utf-8") as stream:
                    included = yaml.safe_load(stream) or {}
                included = included.get("dataset", included)
                merged_sources.extend(copy.deepcopy(included.get("sources") or []))
                merged_tasks.update(copy.deepcopy(included.get("tasks") or {}))
            if includes:
                dataset["sources"] = merged_sources
            task_balance = (dataset.get("sampler") or {}).get("task_balance")
            if isinstance(task_balance, dict) and merged_tasks:
                task_balance.setdefault("params", {}).setdefault(
                    "counts_per_task", merged_tasks
                )
        return value
    if not source:
        return {}
    with open(source, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _rewrite_data_asset_paths(data_config: dict, *, has_normalizer: bool) -> None:
    """Rewrite only known model-side paths; raw dataset locations stay external."""
    dataset = data_config.get("dataset", data_config)
    if not isinstance(dataset, dict):
        return
    epilogue = dataset.get("epilogue")
    processors = dataset.get("processors")
    if isinstance(processors, dict):
        epilogue = processors.get("epilogue", epilogue)
    if isinstance(epilogue, dict):
        params = epilogue.get("params")
        if isinstance(params, dict) and params.get("processor_path"):
            params["processor_path"] = "processor"
        if isinstance(params, dict) and has_normalizer and params.get("stats_file"):
            params["stats_file"] = "normalizer_keys.json"


def ensure_portable_assets(
    *,
    save_path: str,
    config: dict,
    processor: Any,
    effective_data_config: Optional[dict] = None,
    include_pretrained: bool = False,
) -> Path:
    """Create ``<save_path>/assets`` once, atomically, and then reuse it."""
    root = Path(save_path).expanduser().resolve()
    assets = root / "assets"
    signature = _dependency_signature(config, include_pretrained=include_pretrained)
    manifest_path = assets / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("bundle_version") != BUNDLE_VERSION:
            raise RuntimeError(f"Unsupported portable bundle at {assets}")
        if manifest.get("dependency_signature") != signature:
            raise RuntimeError(
                f"Portable assets at {assets} belong to different dependencies; "
                "use a new checkpoint.save_path"
            )
        for item in manifest.get("files", []):
            path = assets / item["path"]
            if not path.is_file() or path.stat().st_size != item["size"]:
                raise RuntimeError(f"Portable asset is missing or truncated: {path}")
        return assets
    if assets.exists():
        raise RuntimeError(f"Incomplete portable assets directory exists: {assets}")

    tmp = Path(tempfile.mkdtemp(prefix=".assets.tmp-", dir=root))
    try:
        model = config.get("model", {})
        data = config.get("data", {})
        norm = data.get("normalizer_config") or {}

        model_config_path = model.get("config_path")
        if model_config_path:
            _copy_file(model_config_path, tmp / "model_config.json")
        processor_path = model.get("processor_path")
        if processor is not None:
            _save_processor_assets(processor, processor_path, tmp / "processor")
        elif processor_path:
            _copy_tree(processor_path, tmp / "processor")
        if include_pretrained and model.get("pretrained_path"):
            _copy_tree(model["pretrained_path"], tmp / "pretrained")

        action_ckpt = model.get("action_tokenizer_checkpoint_path")
        if action_ckpt:
            _copy_file(action_ckpt, tmp / "action_tokenizer" / "latest.pth")
        action_configs = model.get("action_tokenizer_config_dir")
        if action_configs:
            _copy_tree(action_configs, tmp / "action_tokenizer" / "configs")

        norm_path = norm.get("customized_action_statistic_dof")
        if norm_path:
            _copy_file(norm_path, tmp / "normalizer_keys.json")

        if effective_data_config is not None or data.get("dataset_config_path"):
            data_config = _portable_data_config(
                effective_data_config, data.get("dataset_config_path")
            )
            _rewrite_data_asset_paths(data_config, has_normalizer=bool(norm_path))
            _write_yaml(tmp / "data_config.yml", data_config)

        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "model_type": config.get("model_type"),
            "dependency_signature": signature,
            "sources": {
                "model_config": model.get("config_path"),
                "processor": model.get("processor_path"),
                "action_tokenizer_checkpoint": action_ckpt,
                "action_tokenizer_configs": action_configs,
                "normalizer": norm_path,
                "data_config": data.get("dataset_config_path"),
            },
            "files": _asset_inventory(tmp),
        }
        (tmp / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _fsync_tree(tmp)
        os.rename(tmp, assets)
        root_fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
        return assets
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def portable_checkpoint_config(
    config: dict, *, include_pretrained: bool = False
) -> dict:
    """Return a checkpoint-local, relocation-safe copy of an effective config."""
    out = copy.deepcopy(config)
    out["portable_bundle_version"] = BUNDLE_VERSION
    model = out.setdefault("model", {})
    data = out.setdefault("data", {})
    checkpoint = out.setdefault("checkpoint", {})
    norm = data.get("normalizer_config")

    if model.get("config_path"):
        model["config_path"] = "../assets/model_config.json"
    if model.get("processor_path"):
        model["processor_path"] = "../assets/processor"
    if "pretrained_path" in model:
        model["pretrained_path"] = (
            "../assets/pretrained" if include_pretrained else None
        )
    if model.get("action_tokenizer_checkpoint_path"):
        model["action_tokenizer_checkpoint_path"] = (
            "../assets/action_tokenizer/latest.pth"
        )
    if model.get("action_tokenizer_config_dir"):
        model["action_tokenizer_config_dir"] = "../assets/action_tokenizer/configs"
    if data.get("dataset_config_path") or data.get("dataset_config_effective"):
        data["dataset_config_path"] = "../assets/data_config.yml"
    if isinstance(norm, dict) and norm.get("customized_action_statistic_dof"):
        norm["customized_action_statistic_dof"] = "../assets/normalizer_keys.json"
    checkpoint["save_path"] = ".."
    checkpoint["resume_from"] = "."
    checkpoint["resume_data_skip_steps"] = 0
    checkpoint["resume_data_skip_samples_per_rank"] = 0
    return out


def link_checkpoint_compat_assets(ckpt_path: str) -> None:
    """Expose small legacy checkpoint-local names via relative symlinks."""
    checkpoint = Path(ckpt_path)
    checkpoint.mkdir(parents=True, exist_ok=True)
    links = {}
    if (checkpoint.parent / "assets" / "model_config.json").is_file():
        links["config.json"] = "../assets/model_config.json"
    processor = checkpoint.parent / "assets" / "processor"
    if processor.is_dir():
        for source in processor.iterdir():
            if source.is_file():
                links[source.name] = f"../assets/processor/{source.name}"
    for name, target in links.items():
        path = checkpoint / name
        if os.path.lexists(path):
            continue
        path.symlink_to(target)
