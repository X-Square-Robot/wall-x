import inspect

from torch.optim import AdamW

from wall_x.config.hyperparams_config import LRGroupConfig

_OPTIMIZERS = {}


def register_optimizer(name, optimizer_fn):
    _OPTIMIZERS[name] = optimizer_fn


def get_optimizer(name, *args, **kwargs):
    if name not in _OPTIMIZERS:
        raise KeyError(f"Unknown optimizer '{name}'. Registered: {sorted(_OPTIMIZERS)}")
    return _OPTIMIZERS[name](*args, **kwargs)


def _group_weight_decay(opt_cfg):
    return getattr(opt_cfg, "weight_decay", None)


def resolve_lr_group_configs(opt_cfg, default_action_lr_keywords):
    """Return structured LR groups, with legacy action config fallback.

    ``optimizer.lr_groups`` is the preferred path. The legacy
    ``action_expert_learning_rate`` fields are still converted into a single
    action group so older configs keep working.
    """
    if opt_cfg.lr_groups:
        if (
            opt_cfg.action_expert_learning_rate is not None
            or opt_cfg.action_lr_keywords is not None
        ):
            raise ValueError(
                "Use either optimizer.lr_groups or legacy "
                "action_expert_learning_rate/action_lr_keywords, not both."
            )
        return opt_cfg.lr_groups

    if opt_cfg.action_expert_learning_rate is None:
        return []

    action_lr_keywords = (
        opt_cfg.action_lr_keywords
        if opt_cfg.action_lr_keywords is not None
        else default_action_lr_keywords
    )
    return [
        LRGroupConfig(
            name="action_lr_group",
            lr=opt_cfg.action_expert_learning_rate,
            include=action_lr_keywords,
            fail_on_empty=True,
        )
    ]


def uses_legacy_action_lr_groups(opt_cfg) -> bool:
    return not opt_cfg.lr_groups and opt_cfg.action_expert_learning_rate is not None


def _make_param_group(name, params, lr, opt_cfg):
    group = {
        "params": params,
        "lr": lr,
        "group_name": name,
    }
    weight_decay = _group_weight_decay(opt_cfg)
    if weight_decay is not None:
        group["weight_decay"] = weight_decay
    return group


def _validate_lr_group(group: LRGroupConfig, *, base_group_name: str):
    if not group.name:
        raise ValueError("optimizer.lr_groups entries must have a non-empty name")
    if group.name == base_group_name:
        raise ValueError(
            f"optimizer.lr_groups name {group.name!r} is reserved for the base group"
        )
    if "/" in group.name:
        raise ValueError(
            f"optimizer.lr_groups name {group.name!r} must not contain '/'. "
            "DMuon appends '/muon' and '/adamw' to group names."
        )
    if not group.include:
        raise ValueError(
            f"optimizer.lr_groups.{group.name} must define at least one include keyword"
        )


def build_lr_param_groups(model, opt_cfg, lr_groups, *, base_group_name="base"):
    """Split trainable params into named LR groups plus a base group.

    ``lr_groups`` is a list of :class:`LRGroupConfig`. Each group matches
    parameter names by substring. A parameter may match at most one explicit
    group; unmatched trainable parameters remain in the ``base`` group using
    ``opt_cfg.learning_rate``.

    Returns a list of torch.optim-compatible param_group dicts. The
    ``group_name`` key is non-standard but preserved by torch.optim via
    ``setdefault`` in ``add_param_group`` and is consumed downstream for
    per-group lr logging.

    For AdamW / native Muon, each returned group includes ``weight_decay``.
    For DMuon, weight decay is split between Muon and AdamW defaults, so the
    groups only carry lr and metadata; DMuon applies its own per-route defaults.
    The caller is expected to gate on ``opt_cfg.optimizer_type`` before calling
    this.
    """
    if not lr_groups:
        return None

    names = [group.name for group in lr_groups]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(
            f"optimizer.lr_groups contains duplicate names: {duplicate_names}"
        )

    for group in lr_groups:
        _validate_lr_group(group, base_group_name=base_group_name)

    base_params = []
    grouped_params = {group.name: [] for group in lr_groups}
    ambiguous = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matches = [
            group.name
            for group in lr_groups
            if any(keyword in name for keyword in group.include)
        ]
        if len(matches) > 1:
            ambiguous.append((name, matches))
            continue
        if matches:
            grouped_params[matches[0]].append(param)
        else:
            base_params.append(param)

    if ambiguous:
        examples = ", ".join(f"{name} -> {matches}" for name, matches in ambiguous[:10])
        raise ValueError(
            "Some parameters match multiple optimizer.lr_groups. Make group "
            f"include patterns disjoint. Examples: {examples}"
        )

    if opt_cfg.train_action_expert_only:
        assert len(base_params) == 0, (
            f"Expected 0 base_params after pre-wrap freeze, got {len(base_params)}. "
            "Ensure base params are frozen before building the optimizer."
        )

    param_groups = []
    if len(base_params) > 0:
        param_groups.append(
            _make_param_group(
                base_group_name, base_params, opt_cfg.learning_rate, opt_cfg
            )
        )

    for group in lr_groups:
        params = grouped_params[group.name]
        if len(params) == 0:
            if group.fail_on_empty:
                raise ValueError(
                    f"No params found for optimizer.lr_groups.{group.name}. "
                    f"Please check include={group.include!r}."
                )
            continue
        param_groups.append(_make_param_group(group.name, params, group.lr, opt_cfg))

    return param_groups


def build_action_expert_param_groups(model, opt_cfg, action_lr_keywords):
    """Compatibility wrapper for the legacy action-expert LR config."""
    return build_lr_param_groups(
        model,
        opt_cfg,
        [
            LRGroupConfig(
                name="action_lr_group",
                lr=opt_cfg.action_expert_learning_rate,
                include=action_lr_keywords,
                fail_on_empty=True,
            )
        ],
        base_group_name="base_lr_group",
    )


def get_adamw_optimizer(model, *, opt_cfg, param_groups=None):
    """Build AdamW from AdamWConfig."""
    from wall_x.config.hyperparams_config import AdamWConfig

    if not isinstance(opt_cfg, AdamWConfig):
        raise TypeError(
            f"get_adamw_optimizer expects AdamWConfig, got {type(opt_cfg).__name__}"
        )

    if param_groups is None:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        # Per-group lr / weight_decay are preserved by torch.optim (via
        # setdefault in add_param_group), so top-level values act only as
        # defaults. Extra keys like ``group_name`` are kept in-place and used
        # downstream for per-group lr logging.
        params = param_groups

    kw = {
        "lr": opt_cfg.learning_rate,
        "weight_decay": opt_cfg.weight_decay,
        "betas": tuple(opt_cfg.betas),
        "eps": opt_cfg.eps,
    }
    sig_params = inspect.signature(AdamW.__init__).parameters
    if "foreach" in sig_params and opt_cfg.foreach is not None:
        kw["foreach"] = opt_cfg.foreach
    if "fused" in sig_params:
        kw["fused"] = opt_cfg.fused
    return AdamW(params, **kw)


register_optimizer("adamw", get_adamw_optimizer)
