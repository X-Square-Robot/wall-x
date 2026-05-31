"""DMuon optimizer builder."""

import inspect
import logging

from ..utils import register_optimizer

_logger = logging.getLogger(__name__)


def _emit(log_fn, message, *args, level=logging.INFO):
    if args:
        message = message % args
    if log_fn is not None:
        log_fn(message, level=level)
    else:
        _logger.log(level, message)


def _is_rank0():
    try:
        import torch.distributed as dist

        return (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        )
    except Exception:
        return True


def _build_ns_backend(dmuon, opt_cfg):
    coefficients = getattr(opt_cfg, "ns_coefficients", "default")
    if coefficients in (None, "default"):
        return opt_cfg.ns_backend

    if coefficients != "wallx_muon":
        raise ValueError(
            "Unsupported DMuon ns_coefficients="
            f"{coefficients!r}. Supported: 'default', 'wallx_muon'."
        )
    if opt_cfg.ns_backend != "direct":
        raise ValueError(
            "ns_coefficients='wallx_muon' is intended to match Wall-X's "
            "direct-space Muon implementation. Set ns_backend='direct'."
        )

    wallx_coefficients = [[3.4445, -4.7750, 2.0315] for _ in range(opt_cfg.ns_steps)]
    return dmuon.NewtonSchulz(
        backend="direct",
        coefficients=wallx_coefficients,
    )


def get_dmuon_optimizer(model, *, opt_cfg, param_groups=None, log_fn=None):
    """Build dmuon.Muon from a DMuonConfig.

    When ``param_groups`` is provided, Wall-X expects DMuon to preserve the
    PyTorch optimizer group semantics and then split each user group into
    dedicated/Muon and non-dedicated/AdamW subgroups internally.
    """
    from wall_x.config.hyperparams_config import DMuonConfig

    if not isinstance(opt_cfg, DMuonConfig):
        raise TypeError(
            f"get_dmuon_optimizer expects DMuonConfig, got {type(opt_cfg).__name__}"
        )
    import dmuon

    muon_signature = inspect.signature(dmuon.Muon)
    supports_param_groups = "param_groups" in muon_signature.parameters
    if param_groups is not None and not supports_param_groups:
        raise RuntimeError(
            "Wall-X built optimizer param_groups for DMuon, but the installed "
            "dmuon.Muon does not accept a param_groups= argument. Please update "
            "DMuon to the param-group-aware implementation before enabling "
            "action_expert_learning_rate with optimizer_type='dmuon'."
        )

    ns_backend = _build_ns_backend(dmuon, opt_cfg)

    _emit(
        log_fn,
        "DMuon: Muon lr=%s momentum=%s ns_steps=%s; "
        "AdamW lr=%s betas=%s wd=%s; "
        "ns_backend=%s ns_coefficients=%s nesterov=%s",
        opt_cfg.muon_lr,
        opt_cfg.momentum,
        opt_cfg.ns_steps,
        opt_cfg.adamw_lr,
        opt_cfg.adamw_betas,
        opt_cfg.adamw_weight_decay,
        opt_cfg.ns_backend,
        opt_cfg.ns_coefficients,
        opt_cfg.nesterov,
    )
    if param_groups is not None:
        _emit(
            log_fn,
            "DMuon param_groups enabled: %s",
            [
                {
                    "group_name": group.get("group_name", f"group_{idx}"),
                    "lr": group.get("lr"),
                    "num_params": len(group.get("params", [])),
                }
                for idx, group in enumerate(param_groups)
            ],
        )

    kwargs = {}
    if param_groups is not None:
        kwargs["param_groups"] = param_groups

    optimizer = dmuon.Muon(
        model,
        lr=opt_cfg.muon_lr,
        momentum=opt_cfg.momentum,
        weight_decay=opt_cfg.muon_weight_decay,
        ns_steps=opt_cfg.ns_steps,
        adamw_lr=opt_cfg.adamw_lr,
        adamw_betas=tuple(opt_cfg.adamw_betas),
        adamw_weight_decay=opt_cfg.adamw_weight_decay,
        adamw_eps=opt_cfg.adamw_eps,
        ns_backend=ns_backend,
        nesterov=opt_cfg.nesterov,
        **kwargs,
    )

    if param_groups is not None and _is_rank0():
        summarize = getattr(dmuon, "summarize_param_groups", None)
        format_summary = getattr(dmuon, "format_param_group_summary", None)
        if summarize is None or format_summary is None:
            _emit(
                log_fn,
                "DMuon param_groups are enabled, but the installed DMuon package "
                "does not expose param-group diagnostics. Update DMuon if you need "
                "startup verification of the Muon/AdamW subgroup split.",
                level=logging.WARNING,
            )
        else:
            try:
                summary = summarize(model, optimizer, max_rows=80)
                _emit(log_fn, "%s", format_summary(summary))
            except Exception as exc:
                _logger.exception("Failed to summarize DMuon param_groups")
                _emit(
                    log_fn,
                    "Failed to summarize DMuon param_groups: %s",
                    exc,
                    level=logging.WARNING,
                )

    return optimizer


register_optimizer("dmuon", get_dmuon_optimizer)
