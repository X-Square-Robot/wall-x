from typing import Dict, Optional, Union

import torch
from packaging import version
from transformers.modeling_utils import AttentionInterface
from transformers.utils import is_torch_xla_available, logging

ALL_ATTENTION_FUNCTIONS: AttentionInterface = AttentionInterface()
logger = logging.get_logger(__name__)

CUSTOM_ATTENTION_FUNCTIONS = [
    "flash_attention_2_ki",
    "flash_attention_2_triton",
    "flash_mask",
    "flash_mask_ki",
]
ATTENTION_TYPES_WITH_2D_MASK = [
    "flash_attention_2_ki",
    "flash_attention_2_triton",
    "sdpa",
]
ATTENTION_TYPES_WITH_FLASH_MASK = ["flash_mask", "flash_mask_ki"]


class AttentionsSelectorMixin:

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        """
        Automatically checks and dispatches to a default attention implementation. In order of priority:
            1. An implementation specified in `config._attn_implementation` (due for example to the argument attn_implementation="sdpa" in from_pretrained).
            2. DEPRECATED: if use_flash_attention_2 is set to `True` and `flash_attn` is available, flash attention. (`LlamaFlashAttention` for example)
            3. SDPA implementation, if available and supported by the model type. (`LlamaSdpaAttention` for example)
            4. The default model's implementation otherwise (`LlamaAttention` for example) .
        """
        # Here we use config._attn_implementation_internal to check whether the attention implementation was explicitly set by the user.
        # The property `PretrainedConfig._attn_implementation` is never `None`, for backward compatibility (always fall back on "eager").
        # The `hasattr` here is used as some Transformers tests for some reason do not call PretrainedConfig __init__ (e.g. test_no_super_init_config_and_model)
        requested_attn_implementation = None
        if (
            hasattr(config, "_attn_implementation_internal")
            and config._attn_implementation_internal is not None
        ):
            if (
                config._attn_implementation != "flash_attention_2"
                and use_flash_attention_2
            ):
                raise ValueError(
                    f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were used when loading the model, which are not compatible.'
                    ' We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.'
                )

            if (
                not isinstance(config._attn_implementation, dict)
                and config._attn_implementation
                not in ["eager"]
                + ALL_ATTENTION_FUNCTIONS.valid_keys()
                + CUSTOM_ATTENTION_FUNCTIONS
            ):
                message = f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. The only possible arguments are `attn_implementation="eager"` (manual attention implementation)'
                if cls._supports_flash_attn_2:
                    message += ', `"attn_implementation=flash_attention_2"` (implementation using flash attention 2)'
                if cls._supports_sdpa:
                    message += ', `"attn_implementation=sdpa"` (implementation using torch.nn.functional.scaled_dot_product_attention)'
                if cls._supports_flex_attn:
                    message += ', `"attn_implementation=flex_attention"` (implementation using torch\'s flex_attention)'
                raise ValueError(message + ".")

            # If a config is passed with a preset attn_implementation, we skip the automatic dispatch and use the user-provided config, with hard checks that the requested attention implementation is available.
            requested_attn_implementation = config._attn_implementation_internal

        if use_flash_attention_2:
            logger.warning_once(
                'The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.'
            )
            config._attn_implementation = "flash_attention_2"

        if config._attn_implementation == "flash_attention_2":
            cls._check_and_enable_flash_attn_2(
                config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                hard_check_only=False,
                check_device_map=check_device_map,
            )
        elif requested_attn_implementation == "flex_attention":
            config = cls._check_and_enable_flex_attn(config, hard_check_only=True)
        elif (
            requested_attn_implementation in [None, "sdpa"]
            and not is_torch_xla_available()
        ):
            # use_flash_attention_2 takes priority over SDPA, hence SDPA treated in this elif.
            config = cls._check_and_enable_sdpa(
                config,
                hard_check_only=(
                    False if requested_attn_implementation is None else True
                ),
            )

            if (
                torch.version.hip is not None
                and config._attn_implementation == "sdpa"
                and torch.cuda.device_count() > 1
                and version.parse(torch.__version__) < version.parse("2.4.1")
            ):
                logger.warning_once(
                    "Using the `SDPA` attention implementation on multi-gpu setup with ROCM may lead to performance issues due to the FA backend. Disabling it to use alternative backends."
                )
                torch.backends.cuda.enable_flash_sdp(False)
        elif requested_attn_implementation in ALL_ATTENTION_FUNCTIONS.valid_keys():
            config._attn_implementation = requested_attn_implementation
        elif isinstance(requested_attn_implementation, dict):
            config._attn_implementation = None
        elif config._attn_implementation in CUSTOM_ATTENTION_FUNCTIONS:
            pass
        else:
            config._attn_implementation = "eager"

        config._attn_implementation_autoset = True
        return config

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: Optional[str], is_init_check: bool = False
    ) -> str:
        assert (
            attn_implementation
            in ["eager", "flash_attention_2", "sdpa"] + CUSTOM_ATTENTION_FUNCTIONS
        )
        return attn_implementation
