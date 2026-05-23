import os
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional

import torch
from dotenv import load_dotenv


@dataclass
class KronosConfig:
    kronos_model: str = "NeoQuasar/Kronos-base"
    kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-base"
    kronos_device: str = "auto"
    kronos_max_context: int = 512
    kronos_lookback: int = 400
    kronos_pred_len: int = 120
    kronos_temperature: float = 1.0
    kronos_top_p: float = 0.9
    kronos_sample_count: int = 1
    llm_api_base: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_model_id: str = "glm-4-flash"
    llm_api_key: str = ""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    qlib_provider_uri: str = "/home/zxh/qlib_data"
    max_batch_size: int = 50
    output_dir: str = "/home/zxh/quant_projects/kronos/outputs"

    _ENV_MAP: Dict[str, str] = field(
        default_factory=lambda: {
            "kronos_model": "KRONOS_MODEL",
            "kronos_tokenizer": "KRONOS_TOKENIZER",
            "kronos_device": "KRONOS_DEVICE",
            "kronos_max_context": "KRONOS_MAX_CONTEXT",
            "kronos_lookback": "KRONOS_LOOKBACK",
            "kronos_pred_len": "KRONOS_PRED_LEN",
            "kronos_temperature": "KRONOS_TEMPERATURE",
            "kronos_top_p": "KRONOS_TOP_P",
            "kronos_sample_count": "KRONOS_SAMPLE_COUNT",
            "llm_api_base": "LLM_API_BASE",
            "llm_model_id": "LLM_MODEL_ID",
            "llm_api_key": "LLM_API_KEY",
            "api_host": "API_HOST",
            "api_port": "API_PORT",
            "qlib_provider_uri": "QLIB_PROVIDER_URI",
            "max_batch_size": "MAX_BATCH_SIZE",
            "output_dir": "OUTPUT_DIR",
        },
        repr=False,
        compare=False,
    )

    _INT_FIELDS: frozenset = frozenset({
        "kronos_max_context", "kronos_lookback", "kronos_pred_len",
        "kronos_sample_count", "api_port", "max_batch_size",
    })

    _FLOAT_FIELDS: frozenset = frozenset({
        "kronos_temperature", "kronos_top_p",
    })

    MODEL_PRESETS: Dict[str, Dict] = field(
        default_factory=lambda: {
            "mini": {
                "kronos_model": "NeoQuasar/Kronos-mini",
                "kronos_tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
                "kronos_max_context": 2048,
                "kronos_lookback": 400,
                "kronos_pred_len": 120,
                "max_batch_size": 100,
            },
            "small": {
                "kronos_model": "NeoQuasar/Kronos-small",
                "kronos_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                "kronos_max_context": 512,
                "kronos_lookback": 400,
                "kronos_pred_len": 120,
                "max_batch_size": 50,
            },
            "base": {
                "kronos_model": "NeoQuasar/Kronos-base",
                "kronos_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                "kronos_max_context": 512,
                "kronos_lookback": 400,
                "kronos_pred_len": 120,
                "max_batch_size": 50,
            },
        },
        repr=False,
        compare=False,
    )

    @classmethod
    def load_config(cls, env_path: Optional[str] = None) -> "KronosConfig":
        if env_path is not None:
            load_dotenv(env_path, override=True)
        else:
            load_dotenv(override=True)

        instance = cls()
        env_map = instance._ENV_MAP

        for f in fields(instance):
            if f.name.startswith("_"):
                continue
            env_key = env_map.get(f.name)
            if env_key is None:
                continue
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            if f.name in instance._INT_FIELDS:
                setattr(instance, f.name, int(raw))
            elif f.name in instance._FLOAT_FIELDS:
                setattr(instance, f.name, float(raw))
            else:
                setattr(instance, f.name, raw)

        return instance

    def get_device(self) -> str:
        if self.kronos_device != "auto":
            return self.kronos_device
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def detect_model_variant(self) -> Optional[str]:
        model_lower = self.kronos_model.lower()
        for variant in self.MODEL_PRESETS:
            if variant in model_lower:
                return variant
        return None

    def apply_preset(self, variant: str) -> None:
        if variant not in self.MODEL_PRESETS:
            raise ValueError(
                f"Unknown variant '{variant}', available: "
                f"{list(self.MODEL_PRESETS.keys())}"
            )
        preset = self.MODEL_PRESETS[variant]
        for key, value in preset.items():
            setattr(self, key, value)

    def is_llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_api_key.strip())

    def validate(self) -> List[str]:
        errors: List[str] = []

        if self.kronos_max_context <= 0:
            errors.append("kronos_max_context must be positive")
        if self.kronos_lookback <= 0:
            errors.append("kronos_lookback must be positive")
        if self.kronos_pred_len <= 0:
            errors.append("kronos_pred_len must be positive")
        if self.kronos_lookback >= self.kronos_max_context:
            errors.append(
                "kronos_lookback must be less than kronos_max_context"
            )
        if not (0.0 < self.kronos_temperature):
            errors.append("kronos_temperature must be positive")
        if not (0.0 < self.kronos_top_p <= 1.0):
            errors.append("kronos_top_p must be in (0, 1]")
        if self.kronos_sample_count < 1:
            errors.append("kronos_sample_count must be at least 1")
        if self.api_port < 1 or self.api_port > 65535:
            errors.append("api_port must be between 1 and 65535")
        if self.max_batch_size < 1:
            errors.append("max_batch_size must be at least 1")
        if not self.output_dir:
            errors.append("output_dir must not be empty")

        resolved_device = self.get_device()
        if resolved_device.startswith("cuda"):
            if not torch.cuda.is_available():
                errors.append(
                    f"Device resolved to '{resolved_device}' but CUDA is not available"
                )

        return errors
