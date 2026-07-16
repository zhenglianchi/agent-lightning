import glob
import json
import logging
import os
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

import agentlightning as agl
from agentlightning.store.client_server import LightningStoreClient
from datasets import load_dataset

from adaptive_adapter import AdaptiveTraceToTriplet

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

_BOOL_TRUE_VALUES = frozenset({"true", "1", "yes"})

DEFAULT_MAPPING_FILE = "config_mapping.json"
DEFAULT_STORE_ADDRESS = "http://127.0.0.1:4748"


# ==================== 嵌套字典工具 ====================
def set_nested_value(dic: Dict, path: str, value: Any) -> None:
    keys = path.split('.')
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def get_nested_value(dic: Dict, path: str, default: Any = None) -> Any:
    keys = path.split('.')
    current = dic
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _find_env_vars(template: str) -> List[str]:
    return _ENV_VAR_PATTERN.findall(template)


# ==================== 模板展开 ====================
def expand_template(template: str) -> Optional[str]:
    if not isinstance(template, str):
        raise TypeError(f"template must be a string, got {type(template).__name__}")

    env_vars = _find_env_vars(template)
    for var in env_vars:
        val = os.getenv(var)
        if val is None or val == "":
            logger.debug("Environment variable '%s' is not set or empty, template expansion skipped", var)
            return None

    def repl(match):
        return os.getenv(match.group(1), '')

    return _ENV_VAR_PATTERN.sub(repl, template)


# ==================== 类型转换 ====================
def _convert_value(raw: str, typ: str) -> Any:
    if typ == "int":
        return int(raw)
    elif typ == "float":
        return float(raw)
    elif typ == "bool":
        return raw.lower() in _BOOL_TRUE_VALUES
    return raw


def _convert_with_fallback(raw: str, typ: str, default: Any) -> Any:
    try:
        return _convert_value(raw, typ)
    except (ValueError, TypeError) as e:
        logger.debug("Primary conversion failed for raw=%r type=%s: %s", raw, typ, e)
        if default is not None:
            try:
                if typ == "bool" and isinstance(default, bool):
                    return default
                return _convert_value(str(default), typ)
            except (ValueError, TypeError):
                logger.warning("Fallback conversion also failed for default=%r type=%s", default, typ)
        return raw


# ==================== 单条 mapping 解析 ====================
def _resolve_mapping_item(item: Dict[str, Any]) -> Optional[tuple]:
    path = item.get("path")
    if not path:
        logger.warning("Mapping item missing 'path', skipping: %s", item)
        return None

    template = item.get("template")
    default = item.get("default")
    typ = item.get("type", "str")

    if template is not None:
        expanded = expand_template(template)
        if expanded is None:
            if default is None:
                logger.debug("Template '%s' has missing env vars and no default, skipping path=%s", template, path)
                return None
            raw = str(default)
        else:
            raw = expanded
    else:
        if default is None:
            logger.debug("No template and no default for path=%s, skipping", path)
            return None
        raw = str(default)

    final_value = _convert_with_fallback(raw, typ, default)
    logger.debug("Resolved mapping: path=%s, value=%r (type=%s)", path, final_value, typ)
    return path, final_value


# ==================== 配置构建 ====================
def load_mapping_file(mapping_file: str) -> Dict[str, Any]:
    if not os.path.isfile(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Mapping file must contain a JSON object, got {type(data).__name__}")

    return data


def build_config_from_mapping(mapping_file: str = DEFAULT_MAPPING_FILE) -> Dict[str, Any]:
    logger.info("Building config from mapping file: %s", mapping_file)
    data = load_mapping_file(mapping_file)

    config = deepcopy(data.get("fixed", {}))
    mappings = data.get("mappings", [])

    if not isinstance(mappings, list):
        raise ValueError("'mappings' field must be a list")

    resolved_count = 0
    skipped_count = 0
    for item in mappings:
        result = _resolve_mapping_item(item)
        if result is None:
            skipped_count += 1
            continue
        path, value = result
        set_nested_value(config, path, value)
        resolved_count += 1

    logger.info("Config built: %d resolved, %d skipped", resolved_count, skipped_count)
    return config


class ConfigBuilder:
    def __init__(self, base_config: Dict[str, Any]):
        self._base_config = base_config

    def build(self) -> Dict[str, Any]:
        return deepcopy(self._base_config)

    def with_overrides(self, overrides: Dict[str, Any]) -> "ConfigBuilder":
        config = self.build()
        for path, value in overrides.items():
            set_nested_value(config, path, value)
        builder = ConfigBuilder(config)
        return builder


def config_train_npu(base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if base_config is None:
        base_config = build_config_from_mapping(DEFAULT_MAPPING_FILE)

    overrides = {
        "actor_rollout_ref.actor.use_torch_compile": False,
        "trainer.device": "npu",
    }
    return ConfigBuilder(base_config).with_overrides(overrides).build()


# ==================== Store 注入 ====================
def inject_store_into_algorithm(algorithm: Any, store_client: LightningStoreClient) -> None:
    if hasattr(algorithm, "set_store"):
        algorithm.set_store(store_client)
        logger.info("Store injected via set_store()")
    elif hasattr(algorithm, "store"):
        algorithm.store = store_client
        logger.info("Store injected via store attribute")
    else:
        logger.warning(
            "Cannot inject store into algorithm. "
            "Ensure store is accessible via other means (e.g. environment variables)."
        )


# ==================== 数据加载 ====================
def load_parquet_dataset(file_pattern: str, split: str = "train") -> List[Dict[str, Any]]:
    files = glob.glob(file_pattern)
    if not files:
        raise FileNotFoundError(f"No parquet files found matching pattern: {file_pattern}")

    logger.info("Loading parquet dataset from %d file(s) matching '%s'", len(files), file_pattern)
    dataset = load_dataset("parquet", data_files=files, split=split)
    data = [dict(row) for row in dataset]
    logger.info("Loaded %d record(s) from '%s'", len(data), file_pattern)
    return data


# ==================== Adapter 构建 ====================
def build_adaptive_adapter() -> AdaptiveTraceToTriplet:
    adapters = [
        agl.TracerTraceToTriplet(),
        agl.LlmProxyTraceToTriplet(),
    ]
    adapter = AdaptiveTraceToTriplet(
        adapters=adapters,
        use_cached_adapter=True,
    )
    logger.info("Built AdaptiveTraceToTriplet with %d adapter(s)", len(adapters))
    return adapter


# ==================== 主逻辑 ====================
def start(config: Dict[str, Any]) -> None:
    os.environ["AGL_CURRENT_ROLE"] = "algorithm"
    logger.info("Starting algorithm with config: %s", json.dumps(config, indent=2, ensure_ascii=False))

    store_address = os.environ.get("STORE_ADDRESS", DEFAULT_STORE_ADDRESS)
    logger.info("Connecting to store at %s", store_address)
    store_client = LightningStoreClient(server_address=store_address)

    algorithm = agl.VERL(config)
    inject_store_into_algorithm(algorithm, store_client)

    train_data = load_parquet_dataset(config["data"]["train_files"])
    val_data = load_parquet_dataset(config["data"]["val_files"])
    logger.info("Using %d training samples, %d validation samples", len(train_data), len(val_data))

    adapter = build_adaptive_adapter()
    algorithm.set_adapter(adapter)
    algorithm.run(train_dataset=train_data, val_dataset=val_data)  # type: ignore


def setup_default_env() -> None:
    defaults = {
        "AGL_CURRENT_ROLE": "algorithm",
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "AGL_MANAGED_STORE": "false",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def main() -> None:
    setup_default_env()
    config = config_train_npu()
    start(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    main()
