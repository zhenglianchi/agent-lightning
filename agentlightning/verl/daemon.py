# Copyright (c) Microsoft. All rights reserved.

import asyncio
import json
import os
import random
import socket
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
import requests
import torch
from flask import Flask, Response, abort, request
from tensordict import TensorDict
from verl import DataProto
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
import transfer_queue as tq
from transfer_queue import KVBatchMeta

from agentlightning import LLM, AgentLightningServer, NamedResources, RolloutLegacy
from agentlightning.adapter.triplet import TracerTraceToTriplet, TraceToTripletBase
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.store.base import LightningStore
from agentlightning.types import EnqueueRolloutRequest, Rollout, RolloutConfig, Task
from agentlightning.bench_detail import bench_log, now, rss_mb

__all__ = [
    "AgentModeDaemon",
    "get_left_padded_ids_and_attention_mask",
    "get_right_padded_ids_and_attention_mask",
]


def ids_startswith(
    full_ids: List[int], prefix_ids: List[int], tokenizer: Any, debug: bool = False
) -> Tuple[bool, Tuple[bool, bool, bool]]:
    is_prefix: bool
    template_mismatch, retoken_mismatch, others_mismatch = False, False, False
    if full_ids[: len(prefix_ids)] == prefix_ids:
        is_prefix = True
        return True, (template_mismatch, retoken_mismatch, others_mismatch)
    else:
        is_prefix = False

    if not debug:
        return is_prefix, (template_mismatch, retoken_mismatch, others_mismatch)

    def _special_token_sequence(ids: List[int]) -> List[int]:
        return [id for id in ids if id in tokenizer.all_special_ids]

    def _none_special_token_sequence(ids: List[int]) -> List[int]:
        return [id for id in ids if id not in tokenizer.all_special_ids]

    # First, handle special tokens
    full_special_ids = _special_token_sequence(full_ids)
    prefix_special_ids = _special_token_sequence(prefix_ids)
    if sum(1 for a, b in zip(full_special_ids, prefix_special_ids) if a != b) > 0:
        template_mismatch = True

    # Next, handle string content
    full_content_ids = _none_special_token_sequence(full_ids)
    prefix_content_ids = _none_special_token_sequence(prefix_ids)
    full_string = tokenizer.decode(full_ids, skip_special_tokens=True)
    prefix_string = tokenizer.decode(prefix_ids, skip_special_tokens=True)
    if full_content_ids[: len(prefix_content_ids)] != prefix_content_ids and full_string.startswith(prefix_string):
        retoken_mismatch = True
    elif full_content_ids[: len(prefix_content_ids)] != prefix_content_ids and not full_string.startswith(
        prefix_string
    ):
        others_mismatch = True
    return is_prefix, (template_mismatch, retoken_mismatch, others_mismatch)


def log_mismatch_detail(
    diagnostic: Tuple[bool, bool, bool],
    full_ids: List[int],
    prefix_ids: List[int],
    global_steps: int,
    rollout_id: str,
    turn_id: int,
    log_dir: str | None = None,
):
    if log_dir is None:
        return
    os.makedirs(log_dir, exist_ok=True)
    template_mismatch, retoken_mismatch, others_mismatch = diagnostic
    if template_mismatch:
        with open(os.path.join(log_dir, "template_mismatch.log"), "a+") as f:
            print(
                "-" * 10 + f" Global Steps: {global_steps}, Rollout ID: {rollout_id}, Turn ID: {turn_id} " + "-" * 10,
                file=f,
            )
            print(full_ids, file=f)
            print(prefix_ids, file=f)
    if retoken_mismatch:
        with open(os.path.join(log_dir, "retoken_mismatch.log"), "a+") as f:
            print(
                "-" * 10 + f" Global Steps: {global_steps}, Rollout ID: {rollout_id}, Turn ID: {turn_id} " + "-" * 10,
                file=f,
            )
            print(full_ids, file=f)
            print(prefix_ids, file=f)
    if others_mismatch:
        with open(os.path.join(log_dir, "others_mismatch.log"), "a+") as f:
            print(
                "-" * 10 + f" Global Steps: {global_steps}, Rollout ID: {rollout_id}, Turn ID: {turn_id} " + "-" * 10,
                file=f,
            )
            print(full_ids, file=f)
            print(prefix_ids, file=f)


def get_left_padded_ids_and_attention_mask(
    ids: List[int], max_length: int, pad_token_id: int
) -> Tuple[List[int], List[int]]:
    """
    Left-pad (or truncate) a sequence of token IDs to a fixed length,
    and build the corresponding attention mask.

    Args:
        ids:             the original list of token IDs.
        max_length:      desired total length after padding/truncation.
        pad_token_id:    ID to use for padding.

    Returns:
        padded_ids (any):      list of length == max_length.
        attention_mask (any):  list of same length: 1 for non-pad tokens, 0 for pads.
    """
    seq_len = len(ids)

    if seq_len >= max_length:
        # too long → truncate from the left, keep the last max_length tokens
        trimmed = ids[-max_length:]
        attention_mask = [1] * max_length
        return trimmed, attention_mask

    # too short → pad on the left
    pad_len = max_length - seq_len
    padded_ids = [pad_token_id] * pad_len + ids
    attention_mask = [0] * pad_len + [1] * seq_len
    return padded_ids, attention_mask


def get_right_padded_ids_and_attention_mask(
    ids: List[int], max_length: int, pad_token_id: int
) -> Tuple[List[int], List[int]]:
    """
    Right-pad (or truncate) a sequence of token IDs to a fixed length,
    and build the corresponding attention mask.

    Args:
        ids:            the original list of token IDs.
        max_length:     desired total length after padding/truncation.
        pad_token_id:   ID to use for padding.

    Returns:
        padded_ids (any):     list of length == max_length.
        attention_mask (any): list of same length: 1 for non-pad tokens, 0 for pads.
    """
    seq_len = len(ids)

    if seq_len >= max_length:
        # too long → truncate to the first max_length tokens
        trimmed = ids[:max_length]
        attention_mask = [1] * max_length
        return trimmed, attention_mask

    # too short → pad on the right
    pad_len = max_length - seq_len
    padded_ids = ids + [pad_token_id] * pad_len
    attention_mask = [1] * seq_len + [0] * pad_len
    return padded_ids, attention_mask


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _to_native(obj: Any) -> Any:
    """Convert data retrieved from Parquet to data usable in AGL server."""
    # 1) Arrays -> list (then recurse)
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())

    # 2) NumPy scalar types -> Python scalars
    if isinstance(obj, np.generic):
        return _to_native(obj.item())

    # 3) Dict-like -> dict
    if isinstance(obj, Mapping):
        return {_to_native(k): _to_native(v) for k, v in obj.items()}  # type: ignore

    # 4) Lists/Tuples/Sets -> list
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(x) for x in obj]  # type: ignore

    # 5) Anything else: leave as-is
    return obj


class AgentModeDaemon:
    """
    AgentModeDaemon using the AgentLightningServer SDK.

    This class manages the server lifecycle, task queueing, and results
    retrieval, while also running a proxy server for LLM requests. It maintains
    the original interface for compatibility with the RayPPOTrainer.
    """

    def __init__(
        self,
        port: Optional[int],
        train_rollout_n: int,
        train_information: Dict[str, Any],
        tokenizer: Any,
        mini_batch_size: int,
        pad_token_id: int,
        reward_fillna_value: float = 0.0,
        llm_timeout_seconds: float = 1200.0,
        mode: Literal["v0", "v1"] = "v1",
        llm_proxy: LLMProxy | None = None,
        store: LightningStore | None = None,
        adapter: TraceToTripletBase | None = None,
        processor: Any = None,
        image_base_dir: Optional[str] = None,
        trace_aggregator: Dict[str, Any] = {"level": "transition"},
        max_prompt_length: Optional[int] = None,
        max_response_length: Optional[int] = None,
    ):
        self.mode = mode
        self.llm_timeout_seconds = llm_timeout_seconds

        # Server and Task Configuration
        if mode == "v0":
            assert port is not None
            self.server_port = port
            self.server = AgentLightningServer(
                host="0.0.0.0", port=self.server_port, task_timeout_seconds=self.llm_timeout_seconds
            )
            self.proxy_port = _find_available_port()  # Run proxy on a different port
        else:
            assert store is not None
            self.store = store
            if llm_proxy is None:
                resolved_adapter = adapter if adapter is not None else TracerTraceToTriplet()
                self.llm_proxy = LLMProxy(
                    port=_find_available_port(),
                    model_list=[],
                    store=store,
                    adapter=resolved_adapter,
                    max_prompt_length=max_prompt_length,
                    max_response_length=max_response_length,
                )
            else:
                # Reuse the existing LLM proxy (probably configured by user)
                self.llm_proxy = llm_proxy
            self.adapter = resolved_adapter if llm_proxy is None else (adapter if adapter is not None else TracerTraceToTriplet())
            self._internal_loop: Optional[asyncio.AbstractEventLoop] = None
            self._internal_loop_thread = threading.Thread(target=self._internal_loop_runner, daemon=True)
            self._internal_loop_thread.start()

        # Training and Data Configuration
        self.train_rollout_n = train_rollout_n
        self.train_information = train_information
        self.mini_batch_size = mini_batch_size
        self.pad_token_id = pad_token_id
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fillna_value = reward_fillna_value
        self.image_base_dir = image_base_dir
        self.trace_aggregator = trace_aggregator
        self._max_prompt_length = max_prompt_length
        self._max_response_length = max_response_length

        # Check if model requires multimodal position_ids (e.g., Qwen2-VL)
        self._use_mrope = self._is_mrope_model()

        # Internal State
        self.backend_llm_server_addresses: List[str] = []
        self._total_tasks_queued = 0
        self._completed_rollouts_v0: Dict[str, RolloutLegacy] = {}
        self._task_id_to_original_sample: Dict[str, Dict[str, Any]] = {}
        self._server_thread: Optional[threading.Thread] = None
        self._proxy_thread: Optional[threading.Thread] = None
        self.is_train = True

    def _internal_loop_runner(self):
        """Run the internal loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._internal_loop = loop
        loop.run_forever()
        loop.close()

    # Multimodal utilities for M-RoPE position embeddings

    def _is_mrope_model(self) -> bool:
        """Check if processor requires M-RoPE position embeddings."""
        if self.processor is None or not hasattr(self.processor, "image_processor"):
            return False
        name = self.processor.image_processor.__class__.__name__
        return "Qwen2VLImageProcessor" in name or "Qwen3VLImageProcessor" in name

    def _resolve_image_path(self, path: str) -> str:
        """Resolve relative image path with base directory."""
        import os

        if os.path.isabs(path):
            return path
        if self.image_base_dir is None:
            raise ValueError(f"Relative path '{path}' requires 'image_base_dir' to be set.")
        return os.path.join(self.image_base_dir, path)

    def _get_image_grid_thw(self, image_urls: List[str]) -> Optional[torch.Tensor]:
        """Compute image_grid_thw from image URLs for M-RoPE computation.

        Args:
            image_urls: List of image URLs extracted from triplet prompt payload.
                URLs can be http(s):// URLs or file:// URIs, or data: URIs.
        """
        from PIL import Image
        from verl.utils.dataset.vision_utils import process_image  # pyright: ignore[reportUnknownVariableType]

        if self.processor is None or not image_urls:
            return None

        def to_image_uri(url: str) -> str:
            # Already a proper URI (http, https, file, data)
            if url.startswith(("http://", "https://", "file://", "data:")):
                return url
            # Treat as a file path that needs resolution
            resolved = self._resolve_image_path(url)
            return f"file://{resolved}"

        images: List[Image.Image] = [process_image({"image": to_image_uri(url)}) for url in image_urls]
        model_inputs = self.processor(text=["dummy"], images=images, return_tensors="pt")
        return model_inputs.get("image_grid_thw")

    def _compute_mrope_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute 4D position_ids for M-RoPE models."""
        from typing import Callable

        get_rope_index: Callable[..., torch.Tensor]
        if "Qwen3VL" in self.processor.__class__.__name__:
            from verl.models.transformers.qwen3_vl import get_rope_index  # pyright: ignore[reportUnknownVariableType]
        else:
            from verl.models.transformers.qwen2_vl import get_rope_index  # pyright: ignore[reportUnknownVariableType]

        vision_pos = get_rope_index(
            self.processor, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask
        )

        valid_mask = attention_mask.bool()
        text_pos = torch.zeros((1, len(input_ids)), dtype=torch.long, device=input_ids.device)
        text_pos[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids.device)

        return torch.cat([text_pos, vision_pos], dim=0)

    def _start_proxy_server_v0(self):
        """
        Initializes and runs a Flask-based proxy server in a separate thread.
        This proxy load-balances requests to the actual backend LLM servers.
        """
        app = Flask(__name__)

        num_requests = 0
        last_request_time = 0

        @app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
        def proxy(path: str):  # type: ignore
            if not self.backend_llm_server_addresses:
                abort(503, description="No backend LLM servers available.")

            # Randomly choose a backend server for load balancing
            target_server = random.choice(self.backend_llm_server_addresses)
            target_url = f"http://{target_server}/v1/{path}"

            # Copy client request headers, removing the Host header
            headers = {key: value for key, value in request.headers if key.lower() != "host"}

            # Log the request for debugging
            nonlocal num_requests, last_request_time
            current_time = time.time()
            num_requests += 1
            if current_time - last_request_time > 60 or num_requests == 1 or num_requests % 100 == 0:
                print(f"Proxying {request.method} request to {target_server}. Request data: {request.get_data()}")
            last_request_time = current_time

            try:
                # Forward the request to the target backend
                resp = requests.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    params=request.args,  # type: ignore
                    data=request.get_data(),
                    cookies=request.cookies,
                    allow_redirects=False,
                    timeout=self.llm_timeout_seconds,
                )
                # Filter out hop-by-hop headers before returning the response
                excluded_headers = [
                    "content-encoding",
                    "content-length",
                    "transfer-encoding",
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "upgrade",
                ]
                response_headers = [
                    (name, value) for name, value in resp.raw.headers.items() if name.lower() not in excluded_headers
                ]
                if resp.status_code == 200:
                    # NOTE: from Zhiyuan's code.
                    # https://github.com/hzy46/verl_agent_mode/blob/2db65ea9858f645a914120357412a7540f8bd82d/verl/trainer/ppo/ray_trainer.py#L692-L711
                    # request_json = json.loads(request.get_data().decode("utf-8"))
                    response_json = json.loads(resp.content.decode("utf-8"))
                    # response_message = ChatCompletion(**response_json).choices[0].message.model_dump(exclude_unset=True, exclude_none=True)
                    # tool_schemas = request_json.get("tools", None)
                    # prompt_ids = self.tokenizer.apply_chat_template(request_json["messages"], tools=tool_schemas, add_generation_prompt=True, tokenize=True)
                    # full_ids = self.tokenizer.apply_chat_template(request_json["messages"] + [response_message], tools=tool_schemas, add_generation_prompt=False, tokenize=True)
                    # TBD: response_ids sometimes ends with "<eos_id>\n", shall we keep the extra "\n"?
                    # sometimes it has some differences with the hacky method in the end, but this should align with ToolCompletionCallback
                    # response_ids = full_ids[len(prompt_ids):]

                    # NOTE (yuge): They are different. Don't know why.
                    # assert response_json['prompt_token_ids'] == prompt_ids
                    # patched_response_ids = response_json['response_token_ids'][0]
                    # assert patched_response_ids == response_ids[:len(patched_response_ids)], f"{patched_response_ids} != {response_ids[:len(patched_response_ids)]}"
                    # response_json['prompt_token_ids'] = prompt_ids
                    # response_json['response_token_ids'] = [response_ids]
                    replaced_return_content = json.dumps(response_json).encode("utf-8")
                    return Response(replaced_return_content, status=resp.status_code, headers=response_headers)
                return Response(resp.content, resp.status_code, response_headers)
            except requests.exceptions.RequestException as e:
                abort(500, description=f"Error proxying request: {e}")

        def run_app():
            app.run(host="0.0.0.0", port=self.proxy_port, threaded=True, debug=False)

        self._proxy_thread = threading.Thread(target=run_app, daemon=True)
        self._proxy_thread.start()
        print(f"Proxy server running on port {self.proxy_port}")

    async def _update_proxy_server_v1(self):
        model_name = self.train_information.get("model")
        if not model_name:
            raise ValueError("Model name is not set.")
        self.llm_proxy.update_model_list(
            [
                ModelConfig(
                    {
                        "model_name": model_name,
                        "litellm_params": {
                            "model": "hosted_vllm/" + model_name,
                            "api_base": f"http://{address}/v1/",
                        },
                    }
                )
                for address in self.backend_llm_server_addresses
            ],
        )

        await self.llm_proxy.restart()

    def start(self):
        """Starts the main AgentLightningServer and the proxy server."""

        if self.mode == "v0":

            def run_server():
                """Run the AgentLightningServer in a separate thread."""
                asyncio.run(self.server.run_forever())

            self._server_thread = threading.Thread(target=run_server, daemon=True)
            self._server_thread.start()

            # Wait for the server's internal startup event to be set.
            print("Waiting for AgentLightningServer to start...")
            is_ready = self.server.startup_event.wait(timeout=20.0)  # Wait up to 20s
            if not is_ready:
                raise RuntimeError("AgentLightningServer failed to start within the timeout period.")

            print(f"AgentLightningServer control plane running on port {self.server_port}")

            self._start_proxy_server_v0()
        else:
            # Agent lightning server is no longer needed;
            # Start proxy server in _async_set_up
            pass

    async def _async_set_up(self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True, global_steps=None):
        """Async helper to set up data and resources on the server."""
        self.clear_data_and_server()
        if server_addresses != self.backend_llm_server_addresses:
            self.backend_llm_server_addresses = server_addresses
            if self.mode == "v1" and not self.llm_proxy.is_running():
                await self._update_proxy_server_v1()
        self.is_train = is_train

        # 1. Update resources on the server for clients to use
        if self.mode == "v0":
            llm_resource = LLM(
                endpoint=f"http://127.0.0.1:{self.proxy_port}/v1",
                model=self.train_information.get("model", "default-model"),
                sampling_parameters={
                    "temperature": self.train_information.get("temperature", 0.7 if is_train else 0.0),
                    "seed": 42,
                },
            )
        else:
            llm_resource = self.llm_proxy.as_resource(
                sampling_parameters={
                    "temperature": self.train_information.get("temperature", 0.7 if is_train else 0.0),
                    "seed": 42,
                },
            )

        resources: NamedResources = {"main_llm": llm_resource}

        if self.mode == "v0":
            resources_id = await self.server.update_resources(resources)
        else:
            resources_update = await self.store.add_resources(resources)
            resources_id = resources_update.resources_id

        # 2. Queue tasks for agents to process
        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        rollouts_per_sample = self.train_rollout_n if is_train else 1

        enqueue_rollout_requests: List[EnqueueRolloutRequest] = []
        data_id_to_original_sample: Dict[str, Dict[str, Any]] = {}

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = data_id
            data_id_to_original_sample[data_id] = original_sample

            # For training, each sample is rolled out multiple times
            # Data ID is different from Rollout ID, as one data can have multiple rollouts.
            for _ in range(rollouts_per_sample):
                task_metadata = {"data_id": data_id, "is_train": is_train}
                if global_steps is not None:
                    task_metadata["global_steps"] = global_steps
                if self._max_prompt_length is not None:
                    task_metadata["max_prompt_length"] = self._max_prompt_length
                if self._max_response_length is not None:
                    task_metadata["max_response_length"] = self._max_response_length
                if self.mode == "v0":
                    # Queue immediately
                    rollout_id = await self.server.queue_task(
                        sample=_to_native(original_sample),
                        mode="train" if is_train else "val",
                        resources_id=resources_id,
                        metadata=task_metadata,
                    )

                    # Store original sample data to reconstruct batch information later
                    self._task_id_to_original_sample[rollout_id] = original_sample
                    self._total_tasks_queued += 1
                else:
                    # Collect tasks to enqueue in batch and queue them later
                    enqueue_rollout_requests.append(
                        EnqueueRolloutRequest(
                            input=_to_native(original_sample),
                            mode="train" if is_train else "val",
                            resources_id=resources_id,
                            config=RolloutConfig(
                                unresponsive_seconds=self.llm_timeout_seconds,
                                timeout_seconds=self.llm_timeout_seconds,
                            ),
                            metadata=task_metadata,
                        )
                    )

        if self.mode == "v1":
            # Enqueue all the tasks in a single batch
            rollouts = await self.store.enqueue_many_rollouts(enqueue_rollout_requests)
            self._task_id_to_original_sample.update(
                {
                    # Recover the original data and store it for later use.
                    rollout.rollout_id: data_id_to_original_sample[cast(Dict[str, Any], rollout.metadata)["data_id"]]
                    for rollout in rollouts
                }
            )
            self._total_tasks_queued += len(rollouts)

        if global_steps is not None and is_train:
            import transfer_queue as tq
            for rollout_id in self._task_id_to_original_sample:
                tq.kv_put(
                    key=rollout_id,
                    partition_id="train",
                    tag={"global_steps": global_steps, "status": "running"},
                )

    def set_up_data_and_server(self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True, global_steps: int = None):
        """Synchronous wrapper for setting up data and server resources."""
        coro = self._async_set_up(data, server_addresses, is_train, global_steps=global_steps)

        if self.mode == "v0":
            if not self.server.loop or not self.server.startup_event.is_set():
                raise RuntimeError("Server is not running or ready.")

            future = asyncio.run_coroutine_threadsafe(coro, self.server.loop)

        else:
            if self._internal_loop is None:
                raise RuntimeError("Internal loop is not running.")
            future = asyncio.run_coroutine_threadsafe(coro, self._internal_loop)
        try:
            future.result(timeout=300)  # Wait for completion with a timeout
        except Exception as e:
            print(f"Failed to set up data on server: {e}")
            raise

    def _validate_data(self, rollout: RolloutLegacy):
        if rollout.final_reward is None:
            print(
                f"Warning: Reward is None for rollout {rollout.rollout_id}, will be auto-set to {self.reward_fillna_value}."
            )
        if rollout.triplets is None:
            print(f"Warning: Triplet is None for rollout {rollout.rollout_id}.")
        elif len(rollout.triplets) == 0:
            print(f"Warning: Length of triplets is 0 for rollout {rollout.rollout_id}.")
        elif any(not r.response.get("token_ids", []) for r in rollout.triplets):
            print(f"Warning: Rollout {rollout.rollout_id} contains empty response: {rollout.triplets}")
        elif any(not r.prompt.get("token_ids", []) for r in rollout.triplets):
            print(f"Warning: Rollout {rollout.rollout_id} contains empty prompt: {rollout.triplets}")

    async def _validate_data_v1(self, rollout: Rollout) -> RolloutLegacy:
        """Convert Rollout to RolloutLegacy and validate.

        1. Task: construct from Rollout
        2. Triplets: obtained by querying spans and feeding into the adapter
        3. Final reward: extracted from last triplet's reward, searching backwards if not found
        """
        # Query spans for this rollout (latest attempt)
        spans = await self.store.query_spans(rollout.rollout_id, attempt_id="latest")

        # Convert spans to triplets using the adapter
        if not spans:
            # No triplets found, will emit a warning later.
            triplets = []
        else:
            triplets = self.adapter.adapt(spans)

        # Extract final reward from triplets
        final_reward: Optional[float] = None
        if triplets:
            # Search backwards through triplets for the first non-None reward
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    final_reward = triplet.reward
                    break

        # Construct the Task object from Rollout
        task = Task(
            rollout_id=rollout.rollout_id,
            input=rollout.input,
            mode=rollout.mode,
            resources_id=rollout.resources_id,
            metadata=rollout.metadata or {},
        )

        # Create the Rollout object (without trace and logs as per user's note)
        result_rollout = RolloutLegacy(
            rollout_id=rollout.rollout_id,
            task=task,
            final_reward=final_reward,
            triplets=triplets,
            metadata=rollout.metadata or {},
        )

        # Run the same validation as v0
        self._validate_data(result_rollout)

        return result_rollout

    async def _async_run_until_finished(self, verbose: bool = True):
        """Async helper to wait for all tasks to complete."""
        import time as _time
        _wait_start = _time.time()
        while len(self._completed_rollouts_v0) < self._total_tasks_queued:
            if self.mode == "v0":
                completed_batch = await self.server.retrieve_completed_rollouts()
            else:
                completed_batch = await self.store.wait_for_rollouts(
                    rollout_ids=list(self._task_id_to_original_sample.keys()), timeout=0
                )
            for rollout in completed_batch:
                if rollout.rollout_id in self._completed_rollouts_v0:
                    # Already processed, skip
                    continue
                if isinstance(rollout, Rollout):
                    rollout = await self._validate_data_v1(rollout)
                else:
                    self._validate_data(rollout)
                if rollout.rollout_id not in self._task_id_to_original_sample:
                    print(f"Warning: Received unknown rollout ID {rollout.rollout_id}, skipping.")
                else:
                    self._completed_rollouts_v0[rollout.rollout_id] = rollout
            if verbose:
                print(f"[TQ4-DEBUG] daemon: Completed {len(self._completed_rollouts_v0)}/{self._total_tasks_queued} tasks... (elapsed={_time.time()-_wait_start:.1f}s)")
            await asyncio.sleep(5)

        print(f"[TQ4-DEBUG] daemon: All tasks finished. (elapsed={_time.time()-_wait_start:.1f}s)")

    def run_until_all_finished(self, verbose: bool = True):
        """Synchronously waits for all queued tasks to be completed and reported."""
        if self._total_tasks_queued == 0:
            print("Warning: No tasks were queued.")
            return

        if self.mode == "v0":
            if not self.server.loop or not self.server.startup_event.is_set():
                raise RuntimeError("Server is not running or ready.")
            loop = self.server.loop
        else:
            loop = self._internal_loop
            assert loop is not None

        coro = self._async_run_until_finished(verbose)
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            future.result()  # Wait indefinitely for all tasks to complete
        except Exception as e:
            print(f"Error while waiting for tasks to finish: {e}")
            raise

    def get_test_metrics(self):
        """Calculates and returns metrics for a validation run."""
        assert not self.is_train, "This method should only be called during validation."
        assert len(self._completed_rollouts_v0) == self._total_tasks_queued

        sample_stat_list: List[Dict[str, Any]] = []
        sample_stat_list_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(
            list
        )  # FIXME: Evaluate whether grouping stats by source is actually needed.

        for rollout_id, rollout in self._completed_rollouts_v0.items():
            final_reward_raw: Optional[float] = rollout.final_reward
            final_reward = self._fillna_reward(rollout)
            if not rollout.triplets:
                print(f"Warning: No triplets found for test rollout {rollout.rollout_id}.")
                sample_stat_list.append({"reward": final_reward, "has_reward": final_reward_raw is not None})
                continue
            response_length_list = [len(triplet.response.get("token_ids", [])) for triplet in rollout.triplets]

            if "data_source" in self._task_id_to_original_sample[rollout_id]:
                # When a test sample includes a 'data_source' field, record per-source statistics for test results.
                # TODO: This is a flawed design. We should have a better way to handle this.
                data_source = self._task_id_to_original_sample[rollout_id]["data_source"]
                sample_stat_list_by_source[data_source].append(
                    {
                        "sum_response_length": np.sum(response_length_list),
                        "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                        "turn_count": len(rollout.triplets),
                        "reward": final_reward,
                        "has_reward": final_reward_raw is not None,
                    }
                )
            sample_stat_list.append(
                {
                    "sum_response_length": np.sum(response_length_list),
                    "mean_response_length": np.mean(response_length_list) if response_length_list else 0,
                    "turn_count": len(rollout.triplets),
                    "reward": final_reward,
                    "has_reward": final_reward_raw is not None,
                }
            )
        metric_dict: Dict[str, Any] = {}

        stats_w_trace = [stat for stat in sample_stat_list if "sum_response_length" in stat]
        stats_w_trace_by_source = {
            data_source: [stat for stat in sample_stats if "sum_response_length" in stat]
            for data_source, sample_stats in sample_stat_list_by_source.items()
        }
        for data_source, sample_stats in sample_stat_list_by_source.items():
            metric_dict.update(
                {
                    f"val/{data_source}/n_rollouts": len(sample_stats),
                    f"val/{data_source}/n_rollouts_w_trace": len(stats_w_trace_by_source[data_source]),
                    f"val/{data_source}/n_rollouts_w_reward": len(
                        [stat for stat in sample_stats if stat["has_reward"]]
                    ),
                    f"val/{data_source}/reward": np.mean(
                        [stat["reward"] for stat in sample_stats]
                    ),  # each rollout must have a reward (fillna if missing)
                    f"val/{data_source}/mean_response_length": np.mean(
                        [stat["mean_response_length"] for stat in stats_w_trace_by_source[data_source]]
                    ),
                    f"val/{data_source}/sum_response_length": np.mean(
                        [stat["sum_response_length"] for stat in stats_w_trace_by_source[data_source]]
                    ),
                    f"val/{data_source}/turn_count": np.mean(
                        [stat["turn_count"] for stat in stats_w_trace_by_source[data_source]]
                    ),
                }
            )
        metric_dict.update(
            {
                "val/n_rollouts": len(sample_stat_list),
                "val/n_rollouts_w_trace": len(stats_w_trace),
                "val/n_rollouts_w_reward": len([stat for stat in sample_stat_list if stat["has_reward"]]),
                "val/reward": np.mean(
                    [stat["reward"] for stat in sample_stat_list]
                ),  # each rollout must have a reward (fillna if missing)
                "val/mean_response_length": np.mean([stat["mean_response_length"] for stat in stats_w_trace]),
                "val/sum_response_length": np.mean([stat["sum_response_length"] for stat in stats_w_trace]),
                "val/turn_count": np.mean([stat["turn_count"] for stat in stats_w_trace]),
            }
        )
        return metric_dict

    def get_train_data_batch(
        self, max_prompt_length: int, max_response_length: int, device: torch.device, global_steps: int, partition_id: str = "train"
    ):
        """
        Processes completed rollouts to generate a training data batch.

        This function reconstructs the logic from the original AgentModeDaemon,
        using data retrieved from the new server architecture. It handles padding,
        truncation, and tensor creation for the PPO training loop.
        """
        assert self.is_train, "This method should only be called during training."
        assert len(self._completed_rollouts_v0) == self._total_tasks_queued

        # 1. Reconstruct the `finished_id_to_sample_info` structure from completed rollouts
        finished_id_to_sample_info: Dict[str, Dict[str, Any]] = {}
        finished_id_to_final_reward: Dict[str, float] = {}
        sample_with_reward_count = 0
        for rollout_id, rollout in self._completed_rollouts_v0.items():
            original_sample = self._task_id_to_original_sample[rollout_id]
            sample_with_reward_count += int(rollout.final_reward is not None)
            final_reward = self._fillna_reward(rollout)

            if not rollout.triplets:
                finished_id_to_final_reward[rollout_id] = final_reward
                print(f"Warning: No triplets found for training rollout {rollout.rollout_id}, skipping.")
                continue

            # The client should report triplets that contain prompt_ids and response_ids.
            # Example triplet.prompt: {"token_ids": [...], "image_urls": [...]}
            # Example triplet.response: {"token_ids": [...]}
            trace_list = [
                {
                    "prompt_ids": t.prompt.get("token_ids", []),
                    "response_ids": t.response.get("token_ids", []),
                    "image_urls": t.prompt.get("image_urls", []),
                }
                for t in rollout.triplets
            ]
            info = {
                "reward": final_reward,
                "trace_list": trace_list,
                "data_id": original_sample["data_id"],
            }
            finished_id_to_sample_info[rollout_id] = info
            finished_id_to_final_reward[rollout_id] = final_reward
        #
        # --- Data processing and tensor creation logic ---
        # Get all the reported data.
        # prompt_ids are left-padded.
        # response_ids are right-padded.
        # They are concatenated in the middle.
        # Discard handling:
        #   - Those exceeding max_prompt_length will be marked for discard, but not
        #     discarded here. They are only truncated and marked, to be discarded later.
        #     This is for the correctness of the advantage calculation.
        #   - The discard for the PPO mini-batch should also be handled this way.
        # TQ写入：直接构建unpadded变长tensor，无需先pad再unpad
        # 逐样本收集原始数据，跳过padding步骤
        reward_list: List[float] = []
        data_id_list: List[str] = []
        rollout_id_list: List[str] = []
        turn_index_list: List[int] = []
        is_drop_list: List[bool] = []
        image_grid_thw_list: List[Optional[torch.Tensor]] = []
        n_trunc_sample_because_of_response = 0
        # 逐样本的unpadded原始数据，用于构建TQ的NestedTensor
        raw_prompt_ids_list: List[List[int]] = []
        raw_response_ids_list: List[List[int]] = []
        raw_response_mask_list: List[List[int]] = []

        if self.trace_aggregator.get("level", "transition") == "transition":
            for rollout_id, sample_info in finished_id_to_sample_info.items():
                for turn_index, trace in enumerate(sample_info["trace_list"]):

                    reward_list.append(sample_info["reward"])
                    prompt_ids, response_ids = trace["prompt_ids"], trace["response_ids"]

                    if len(prompt_ids) > max_prompt_length:
                        prompt_ids = prompt_ids[:max_prompt_length]
                        is_drop_list.append(True)
                    else:
                        is_drop_list.append(False)

                    if len(response_ids) > max_response_length:
                        response_ids = response_ids[:max_response_length]
                        n_trunc_sample_because_of_response += 1

                    # 不再pad，直接保存原始变长数据
                    raw_prompt_ids_list.append(prompt_ids)
                    raw_response_ids_list.append(response_ids)
                    data_id_list.append(sample_info["data_id"])
                    rollout_id_list.append(rollout_id)
                    turn_index_list.append(turn_index)

                    if self._use_mrope:
                        image_urls = trace.get("image_urls", [])
                        image_grid_thw_list.append(self._get_image_grid_thw(image_urls))

        elif self.trace_aggregator.get("level", "transition") == "trajectory":
            assert not self._use_mrope, "M-RoPE is not supported in trajectory level yet."

            unmerged_count: int = 0
            template_mismatch_count, retoken_mismatch_count, others_mismatch_count = 0, 0, 0
            response_per_turn_list: List[int] = []

            for rollout_id, sample_info in finished_id_to_sample_info.items():
                merged_trace_idx: List[List[int]] = []

                current_merged_trace_idx: List[int] = []
                current_context: List[int] = []
                for turn_index, trace in enumerate(sample_info["trace_list"]):
                    response_per_turn_list.append(len(trace["response_ids"]))
                    is_prefix, diagnostic = ids_startswith(
                        trace["prompt_ids"] + trace["response_ids"],
                        current_context,
                        self.tokenizer,
                        self.trace_aggregator.get("debug", False),
                    )
                    if not is_prefix and self.trace_aggregator.get("debug", False) == True:
                        template_mismatch_count += diagnostic[0]
                        retoken_mismatch_count += diagnostic[1]
                        others_mismatch_count += diagnostic[2]
                        log_mismatch_detail(
                            diagnostic,
                            trace["prompt_ids"] + trace["response_ids"],
                            current_context,
                            global_steps,
                            rollout_id,
                            turn_index,
                            self.trace_aggregator.get("mismatch_log_dir", None),
                        )

                    if is_prefix:
                        current_context = trace["prompt_ids"] + trace["response_ids"]
                        current_merged_trace_idx.append(turn_index)
                    else:
                        merged_trace_idx.append(current_merged_trace_idx)
                        current_merged_trace_idx = [turn_index]
                        current_context = trace["prompt_ids"] + trace["response_ids"]

                if current_merged_trace_idx not in merged_trace_idx:
                    merged_trace_idx.append(current_merged_trace_idx)

                if len(merged_trace_idx) > 1:
                    unmerged_count += 1

                for current_merged_trace_idx in merged_trace_idx:
                    prompt_ids = sample_info["trace_list"][current_merged_trace_idx[0]]["prompt_ids"]

                    if current_merged_trace_idx[0] > 0 and len(prompt_ids) > max_prompt_length:
                        response_ids = prompt_ids[max_prompt_length:]
                        prompt_ids = prompt_ids[:max_prompt_length]
                        response_mask = [1] * len(response_ids)
                    else:
                        response_ids = []
                        response_mask = []

                    prompt_length = len(prompt_ids)
                    first_turn_response = sample_info["trace_list"][current_merged_trace_idx[0]]["response_ids"]
                    response_ids += first_turn_response
                    response_mask += [1] * len(first_turn_response)
                    for turn_index in current_merged_trace_idx[1:]:
                        trace = sample_info["trace_list"][turn_index]
                        new_prompt_length = len(trace["prompt_ids"]) - len(response_ids) - prompt_length
                        response_ids += trace["prompt_ids"][-new_prompt_length:]
                        response_ids += trace["response_ids"]
                        response_mask += [0] * new_prompt_length
                        response_mask += [1] * len(trace["response_ids"])

                    reward_list.append(sample_info["reward"])

                    if len(prompt_ids) > max_prompt_length:
                        prompt_ids = prompt_ids[:max_prompt_length]
                        is_drop_list.append(True)
                    else:
                        is_drop_list.append(False)

                    if len(response_ids) > max_response_length:
                        response_ids = response_ids[:max_response_length]
                        response_mask = response_mask[:max_response_length]
                        n_trunc_sample_because_of_response += 1

                    # 不再pad，直接保存原始变长数据
                    raw_prompt_ids_list.append(prompt_ids)
                    raw_response_ids_list.append(response_ids)
                    raw_response_mask_list.append(response_mask)
                    data_id_list.append(sample_info["data_id"])
                    rollout_id_list.append(rollout_id)
        else:
            raise ValueError(f"Unknown trace_aggregator level: {self.trace_aggregator.get('level')}")

        n_transition = len(raw_prompt_ids_list)

        # 逐样本构建unpadded变长tensor，写入TQ时使用NestedTensor格式
        # PPOTrainer._compute_metrics中data["prompts"].offsets().diff()依赖NestedTensor获取真实长度
        fields_list = []
        for i in range(n_transition):
            prompt_ids = raw_prompt_ids_list[i]
            response_ids = raw_response_ids_list[i]
            seq_ids = prompt_ids + response_ids
            prompt_len = len(prompt_ids)
            response_len = len(response_ids)
            seq_len = prompt_len + response_len

            # token_level_scores: reward放在response最后一个token位置
            token_level_scores_i = [0.0] * response_len
            token_level_scores_i[-1] = reward_list[i]
            token_level_scores_tensor = torch.tensor(token_level_scores_i, dtype=torch.float32).to(device)

            # position_ids: unpadded序列全为真实token，等价于cumsum(all-1 mask) - 1
            position_ids_i = torch.arange(seq_len, dtype=torch.long).to(device)

            # mrope需要特殊处理position_ids
            if self._use_mrope:
                input_ids_tensor = torch.tensor([seq_ids], dtype=torch.long).to(device)
                attention_mask_tensor = torch.ones(1, seq_len, dtype=torch.long).to(device)
                image_grid_thw = image_grid_thw_list[i] if image_grid_thw_list else None
                position_ids_i = self._compute_mrope_position_ids(
                    input_ids=input_ids_tensor[0],
                    attention_mask=attention_mask_tensor[0],
                    image_grid_thw=image_grid_thw,
                )

            field = {
                "prompts": torch.tensor(prompt_ids, dtype=torch.long).to(device),
                "responses": torch.tensor(response_ids, dtype=torch.long).to(device),
                "input_ids": torch.tensor(seq_ids, dtype=torch.long).to(device),
                "attention_mask": torch.ones(seq_len, dtype=torch.long).to(device),
                "position_ids": position_ids_i,
                "token_level_scores": token_level_scores_tensor,
                "uid": data_id_list[i],
                "rm_scores": token_level_scores_tensor,
                "num_turns": 1,
            }

            # response_mask/loss_mask：trajectory模式从trace生成，transition模式全1
            if self.trace_aggregator.get("level") == "trajectory":
                field["response_mask"] = torch.tensor(raw_response_mask_list[i], dtype=torch.long).to(device)
                field["loss_mask"] = torch.tensor(raw_response_mask_list[i], dtype=torch.long).to(device)
            else:
                field["response_mask"] = torch.ones(response_len, dtype=torch.long).to(device)
                field["loss_mask"] = torch.ones(response_len, dtype=torch.long).to(device)

            fields_list.append(field)

        # list_of_dict_to_tensordict: 变长tensor→NestedTensor(jagged), 非tensor→NonTensorStack
        fields = list_of_dict_to_tensordict(fields_list)

        # 生成唯一key：{data_id}_{rollout_id}_{turn_index}
        keys = []
        tags = []
        for i in range(n_transition):
            if self.trace_aggregator.get("level") == "transition":
                key = f"{data_id_list[i]}_{rollout_id_list[i]}_{turn_index_list[i]}"
            else:
                key = f"{data_id_list[i]}_{rollout_id_list[i]}_{i}"
            keys.append(key)
            prompt_len = len(raw_prompt_ids_list[i])
            response_len = len(raw_response_ids_list[i])
            tags.append({"seq_len": prompt_len + response_len, "is_drop": is_drop_list[i]})

        _t_tq_write = now()
        tq.kv_batch_put(keys=keys, partition_id=partition_id, fields=fields, tags=tags)
        _t_tq_write_end = now()
        _seq_lens = [t.get("seq_len", 0) for t in tags]
        bench_log("dualwrite", global_steps, "daemon_tq_write",
            daemon_tq_write=round(_t_tq_write_end - _t_tq_write, 4),
            n_triplets=n_transition,
            unpadded_bytes=sum(sl * 8 for sl in _seq_lens),
            avg_seq_len=round(float(np.mean(_seq_lens)), 1) if _seq_lens else 0,
            max_seq_len=max(_seq_lens) if _seq_lens else 0,
            min_seq_len=min(_seq_lens) if _seq_lens else 0,
            rss_mb=rss_mb())

        batch_meta = KVBatchMeta(
            keys=keys,
            tags=tags,
            partition_id=partition_id,
            fields=set(fields.keys()),
        )

        data_metrics = {
            "training/reward": np.mean(list(finished_id_to_final_reward.values())),
            "training/n_rollouts": len(finished_id_to_final_reward),
            "training/n_rollouts_w_trace": len(finished_id_to_sample_info),
            "training/n_rollouts_w_reward": sample_with_reward_count,
            "training/n_truncated_triplets": n_trunc_sample_because_of_response,
            "training/n_triplets": n_transition,
            **(
                {
                    "training/n_unmerged_rollouts": unmerged_count,  # type: ignore
                    "training/n_triplets_by_turn": len(response_per_turn_list),  # type: ignore
                    "training/avg_response_length_by_turn": np.mean(response_per_turn_list),  # type: ignore
                    "training/max_response_length_by_turn": np.max(response_per_turn_list),  # type: ignore
                    "training/min_response_length_by_turn": np.min(response_per_turn_list),  # type: ignore
                }
                if self.trace_aggregator.get("level", "transition") == "trajectory"
                else {}
            ),
            **(
                {
                    "training/template_mismatch_triplets": template_mismatch_count,  # type: ignore
                    "training/retoken_mismatch_triplets": retoken_mismatch_count,  # type: ignore
                    "training/others_mismatch_triplets": others_mismatch_count,  # type: ignore
                    "training/template_mismatch_ratio": template_mismatch_count / len(response_per_turn_list),  # type: ignore
                    "training/retoken_mismatch_ratio": retoken_mismatch_count / len(response_per_turn_list),  # type: ignore
                    "training/others_mismatch_ratio": others_mismatch_count / len(response_per_turn_list),  # type: ignore
                }
                if self.trace_aggregator.get("level", "transition") == "trajectory"
                and self.trace_aggregator.get("debug", False)
                else {}
            ),
        }

        return batch_meta, data_metrics

    def clear_data_and_server(self):
        """Resets the internal state of the daemon for the next run."""
        self.backend_llm_server_addresses = []
        self._completed_rollouts_v0.clear()
        self._task_id_to_original_sample.clear()
        self._total_tasks_queued = 0
        # For a true reset, the server's internal queues would also need clearing.
        # This implementation assumes that `set_up_data_and_server` is called
        # for each new run, effectively starting a fresh batch.

    def _fillna_reward(self, rollout: RolloutLegacy):
        if rollout.final_reward is None:
            if self.reward_fillna_value is not None:  # type: ignore
                final_reward = self.reward_fillna_value
            else:
                raise ValueError(f"Reward is None for rollout {rollout.rollout_id}, please check the reward function.")
        else:
            final_reward = rollout.final_reward
        return final_reward
