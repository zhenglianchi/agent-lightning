# Copyright (c) Microsoft. All rights reserved.

import logging
from typing import List, Sequence, Union, Optional

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning import Triplet
from agentlightning import Span
from agentlightning import TraceToTripletBase

logger = logging.getLogger(__name__)


class AdaptiveTraceToTriplet(TraceToTripletBase):
    """尝试多个适配器，返回第一个能成功提取非空Triplet的结果。"""

    def __init__(
        self,
        adapters: List[TraceToTripletBase],
        fallback_on_empty: bool = True,
        fallback_on_exception: bool = True,
        use_cached_adapter: bool = True,
    ):
        """
        Args:
            adapters: 按优先级顺序排列的适配器列表。
            fallback_on_empty: 若返回的Triplet列表为空，是否继续尝试下一个适配器。
            fallback_on_exception: 若adapt()抛出异常，是否捕获并继续尝试下一个适配器。
            use_cached_adapter: 是否使用上次成功选择的adapter记录，若为True则优先使用记录的adapter。
        """
        if not adapters:
            raise ValueError("adapters must not be empty")
        for i, adapter in enumerate(adapters):
            if not isinstance(adapter, TraceToTripletBase):
                raise TypeError(f"adapters[{i}] must be a TraceToTripletBase instance, got {type(adapter).__name__}")

        self.adapters = adapters
        self.fallback_on_empty = fallback_on_empty
        self.fallback_on_exception = fallback_on_exception
        self.use_cached_adapter = use_cached_adapter
        self._cached_adapter_name: Optional[str] = None

    def _get_adapter_by_name(self, name: str) -> Optional[TraceToTripletBase]:
        if not isinstance(name, str) or not name:
            logger.warning("Invalid adapter name provided: %r", name)
            return None
        for adapter in self.adapters:
            if type(adapter).__name__ == name:
                return adapter
        return None

    def _try_cached_adapter(self, source: Union[Sequence[Span], Sequence[ReadableSpan]]) -> Optional[List[Triplet]]:
        if not self.use_cached_adapter or self._cached_adapter_name is None:
            return None

        cached_adapter = self._get_adapter_by_name(self._cached_adapter_name)
        if cached_adapter is None:
            logger.warning(
                "Cached adapter '%s' not found in adapter list, falling back to full selection",
                self._cached_adapter_name,
            )
            self._cached_adapter_name = None
            return None

        logger.info("Found cached adapter: %s, using it directly", self._cached_adapter_name)
        try:
            triplets = cached_adapter.adapt(source)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning(
                "Cached adapter %s raised %s: %s, falling back to full selection",
                self._cached_adapter_name,
                type(e).__name__,
                e,
            )
            self._cached_adapter_name = None
            return None

        if triplets:
            logger.info(
                "Cached adapter %s succeeded: %d triplet(s) extracted",
                self._cached_adapter_name,
                len(triplets),
            )
            return triplets

        logger.info(
            "Cached adapter %s returned empty, falling back to full selection",
            self._cached_adapter_name,
        )
        self._cached_adapter_name = None
        return None

    def _try_adapter(
        self,
        adapter: TraceToTripletBase,
        idx: int,
        total: int,
        source: Union[Sequence[Span], Sequence[ReadableSpan]],
    ) -> Optional[List[Triplet]]:
        adapter_name = type(adapter).__name__
        logger.info("Trying adapter %d/%d: %s", idx, total, adapter_name)
        try:
            triplets = adapter.adapt(source)  # type: ignore[arg-type]
        except Exception as e:
            logger.warning("Adapter %s raised %s: %s", adapter_name, type(e).__name__, e)
            if self.fallback_on_exception:
                logger.info("Fallback to next adapter (fallback_on_exception=True)")
                return None
            logger.error("Rethrowing exception (fallback_on_exception=False)")
            raise

        if not triplets:
            logger.info("Adapter %s returned empty list (length 0)", adapter_name)
            if self.fallback_on_empty:
                logger.info("Fallback to next adapter (fallback_on_empty=True)")
                return None
            logger.info("Stopping and returning empty (fallback_on_empty=False)")
            return []

        logger.info("Adapter %s succeeded: %d triplet(s) extracted", adapter_name, len(triplets))
        if self.use_cached_adapter:
            self._cached_adapter_name = adapter_name
            logger.info("Cached adapter '%s' for future use", adapter_name)
        return triplets

    def adapt(self, source: Union[Sequence[Span], Sequence[ReadableSpan]]) -> List[Triplet]:
        total = len(self.adapters)
        logger.info(
            "Starting adaptation with %d adapter(s), fallback_on_empty=%s, fallback_on_exception=%s",
            total,
            self.fallback_on_empty,
            self.fallback_on_exception,
        )

        cached_result = self._try_cached_adapter(source)
        if cached_result is not None:
            return cached_result

        last_exception: Optional[Exception] = None
        for idx, adapter in enumerate(self.adapters, start=1):
            try:
                result = self._try_adapter(adapter, idx, total, source)
            except Exception as e:
                last_exception = e
                continue

            if result is not None:
                return result

        if last_exception is not None:
            logger.error(
                "All adapters failed, re-raising last exception from %s",
                type(self.adapters[-1]).__name__,
            )
            raise last_exception

        logger.warning("All adapters returned empty and fallback_on_empty is True, returning []")
        return []
