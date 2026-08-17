from asyncio import Runner
import functools
from typing import Any
from agentlightning.execution.client_server import ClientServerExecutionStrategy
from agentlightning.execution.events import ExecutionEvent
from agentlightning.runner.agent import LitAgentRunner
from agentlightning.store.base import LightningStore
from agentlightning.store.client_server import LightningStoreClient
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import Tracer
import calc_agent

class CalcAgentRunner:
    def __init__(self, tracer: Tracer, store: LightningStore) -> None:
        self.tracer = tracer
        self.store = store

    def run(self):
        exec = ClientServerExecutionStrategy(role="runner", n_runners=2, managed_store=False)
        runner_bundle = functools.partial(self._run, agent=calc_agent.calc_agent)
        exec.execute(runner=runner_bundle, algorithm=None, store=self.store) # type: ignore


    async def _run(self, store: LightningStore, worker_id: int, event: ExecutionEvent, agent) -> None:
        """Internal entry point executed by the strategy for each runner role.

        The bundle materializes the configured runner, binds the agent and hooks, associates
        the worker with the shared store, and then drives the runner's [`iter`][agentlightning.Runner.iter]
        loop until the execution event is set or an exception occurs. Cleanup mirrors the initialization
        sequence to keep tracer state, hooks, and agent resources consistent across restarts.
        """
        runner_instance: Runner[Any] | None = None # type: ignore
        runner_initialized = False
        worker_initialized = False
        try:
            # If not using shm execution strategy, we are already in the forked process
            runner_instance = LitAgentRunner(tracer=self.tracer)
            runner_instance.init(agent=agent)
            runner_initialized = True
            runner_instance.init_worker(worker_id, store)
            worker_initialized = True
            await runner_instance.iter(event=event)
        except Exception:
            print("Runner bundle encountered an error (worker_id=%s).", worker_id)
            raise
        finally:
            if runner_instance is not None:
                if worker_initialized:
                    try:
                        runner_instance.teardown_worker(worker_id)
                    except Exception:
                        print("Error during runner worker teardown (worker_id=%s).", worker_id)
                if runner_initialized:
                    try:
                        runner_instance.teardown()
                    except Exception:
                        print("Error during runner teardown (worker_id=%s).", worker_id)

if __name__ == "__main__":
    store = LightningStoreClient("http://7.150.14.159:4747")
    tracer = AgentOpsTracer(agentops_managed=True, instrument_managed=True, daemon=True)
    runner = CalcAgentRunner(tracer=tracer, store=store)
    runner.run()

    