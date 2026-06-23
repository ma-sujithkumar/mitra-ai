from abc import ABC, abstractmethod

from backend.agents.feature_engineering.state import PipelineState


class PreconditionError(RuntimeError):
    pass


class PostconditionError(RuntimeError):
    pass


class BaseTool(ABC):
    @abstractmethod
    def precondition(self, state: PipelineState) -> None:
        ...

    @abstractmethod
    def run(self, state: PipelineState) -> None:
        ...

    @abstractmethod
    def postcondition(self, state: PipelineState) -> None:
        ...

    def __call__(self, state: PipelineState) -> PipelineState:
        self.precondition(state)
        self.run(state)
        self.postcondition(state)
        return state
