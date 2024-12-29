from typing import Protocol, runtime_checkable


@runtime_checkable
class SamplingStrategy(Protocol):
    pass


class ChooseWithReplacement(SamplingStrategy):
    pass
