from dataclasses import dataclass


@dataclass(frozen=True)
class PackSpec:
    name: str
    version: str
    url: str
    sha256: str | None = None


POINTSOURCE = PackSpec(
    name="pointsource_noises",
    version="main",
    url="https://huggingface.co/datasets/gfdb/pointsource_noises/resolve/main/pointsource_noises.tar.gz",
    sha256="f2a3ee1c9c80443f38422b16936361fadc3c7d571a3a808f04fa3fac8302a94d",
)

REGISTRY = {"pointsource_noises": POINTSOURCE}
DEFAULT_PACK = POINTSOURCE
