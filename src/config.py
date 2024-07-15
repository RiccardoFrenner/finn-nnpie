from dataclasses import dataclass


@dataclass
class Config:
    nx: int
    nt: int
    T: float
