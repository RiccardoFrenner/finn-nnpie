import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

import numpy as np

# public API
__all__ = [
    "load_json",
    "save_json",
    "MyFile",
    "MySmartFile",
    "MyFolder",
    "MySmartFolder",
    "MyDir",
]

T = TypeVar("T")
K = TypeVar("K")


def load_json(path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text())


def save_json(path, data) -> None:
    path = Path(path)
    path.write_text(json.dumps(data, indent=4))


def load_save_funs_from_suffix(suffix: str):
    if suffix == ".npy":
        load_fun = np.load
        save_fun = np.save
    elif suffix == ".txt":
        load_fun = np.loadtxt  # type: ignore
        save_fun = np.savetxt  # type: ignore
    elif suffix == ".json":
        load_fun = load_json  # type: ignore
        save_fun = save_json  # type: ignore
    else:
        raise ValueError(
            f"Cannot determine load/save functions from file suffix '{suffix}'"
        )
    return load_fun, save_fun


@dataclass
class MyFile(Generic[T]):
    path: Path
    _load_fun: Callable[[Path], T]
    _save_fun: Callable[[Path, T], None]

    def save(self, data: T):
        self._save_fun(self.path, data)

    def load(self) -> T:
        return self._load_fun(self.path)


class MySmartFile(MyFile):
    def __init__(self, path: Path) -> None:
        load_fun = load_save_funs_from_suffix(path.suffix)[0]
        save_fun = load_save_funs_from_suffix(path.suffix)[1]
        super().__init__(path, load_fun, save_fun)  # type: ignore


@dataclass
class MyFolder(Generic[T]):
    path: Path
    load_fun: Callable[[Path], T]
    save_fun: Callable[[Path, T], None]
    file_template: str

    def __post_init__(self):
        assert "." in self.file_template, "Template contains no file extension."
        assert "{}" in self.file_template, "Template contains no key placeholder."

        self.path.mkdir(exist_ok=True, parents=True)

    def file_path(self, key) -> Path:
        return self.path / self.file_template.format(key)

    def save(self, data: T, key):
        self.save_fun(self.file_path(key), data)

    def load(self, key: int) -> T:
        return self.load_fun(self.file_path(key))

    def iter_all(self, sort_it: bool = False):
        paths = self.path.glob(self.file_template.replace("{}", "*"))
        keys = [MyFolder._get_key_from_path(p) for p in paths]
        if sort_it:
            keys = sorted(keys)

        for k in keys:
            yield k, self.load(k)

    def load_all(self, sort_it: bool = False) -> tuple[list[int], list[T]]:
        all_data = []
        all_keys = []
        for k, d in self.iter_all(sort_it=sort_it):
            all_keys.append(k)
            all_data.append(d)
        return all_keys, all_data

    @staticmethod
    def _get_key_from_path(p: Path) -> int:
        return int(p.stem.split("_")[-1])


class MySmartFolder(MyFolder):
    def __init__(self, path: Path, file_template: str) -> None:
        load_fun = load_save_funs_from_suffix(path.suffix)[0]
        save_fun = load_save_funs_from_suffix(path.suffix)[1]
        super().__init__(path, load_fun, save_fun, file_template)  # type: ignore


class MyDir:
    def __init__(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        self.path = path

    def new_ts(
        self,
        name: str,
        load_fun: Callable[[Path], T],
        save_fun: Callable[[Path, T], None],
        file_template: str,
    ):
        return MyFolder(self.path / name, load_fun, save_fun, file_template)


if __name__ == "__main__":

    class MyExampleDir(MyDir):
        def __init__(self, path):
            path = Path(path)
            self.path = path
            self.config = MySmartFile(path / "config.json")
            self.data = MySmartFile(path / "data.npy")
            self.ts1 = MySmartFolder(path / "ts1", "ts1data_{}.npy")
            self.ts2 = MySmartFolder(path / "ts2", "ts1data_{}.json")

        def new_ts(
            self,
            name: str,
            load_fun: Callable[[Path], T],
            save_fun: Callable[[Path, T], None],
            file_template: str,
        ):
            return MyFolder(self.path / name, load_fun, save_fun, file_template)

    mydir = MyExampleDir("/tmp/testdir1")
    mydir.config.save({"a": 2, "b": "test"})
    config = mydir.config.load()
    print(f"Config is saved at {mydir.config.path}")

    for i in range(10):
        ts_data = np.random.uniform(0, 1, 5)
        mydir.ts1.save(ts_data, key=i)

    all_ts_keys, all_ts_data = mydir.ts1.load_all(sort_it=False)
    example_ts_data = mydir.ts1.load(key=8)

    dynamic_ts_1 = mydir.new_ts("dynamic_ts_1", np.load, np.save, "dyna_{}.npy")
    for i in range(20):
        ts_data = np.random.uniform(0, 1, 5)
        dynamic_ts_1.save(ts_data, key=i)

    dynamic_ts_1_mean = np.mean(dynamic_ts_1.load_all(sort_it=False)[0], axis=0)

    for p in mydir.path.rglob("*"):
        print(p)
