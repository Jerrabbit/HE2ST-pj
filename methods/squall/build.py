"""最小 MODELS registry（官方 build.py 的轻量替代，仅用于 Squall 装饰器）。

官方 build.py 依赖完整的 utils/registry 框架；本仓库 benchmark 只加载冻结
Squall 模型，不需要训练注册表，故提供最小 register_module 装饰器。
架构不受影响：@MODELS.register_module() 仅把类登记到 registry。
"""


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._modules: dict = {}

    def register_module(self, name: str | None = None):
        def wrapper(cls):
            key = name or cls.__name__
            self._modules[key] = cls
            return cls
        return wrapper

    def get(self, key: str):
        return self._modules[key]

    def __contains__(self, key: str) -> bool:
        return key in self._modules


MODELS = Registry("MODELS")
