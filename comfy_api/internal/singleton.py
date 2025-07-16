from typing import Optional, Type, TypeVar
import threading

def local_execution(func):
    """Decorator to mark a method for local execution instead of RPC"""
    func._is_local_execution = True
    return func


class LocalMethodRegistry:
    """Registry for local method implementations in proxied singletons"""
    _instance: Optional['LocalMethodRegistry'] = None
    _lock = threading.Lock()

    def __init__(self):
        self._local_implementations: dict[Type, object] = {}
        self._local_methods: dict[Type, set[str]] = {}

    @classmethod
    def get_instance(cls) -> 'LocalMethodRegistry':
        """Get the singleton instance of LocalMethodRegistry"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def register_class(self, cls: Type, proxy_instance: object) -> None:
        """Register a class with its local method implementations"""
        # Create a local instance by bypassing the singleton mechanism
        # We call the base object.__new__ directly to avoid getting the existing singleton
        local_instance = object.__new__(cls)  # type: ignore[misc]
        cls.__init__(local_instance)
        self._local_implementations[cls] = local_instance

        # Track which methods are marked for local execution
        local_methods = set()
        for name in dir(cls):
            if not name.startswith('_'):
                attr = getattr(cls, name, None)
                if callable(attr) and getattr(attr, '_is_local_execution', False):
                    local_methods.add(name)

        self._local_methods[cls] = local_methods

    def is_local_method(self, cls: Type, method_name: str) -> bool:
        """Check if a method should be executed locally"""
        return cls in self._local_methods and method_name in self._local_methods[cls]

    def get_local_method(self, cls: Type, method_name: str):
        """Get the local implementation of a method"""
        if cls not in self._local_implementations:
            raise ValueError(f"Class {cls} not registered for local execution")

        local_instance = self._local_implementations[cls]
        return getattr(local_instance, method_name)


class SingletonMetaclass(type):
    T = TypeVar('T', bound='SingletonMetaclass')
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonMetaclass, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

    def inject_instance(cls: Type[T], instance: T) -> None:
        assert cls not in SingletonMetaclass._instances, "Cannot inject instance after first instantiation"
        SingletonMetaclass._instances[cls] = instance

    def get_instance(cls: Type[T], *args, **kwargs) -> T:
        """
        Gets the singleton instance of the class, creating it if it doesn't exist.
        """
        if cls not in SingletonMetaclass._instances:
            SingletonMetaclass._instances[cls] = super(SingletonMetaclass, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ProxiedSingleton(object, metaclass=SingletonMetaclass):
    def __init__(self):
        super().__init__()