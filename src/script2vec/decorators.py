import asyncio
import inspect
import functools
from typing import Callable, Any, Dict, Optional, List
from .script2vec import Script2Vec
from .ctrm_script_interface import CTRMScriptInterface

def auto_vectorize(purpose: str = "function_optimization",
                  ctrm_manager=None,
                  store_in_ctrm: bool = True):
    """
    Decorator to automatically vectorize Python functions

    Args:
        purpose: Purpose of vectorization
        ctrm_manager: Optional CTRM manager for direct integration
        store_in_ctrm: Whether to store in CTRM automatically

    Returns:
        Decorated function
    """
    def decorator(func):
        script2vec = Script2Vec()
        ctrm_interface = CTRMScriptInterface(ctrm_manager=ctrm_manager)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get function source code
            source_code = inspect.getsource(func)

            # Vectorize the function
            vector_result = script2vec.python_to_vector(source_code)

            # Store in CTRM if enabled
            if store_in_ctrm:
                try:
                    ctrm_result = await ctrm_interface.function_to_ctrm(func)

                    # Add vector metadata to function
                    func._vector_metadata = {
                        "vector_hash": ctrm_result["script_hash"],
                        "ctrm_vector_hash": ctrm_result["ctrm_vector_hash"],
                        "vector": vector_result["vector"],
                        "concepts": vector_result.get("concepts", []),
                        "purpose": purpose
                    }
                except Exception as e:
                    print(f"⚠️  Failed to store in CTRM: {e}")
                    # Still add basic metadata
                    func._vector_metadata = {
                        "vector_hash": vector_result["script_hash"],
                        "vector": vector_result["vector"],
                        "concepts": vector_result.get("concepts", []),
                        "purpose": purpose,
                        "error": str(e)
                    }

            # Execute the original function
            result = await func(*args, **kwargs)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get function source code
            source_code = inspect.getsource(func)

            # Vectorize the function
            vector_result = script2vec.python_to_vector(source_code)

            # Store in CTRM if enabled (run async in background)
            if store_in_ctrm:
                async def store_in_ctrm_async():
                    ctrm_result = await ctrm_interface.function_to_ctrm(func)

                    # Add vector metadata to function
                    func._vector_metadata = {
                        "vector_hash": ctrm_result["script_hash"],
                        "ctrm_vector_hash": ctrm_result["ctrm_vector_hash"],
                        "vector": vector_result["vector"],
                        "concepts": vector_result.get("concepts", []),
                        "purpose": purpose
                    }

                # Run in background without blocking
                asyncio.create_task(store_in_ctrm_async())

            # Execute the original function
            result = func(*args, **kwargs)

            return result

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def vectorize_class(purpose: str = "class_optimization",
                   ctrm_manager=None,
                   store_in_ctrm: bool = True):
    """
    Decorator to automatically vectorize Python classes

    Args:
        purpose: Purpose of vectorization
        ctrm_manager: Optional CTRM manager for direct integration
        store_in_ctrm: Whether to store in CTRM automatically

    Returns:
        Decorated class
    """
    def decorator(cls):
        script2vec = Script2Vec()
        ctrm_interface = CTRMScriptInterface(ctrm_manager=ctrm_manager)

        # Get class source code
        source_code = inspect.getsource(cls)

        # Vectorize the class
        vector_result = script2vec.python_to_vector(source_code)

        # Store in CTRM if enabled
        if store_in_ctrm:
            async def store_in_ctrm_async():
                ctrm_result = await ctrm_interface.script_to_ctrm(
                    source_code,
                    purpose,
                    class_name=cls.__name__,
                    module=cls.__module__,
                    qualname=cls.__qualname__
                )

                # Add vector metadata to class
                cls._vector_metadata = {
                    "vector_hash": ctrm_result["script_hash"],
                    "ctrm_vector_hash": ctrm_result["ctrm_vector_hash"],
                    "vector": vector_result["vector"],
                    "concepts": vector_result.get("concepts", []),
                    "purpose": purpose
                }

            # Run in background without blocking
            asyncio.create_task(store_in_ctrm_async())

        return cls

    return decorator

def track_script_evolution(purpose: str = "evolution_tracking",
                          ctrm_manager=None):
    """
    Decorator to track script evolution over time

    Args:
        purpose: Purpose of evolution tracking
        ctrm_manager: Optional CTRM manager for direct integration

    Returns:
        Decorated function
    """
    def decorator(func):
        script2vec = Script2Vec()
        ctrm_interface = CTRMScriptInterface(ctrm_manager=ctrm_manager)

        # Store original version
        original_source = inspect.getsource(func)
        original_vector = script2vec.python_to_vector(original_source)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get current source code
            current_source = inspect.getsource(func)

            # Check if function has changed
            current_vector = script2vec.python_to_vector(current_source)

            if current_vector["script_hash"] != original_vector["script_hash"]:
                # Function has changed - track evolution
                async def track_evolution_async():
                    nonlocal original_vector, original_source

                    evolution_result = await ctrm_interface.track_script_evolution(
                        current_source,
                        original_source
                    )

                    # Update metadata
                    if not hasattr(func, '_evolution_history'):
                        func._evolution_history = []

                    func._evolution_history.append({
                        "evolution_id": evolution_result["evolution_id"],
                        "generation": evolution_result["generation"],
                        "changes": evolution_result["changes"],
                        "timestamp": datetime.now().isoformat(),
                        "improvement_score": evolution_result["improvement_score"]
                    })

                    # Update original for next comparison
                    original_vector = current_vector
                    original_source = current_source

                # Run in background
                asyncio.create_task(track_evolution_async())

            # Execute the original function
            result = func(*args, **kwargs)

            return result

        return wrapper

    return decorator

class ScriptVectorizer:
    """Context manager for temporary script vectorization"""

    def __init__(self, ctrm_manager=None, purpose: str = "temporary_analysis"):
        self.ctrm_manager = ctrm_manager
        self.purpose = purpose
        self.script2vec = Script2Vec()
        self.ctrm_interface = CTRMScriptInterface(ctrm_manager=ctrm_manager)
        self.vectorized_scripts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clean up - store all vectorized scripts in CTRM
        if self.vectorized_scripts:
            await self.ctrm_interface.batch_process_scripts(
                [script for script, _ in self.vectorized_scripts],
                self.purpose
            )

    def vectorize(self, script: str) -> Dict:
        """Vectorize a script and track it for later storage"""
        vector_result = self.script2vec.python_to_vector(script)
        self.vectorized_scripts.append((script, vector_result))
        return vector_result

    async def find_similar(self, script: str, threshold: float = 0.8) -> List[Dict]:
        """Find similar scripts in CTRM"""
        return await self.ctrm_interface.find_similar_code_in_ctrm(script, threshold)

    async def improve(self, script: str, improvement_type: str = "optimize") -> Dict:
        """Improve a script using CTRM"""
        return await self.ctrm_interface.improve_script_via_ctrm(script, improvement_type)

def get_vector_metadata(obj) -> Optional[Dict]:
    """Get vector metadata from a decorated object"""
    return getattr(obj, '_vector_metadata', None)

def get_evolution_history(obj) -> Optional[List[Dict]]:
    """Get evolution history from a decorated object"""
    return getattr(obj, '_evolution_history', None)