"""
Script2Vec - Python to Vector Conversion Library for CTRM Integration

This library provides comprehensive tools for converting Python scripts directly
to semantic vectors, enabling seamless integration with CTRM systems.
"""

from .script2vec import Script2Vec
from .ctrm_script_interface import CTRMScriptInterface
from .decorators import (
    auto_vectorize,
    vectorize_class,
    track_script_evolution,
    ScriptVectorizer,
    get_vector_metadata,
    get_evolution_history
)
from .cli import Script2VecCLI, main as cli_main
from .web_interface import Script2VecWebInterface, create_web_interface

# Version information
__version__ = "1.0.0"
__author__ = "CTRM Architect"
__license__ = "MIT"
__description__ = "Python to Vector Conversion Library for CTRM Integration"

# Main exports
__all__ = [
    # Core classes
    'Script2Vec',
    'CTRMScriptInterface',
    'Script2VecCLI',
    'Script2VecWebInterface',

    # Decorators
    'auto_vectorize',
    'vectorize_class',
    'track_script_evolution',
    'ScriptVectorizer',
    'get_vector_metadata',
    'get_evolution_history',

    # Utility functions
    'create_web_interface',
    'cli_main',

    # Version info
    '__version__',
    '__author__',
    '__license__',
    '__description__'
]

def create_script2vec(embedding_model=None):
    """Create a Script2Vec instance"""
    return Script2Vec(embedding_model=embedding_model)

def create_ctrm_interface(ctrm_url="http://localhost:8000", ctrm_manager=None):
    """Create a CTRM script interface"""
    return CTRMScriptInterface(ctrm_url=ctrm_url, ctrm_manager=ctrm_manager)

def create_cli():
    """Create a CLI instance"""
    return Script2VecCLI()

# Convenience functions for common operations
async def script_to_vector(script: str, strategy: str = "hybrid"):
    """Convert script to vector"""
    s2v = Script2Vec()
    return s2v.python_to_vector(script, strategy=f"{strategy}_embedding")

async def file_to_vector(filepath: str, strategy: str = "hybrid"):
    """Convert file to vector"""
    s2v = Script2Vec()
    return s2v.embed_file(filepath)

async def script_to_ctrm(script: str, purpose: str = "store", **kwargs):
    """Convert script to vector and store in CTRM"""
    interface = CTRMScriptInterface()
    return await interface.script_to_ctrm(script, purpose, **kwargs)

async def find_similar_code(script: str, threshold: float = 0.8):
    """Find similar code in CTRM"""
    interface = CTRMScriptInterface()
    return await interface.find_similar_code_in_ctrm(script, threshold)

async def improve_script(script: str, improvement_type: str = "optimize"):
    """Improve script using CTRM"""
    interface = CTRMScriptInterface()
    return await interface.improve_script_via_ctrm(script, improvement_type)