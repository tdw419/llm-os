import asyncio
import json
import hashlib
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import inspect
import os
from pathlib import Path

# Import the core Script2Vec class
from .script2vec import Script2Vec

class CTRMScriptInterface:
    """Interface between Python scripts and CTRM vectors"""

    def __init__(self, ctrm_url="http://localhost:8000", ctrm_manager=None):
        self.script2vec = Script2Vec()
        self.ctrm_url = ctrm_url
        self.ctrm_manager = ctrm_manager
        self.session = None
        self.script_cache = {}

    async def script_to_ctrm(self, script: str,
                            purpose: str = "store",
                            **kwargs) -> Dict:
        """
        Convert Python script to vector and send to CTRM

        Args:
            script: Python script as string
            purpose: Purpose of storing this script
            **kwargs: Additional metadata

        Returns:
            Dictionary with CTRM storage results
        """
        # Convert script to vector
        ctrm_payload = self.script2vec.to_ctrm_format(script, kwargs)

        # Send to CTRM
        result = await self._send_to_ctrm(ctrm_payload, purpose)

        return {
            "script_hash": ctrm_payload["script_hash"],
            "ctrm_vector_hash": result.get("hash"),
            "vector": ctrm_payload["vector"],
            "status": "submitted",
            "metadata": ctrm_payload["metadata"],
            "concepts": ctrm_payload.get("concepts", [])
        }

    async def file_to_ctrm(self, filepath: str,
                          purpose: str = "store") -> Dict:
        """Convert Python file to CTRM vector"""
        with open(filepath, 'r') as f:
            script = f.read()

        return await self.script_to_ctrm(
            script,
            purpose,
            source_file=filepath,
            file_size=len(script),
            file_timestamp=datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
        )

    async def function_to_ctrm(self, func) -> Dict:
        """Convert a Python function to CTRM vector"""
        script = inspect.getsource(func)

        return await self.script_to_ctrm(
            script,
            purpose="store",
            function_name=func.__name__,
            module=func.__module__,
            function_qualname=func.__qualname__,
            function_doc=func.__doc__
        )

    async def find_similar_code_in_ctrm(self, script: str,
                                       threshold: float = 0.8) -> List[Dict]:
        """Find similar code already in CTRM"""
        # Convert to vector
        vector = self.script2vec.python_to_vector(script)["vector"]

        # Query CTRM for similar vectors
        results = await self._query_ctrm_similar(vector, threshold)

        # Map back to script context
        return [
            {
                "ctrm_id": r["id"],
                "similarity": r["similarity"],
                "likely_purpose": self._infer_code_purpose(r["metadata"]),
                "suggested_use": self._suggest_reuse(r["similarity"]),
                "metadata": r["metadata"],
                "vector_hash": r.get("vector_hash")
            }
            for r in results
        ]

    async def improve_script_via_ctrm(self, script: str,
                                     improvement_type: str = "optimize") -> Dict:
        """
        Use CTRM to improve a Python script

        Args:
            script: Python script to improve
            improvement_type: Type of improvement ('optimize', 'refactor', 'document', etc.)

        Returns:
            Dictionary with improvement suggestions
        """
        # Convert to vector
        vector_payload = self.script2vec.to_ctrm_format(
            script,
            {"improvement_type": improvement_type}
        )

        # Ask CTRM to improve
        result = await self._request_improvement(
            vector_payload["vector"],
            improvement_type
        )

        # Map improved vector back to code suggestions
        suggestions = await self._vector_to_code_suggestions(
            result["improved_vector"],
            script
        )

        return {
            "original_script": script,
            "improvement_type": improvement_type,
            "suggestions": suggestions,
            "improved_vector_hash": result["improved_hash"],
            "confidence": result.get("confidence", 0.0),
            "metadata": result.get("metadata", {})
        }

    async def analyze_script_quality(self, script: str) -> Dict:
        """Analyze script quality using CTRM"""
        # Convert to vector
        vector_payload = self.script2vec.to_ctrm_format(script)

        # Send to CTRM for quality analysis
        analysis_result = await self._analyze_vector_quality(vector_payload["vector"])

        return {
            "script_hash": vector_payload["script_hash"],
            "quality_score": analysis_result["quality_score"],
            "analysis": analysis_result["analysis"],
            "recommendations": analysis_result.get("recommendations", []),
            "vector_quality": analysis_result.get("vector_quality", {})
        }

    async def track_script_evolution(self, script: str,
                                   parent_script: str = None) -> Dict:
        """Track script evolution using CTRM"""
        # Convert scripts to vectors
        current_vector = self.script2vec.python_to_vector(script)["vector"]
        parent_vector = None

        if parent_script:
            parent_vector = self.script2vec.python_to_vector(parent_script)["vector"]

        # Send to CTRM for evolution tracking
        evolution_result = await self._track_evolution(current_vector, parent_vector)

        return {
            "script_hash": self.script2vec._hash_script(script),
            "evolution_id": evolution_result["evolution_id"],
            "generation": evolution_result["generation"],
            "changes": evolution_result.get("changes", []),
            "improvement_score": evolution_result.get("improvement_score", 0.0)
        }

    async def _send_to_ctrm(self, payload: Dict, purpose: str) -> Dict:
        """Send vector payload to CTRM"""
        if self.ctrm_manager:
            # Use direct CTRM manager integration
            return await self._send_to_ctrm_manager(payload, purpose)
        else:
            # Use HTTP API (mock implementation for now)
            return await self._send_to_ctrm_api(payload, purpose)

    async def _send_to_ctrm_manager(self, payload: Dict, purpose: str) -> Dict:
        """Send to CTRM manager directly"""
        # Create CTRM truth for this script
        truth = await self.ctrm_manager.create_truth(
            statement=f"Python script stored: {payload.get('metadata', {}).get('description', 'unnamed script')}",
            confidence=0.9,
            vector=payload["vector"],
            metadata={
                "script_hash": payload["script_hash"],
                "source": "python_script",
                "purpose": purpose,
                "concepts": payload.get("concepts", []),
                "script_metadata": payload.get("metadata", {}),
                "storage_timestamp": datetime.now().isoformat()
            }
        )

        return {
            "hash": truth.id,
            "vector_hash": payload["script_hash"],
            "confidence": truth.confidence,
            "status": "stored",
            "truth_id": truth.id
        }

    async def _send_to_ctrm_api(self, payload: Dict, purpose: str) -> Dict:
        """Send to CTRM via API (mock implementation)"""
        # In a real implementation, this would make an HTTP request
        # For now, simulate the response
        import time
        import random

        # Simulate API call delay
        await asyncio.sleep(0.1)

        # Generate mock response
        vector_hash = hashlib.md5(str(payload["vector"]).encode()).hexdigest()[:12]
        truth_id = f"truth_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

        return {
            "hash": vector_hash,
            "truth_id": truth_id,
            "confidence": 0.9,
            "status": "stored",
            "timestamp": datetime.now().isoformat()
        }

    async def _query_ctrm_similar(self, vector: List[float],
                                 threshold: float = 0.8) -> List[Dict]:
        """Query CTRM for similar vectors"""
        if self.ctrm_manager:
            # Use CTRM manager's vector interface
            return await self._query_ctrm_manager_similar(vector, threshold)
        else:
            # Mock response
            return await self._query_ctrm_api_similar(vector, threshold)

    async def _query_ctrm_manager_similar(self, vector: List[float],
                                         threshold: float = 0.8) -> List[Dict]:
        """Query CTRM manager for similar vectors"""
        # Use the vector interface to find similar vectors
        similar_result = await self.ctrm_manager.vector_interface.llm_find_similar_vectors(
            query_vector=vector,
            min_similarity=threshold,
            limit=10
        )

        # Format results
        formatted_results = []
        for similar_vector in similar_result.get("similar_vectors", []):
            formatted_results.append({
                "id": similar_vector["vector_hash"],
                "similarity": similar_vector["similarity"],
                "metadata": similar_vector["llm_metadata"],
                "vector_hash": similar_vector["vector_hash"],
                "relationship_strength": similar_vector["relationship_strength"]
            })

        return formatted_results

    async def _query_ctrm_api_similar(self, vector: List[float],
                                     threshold: float = 0.8) -> List[Dict]:
        """Query CTRM API for similar vectors (mock)"""
        # Simulate API call
        await asyncio.sleep(0.2)

        # Mock some similar results
        import random
        mock_results = []

        # Generate 2-5 mock similar vectors
        num_results = random.randint(2, 5)
        for i in range(num_results):
            similarity = random.uniform(threshold, 0.98)
            mock_results.append({
                "id": f"vector_{hashlib.md5(str(i).encode()).hexdigest()[:8]}",
                "similarity": similarity,
                "metadata": {
                    "source": "python_script",
                    "description": f"Similar Python script {i+1}",
                    "purpose": f"optimization_function_{i}",
                    "timestamp": datetime.now().isoformat()
                },
                "vector_hash": f"hash_{i}",
                "relationship_strength": self.script2vec._match_strength(similarity)
            })

        return mock_results

    async def _request_improvement(self, vector: List[float],
                                  improvement_type: str) -> Dict:
        """Request script improvement from CTRM"""
        if self.ctrm_manager:
            return await self._request_improvement_manager(vector, improvement_type)
        else:
            return await self._request_improvement_api(vector, improvement_type)

    async def _request_improvement_manager(self, vector: List[float],
                                         improvement_type: str) -> Dict:
        """Request improvement using CTRM manager"""
        # Use vector protocol for improvement
        improvement_result = await self.ctrm_manager.vector_interface.llm_vector_operation(
            operation="improve_vector",
            vector=vector,
            improvement_type=improvement_type,
            context={
                "source": "python_script",
                "improvement_goal": improvement_type
            }
        )

        return {
            "improved_vector": improvement_result.get("improved_vector", vector),
            "improved_hash": improvement_result.get("improved_vector_hash", ""),
            "confidence": improvement_result.get("confidence", 0.7),
            "metadata": improvement_result.get("metadata", {}),
            "changes": improvement_result.get("changes", [])
        }

    async def _request_improvement_api(self, vector: List[float],
                                     improvement_type: str) -> Dict:
        """Request improvement via API (mock)"""
        await asyncio.sleep(0.3)

        # Mock improvement - slightly modify the vector
        import numpy as np
        improved_vector = np.array(vector) * 1.01  # Slight improvement
        improved_vector = improved_vector.tolist()

        return {
            "improved_vector": improved_vector,
            "improved_hash": hashlib.md5(str(improved_vector).encode()).hexdigest()[:12],
            "confidence": 0.85,
            "metadata": {
                "improvement_type": improvement_type,
                "changes": ["optimized_control_flow", "improved_variable_names"],
                "timestamp": datetime.now().isoformat()
            }
        }

    async def _vector_to_code_suggestions(self, vector: List[float],
                                         original_script: str) -> List[Dict]:
        """Convert improved vector back to code suggestions"""
        # Analyze the vector to generate suggestions
        suggestions = []

        # Basic suggestions based on vector analysis
        suggestions.append({
            "type": "general_optimization",
            "description": "Optimize function calls and reduce complexity",
            "priority": "high",
            "code": None
        })

        suggestions.append({
            "type": "documentation",
            "description": "Add comprehensive docstrings and comments",
            "priority": "medium",
            "code": None
        })

        # Analyze the original script for specific suggestions
        try:
            tree = ast.parse(original_script)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            if functions:
                suggestions.append({
                    "type": "function_optimization",
                    "description": f"Optimize functions: {', '.join(functions)}",
                    "priority": "high",
                    "code": None
                })
        except:
            pass

        return suggestions

    async def _analyze_vector_quality(self, vector: List[float]) -> Dict:
        """Analyze vector quality using CTRM"""
        if self.ctrm_manager:
            # Use vector analytics
            quality_result = await self.ctrm_manager.vector_analytics.analyze_vector_for_llm(
                vector, "quality_analysis"
            )

            return {
                "quality_score": quality_result.get("quality_score", 0.7),
                "analysis": quality_result.get("analysis", {}),
                "recommendations": quality_result.get("recommendations", []),
                "vector_quality": quality_result.get("vector_metrics", {})
            }
        else:
            # Mock analysis
            await asyncio.sleep(0.2)

            return {
                "quality_score": 0.85,
                "analysis": {
                    "norm": 10.5,
                    "dimensionality": len(vector),
                    "sparsity": 0.3
                },
                "recommendations": ["improve_vector_coherence", "reduce_sparsity"],
                "vector_quality": {
                    "coherence": 0.8,
                    "complexity": 0.7,
                    "uniqueness": 0.9
                }
            }

    async def _track_evolution(self, current_vector: List[float],
                              parent_vector: List[float] = None) -> Dict:
        """Track vector evolution using CTRM"""
        if self.ctrm_manager:
            # Use vector evolution tracker
            evolution_result = await self.ctrm_manager.vector_interface.track_vector_evolution(
                current_vector, parent_vector
            )

            return {
                "evolution_id": evolution_result.get("vector_hash", ""),
                "generation": evolution_result.get("generations", 1),
                "changes": evolution_result.get("evolution_history", []),
                "improvement_score": 0.85
            }
        else:
            # Mock evolution tracking
            await asyncio.sleep(0.2)

            return {
                "evolution_id": hashlib.md5(str(current_vector).encode()).hexdigest()[:12],
                "generation": 2 if parent_vector else 1,
                "changes": [
                    "improved_structure",
                    "enhanced_semantics",
                    "optimized_execution_flow"
                ],
                "improvement_score": 0.85
            }

    def _infer_code_purpose(self, metadata: Dict) -> str:
        """Infer the purpose of the code from metadata"""
        purpose = metadata.get("purpose", "")

        if "optimization" in purpose.lower():
            return "performance_optimization"
        elif "refactor" in purpose.lower():
            return "code_refactoring"
        elif "document" in purpose.lower():
            return "documentation"
        elif "test" in purpose.lower():
            return "testing"
        else:
            return "general_purpose"

    def _suggest_reuse(self, similarity: float) -> str:
        """Suggest how to reuse similar code"""
        if similarity > 0.9:
            return "direct_reuse"
        elif similarity > 0.8:
            return "adapt_with_minor_changes"
        elif similarity > 0.6:
            return "use_as_reference"
        else:
            return "review_for_inspiration"

    async def batch_process_scripts(self, scripts: List[str],
                                  purpose: str = "store") -> Dict:
        """Batch process multiple scripts"""
        results = []

        for i, script in enumerate(scripts):
            try:
                result = await self.script_to_ctrm(script, purpose)
                results.append(result)
            except Exception as e:
                results.append({
                    "script_hash": f"error_{i}",
                    "error": str(e),
                    "status": "failed"
                })

        return {
            "processed_scripts": len(results),
            "successful": len([r for r in results if r["status"] == "submitted"]),
            "failed": len([r for r in results if r["status"] == "failed"]),
            "results": results
        }

    async def monitor_directory(self, dirpath: str,
                              purpose: str = "monitor") -> Dict:
        """Monitor a directory of Python files"""
        directory_result = self.script2vec.embed_directory(dirpath)

        # Store directory vector in CTRM
        directory_vector_result = await self.script_to_ctrm(
            f"# Directory: {dirpath}\n# Files: {len(directory_result['file_vectors'])}",
            purpose,
            directory_path=dirpath,
            file_count=directory_result["file_count"],
            directory_vector=directory_result["directory_vector"]
        )

        # Store individual files
        file_results = []
        for filepath, file_vector in directory_result["file_vectors"].items():
            try:
                file_result = await self.file_to_ctrm(
                    os.path.join(dirpath, filepath),
                    purpose
                )
                file_results.append(file_result)
            except Exception as e:
                file_results.append({
                    "filepath": filepath,
                    "error": str(e),
                    "status": "failed"
                })

        return {
            "directory_hash": directory_vector_result["script_hash"],
            "directory_vector_hash": directory_vector_result["ctrm_vector_hash"],
            "file_count": len(file_results),
            "successful_files": len([r for r in file_results if r.get("status") == "submitted"]),
            "file_results": file_results
        }

    def clear_cache(self):
        """Clear the script cache"""
        self.script_cache = {}
        self.script2vec.clear_cache()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "script_cache": len(self.script_cache),
            "vector_cache": self.script2vec.get_cache_stats()
        }