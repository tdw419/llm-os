#!/usr/bin/env python3
"""
CTRM Integration Layer for LLM OS Builder

This module provides seamless integration between the LLM OS Builder
and the CTRM (Confidence-Based Truth Management) system.
"""

import aiohttp
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import hashlib
from datetime import datetime

@dataclass
class CTRMTruth:
    """CTRM truth entry"""
    id: str
    statement: str
    confidence: float
    vector: List[float]
    metadata: Dict[str, Any]

class CTRMInterface:
    """Interface to CTRM system for vector storage"""

    def __init__(self, ctrm_url: str = "http://localhost:8000"):
        self.ctrm_url = ctrm_url
        self.session = None

    async def connect(self):
        """Establish connection to CTRM"""
        self.session = aiohttp.ClientSession()

    async def disconnect(self):
        """Close connection"""
        if self.session:
            await self.session.close()

    async def store_component_truth(self, component) -> str:
        """Store component as CTRM truth"""
        if not self.session:
            await self.connect()

        truth_data = {
            "statement": f"OS Component: {component.name} - {component.vector.semantic_summary}",
            "confidence": 0.8 if component.execution_results.get("tests_passed", 0) > 0 else 0.5,
            "vector": component.vector.vector,
            "metadata": {
                "type": "os_component",
                "component_id": component.id,
                "component_name": component.name,
                "code_hash": component.vector.script_hash,
                "requirements": component.requirements,
                "dependencies": component.dependencies,
                "tests_passed": component.execution_results.get("tests_passed", 0),
                "tests_total": component.execution_results.get("tests_total", 0),
                "concepts": component.vector.concepts[:20],  # First 20 concepts
                "embedding_type": component.vector.embedding_type,
                "created_at": component.created_at
            }
        }

        try:
            async with self.session.post(
                f"{self.ctrm_url}/truths",
                json=truth_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("truth_id", "unknown")
                else:
                    print(f"CTRM store failed: {response.status}")
                    return "store_failed"
        except Exception as e:
            print(f"CTRM connection error: {e}")
            return "connection_error"

    async def query_similar_truths(self, vector: List[float],
                                  threshold: float = 0.7) -> List[Dict]:
        """Query CTRM for similar truths"""
        if not self.session:
            await self.connect()

        query_data = {
            "query_vector": vector,
            "threshold": threshold,
            "limit": 10
        }

        try:
            async with self.session.post(
                f"{self.ctrm_url}/truths/query",
                json=query_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("matches", [])
                else:
                    return []
        except:
            return []

    async def store_execution_result(self, component_id: str,
                                   result: Dict[str, Any]) -> bool:
        """Store execution results in CTRM"""
        if not self.session:
            await self.connect()

        execution_data = {
            "component_id": component_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }

        try:
            async with self.session.post(
                f"{self.ctrm_url}/executions",
                json=execution_data
            ) as response:
                return response.status == 200
        except:
            return False

    async def store_component_vector(self, component) -> str:
        """Store component vector in CTRM vector database"""
        if not self.session:
            await self.connect()

        vector_data = {
            "vector": component.vector.vector,
            "metadata": {
                "component_id": component.id,
                "component_name": component.name,
                "type": "os_component",
                "concepts": component.vector.concepts,
                "dependencies": component.dependencies,
                "created_at": component.created_at
            }
        }

        try:
            async with self.session.post(
                f"{self.ctrm_url}/vectors",
                json=vector_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("vector_id", "unknown")
                else:
                    return "store_failed"
        except Exception as e:
            print(f"CTRM vector store error: {e}")
            return "connection_error"

    async def find_similar_components(self, requirement: str) -> List[Dict]:
        """Find similar components in CTRM"""
        if not self.session:
            await self.connect()

        # Convert requirement to vector
        from script2vec.script2vec import Script2Vec
        s2v = Script2Vec()
        vector_data = s2v.python_to_vector(f"# Requirement: {requirement}")
        requirement_vector = vector_data["vector"]

        # Query CTRM
        similar_truths = await self.query_similar_truths(requirement_vector, threshold=0.6)

        # Filter for OS components
        similar_components = []
        for truth in similar_truths:
            metadata = truth.get("metadata", {})
            if metadata.get("type") == "os_component":
                similar_components.append({
                    "component_id": metadata.get("component_id"),
                    "component_name": metadata.get("component_name"),
                    "similarity": truth.get("similarity", 0),
                    "concepts": metadata.get("concepts", []),
                    "dependencies": metadata.get("dependencies", [])
                })

        return similar_components

    async def log_os_building_event(self, event_type: str, data: Dict) -> bool:
        """Log OS building events in CTRM"""
        if not self.session:
            await self.connect()

        event_data = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }

        try:
            async with self.session.post(
                f"{self.ctrm_url}/events",
                json=event_data
            ) as response:
                return response.status == 200
        except:
            return False

    async def get_os_architecture(self) -> Dict:
        """Get current OS architecture from CTRM"""
        if not self.session:
            await self.connect()

        try:
            async with self.session.get(
                f"{self.ctrm_url}/architecture"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": "Failed to get architecture"}
        except:
            return {"error": "Connection failed"}

    async def update_os_architecture(self, architecture: Dict) -> bool:
        """Update OS architecture in CTRM"""
        if not self.session:
            await self.connect()

        try:
            async with self.session.post(
                f"{self.ctrm_url}/architecture",
                json=architecture
            ) as response:
                return response.status == 200
        except:
            return False

    async def get_component_history(self, component_id: str) -> List[Dict]:
        """Get history of a component from CTRM"""
        if not self.session:
            await self.connect()

        try:
            async with self.session.get(
                f"{self.ctrm_url}/components/{component_id}/history"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
        except:
            return []

    async def store_os_composition(self, composition: Dict) -> str:
        """Store OS composition in CTRM"""
        if not self.session:
            await self.connect()

        composition_data = {
            "composition": composition,
            "timestamp": datetime.now().isoformat()
        }

        try:
            async with self.session.post(
                f"{self.ctrm_url}/compositions",
                json=composition_data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("composition_id", "unknown")
                else:
                    return "store_failed"
        except Exception as e:
            print(f"CTRM composition store error: {e}")
            return "connection_error"

    async def get_os_compositions(self) -> List[Dict]:
        """Get all OS compositions from CTRM"""
        if not self.session:
            await self.connect()

        try:
            async with self.session.get(
                f"{self.ctrm_url}/compositions"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return []
        except:
            return []

    async def get_os_health(self) -> Dict:
        """Get OS health metrics from CTRM"""
        if not self.session:
            await self.connect()

        try:
            async with self.session.get(
                f"{self.ctrm_url}/health"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unknown"}
        except:
            return {"status": "connection_error"}

    async def backup_os_state(self) -> bool:
        """Backup current OS state to CTRM"""
        if not self.session:
            await self.connect()

        try:
            async with self.session.post(
                f"{self.ctrm_url}/backup"
            ) as response:
                return response.status == 200
        except:
            return False

    async def restore_os_state(self, backup_id: str) -> bool:
        """Restore OS state from CTRM backup"""
        if not self.session:
            await self.connect()

        try:
            async with self.session.post(
                f"{self.ctrm_url}/restore/{backup_id}"
            ) as response:
                return response.status == 200
        except:
            return False