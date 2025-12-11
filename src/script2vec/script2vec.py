import ast
import inspect
import hashlib
import json
from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime
import os
from pathlib import Path

class Script2Vec:
    """Convert Python scripts directly to semantic vectors"""

    def __init__(self, embedding_model=None):
        self.model = embedding_model
        self.vector_cache = {}
        self.default_vector_dim = 1536  # Standard embedding dimension

    def python_to_vector(self, script: str,
                        strategy: str = "semantic") -> Dict:
        """
        Convert Python script to vector representation

        Args:
            script: Python script as string
            strategy: Embedding strategy ('semantic', 'ast', 'execution', 'hybrid')

        Returns:
            Dictionary containing vector and metadata
        """
        strategies = {
            "semantic": self._semantic_embedding,
            "ast": self._ast_embedding,
            "execution": self._execution_embedding,
            "hybrid": self._hybrid_embedding
        }

        # Check cache first
        script_hash = self._hash_script(script)
        cache_key = f"{script_hash}_{strategy}"
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]

        result = strategies[strategy](script)
        self.vector_cache[cache_key] = result
        return result

    def _semantic_embedding(self, script: str) -> Dict:
        """Embed the semantic meaning of the code"""
        # Extract semantic concepts from code
        concepts = self._extract_semantic_concepts(script)

        # Convert to vector (using your favorite embedding method)
        if self.model:
            vector = self.model.embed(" ".join(concepts))
        else:
            # Fallback: simple concept-based vector
            vector = self._concept_vector(concepts)

        return {
            "vector": vector,
            "type": "semantic",
            "concepts": concepts,
            "script_hash": self._hash_script(script),
            "dimensionality": len(vector),
            "strategy": "semantic_embedding"
        }

    def _ast_embedding(self, script: str) -> Dict:
        """Embed AST structure as vector"""
        try:
            tree = ast.parse(script)

            # Extract AST features
            features = self._extract_ast_features(tree)

            # Convert to vector
            vector = self._features_to_vector(features)

            return {
                "vector": vector,
                "type": "ast",
                "features": features,
                "node_count": len(list(ast.walk(tree))),
                "script_hash": self._hash_script(script),
                "dimensionality": len(vector),
                "strategy": "ast_embedding"
            }
        except SyntaxError:
            # Fallback to semantic embedding if not valid Python
            return self._semantic_embedding(script)

    def _execution_embedding(self, script: str) -> Dict:
        """Embed based on execution patterns"""
        # Analyze execution flow
        flow = self._analyze_execution_flow(script)

        # Create vector from execution patterns
        vector = self._execution_pattern_to_vector(flow)

        return {
            "vector": vector,
            "type": "execution",
            "flow_pattern": flow,
            "script_hash": self._hash_script(script),
            "dimensionality": len(vector),
            "strategy": "execution_embedding"
        }

    def _hybrid_embedding(self, script: str) -> Dict:
        """Combine all embedding strategies"""
        semantic = self._semantic_embedding(script)
        ast_emb = self._ast_embedding(script)
        execution = self._execution_embedding(script)

        # Combine vectors
        combined_vector = np.mean([
            semantic["vector"],
            ast_emb["vector"],
            execution["vector"]
        ], axis=0)

        return {
            "vector": combined_vector.tolist(),
            "type": "hybrid",
            "components": {
                "semantic": semantic,
                "ast": ast_emb,
                "execution": execution
            },
            "script_hash": self._hash_script(script),
            "dimensionality": len(combined_vector),
            "strategy": "hybrid_embedding"
        }

    def _extract_semantic_concepts(self, script: str) -> List[str]:
        """Extract key concepts from Python code"""
        concepts = []

        # 1. Extract imports (libraries used)
        imports = self._extract_imports(script)
        concepts.extend(imports)

        # 2. Extract function/class names
        entities = self._extract_entities(script)
        concepts.extend(entities)

        # 3. Extract key variable names
        variables = self._extract_key_variables(script)
        concepts.extend(variables)

        # 4. Extract comments/docstrings
        comments = self._extract_comments(script)
        concepts.extend(comments)

        # 5. Extract string literals (often contain semantics)
        strings = self._extract_string_literals(script)
        concepts.extend(strings)

        return list(set(concepts))  # Deduplicate

    def _extract_imports(self, script: str) -> List[str]:
        """Extract imported libraries"""
        imports = []
        try:
            tree = ast.parse(script)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import:{alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(f"from:{node.module}")
        except:
            pass
        return imports

    def _extract_entities(self, script: str) -> List[str]:
        """Extract function and class names"""
        entities = []
        try:
            tree = ast.parse(script)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entities.append(f"function:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    entities.append(f"class:{node.name}")
        except:
            pass
        return entities

    def _extract_key_variables(self, script: str) -> List[str]:
        """Extract key variable names"""
        variables = []
        try:
            tree = ast.parse(script)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append(f"var:{target.id}")
                elif isinstance(node, ast.AnnAssign):
                    if isinstance(node.target, ast.Name):
                        variables.append(f"var:{node.target.id}")
        except:
            pass
        return variables

    def _extract_comments(self, script: str) -> List[str]:
        """Extract comments and docstrings"""
        comments = []
        lines = script.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                comments.append(f"comment:{line[1:].strip()}")
            elif line.startswith('"""') or line.startswith("'''"):
                comments.append(f"docstring:{line.strip()}")

        return comments

    def _extract_string_literals(self, script: str) -> List[str]:
        """Extract string literals"""
        strings = []
        try:
            tree = ast.parse(script)
            for node in ast.walk(tree):
                if isinstance(node, ast.Str):  # Python 3.7 and earlier
                    strings.append(f"string:{node.s}")
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):  # Python 3.8+
                    strings.append(f"string:{node.value}")
        except:
            pass
        return strings

    def _extract_ast_features(self, tree: ast.AST) -> Dict:
        """Extract structural features from AST"""
        features = {
            "imports": 0,
            "functions": 0,
            "classes": 0,
            "loops": 0,
            "conditionals": 0,
            "assignments": 0,
            "calls": 0,
            "decorators": 0,
            "exceptions": 0,
            "returns": 0
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                features["imports"] += 1
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features["functions"] += 1
                if len(node.decorator_list) > 0:
                    features["decorators"] += len(node.decorator_list)
            elif isinstance(node, ast.ClassDef):
                features["classes"] += 1
            elif isinstance(node, (ast.For, ast.While, ast.AsyncFor)):
                features["loops"] += 1
            elif isinstance(node, (ast.If, ast.IfExp)):
                features["conditionals"] += 1
            elif isinstance(node, ast.Assign):
                features["assignments"] += 1
            elif isinstance(node, ast.Call):
                features["calls"] += 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                features["exceptions"] += 1
            elif isinstance(node, ast.Return):
                features["returns"] += 1

        return features

    def _concept_vector(self, concepts: List[str]) -> List[float]:
        """Simple concept-based vector generation"""
        # Create a vector based on concept presence
        vector = [0.0] * self.default_vector_dim

        for concept in concepts:
            # Hash concept to determine which dimensions to activate
            concept_hash = hashlib.md5(concept.encode()).hexdigest()

            # Use first 8 hex chars to determine dimensions
            for i in range(0, 8, 2):
                dim = int(concept_hash[i:i+2], 16) % self.default_vector_dim
                vector[dim] += 0.1  # Slight activation

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = (vector / norm).tolist()

        return vector

    def _features_to_vector(self, features: Dict) -> List[float]:
        """Convert AST features to vector"""
        # Create feature vector
        vector = [0.0] * self.default_vector_dim

        # Map features to specific dimensions
        feature_names = list(features.keys())
        for i, feature_name in enumerate(feature_names):
            # Hash feature name to get base dimension
            feature_hash = hashlib.md5(feature_name.encode()).hexdigest()
            base_dim = int(feature_hash[:4], 16) % (self.default_vector_dim // 2)

            # Distribute feature value across multiple dimensions
            value = features[feature_name]
            for offset in range(3):  # Spread across 3 dimensions
                dim = (base_dim + offset) % self.default_vector_dim
                vector[dim] += value * 0.1  # Scale down

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = (vector / norm).tolist()

        return vector

    def _analyze_execution_flow(self, script: str) -> Dict:
        """Analyze execution flow patterns"""
        flow = {
            "control_structures": 0,
            "function_calls": 0,
            "error_handling": 0,
            "complexity": 0,
            "depth": 0,
            "branches": 0
        }

        try:
            tree = ast.parse(script)

            # Count control structures and complexity
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.IfExp, ast.For, ast.While, ast.Try)):
                    flow["control_structures"] += 1
                    flow["complexity"] += 1

                if isinstance(node, ast.Call):
                    flow["function_calls"] += 1

                if isinstance(node, (ast.Try, ast.ExceptHandler)):
                    flow["error_handling"] += 1

                # Calculate depth
                current_depth = 0
                current_node = node
                while hasattr(current_node, 'parent'):
                    current_depth += 1
                    current_node = current_node.parent
                flow["depth"] = max(flow["depth"], current_depth)

            # Estimate branches
            flow["branches"] = flow["control_structures"] * 2

        except:
            pass

        return flow

    def _execution_pattern_to_vector(self, flow: Dict) -> List[float]:
        """Convert execution patterns to vector"""
        # Create execution pattern vector
        vector = [0.0] * self.default_vector_dim

        # Map execution patterns to dimensions
        pattern_names = list(flow.keys())
        for i, pattern_name in enumerate(pattern_names):
            # Hash pattern name to get base dimension
            pattern_hash = hashlib.md5(pattern_name.encode()).hexdigest()
            base_dim = int(pattern_hash[:4], 16) % (self.default_vector_dim // 2) + (self.default_vector_dim // 2)

            # Distribute pattern value
            value = flow[pattern_name]
            for offset in range(3):  # Spread across 3 dimensions
                dim = (base_dim + offset) % self.default_vector_dim
                vector[dim] += value * 0.1  # Scale down

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = (vector / norm).tolist()

        return vector

    def _hash_script(self, script: str) -> str:
        """Create unique hash for script"""
        return hashlib.md5(script.encode()).hexdigest()[:12]

    def embed_file(self, filepath: str) -> Dict:
        """Embed an entire Python file"""
        with open(filepath, 'r') as f:
            script = f.read()

        return self.python_to_vector(script)

    def embed_directory(self, dirpath: str) -> Dict:
        """Embed all Python files in a directory"""
        vectors = {}

        for filepath in Path(dirpath).rglob("*.py"):
            relative_path = str(filepath.relative_to(dirpath))
            vectors[relative_path] = self.embed_file(filepath)

        # Create directory-level vector (average of all files)
        all_vectors = [v["vector"] for v in vectors.values()]
        if all_vectors:
            directory_vector = np.mean(all_vectors, axis=0).tolist()
        else:
            directory_vector = [0.0] * self.default_vector_dim

        return {
            "directory_vector": directory_vector,
            "file_vectors": vectors,
            "file_count": len(vectors)
        }

    def find_similar_scripts(self, script: str,
                            script_database: Dict[str, List[float]],
                            threshold: float = 0.7) -> List[Dict]:
        """Find similar scripts in database"""
        query_vec = self.python_to_vector(script)["vector"]

        results = []
        for script_id, vector in script_database.items():
            similarity = self.cosine_similarity(query_vec, vector)

            if similarity >= threshold:
                results.append({
                    "script_id": script_id,
                    "similarity": float(similarity),
                    "match_strength": self._match_strength(similarity)
                })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

    def _match_strength(self, similarity: float) -> str:
        """Map similarity to strength description"""
        if similarity > 0.9:
            return "very_strong"
        elif similarity > 0.8:
            return "strong"
        elif similarity > 0.6:
            return "moderate"
        elif similarity > 0.4:
            return "weak"
        else:
            return "very_weak"

    def to_ctrm_format(self, script: str, metadata: Dict = None) -> Dict:
        """Convert script to CTRM-ready format"""
        embedding = self.python_to_vector(script)

        return {
            "vector": embedding["vector"],
            "source": "python_script",
            "script_hash": embedding.get("script_hash"),
            "concepts": embedding.get("concepts", []),
            "metadata": {
                "embedding_type": embedding.get("type"),
                "lines_of_code": len(script.split('\n')),
                "extraction_timestamp": datetime.now().isoformat(),
                **(metadata or {})
            }
        }

    def clear_cache(self):
        """Clear the vector cache"""
        self.vector_cache = {}

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cached_vectors": len(self.vector_cache),
            "cache_size_mb": sum(len(str(v)) for v in self.vector_cache.values()) / (1024 * 1024)
        }