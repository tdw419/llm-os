#!/usr/bin/env python3
"""
LLM OS Builder Core - Self-building LLM Operating System

This module implements the core functionality for building an LLM OS where
LLMs write their own operating system components using Python scripts
that auto-convert to vectors.
"""

import asyncio
import json
import os
import sys
import tempfile
import subprocess
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import inspect
import traceback
import hashlib
import numpy as np

# Import Script2Vec components
from script2vec.script2vec import Script2Vec
from script2vec.ctrm_script_interface import CTRMScriptInterface

@dataclass
class VectorEmbedding:
    """Container for vector embedding with metadata"""
    vector: List[float]
    script_hash: str
    concepts: List[str]
    ast_features: Dict[str, int]
    semantic_summary: str
    dependencies: List[str]
    code_preview: str
    embedding_type: str
    timestamp: str

@dataclass
class OSComponent:
    """An OS component built by LLM"""
    id: str
    name: str
    code: str
    vector: VectorEmbedding
    requirements: List[str]
    dependencies: List[str]
    tests: List[str]
    execution_results: Dict[str, Any]
    created_at: str
    version: int = 1

class LLMOSBuilder:
    """LLM builds its own operating system"""

    def __init__(self,
                 llm_endpoint: str = "http://localhost:1234/v1/completions",
                 vector_size: int = 1536,
                 workspace_dir: str = "./llm_os_workspace"):
        """
        Initialize the LLM OS Builder

        Args:
            llm_endpoint: URL for LM Studio API
            vector_size: Size of vectors to generate
            workspace_dir: Directory for storing built components
        """
        self.llm_endpoint = llm_endpoint
        self.script2vec = Script2Vec()
        self.ctrm_interface = CTRMScriptInterface()
        self.workspace_dir = workspace_dir
        self.components: Dict[str, OSComponent] = {}
        self.component_registry: Dict[str, List[str]] = {}  # intent -> component_ids

        # Create workspace
        os.makedirs(workspace_dir, exist_ok=True)
        os.makedirs(f"{workspace_dir}/components", exist_ok=True)
        os.makedirs(f"{workspace_dir}/tests", exist_ok=True)
        os.makedirs(f"{workspace_dir}/vectors", exist_ok=True)

    async def build_component(self,
                            requirement: str,
                            component_name: Optional[str] = None) -> OSComponent:
        """
        Build an OS component from requirement

        Args:
            requirement: Natural language requirement for the component
            component_name: Optional name for the component

        Returns:
            OSComponent object with all metadata
        """
        print(f"🔨 Building component: {requirement}")

        # Step 1: Design the component
        design = await self._llm_design(requirement)

        # Step 2: Write the code
        code = await self._llm_write_code(design, component_name)

        # Step 3: Write tests
        tests = await self._llm_write_tests(code, requirement)

        # Step 4: Convert to vector
        vector = self.script2vec.python_to_vector(code)

        # Step 5: Execute and test
        execution_results = await self._execute_and_test(code, tests)

        # Step 6: Create component object
        component_id = f"comp_{vector['script_hash']}"

        component = OSComponent(
            id=component_id,
            name=component_name or f"component_{vector['script_hash'][:8]}",
            code=code,
            vector=self._create_vector_embedding(vector, code),
            requirements=[requirement],
            dependencies=self._extract_dependencies_from_code(code),
            tests=tests,
            execution_results=execution_results,
            created_at=datetime.now().isoformat()
        )

        # Step 7: Store component
        self.components[component_id] = component

        # Index by intent
        intent_key = requirement.lower()[:50]
        if intent_key not in self.component_registry:
            self.component_registry[intent_key] = []
        self.component_registry[intent_key].append(component_id)

        # Step 8: Save to filesystem
        await self._save_component(component)

        print(f"✅ Component built: {component_id}")
        print(f"   Tests: {execution_results.get('tests_passed', 0)}/{execution_results.get('tests_total', 0)} passed")

        return component

    def _create_vector_embedding(self, vector_data: Dict, code: str) -> VectorEmbedding:
        """Create VectorEmbedding object from vector data"""
        return VectorEmbedding(
            vector=vector_data["vector"],
            script_hash=vector_data["script_hash"],
            concepts=vector_data.get("concepts", []),
            ast_features=vector_data.get("features", {}),
            semantic_summary=self._generate_semantic_summary(code),
            dependencies=self._extract_dependencies_from_code(code),
            code_preview=code[:500] + ("..." if len(code) > 500 else ""),
            embedding_type=vector_data["type"],
            timestamp=datetime.now().isoformat()
        )

    def _generate_semantic_summary(self, code: str) -> str:
        """Generate semantic summary from code"""
        # Count lines and functions
        lines = len(code.split('\n'))
        functions = len(re.findall(r'def\s+\w+', code))
        classes = len(re.findall(r'class\s+\w+', code))

        # Extract main purpose from first function/class
        first_def = re.search(r'def\s+(\w+)', code)
        first_class = re.search(r'class\s+(\w+)', code)

        purpose = "unknown"
        if first_def:
            purpose = f"function {first_def.group(1)}"
        elif first_class:
            purpose = f"class {first_class.group(1)}"

        return f"Python code ({lines} lines, {functions} functions, {classes} classes) implementing {purpose}"

    async def _llm_design(self, requirement: str) -> Dict[str, Any]:
        """Ask LLM to design the component"""
        prompt = f"""
        You are designing a Python component for an LLM Operating System.

        REQUIREMENT: {requirement}

        Design a Python component that meets this requirement. Consider:
        1. What is the main purpose?
        2. What functions/methods are needed?
        3. What data structures will it use?
        4. How will it interface with other OS components?
        5. What error cases need handling?

        Return your design as JSON with these fields:
        - name: Component name
        - purpose: One sentence purpose
        - functions: List of function signatures with descriptions
        - data_structures: List of data structures needed
        - dependencies: List of external libraries needed
        - interface: How other components will use it
        - error_handling: Key error cases to handle

        JSON ONLY, no other text.
        """

        response = await self._call_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback design
            return {
                "name": f"component_for_{requirement[:20].replace(' ', '_')}",
                "purpose": f"Handle {requirement}",
                "functions": ["process(data)", "validate(input)"],
                "dependencies": [],
                "interface": "Call process() method",
                "error_handling": ["Handle invalid input", "Log errors"]
            }

    async def _llm_write_code(self, design: Dict[str, Any],
                            component_name: Optional[str]) -> str:
        """LLM writes Python code from design"""
        name = component_name or design.get("name", "UnnamedComponent")

        prompt = f"""
        Write Python 3.10+ code for this OS component design:

        COMPONENT NAME: {name}
        DESIGN: {json.dumps(design, indent=2)}

        Requirements:
        1. Write complete, runnable Python code
        2. Include proper imports
        3. Add docstrings for all functions/classes
        4. Include error handling
        5. Make it modular and testable
        6. Follow PEP 8 style
        7. Include a main class or function that represents the component
        8. NO placeholder comments like "TODO" or "implement later"

        Return ONLY the Python code, no explanations.
        """

        return await self._call_llm(prompt)

    async def _llm_write_tests(self, code: str, requirement: str) -> List[str]:
        """LLM writes tests for the component"""
        prompt = f"""
        Write pytest tests for this Python OS component:

        REQUIREMENT: {requirement}

        CODE:
        {code}

        Write 3-5 test functions that:
        1. Test the main functionality
        2. Test error cases
        3. Test edge cases
        4. Are independent and fast

        Return ONLY the Python test code starting with 'import pytest'.
        """

        test_code = await self._call_llm(prompt)

        # Extract test functions
        tests = []
        lines = test_code.split('\n')
        current_test = []
        in_test = False

        for line in lines:
            if line.strip().startswith('def test_'):
                if current_test and in_test:
                    tests.append('\n'.join(current_test))
                    current_test = []
                in_test = True
                current_test.append(line)
            elif in_test:
                if line.strip() and not line.startswith(' ' * 4) and not line.startswith('\t'):
                    # End of test function
                    tests.append('\n'.join(current_test))
                    current_test = []
                    in_test = False
                else:
                    current_test.append(line)

        if current_test:
            tests.append('\n'.join(current_test))

        return tests

    async def _execute_and_test(self, code: str, tests: List[str]) -> Dict[str, Any]:
        """Execute code and run tests"""
        results = {
            "compilation_success": False,
            "tests_total": 0,
            "tests_passed": 0,
            "test_results": [],
            "execution_time": 0,
            "errors": []
        }

        try:
            # Create temporary module
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                module_file = f.name

            # Try to import/compile
            spec = importlib.util.spec_from_file_location("temp_module", module_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                results["compilation_success"] = True

            # Run tests if any
            if tests:
                test_results = []
                for i, test_code in enumerate(tests):
                    test_name = f"test_{i}"
                    try:
                        # Create test module
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                            # Create a test file with both the component and test
                            f.write(code + "\n\n" + test_code)
                            test_file = f.name

                        # Run pytest on this test
                        start_time = datetime.now()
                        result = subprocess.run(
                            [sys.executable, "-m", "pytest", test_file, "-v"],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        end_time = datetime.now()

                        passed = result.returncode == 0
                        test_results.append({
                            "test_id": i,
                            "passed": passed,
                            "output": result.stdout[-500:],  # Last 500 chars
                            "error": result.stderr[-500:] if result.stderr else None,
                            "duration": (end_time - start_time).total_seconds()
                        })

                        if passed:
                            results["tests_passed"] += 1
                        results["tests_total"] += 1

                    except Exception as e:
                        test_results.append({
                            "test_id": i,
                            "passed": False,
                            "error": str(e),
                            "duration": 0
                        })
                        results["tests_total"] += 1

                results["test_results"] = test_results

            # Clean up
            os.unlink(module_file)

        except Exception as e:
            results["errors"].append(f"Execution failed: {str(e)}")
            results["compilation_success"] = False

        return results

    def _extract_dependencies_from_code(self, code: str) -> List[str]:
        """Extract import dependencies from code"""
        imports = set()

        # Simple regex extraction
        import_patterns = [
            r'import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)',
            r'from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import'
        ]

        for pattern in import_patterns:
            matches = re.findall(pattern, code)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                imports.add(match.split('.')[0])  # Just top-level package

        return list(imports)

    async def _save_component(self, component: OSComponent):
        """Save component to filesystem"""
        # Save code
        code_file = f"{self.workspace_dir}/components/{component.id}.py"
        with open(code_file, 'w') as f:
            f.write(component.code)

        # Save tests
        if component.tests:
            test_file = f"{self.workspace_dir}/tests/test_{component.id}.py"
            with open(test_file, 'w') as f:
                f.write("\n\n".join(component.tests))

        # Save vector
        vector_file = f"{self.workspace_dir}/vectors/{component.id}.json"
        with open(vector_file, 'w') as f:
            json.dump({
                "vector": component.vector.vector,
                "concepts": component.vector.concepts,
                "hash": component.vector.script_hash
            }, f, indent=2)

        # Save metadata
        meta_file = f"{self.workspace_dir}/components/{component.id}.meta.json"
        with open(meta_file, 'w') as f:
            json.dump({
                "id": component.id,
                "name": component.name,
                "requirements": component.requirements,
                "dependencies": component.dependencies,
                "created_at": component.created_at,
                "version": component.version,
                "execution_results": component.execution_results
            }, f, indent=2)

    async def _call_llm(self, prompt: str) -> str:
        """Call LM Studio API"""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.llm_endpoint,
                    json={
                        "prompt": prompt,
                        "max_tokens": 2000,
                        "temperature": 0.2,
                        "stop": ["###", "```"]
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("choices", [{}])[0].get("text", "").strip()
                    else:
                        return f"# LLM Error: {response.status}"
        except Exception as e:
            return f"# LLM Connection Error: {str(e)}\n# Using fallback implementation."

    async def find_similar_component(self, requirement: str) -> Optional[OSComponent]:
        """Find existing component that could meet requirement"""
        # Convert requirement to vector
        requirement_vector = self.script2vec.python_to_vector(
            f"# Requirement: {requirement}",
            strategy="semantic"
        )

        best_match = None
        best_similarity = 0.0

        for component_id, component in self.components.items():
            similarity = self.script2vec.cosine_similarity(
                requirement_vector["vector"],
                component.vector.vector
            )

            if similarity > best_similarity and similarity > 0.6:
                best_similarity = similarity
                best_match = component

        return best_match

    async def improve_component(self, component_id: str,
                              issue: str) -> OSComponent:
        """Improve an existing component"""
        if component_id not in self.components:
            raise ValueError(f"Component {component_id} not found")

        component = self.components[component_id]

        print(f"🔄 Improving component: {component.name}")

        # Ask LLM to improve the code
        prompt = f"""
        Improve this Python OS component:

        CURRENT CODE:
        {component.code}

        ISSUE TO FIX: {issue}

        EXECUTION HISTORY:
        {json.dumps(component.execution_results, indent=2)}

        Requirements for improvement:
        1. Fix the issue mentioned
        2. Don't break existing functionality
        3. Add tests if needed
        4. Improve documentation if needed
        5. Follow PEP 8

        Return ONLY the improved Python code.
        """

        improved_code = await self._call_llm(prompt)

        # Create new version
        new_vector = self.script2vec.python_to_vector(improved_code)
        new_tests = await self._llm_write_tests(improved_code, f"Improved: {issue}")
        new_execution = await self._execute_and_test(improved_code, new_tests)

        new_component = OSComponent(
            id=f"{component_id}_v{component.version + 1}",
            name=f"{component.name}_improved",
            code=improved_code,
            vector=self._create_vector_embedding(new_vector, improved_code),
            requirements=component.requirements + [f"Fix: {issue}"],
            dependencies=self._extract_dependencies_from_code(improved_code),
            tests=new_tests,
            execution_results=new_execution,
            created_at=datetime.now().isoformat(),
            version=component.version + 1
        )

        # Store new version
        self.components[new_component.id] = new_component

        # Link to previous version
        await self._save_component(new_component)

        print(f"✅ Component improved: {new_component.id}")

        return new_component

    async def compose_os(self, component_ids: List[str]) -> str:
        """Compose multiple components into a working OS"""
        # Get components
        components = [self.components[cid] for cid in component_ids if cid in self.components]

        if not components:
            return "# No components to compose"

        # Ask LLM to compose them
        prompt = f"""
        Compose these LLM OS components into a single working system:

        COMPONENTS:
        {json.dumps([{"id": c.id, "name": c.name, "purpose": c.vector.semantic_summary} for c in components], indent=2)}

        Create a main.py that:
        1. Imports all components
        2. Initializes them in correct order
        3. Sets up communication between components
        4. Provides a unified API or CLI
        5. Includes error handling
        6. Has a main() function that starts the system

        The system should be runnable with: python main.py

        Return ONLY the Python code.
        """

        os_code = await self._call_llm(prompt)

        # Save OS composition
        os_file = f"{self.workspace_dir}/llm_os_main.py"
        with open(os_file, 'w') as f:
            f.write(os_code)

        # Also save a bootstrap script
        bootstrap = self._create_bootstrap_script(components, os_code)
        bootstrap_file = f"{self.workspace_dir}/bootstrap.py"
        with open(bootstrap_file, 'w') as f:
            f.write(bootstrap)

        return os_code

    def _create_bootstrap_script(self, components: List[OSComponent],
                               os_code: str) -> str:
        """Create bootstrap script to set up the OS"""
        bootstrap = '''"""
LLM OS Bootstrap Script
Auto-generated by LLM OS Builder
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def install_requirements():
    """Install required packages"""
    requirements = set()

    # Check each component's dependencies
    components_dir = Path(__file__).parent / "components"

    for meta_file in components_dir.glob("*.meta.json"):
        try:
            import json
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                for dep in meta.get("dependencies", []):
                    requirements.add(dep)
        except:
            pass

    # Install requirements
    if requirements:
        print(f"Installing requirements: {', '.join(requirements)}")
        for req in requirements:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            except:
                print(f"Warning: Could not install {req}")

    print("All requirements installed.")

def setup_environment():
    """Set up environment variables and paths"""
    # Add components directory to Python path
    components_dir = Path(__file__).parent / "components"
    if str(components_dir) not in sys.path:
        sys.path.insert(0, str(components_dir))

    # Set up workspace
    workspace = Path(__file__).parent
    os.environ["LLM_OS_WORKSPACE"] = str(workspace)

    print(f"Environment set up. Workspace: {workspace}")

def run_tests():
    """Run component tests"""
    import pytest

    test_dir = Path(__file__).parent / "tests"
    if test_dir.exists():
        print("Running component tests...")
        exit_code = pytest.main([str(test_dir), "-v"])
        if exit_code == 0:
            print("All tests passed!")
        else:
            print("Some tests failed.")
    else:
        print("No tests directory found.")

def start_os():
    """Start the LLM OS"""
    print("Starting LLM OS...")

    # Import and run main OS
    os_main = Path(__file__).parent / "llm_os_main.py"

    if os_main.exists():
        spec = importlib.util.spec_from_file_location("llm_os", os_main)
        module = importlib.util.module_from_spec(spec)
        sys.modules["llm_os"] = module
        spec.loader.exec_module(module)

        # Look for main() function
        if hasattr(module, "main"):
            module.main()
        elif hasattr(module, "run"):
            module.run()
        else:
            print("No main() or run() function found in OS.")
    else:
        print(f"OS main file not found: {os_main}")

if __name__ == "__main__":
    print("=" * 60)
    print("LLM OS Bootstrap")
    print("=" * 60)

    install_requirements()
    setup_environment()
    run_tests()
    start_os()
'''

        return bootstrap