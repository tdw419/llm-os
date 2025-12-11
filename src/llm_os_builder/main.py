#!/usr/bin/env python3
"""
LLM OS Builder - Main Execution Script

This script provides the main interface for building a complete
LLM OS using the self-referential system where LLMs build
their own operating system through Python scripts that
auto-convert to vectors.
"""

import asyncio
import sys
import os
import json
import glob
from typing import List, Dict, Any
from datetime import datetime

from llm_os_builder.core import LLMOSBuilder
from llm_os_builder.ctrm_integration import CTRMInterface

async def build_sample_os():
    """Build a sample LLM OS with core components"""
    print("🚀 Starting LLM OS Builder")
    print("=" * 60)

    # Initialize builder
    builder = LLMOSBuilder(
        llm_endpoint="http://localhost:1234/v1/completions",
        workspace_dir="./llm_os_output"
    )

    # Initialize CTRM
    ctrm = CTRMInterface()
    await ctrm.connect()

    # Define core OS components to build
    core_components = [
        {
            "name": "vector_memory",
            "requirement": "vector-based memory system with semantic search and persistence"
        },
        {
            "name": "task_scheduler",
            "requirement": "task scheduler for parallel LLM operations with priority queues"
        },
        {
            "name": "plugin_manager",
            "requirement": "plugin system for dynamic loading and unloading of capabilities"
        },
        {
            "name": "monitoring_system",
            "requirement": "self-monitoring system that tracks performance and errors"
        },
        {
            "name": "api_gateway",
            "requirement": "REST API gateway for human and programmatic interaction"
        }
    ]

    built_components = []

    # Build each component
    for comp_def in core_components:
        print(f"\n📦 Building: {comp_def['name']}")
        print(f"   Requirement: {comp_def['requirement']}")

        # Check if similar component exists
        similar = await builder.find_similar_component(comp_def["requirement"])
        if similar:
            print(f"   Similar component found: {similar.name}")
            use_existing = input("   Use existing? (y/n): ").lower().strip() == 'y'
            if use_existing:
                built_components.append(similar.id)
                continue

        # Build new component
        try:
            component = await builder.build_component(
                requirement=comp_def["requirement"],
                component_name=comp_def["name"]
            )

            # Store in CTRM
            truth_id = await ctrm.store_component_truth(component)
            if truth_id:
                print(f"   Stored in CTRM as truth: {truth_id}")

            # Store vector in CTRM
            vector_id = await ctrm.store_component_vector(component)
            if vector_id:
                print(f"   Stored vector in CTRM: {vector_id}")

            built_components.append(component.id)

        except Exception as e:
            print(f"   Error building component: {e}")
            continue

    # Compose OS from built components
    if built_components:
        print(f"\n🧩 Composing OS from {len(built_components)} components...")

        os_code = await builder.compose_os(built_components)

        # Store OS composition in CTRM
        composition_data = {
            "components": built_components,
            "timestamp": datetime.now().isoformat(),
            "description": "Sample LLM OS composition"
        }

        composition_id = await ctrm.store_os_composition(composition_data)
        if composition_id:
            print(f"   Stored OS composition in CTRM: {composition_id}")

        # Save final OS
        os_file = "./llm_os_output/final_os.py"
        with open(os_file, 'w') as f:
            f.write(os_code)

        print(f"✅ OS composition complete!")
        print(f"   Main file: {os_file}")
        print(f"   Bootstrap: ./llm_os_output/bootstrap.py")
        print(f"   Components: ./llm_os_output/components/")

        # Create run script
        run_script = '''#!/bin/bash
echo "Starting LLM OS..."
cd "$(dirname "$0")"
python bootstrap.py
'''

        with open("./llm_os_output/run.sh", 'w') as f:
            f.write(run_script)
        os.chmod("./llm_os_output/run.sh", 0o755)

        print(f"   Run script: ./llm_os_output/run.sh")

        # Summary
        print(f"\n📊 Summary:")
        print(f"   Total components built: {len(built_components)}")

        for comp_id in built_components:
            comp = builder.components[comp_id]
            tests_passed = comp.execution_results.get('tests_passed', 0)
            tests_total = comp.execution_results.get('tests_total', 0)
            print(f"   - {comp.name}: {tests_passed}/{tests_total} tests passed")

    else:
        print("❌ No components were built successfully.")

    # Clean up
    await ctrm.disconnect()

    print("\n🎉 LLM OS Builder finished!")
    print("=" * 60)

async def interactive_builder():
    """Interactive mode for building custom components"""
    builder = LLMOSBuilder()
    ctrm = CTRMInterface()
    await ctrm.connect()

    print("🤖 LLM OS Interactive Builder")
    print("Type 'quit' to exit, 'list' to show components")
    print("=" * 60)

    while True:
        command = input("\n> ").strip()

        if command.lower() == 'quit':
            break
        elif command.lower() == 'list':
            print("\nBuilt components:")
            for comp_id, comp in builder.components.items():
                print(f"  {comp.name} ({comp.id})")
                print(f"    {comp.vector.semantic_summary}")
        elif command.lower() == 'compose':
            if builder.components:
                comp_ids = list(builder.components.keys())
                os_code = await builder.compose_os(comp_ids)
                print("OS composed. Check llm_os_output/ directory.")
            else:
                print("No components to compose.")
        else:
            # Treat as requirement for new component
            print(f"\nBuilding component for: {command}")

            # Get component name
            name = input("Component name (or Enter for auto): ").strip()
            if not name:
                name = None

            try:
                component = await builder.build_component(command, name)
                print(f"✅ Built: {component.name}")

                # Store in CTRM
                truth_id = await ctrm.store_component_truth(component)
                if truth_id:
                    print(f"   Stored in CTRM: {truth_id}")

            except Exception as e:
                print(f"❌ Error: {e}")

    await ctrm.disconnect()

async def improve_component_mode():
    """Mode for improving existing components"""
    builder = LLMOSBuilder()
    ctrm = CTRMInterface()
    await ctrm.connect()

    # Load existing components
    meta_files = glob.glob("./llm_os_output/components/*.meta.json")

    if not meta_files:
        print("No existing components found.")
        return

    print("Existing components:")
    for i, meta_file in enumerate(meta_files[:10], 1):
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            print(f"{i}. {meta['name']} ({meta['id']})")
            print(f"   {meta.get('requirements', ['No requirements'])[0]}")

    comp_choice = int(input("Select component to improve (number): ")) - 1
    selected_meta = meta_files[comp_choice]

    with open(selected_meta, 'r') as f:
        meta = json.load(f)

    issue = input("What needs improvement? ").strip()

    # Load component code
    code_file = f"./llm_os_output/components/{meta['id']}.py"
    with open(code_file, 'r') as f:
        code = f.read()

    # Create component object
    from llm_os_builder.core import OSComponent
    from llm_os_builder.core import VectorEmbedding

    vector = VectorEmbedding(
        vector=[],  # Will be generated
        script_hash="",
        concepts=[],
        ast_features={},
        semantic_summary=meta.get('requirements', [''])[0],
        dependencies=meta.get('dependencies', []),
        code_preview=code[:500],
        embedding_type="semantic",
        timestamp=meta.get('created_at', '')
    )

    component = OSComponent(
        id=meta['id'],
        name=meta['name'],
        code=code,
        vector=vector,
        requirements=meta.get('requirements', []),
        dependencies=meta.get('dependencies', []),
        tests=[],
        execution_results=meta.get('execution_results', {}),
        created_at=meta.get('created_at', '')
    )

    builder.components[component.id] = component

    # Improve it
    improved = await builder.improve_component(component.id, issue)
    print(f"✅ Improved: {improved.id}")

    # Store in CTRM
    truth_id = await ctrm.store_component_truth(improved)
    if truth_id:
        print(f"   Stored in CTRM: {truth_id}")

    await ctrm.disconnect()

async def monitor_os_health():
    """Monitor the health of the LLM OS"""
    ctrm = CTRMInterface()
    await ctrm.connect()

    print("🏥 Monitoring LLM OS Health")
    print("=" * 60)

    # Get OS health
    health = await ctrm.get_os_health()
    print(f"OS Health Status: {health.get('status', 'unknown')}")

    # Get components
    components = []
    meta_files = glob.glob("./llm_os_output/components/*.meta.json")
    for meta_file in meta_files:
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            components.append(meta)

    print(f"\nComponents: {len(components)}")
    healthy = 0
    for comp in components:
        tests_passed = comp.get('execution_results', {}).get('tests_passed', 0)
        tests_total = comp.get('execution_results', {}).get('tests_total', 0)
        if tests_total > 0 and tests_passed == tests_total:
            healthy += 1

    print(f"Healthy components: {healthy}/{len(components)}")

    # Get architecture
    architecture = await ctrm.get_os_architecture()
    print(f"\nArchitecture: {architecture.get('description', 'Not available')}")

    await ctrm.disconnect()

async def backup_and_restore():
    """Backup and restore OS state"""
    ctrm = CTRMInterface()
    await ctrm.connect()

    print("💾 Backup and Restore")
    print("=" * 60)

    # Backup current state
    print("Creating backup...")
    backup_success = await ctrm.backup_os_state()
    if backup_success:
        print("✅ Backup created successfully")
    else:
        print("❌ Backup failed")

    # Show available backups
    print("\nAvailable backups:")
    # In a real implementation, this would query CTRM for backups
    print("   (Backup functionality would be implemented in CTRM)")

    await ctrm.disconnect()

async def main():
    """Main entry point"""
    # Check if LM Studio is running
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', 1234))
        if result != 0:
            print("⚠️  LM Studio not detected on localhost:1234")
            print("Please start LM Studio first.")
            sys.exit(1)
    except:
        pass

    # Choose mode
    print("🎯 LLM OS Builder - Self-Building LLM Operating System")
    print("=" * 60)
    print("Select mode:")
    print("1. Build sample LLM OS")
    print("2. Interactive builder")
    print("3. Improve existing component")
    print("4. Monitor OS health")
    print("5. Backup and restore")

    choice = input("Choice (1-5): ").strip()

    if choice == "1":
        await build_sample_os()
    elif choice == "2":
        await interactive_builder()
    elif choice == "3":
        await improve_component_mode()
    elif choice == "4":
        await monitor_os_health()
    elif choice == "5":
        await backup_and_restore()
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    asyncio.run(main())