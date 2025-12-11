#!/usr/bin/env python3

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("script2vec-cli")

# Add script2vec to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from script2vec.script2vec import Script2Vec
from script2vec.ctrm_script_interface import CTRMScriptInterface
from script2vec.decorators import auto_vectorize

class Script2VecCLI:
    """Command Line Interface for Script2Vec"""

    def __init__(self):
        self.script2vec = Script2Vec()
        self.ctrm_interface = CTRMScriptInterface()
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="Script2Vec - Convert Python scripts to vectors for CTRM integration",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  script2vec convert my_script.py --output my_script.vector.json
  script2vec embed-dir ./src --output src_vectors.json
  script2vec find-similar query.py --ctrm-url http://localhost:8000
  script2vec improve script.py --type optimize --output improved.py
  script2vec watch ./src --webhook http://ctrm/vector/update
"""
        )

        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Convert command
        convert_parser = subparsers.add_parser('convert', help='Convert Python file to vector')
        convert_parser.add_argument('filepath', help='Path to Python file')
        convert_parser.add_argument('--output', '-o', help='Output JSON file')
        convert_parser.add_argument('--strategy', choices=['semantic', 'ast', 'execution', 'hybrid'],
                                  default='hybrid', help='Embedding strategy')
        convert_parser.add_argument('--ctrm', action='store_true', help='Store in CTRM')
        convert_parser.add_argument('--purpose', default='store', help='Purpose for CTRM storage')

        # Embed directory command
        embed_parser = subparsers.add_parser('embed-dir', help='Embed all Python files in directory')
        embed_parser.add_argument('dirpath', help='Path to directory')
        embed_parser.add_argument('--output', '-o', help='Output JSON file')
        embed_parser.add_argument('--ctrm', action='store_true', help='Store in CTRM')
        embed_parser.add_argument('--purpose', default='directory_analysis', help='Purpose for CTRM storage')

        # Find similar command
        similar_parser = subparsers.add_parser('find-similar', help='Find similar scripts in CTRM')
        similar_parser.add_argument('filepath', help='Path to query Python file')
        similar_parser.add_argument('--ctrm-url', default='http://localhost:8000', help='CTRM URL')
        similar_parser.add_argument('--threshold', type=float, default=0.8, help='Similarity threshold')
        similar_parser.add_argument('--output', '-o', help='Output JSON file')

        # Improve command
        improve_parser = subparsers.add_parser('improve', help='Improve script using CTRM')
        improve_parser.add_argument('filepath', help='Path to Python file to improve')
        improve_parser.add_argument('--type', choices=['optimize', 'refactor', 'document', 'test'],
                                  default='optimize', help='Improvement type')
        improve_parser.add_argument('--output', '-o', help='Output file for suggestions')
        improve_parser.add_argument('--ctrm-url', default='http://localhost:8000', help='CTRM URL')

        # Watch command
        watch_parser = subparsers.add_parser('watch', help='Monitor directory for changes')
        watch_parser.add_argument('dirpath', help='Path to directory to watch')
        watch_parser.add_argument('--webhook', help='Webhook URL for notifications')
        watch_parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
        watch_parser.add_argument('--ctrm', action='store_true', help='Store in CTRM')

        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Analyze script quality')
        analyze_parser.add_argument('filepath', help='Path to Python file to analyze')
        analyze_parser.add_argument('--output', '-o', help='Output JSON file')

        # Batch command
        batch_parser = subparsers.add_parser('batch', help='Batch process multiple files')
        batch_parser.add_argument('filepaths', nargs='+', help='Paths to Python files')
        batch_parser.add_argument('--output', '-o', help='Output JSON file')
        batch_parser.add_argument('--ctrm', action='store_true', help='Store in CTRM')
        batch_parser.add_argument('--purpose', default='batch_processing', help='Purpose for CTRM storage')

        return parser

    async def run(self, args: argparse.Namespace):
        """Run the appropriate command"""
        try:
            if args.command == 'convert':
                return await self._handle_convert(args)
            elif args.command == 'embed-dir':
                return await self._handle_embed_dir(args)
            elif args.command == 'find-similar':
                return await self._handle_find_similar(args)
            elif args.command == 'improve':
                return await self._handle_improve(args)
            elif args.command == 'watch':
                return await self._handle_watch(args)
            elif args.command == 'analyze':
                return await self._handle_analyze(args)
            elif args.command == 'batch':
                return await self._handle_batch(args)
            else:
                self.parser.print_help()
                return {"error": "No command specified"}

        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {"error": str(e)}

    async def _handle_convert(self, args) -> Dict:
        """Handle convert command"""
        logger.info(f"Converting {args.filepath} to vector...")

        # Read script
        with open(args.filepath, 'r') as f:
            script = f.read()

        # Convert to vector
        vector_result = self.script2vec.python_to_vector(script, strategy=f"{args.strategy}_embedding")

        # Store in CTRM if requested
        if args.ctrm:
            ctrm_result = await self.ctrm_interface.script_to_ctrm(
                script,
                purpose=args.purpose,
                source_file=args.filepath
            )
            vector_result['ctrm'] = ctrm_result

        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(vector_result, f, indent=2)
            logger.info(f"Vector saved to {args.output}")

        return vector_result

    async def _handle_embed_dir(self, args) -> Dict:
        """Handle embed-dir command"""
        logger.info(f"Embedding directory {args.dirpath}...")

        # Embed directory
        directory_result = self.script2vec.embed_directory(args.dirpath)

        # Store in CTRM if requested
        if args.ctrm:
            ctrm_result = await self.ctrm_interface.monitor_directory(
                args.dirpath,
                purpose=args.purpose
            )
            directory_result['ctrm'] = ctrm_result

        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(directory_result, f, indent=2)
            logger.info(f"Directory vectors saved to {args.output}")

        return directory_result

    async def _handle_find_similar(self, args) -> Dict:
        """Handle find-similar command"""
        logger.info(f"Finding similar scripts to {args.filepath}...")

        # Read query script
        with open(args.filepath, 'r') as f:
            query_script = f.read()

        # Find similar scripts
        similar_results = await self.ctrm_interface.find_similar_code_in_ctrm(
            query_script,
            threshold=args.threshold
        )

        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(similar_results, f, indent=2)
            logger.info(f"Similar scripts saved to {args.output}")

        return similar_results

    async def _handle_improve(self, args) -> Dict:
        """Handle improve command"""
        logger.info(f"Improving {args.filepath} using CTRM...")

        # Read script
        with open(args.filepath, 'r') as f:
            script = f.read()

        # Improve script
        improvement_result = await self.ctrm_interface.improve_script_via_ctrm(
            script,
            improvement_type=args.type
        )

        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(improvement_result, f, indent=2)
            logger.info(f"Improvement suggestions saved to {args.output}")

        return improvement_result

    async def _handle_watch(self, args) -> Dict:
        """Handle watch command"""
        logger.info(f"Watching directory {args.dirpath} for changes...")

        # Initial scan
        initial_files = list(Path(args.dirpath).rglob("*.py"))
        logger.info(f"Found {len(initial_files)} Python files initially")

        # Monitor directory
        try:
            while True:
                # Check for changes
                current_files = list(Path(args.dirpath).rglob("*.py"))

                # Detect new or modified files
                new_files = [f for f in current_files if f not in initial_files]
                modified_files = []

                for f in initial_files:
                    if f in current_files:
                        if f.stat().st_mtime > f.stat().st_mtime:  # Simple change detection
                            modified_files.append(f)

                # Process changes
                if new_files or modified_files:
                    logger.info(f"Detected changes: {len(new_files)} new, {len(modified_files)} modified")

                    # Process new files
                    for filepath in new_files:
                        await self._process_file_change(filepath, args)

                    # Process modified files
                    for filepath in modified_files:
                        await self._process_file_change(filepath, args)

                    # Update initial files
                    initial_files = current_files

                # Wait for next interval
                await asyncio.sleep(args.interval)

        except KeyboardInterrupt:
            logger.info("Stopping directory watch...")
            return {"status": "stopped"}

    async def _process_file_change(self, filepath: Path, args) -> Dict:
        """Process a file change"""
        logger.info(f"Processing changed file: {filepath}")

        # Read the file
        with open(filepath, 'r') as f:
            script = f.read()

        # Convert to vector
        vector_result = self.script2vec.python_to_vector(script)

        # Store in CTRM if requested
        if args.ctrm:
            ctrm_result = await self.ctrm_interface.file_to_ctrm(str(filepath))
            vector_result['ctrm'] = ctrm_result

        # Send webhook notification if configured
        if args.webhook:
            await self._send_webhook_notification(args.webhook, filepath, vector_result)

        return vector_result

    async def _send_webhook_notification(self, webhook_url: str,
                                        filepath: Path,
                                        vector_result: Dict) -> bool:
        """Send webhook notification"""
        try:
            # In a real implementation, this would make an HTTP request
            logger.info(f"Would send webhook to {webhook_url} for {filepath}")
            logger.info(f"Vector hash: {vector_result['script_hash']}")
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False

    async def _handle_analyze(self, args) -> Dict:
        """Handle analyze command"""
        logger.info(f"Analyzing {args.filepath}...")

        # Read script
        with open(args.filepath, 'r') as f:
            script = f.read()

        # Analyze script quality
        analysis_result = await self.ctrm_interface.analyze_script_quality(script)

        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            logger.info(f"Analysis results saved to {args.output}")

        return analysis_result

    async def _handle_batch(self, args) -> Dict:
        """Handle batch command"""
        logger.info(f"Batch processing {len(args.filepaths)} files...")

        # Read all scripts
        scripts = []
        for filepath in args.filepaths:
            with open(filepath, 'r') as f:
                scripts.append(f.read())

        # Batch process
        batch_result = await self.ctrm_interface.batch_process_scripts(
            scripts,
            purpose=args.purpose
        )

        # Save output if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(batch_result, f, indent=2)
            logger.info(f"Batch results saved to {args.output}")

        return batch_result

def main():
    """Main CLI entry point"""
    cli = Script2VecCLI()
    args = cli.parser.parse_args()

    if not args.command:
        cli.parser.print_help()
        sys.exit(1)

    # Run the command
    result = asyncio.run(cli.run(args))

    # Print summary
    if 'error' in result:
        logger.error(f"Error: {result['error']}")
        sys.exit(1)
    else:
        logger.info("Operation completed successfully")
        if args.command != 'watch':  # Don't print full result for watch
            print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()