import asyncio
import json
import os
import tempfile
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging
from pathlib import Path

# FastAPI for web interface
try:
    from fastapi import FastAPI, UploadFile, File, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    from starlette.middleware.cors import CORSMiddleware
except ImportError:
    # Fallback for environments without FastAPI
    FastAPI = None
    UploadFile = None
    File = None
    HTTPException = Exception
    HTMLResponse = str
    JSONResponse = dict
    StaticFiles = None
    Jinja2Templates = None
    uvicorn = None
    CORSMiddleware = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("script2vec-web")

from .script2vec import Script2Vec
from .ctrm_script_interface import CTRMScriptInterface

class Script2VecWebInterface:
    """Web interface for Script2Vec"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8001):
        self.host = host
        self.port = port
        self.script2vec = Script2Vec()
        self.ctrm_interface = CTRMScriptInterface()

        # Check if FastAPI is available
        if FastAPI is None:
            logger.warning("FastAPI not available - web interface will be limited")
            self.available = False
            return

        self.available = True
        self.app = FastAPI(
            title="Script2Vec Web Interface",
            description="Web interface for converting Python scripts to vectors",
            version="1.0.0"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Set up templates
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        os.makedirs(template_dir, exist_ok=True)
        self.templates = Jinja2Templates(directory=template_dir)

        # Create static directory
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(static_dir, exist_ok=True)
        self.app.mount("/static", StaticFiles(directory=static_dir), name="static")

        # Set up routes
        self._setup_routes()

        # Create default templates
        self._create_default_templates()

    def _setup_routes(self):
        """Set up web routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def index(request: Request):
            """Main page"""
            return self.templates.TemplateResponse(
                "index.html",
                {"request": request, "title": "Script2Vec Web Interface"}
            )

        @self.app.post("/upload-script", response_class=JSONResponse)
        async def upload_script(file: UploadFile = File(...)):
            """Upload Python script and convert to vector"""
            try:
                contents = await file.read()
                script = contents.decode('utf-8')

                # Convert to vector
                vector_data = self.script2vec.python_to_vector(script)

                # Send to CTRM
                ctrm_result = await self.ctrm_interface.script_to_ctrm(
                    script,
                    purpose="web_upload",
                    original_filename=file.filename,
                    upload_timestamp=datetime.now().isoformat()
                )

                return {
                    "filename": file.filename,
                    "vector_hash": ctrm_result["ctrm_vector_hash"],
                    "concepts": vector_data.get("concepts", []),
                    "vector_dimensions": len(vector_data["vector"]),
                    "script_hash": vector_data["script_hash"],
                    "status": "success",
                    "preview_vector": vector_data["vector"][:10],  # First 10 dimensions
                    "metadata": ctrm_result["metadata"]
                }
            except Exception as e:
                logger.error(f"Error processing upload: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/upload-multiple", response_class=JSONResponse)
        async def upload_multiple(files: List[UploadFile] = File(...)):
            """Upload multiple Python scripts"""
            results = []

            for file in files:
                try:
                    contents = await file.read()
                    script = contents.decode('utf-8')

                    # Convert to vector
                    vector_data = self.script2vec.python_to_vector(script)

                    # Send to CTRM
                    ctrm_result = await self.ctrm_interface.script_to_ctrm(
                        script,
                        purpose="web_batch_upload",
                        original_filename=file.filename
                    )

                    results.append({
                        "filename": file.filename,
                        "vector_hash": ctrm_result["ctrm_vector_hash"],
                        "status": "success",
                        "concepts": vector_data.get("concepts", [])
                    })
                except Exception as e:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": str(e)
                    })

            return {
                "processed_files": len(results),
                "successful": len([r for r in results if r["status"] == "success"]),
                "results": results
            }

        @self.app.post("/find-similar", response_class=JSONResponse)
        async def find_similar(request: Request):
            """Find similar scripts in CTRM"""
            try:
                data = await request.json()
                script = data.get("script", "")
                threshold = data.get("threshold", 0.8)

                if not script:
                    raise HTTPException(status_code=400, detail="No script provided")

                similar_results = await self.ctrm_interface.find_similar_code_in_ctrm(
                    script,
                    threshold=threshold
                )

                return {
                    "query_script": script[:100] + "..." if len(script) > 100 else script,
                    "similar_scripts": similar_results,
                    "count": len(similar_results),
                    "threshold": threshold
                }
            except Exception as e:
                logger.error(f"Error finding similar scripts: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/improve-script", response_class=JSONResponse)
        async def improve_script(request: Request):
            """Improve script using CTRM"""
            try:
                data = await request.json()
                script = data.get("script", "")
                improvement_type = data.get("improvement_type", "optimize")

                if not script:
                    raise HTTPException(status_code=400, detail="No script provided")

                improvement_result = await self.ctrm_interface.improve_script_via_ctrm(
                    script,
                    improvement_type=improvement_type
                )

                return {
                    "original_script": script[:100] + "..." if len(script) > 100 else script,
                    "improvement_type": improvement_type,
                    "suggestions": improvement_result["suggestions"],
                    "confidence": improvement_result["confidence"],
                    "improved_vector_hash": improvement_result["improved_vector_hash"]
                }
            except Exception as e:
                logger.error(f"Error improving script: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.post("/analyze-script", response_class=JSONResponse)
        async def analyze_script(request: Request):
            """Analyze script quality"""
            try:
                data = await request.json()
                script = data.get("script", "")

                if not script:
                    raise HTTPException(status_code=400, detail="No script provided")

                analysis_result = await self.ctrm_interface.analyze_script_quality(script)

                return {
                    "script_hash": analysis_result["script_hash"],
                    "quality_score": analysis_result["quality_score"],
                    "analysis": analysis_result["analysis"],
                    "recommendations": analysis_result["recommendations"],
                    "vector_quality": analysis_result["vector_quality"]
                }
            except Exception as e:
                logger.error(f"Error analyzing script: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/stats", response_class=JSONResponse)
        async def get_stats():
            """Get system statistics"""
            return {
                "script2vec_cache": self.script2vec.get_cache_stats(),
                "ctrm_interface_cache": self.ctrm_interface.get_cache_stats(),
                "timestamp": datetime.now().isoformat(),
                "status": "operational"
            }

        @self.app.get("/health", response_class=JSONResponse)
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "script2vec": "available",
                "ctrm_interface": "available"
            }

        @self.app.post("/batch-process", response_class=JSONResponse)
        async def batch_process(request: Request):
            """Batch process multiple scripts"""
            try:
                data = await request.json()
                scripts = data.get("scripts", [])
                purpose = data.get("purpose", "batch_processing")

                if not scripts:
                    raise HTTPException(status_code=400, detail="No scripts provided")

                batch_result = await self.ctrm_interface.batch_process_scripts(
                    scripts,
                    purpose=purpose
                )

                return {
                    "processed_scripts": batch_result["processed_scripts"],
                    "successful": batch_result["successful"],
                    "failed": batch_result["failed"],
                    "results": batch_result["results"]
                }
            except Exception as e:
                logger.error(f"Error batch processing: {e}")
                raise HTTPException(status_code=400, detail=str(e))

        @self.app.get("/upload-form", response_class=HTMLResponse)
        async def upload_form(request: Request):
            """Simple HTML upload form"""
            return self.templates.TemplateResponse(
                "upload_form.html",
                {"request": request}
            )

        @self.app.get("/advanced-form", response_class=HTMLResponse)
        async def advanced_form(request: Request):
            """Advanced upload form"""
            return self.templates.TemplateResponse(
                "advanced_form.html",
                {"request": request}
            )

    def _create_default_templates(self):
        """Create default HTML templates"""
        template_dir = os.path.join(os.path.dirname(__file__), "templates")

        # Index template
        index_template = """<!DOCTYPE html>
<html>
<head>
    <title>Script2Vec Web Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .header { background: #4a6fa5; color: white; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .btn { background: #4a6fa5; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #3a5a8f; }
        textarea { width: 100%; height: 150px; padding: 10px; border-radius: 5px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Script2Vec Web Interface</h1>
            <p>Convert Python scripts to vectors for CTRM integration</p>
        </div>

        <div class="section">
            <h2>📝 Upload Python Script</h2>
            <form action="/upload-script" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".py" required>
                <button type="submit" class="btn">Convert to Vector</button>
            </form>
        </div>

        <div class="section">
            <h2>🔍 Find Similar Code</h2>
            <form id="similarForm">
                <textarea placeholder="Paste your Python script here..." required></textarea>
                <div style="margin: 10px 0;">
                    <label>Similarity Threshold: </label>
                    <input type="number" name="threshold" value="0.8" step="0.1" min="0" max="1">
                </div>
                <button type="button" onclick="findSimilar()" class="btn">Find Similar Code</button>
            </form>
            <div id="similarResults" style="margin-top: 15px;"></div>
        </div>

        <div class="section">
            <h2>💡 Improve Script</h2>
            <form id="improveForm">
                <textarea placeholder="Paste your Python script here..." required></textarea>
                <div style="margin: 10px 0;">
                    <label>Improvement Type: </label>
                    <select name="improvement_type">
                        <option value="optimize">Optimize</option>
                        <option value="refactor">Refactor</option>
                        <option value="document">Document</option>
                        <option value="test">Test</option>
                    </select>
                </div>
                <button type="button" onclick="improveScript()" class="btn">Get Improvement Suggestions</button>
            </form>
            <div id="improveResults" style="margin-top: 15px;"></div>
        </div>

        <div class="section">
            <h2>📊 System Stats</h2>
            <button onclick="getStats()" class="btn">Get System Statistics</button>
            <div id="statsResults" style="margin-top: 15px;"></div>
        </div>
    </div>

    <script>
        async function findSimilar() {
            const form = document.getElementById('similarForm');
            const textarea = form.querySelector('textarea');
            const threshold = form.querySelector('input[name="threshold"]').value;
            const resultsDiv = document.getElementById('similarResults');

            if (!textarea.value.trim()) {
                resultsDiv.innerHTML = '<p style="color: red;">Please enter a script</p>';
                return;
            }

            resultsDiv.innerHTML = '<p>Searching for similar code...</p>';

            try {
                const response = await fetch('/find-similar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        script: textarea.value,
                        threshold: parseFloat(threshold)
                    })
                });

                const data = await response.json();

                if (data.similar_scripts && data.similar_scripts.length > 0) {
                    let html = `<h3>Found ${data.count} similar scripts:</h3><ul>`;
                    data.similar_scripts.forEach(script => {
                        html += `<li>
                            <strong>Similarity: ${(script.similarity * 100).toFixed(1)}%</strong><br>
                            Purpose: ${script.likely_purpose}<br>
                            Suggested use: ${script.suggested_use}<br>
                            CTRM ID: ${script.ctrm_id}
                        </li>`;
                    });
                    html += '</ul>';
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = '<p>No similar scripts found</p>';
                }
            } catch (error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }

        async function improveScript() {
            const form = document.getElementById('improveForm');
            const textarea = form.querySelector('textarea');
            const improvementType = form.querySelector('select[name="improvement_type"]').value;
            const resultsDiv = document.getElementById('improveResults');

            if (!textarea.value.trim()) {
                resultsDiv.innerHTML = '<p style="color: red;">Please enter a script</p>';
                return;
            }

            resultsDiv.innerHTML = '<p>Analyzing script for improvements...</p>';

            try {
                const response = await fetch('/improve-script', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        script: textarea.value,
                        improvement_type: improvementType
                    })
                });

                const data = await response.json();

                if (data.suggestions && data.suggestions.length > 0) {
                    let html = `<h3>Improvement Suggestions (Confidence: ${data.confidence}):</h3><ul>`;
                    data.suggestions.forEach(suggestion => {
                        html += `<li>
                            <strong>${suggestion.type}</strong>: ${suggestion.description}<br>
                            Priority: ${suggestion.priority}
                        </li>`;
                    });
                    html += '</ul>';
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = '<p>No improvement suggestions available</p>';
                }
            } catch (error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }

        async function getStats() {
            const resultsDiv = document.getElementById('statsResults');
            resultsDiv.innerHTML = '<p>Fetching statistics...</p>';

            try {
                const response = await fetch('/stats');
                const data = await response.json();

                let html = `<h3>System Statistics:</h3>
                <p><strong>Timestamp:</strong> ${data.timestamp}</p>
                <p><strong>Status:</strong> ${data.status}</p>

                <h4>Script2Vec Cache:</h4>
                <p>Cached vectors: ${data.script2vec_cache.cached_vectors}</p>
                <p>Cache size: ${data.script2vec_cache.cache_size_mb.toFixed(2)} MB</p>

                <h4>CTRM Interface Cache:</h4>
                <p>Script cache: ${data.ctrm_interface_cache.script_cache}</p>
                <p>Vector cache: ${data.ctrm_interface_cache.vector_cache.cached_vectors}</p>
                `;

                resultsDiv.innerHTML = html;
            } catch (error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>"""

        with open(os.path.join(template_dir, "index.html"), "w") as f:
            f.write(index_template)

        # Upload form template
        upload_template = """<!DOCTYPE html>
<html>
<head>
    <title>Upload Python Script</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: 0 auto; }
        .form-group { margin: 15px 0; }
        .btn { background: #4a6fa5; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #3a5a8f; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload Python Script</h1>
        <form action="/upload-script" method="post" enctype="multipart/form-data">
            <div class="form-group">
                <label for="file">Python File:</label>
                <input type="file" name="file" id="file" accept=".py" required>
            </div>
            <div class="form-group">
                <button type="submit" class="btn">Convert to Vector</button>
            </div>
        </form>
    </div>
</body>
</html>"""

        with open(os.path.join(template_dir, "upload_form.html"), "w") as f:
            f.write(upload_template)

        # Advanced form template
        advanced_template = """<!DOCTYPE html>
<html>
<head>
    <title>Advanced Script Processing</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .form-group { margin: 15px 0; }
        .btn { background: #4a6fa5; color: white; padding: 10px 15px; border: none; border-radius: 5px; cursor: pointer; }
        .btn:hover { background: #3a5a8f; }
        textarea { width: 100%; height: 200px; padding: 10px; border-radius: 5px; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Advanced Script Processing</h1>

        <div class="form-group">
            <h2>Find Similar Code</h2>
            <form id="similarForm">
                <textarea placeholder="Paste your Python script here..." required></textarea>
                <div class="form-group">
                    <label>Similarity Threshold: </label>
                    <input type="number" name="threshold" value="0.8" step="0.1" min="0" max="1">
                </div>
                <button type="button" onclick="findSimilar()" class="btn">Find Similar Code</button>
            </form>
            <div id="similarResults" style="margin-top: 15px;"></div>
        </div>

        <div class="form-group">
            <h2>Improve Script</h2>
            <form id="improveForm">
                <textarea placeholder="Paste your Python script here..." required></textarea>
                <div class="form-group">
                    <label>Improvement Type: </label>
                    <select name="improvement_type">
                        <option value="optimize">Optimize</option>
                        <option value="refactor">Refactor</option>
                        <option value="document">Document</option>
                        <option value="test">Test</option>
                    </select>
                </div>
                <button type="button" onclick="improveScript()" class="btn">Get Improvement Suggestions</button>
            </form>
            <div id="improveResults" style="margin-top: 15px;"></div>
        </div>

        <div class="form-group">
            <h2>Analyze Script Quality</h2>
            <form id="analyzeForm">
                <textarea placeholder="Paste your Python script here..." required></textarea>
                <button type="button" onclick="analyzeScript()" class="btn">Analyze Quality</button>
            </form>
            <div id="analyzeResults" style="margin-top: 15px;"></div>
        </div>
    </div>

    <script>
        async function findSimilar() {
            const form = document.getElementById('similarForm');
            const textarea = form.querySelector('textarea');
            const threshold = form.querySelector('input[name="threshold"]').value;
            const resultsDiv = document.getElementById('similarResults');

            if (!textarea.value.trim()) {
                resultsDiv.innerHTML = '<p style="color: red;">Please enter a script</p>';
                return;
            }

            resultsDiv.innerHTML = '<p>Searching for similar code...</p>';

            try {
                const response = await fetch('/find-similar', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        script: textarea.value,
                        threshold: parseFloat(threshold)
                    })
                });

                const data = await response.json();

                if (data.similar_scripts && data.similar_scripts.length > 0) {
                    let html = `<h3>Found ${data.count} similar scripts:</h3><ul>`;
                    data.similar_scripts.forEach(script => {
                        html += `<li>
                            <strong>Similarity: ${(script.similarity * 100).toFixed(1)}%</strong><br>
                            Purpose: ${script.likely_purpose}<br>
                            Suggested use: ${script.suggested_use}<br>
                            CTRM ID: ${script.ctrm_id}
                        </li>`;
                    });
                    html += '</ul>';
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = '<p>No similar scripts found</p>';
                }
            } catch (error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }

        async function improveScript() {
            const form = document.getElementById('improveForm');
            const textarea = form.querySelector('textarea');
            const improvementType = form.querySelector('select[name="improvement_type"]').value;
            const resultsDiv = document.getElementById('improveResults');

            if (!textarea.value.trim()) {
                resultsDiv.innerHTML = '<p style="color: red;">Please enter a script</p>';
                return;
            }

            resultsDiv.innerHTML = '<p>Analyzing script for improvements...</p>';

            try {
                const response = await fetch('/improve-script', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        script: textarea.value,
                        improvement_type: improvementType
                    })
                });

                const data = await response.json();

                if (data.suggestions && data.suggestions.length > 0) {
                    let html = `<h3>Improvement Suggestions (Confidence: ${data.confidence}):</h3><ul>`;
                    data.suggestions.forEach(suggestion => {
                        html += `<li>
                            <strong>${suggestion.type}</strong>: ${suggestion.description}<br>
                            Priority: ${suggestion.priority}
                        </li>`;
                    });
                    html += '</ul>';
                    resultsDiv.innerHTML = html;
                } else {
                    resultsDiv.innerHTML = '<p>No improvement suggestions available</p>';
                }
            } catch (error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }

        async function analyzeScript() {
            const form = document.getElementById('analyzeForm');
            const textarea = form.querySelector('textarea');
            const resultsDiv = document.getElementById('analyzeResults');

            if (!textarea.value.trim()) {
                resultsDiv.innerHTML = '<p style="color: red;">Please enter a script</p>';
                return;
            }

            resultsDiv.innerHTML = '<p>Analyzing script quality...</p>';

            try {
                const response = await fetch('/analyze-script', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        script: textarea.value
                    })
                });

                const data = await response.json();

                let html = `<h3>Script Analysis Results:</h3>
                <p><strong>Quality Score:</strong> ${data.quality_score}</p>

                <h4>Analysis:</h4>
                <pre>${JSON.stringify(data.analysis, null, 2)}</pre>

                <h4>Recommendations:</h4>
                <ul>`;

                data.recommendations.forEach(rec => {
                    html += `<li>${rec}</li>`;
                });

                html += `</ul>

                <h4>Vector Quality:</h4>
                <pre>${JSON.stringify(data.vector_quality, null, 2)}</pre>`;

                resultsDiv.innerHTML = html;
            } catch (error) {
                resultsDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
            }
        }
    </script>
</body>
</html>"""

        with open(os.path.join(template_dir, "advanced_form.html"), "w") as f:
            f.write(advanced_template)

    def run(self):
        """Run the web interface"""
        if not self.available:
            logger.error("Web interface not available - FastAPI not installed")
            return

        logger.info(f"Starting Script2Vec web interface on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=True
        )

    def run_in_background(self):
        """Run the web interface in a background thread"""
        if not self.available:
            return

        import threading
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        return thread

def create_web_interface(host: str = "0.0.0.0", port: int = 8001):
    """Create and return a web interface instance"""
    return Script2VecWebInterface(host, port)

if __name__ == "__main__":
    # Create and run web interface
    web_interface = create_web_interface()
    web_interface.run()