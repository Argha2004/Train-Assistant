import os
import sys
import json

# Add mcp directory to path
sys.path.append(os.path.abspath("mcp"))

import importlib.util

# Load local server module dynamically to avoid namespace collision with the official 'mcp' package
try:
    spec = importlib.util.spec_from_file_location("local_mcp_server", os.path.abspath("mcp/server.py"))
    local_server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(local_server)
    gpu_monitor = local_server.gpu_monitor
    dataset_analyzer = local_server.dataset_analyzer
    print("Successfully imported local MCP server tools!")
except Exception as e:
    print(f"Failed to import local MCP server tools: {e}")
    sys.exit(1)

def test_gpu_monitor():
    print("\n--- Testing GPU Monitor ---")
    res = gpu_monitor()
    try:
        data = json.loads(res)
        print("GPU Monitor Output parsed successfully:")
        print(json.dumps(data, indent=2))
        assert data["status"] == "success"
        print("[OK] GPU Monitor Test Passed!")
    except Exception as e:
        print(f"[FAIL] GPU Monitor Test Failed: {e}")

def test_dataset_analyzer():
    print("\n--- Testing Dataset Analyzer ---")
    res = dataset_analyzer("sample_dataset_path")
    try:
        data = json.loads(res)
        print("Dataset Analyzer Output parsed successfully:")
        print(json.dumps(data, indent=2))
        assert data["status"] == "success"
        print("[OK] Dataset Analyzer Test Passed!")
    except Exception as e:
        print(f"[FAIL] Dataset Analyzer Test Failed: {e}")

if __name__ == "__main__":
    test_gpu_monitor()
    test_dataset_analyzer()
    print("\nAll MCP tests finished successfully!")
