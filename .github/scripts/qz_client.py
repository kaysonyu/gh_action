"""
Qizhi CI/CD Runner
"""

import os
import sys
import time
import argparse
import requests
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ComputeConfig:
    """Holds platform resource configurations and default job settings."""
    # --- API Configuration ---
    base_url: str = os.getenv("QZ_BASE_URL", "https://qz.sii.edu.cn")
    
    # --- Resource IDs (Environment) ---
    project_id: str = os.getenv("QZ_PROJECT_ID", "project-c67c548f-f02c-453b-ba5b-8745db6886e7")
    workspace_id: str = os.getenv("QZ_WORKSPACE_ID", "ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6")
    group_id: str = os.getenv("QZ_COMPUTE_GROUP_ID", "lcg-79b2ad0e-a375-43f3-a0b1-b4ce79710fd7")
    spec_id: str = os.getenv("QZ_SPEC_ID", "b618f5cb-c119-4422-937e-f39131853076")
    image: str = os.getenv("QZ_IMAGE", "docker.sii.shaipower.online/inspire-studio/sg2:np226-orjson-ffmpeg-tiktoken-250825")

    # --- Default Job Parameters ---
    framework: str = "pytorch"
    task_priority: int = 9
    shm_gi: int = 60
    # API requires time values as strings
    reserve_on_success_ms: str = "60000"    # 1 minute
    reserve_on_fail_ms: str = "3600000"     # 1 hour


class JobFactory:
    """Defines HOW and WHERE the job runs (Business Logic)."""
    
    # Path is specific to this project's job definition
    WORK_DIR = "/inspire/ssd/project/embodied-multimodality/public/mschen/2510-mossLite/mossLite_MossVoiceAXS"
    
    _GIT_SETUP = """
set -e
cd "{work_dir}"

export CI_JOB_ID=$JOB_ID
export GIT_BRANCH={branch}
export GIT_COMMIT={commit}
"""

    @classmethod
    def create_spec(cls, job_type: str, branch: str, commit: str) -> Dict[str, Any]:
        git_cmd = cls._GIT_SETUP.format(
            work_dir=cls.WORK_DIR, 
            branch=branch, 
            commit=commit
        )
        
        specs = {
            "unit": {
                "instances": 1,
                "timeout_ms": 30 * 60 * 1000,
                "command": f"{git_cmd}\n bash tests/run_unit_tests.sh"
            },
            "e2e": {
                "instances": 4,
                "timeout_ms": 120 * 60 * 1000,
                "command": f"{git_cmd}\n bash tests/run_e2e_tests.sh"
            },
            "tmp": {
                "instances": 2,
                "timeout_ms": 120 * 60 * 1000,
                "command": f"{git_cmd}\n echo ${GIT_BRANCH}"
            },
        }
        
        if job_type not in specs:
            raise ValueError(f"Invalid job type: {job_type}. Supported: {list(specs.keys())}")
            
        return specs[job_type]


class QzClient:
    """Handles API communication using ComputeConfig defaults."""

    def __init__(self, config: ComputeConfig):
        self.config = config
        self.username = os.getenv("QZ_USERNAME")
        self.password = os.getenv("QZ_PASSWORD")
        self.token = os.getenv("QZ_TOKEN")

        if not self.token and not (self.username and self.password):
            print("Error: Missing credentials (QZ_USERNAME/QZ_PASSWORD or QZ_TOKEN)", file=sys.stderr)
            sys.exit(1)

    def _authenticate(self):
        """Refreshes the access token."""
        resp = requests.post(
            f"{self.config.base_url}/auth/token",
            json={"username": self.username, "password": self.password},
            timeout=10
        )
        resp.raise_for_status()
        
        payload = resp.json()
        if payload.get("code") != 0:
            raise Exception(f"Auth Logic Error: {payload.get('message')}")

        self.token = payload["data"]["access_token"]

    def _call(self, method: str, path: str, json_data: dict = None, retry: bool = True) -> dict:
        """Executes HTTP request with wrapper handling and 401 retry."""
        if not self.token and self.username:
            self._authenticate()

        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        url = f"{self.config.base_url}{path}"
        
        resp = requests.request(method, url, json=json_data, headers=headers, timeout=30)

        if resp.status_code == 401 and retry and self.username:
            self._authenticate()
            return self._call(method, path, json_data, retry=False)

        if resp.status_code != 200:
            raise Exception(f"API HTTP Error {resp.status_code}: {resp.text}")

        payload = resp.json()
        if payload.get("code") != 0:
            raise Exception(f"Platform Logic Error {payload.get('code')}: {payload.get('message')}")

        return payload.get("data", {})

    def submit_job(self, name: str, spec: dict) -> str:
        """Submits the job using Spec (Job) and Config (Platform) details."""
        payload = {
            "name": name,
            "logic_compute_group_id": self.config.group_id,
            "project_id": self.config.project_id,
            "workspace_id": self.config.workspace_id,
            "auto_fault_tolerance": False,
            
            # --- Job Specifics (from spec) ---
            "command": spec["command"],
            "max_running_time_ms": str(spec["timeout_ms"]),
            
            # --- Platform Defaults (from config) ---
            "reserve_on_success_ms": self.config.reserve_on_success_ms,
            "reserve_on_fail_ms": self.config.reserve_on_fail_ms,
            "framework": self.config.framework,
            "task_priority": self.config.task_priority,
            
            "framework_config": [{
                "image_type": "SOURCE_PUBLIC",
                "image": self.config.image,
                "instance_count": spec["instances"],
                "shm_gi": self.config.shm_gi,
                "spec_id": self.config.spec_id
            }]
        }
        
        data = self._call("POST", "/openapi/v1/train_job/create", payload)
        return data["job_id"]
    
    def monitor_job(self, job_id: str, timeout_sec: int) -> bool:
        """Polls job status until terminal state."""
        start = time.time()
        print(f"[INFO] Monitoring job {job_id} (Timeout: {timeout_sec}s)...", flush=True)

        SUCCESS_STATES = {"job_succeeded"}
        FAILURE_STATES = {"job_stopped", "job_failed"}

        while True:
            if time.time() - start > timeout_sec:
                print(f"[ERROR] Job timed out after {timeout_sec}s", file=sys.stderr)
                return False
            
            try:
                detail = self._call("POST", "/openapi/v1/train_job/detail", {"job_id": job_id})
                status = detail.get("status")
            except Exception as e:
                print(f"[WARN] Failed to get status: {e}. Retrying...", flush=True)
                time.sleep(10)
                continue

            if status in SUCCESS_STATES:
                print(f"[INFO] Job finished successfully: {status}", flush=True)
                return True
            
            if status in FAILURE_STATES:
                print(f"[ERROR] Job failed/stopped with status: {status}", file=sys.stderr)
                return False

            if int(time.time()) % 60 < 10:
                print(f"[INFO] Job is {status}...", flush=True)

            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Qizhi CI/CD Runner")
    parser.add_argument("--type", required=True, choices=["unit", "e2e"], help="Test type")
    parser.add_argument("--branch", required=True, help="Git branch")
    parser.add_argument("--commit", required=True, help="Git commit SHA")
    parser.add_argument("--timeout", type=int, default=7200, help="Wait timeout in seconds")
    
    args = parser.parse_args()

    # 1. Load Platform Config
    config = ComputeConfig()
    client = QzClient(config)
    
    # 2. Generate Job Spec (using internal WORK_DIR)
    spec = JobFactory.create_spec(args.type, args.branch, args.commit)
    job_name = f"CI-{args.type}-{args.commit[:7]}"
    
    # 3. Execute
    job_id = client.submit_job(job_name, spec)
    print(f"[INFO] Job ID: {job_id}", flush=True)
    
    # 4. Monitor
    success = client.monitor_job(job_id, args.timeout)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()