import os
import json
import uuid
from datetime import datetime
from threading import Lock

DB_FILE = "./uploads/db.json"

class JSONDatabase:
    def __init__(self):
        self.lock = Lock()
        os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
        with self.lock:
            if not os.path.exists(DB_FILE):
                self.data = {
                    "conversations": {},
                    "uploads": [
                        {
                            "id": "sample-run-a-uuid",
                            "filename": "sample_run.csv",
                            "size_bytes": 1024,
                            "columns": ["epoch", "loss", "val_loss", "accuracy", "val_accuracy", "val_auc", "lr"],
                            "uploaded_at": "2026-06-22T10:00:00.000000"
                        },
                        {
                            "id": "sample-run-b-uuid",
                            "filename": "sample_run_b.csv",
                            "size_bytes": 1024,
                            "columns": ["epoch", "loss", "val_loss", "accuracy", "val_accuracy", "val_auc", "lr"],
                            "uploaded_at": "2026-06-22T10:05:00.000000"
                        }
                    ],
                    "reports": [],
                    "settings": {
                        "gemini_model": "gemini-2.5-flash",
                        "theme": "dark",
                        "system_prompt": "You are a senior ML engineer and AI researcher. You help users analyze training logs and recommend adjustments. Explain your logic step by step. Use the available MCP tools.",
                        "upload_dir": "./uploads",
                        "report_dir": "./reports",
                        "auto_save": True,
                        "streaming_toggle": True
                    }
                }
                self._save_unlocked()
            else:
                self._load_unlocked()

    def _load_unlocked(self):
        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except Exception:
            self.data = {
                "conversations": {},
                "uploads": [],
                "reports": [],
                "settings": {
                    "gemini_model": "gemini-2.5-flash",
                    "theme": "dark",
                    "system_prompt": "You are a senior ML engineer and AI researcher. You help users analyze training logs and recommend adjustments. Explain your logic step by step. Use the available MCP tools.",
                    "upload_dir": "./uploads",
                    "report_dir": "./reports",
                    "auto_save": True,
                    "streaming_toggle": True
                }
            }
            self._save_unlocked()

    def _save_unlocked(self):
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)

    def load(self):
        with self.lock:
            self._load_unlocked()
            return self.data

    def save(self):
        with self.lock:
            self._save_unlocked()

    # Conversation methods
    def get_conversations(self):
        with self.lock:
            self._load_unlocked()
            return [
                {
                    "id": cid,
                    "title": cinfo.get("title", "Untitled Chat"),
                    "created_at": cinfo.get("created_at"),
                    "updated_at": cinfo.get("updated_at"),
                }
                for cid, cinfo in self.data["conversations"].items()
            ]

    def get_conversation(self, cid):
        with self.lock:
            self._load_unlocked()
            return self.data["conversations"].get(cid)

    def save_conversation(self, cid, title, messages):
        with self.lock:
            self._load_unlocked()
            now = datetime.utcnow().isoformat()
            if cid not in self.data["conversations"]:
                self.data["conversations"][cid] = {
                    "id": cid,
                    "title": title,
                    "created_at": now,
                    "updated_at": now,
                    "messages": []
                }
            self.data["conversations"][cid]["messages"] = messages
            self.data["conversations"][cid]["updated_at"] = now
            self._save_unlocked()

    def update_conversation_title(self, cid, title):
        with self.lock:
            self._load_unlocked()
            if cid in self.data["conversations"]:
                self.data["conversations"][cid]["title"] = title
                self.data["conversations"][cid]["updated_at"] = datetime.utcnow().isoformat()
                self._save_unlocked()
                return True
            return False

    def delete_conversation(self, cid):
        with self.lock:
            self._load_unlocked()
            if cid in self.data["conversations"]:
                del self.data["conversations"][cid]
                self._save_unlocked()
                return True
            return False

    # Upload methods
    def add_upload(self, filename, size, columns):
        with self.lock:
            self._load_unlocked()
            upload_item = {
                "id": str(uuid.uuid4()),
                "filename": filename,
                "size_bytes": size,
                "columns": columns,
                "uploaded_at": datetime.utcnow().isoformat()
            }
            self.data["uploads"].append(upload_item)
            self._save_unlocked()
            return upload_item

    def get_uploads(self):
        with self.lock:
            self._load_unlocked()
            return self.data["uploads"]

    def delete_upload(self, uid):
        with self.lock:
            self._load_unlocked()
            new_uploads = [u for u in self.data["uploads"] if u["id"] != uid]
            if len(new_uploads) < len(self.data["uploads"]):
                deleted = [u for u in self.data["uploads"] if u["id"] == uid]
                if deleted:
                    filename = deleted[0]["filename"]
                    # Remove from uploads dir
                    upload_path = os.path.join(self.data["settings"].get("upload_dir", "./uploads"), filename)
                    if os.path.exists(upload_path):
                        try:
                            os.remove(upload_path)
                        except Exception:
                            pass
                    # Remove from training_logs dir
                    logs_path = os.path.join("./training_logs", filename)
                    if os.path.exists(logs_path):
                        try:
                            os.remove(logs_path)
                        except Exception:
                            pass
                self.data["uploads"] = new_uploads
                self._save_unlocked()
                return True
            return False

    # Report methods
    def add_report(self, report_name, filename, path):
        with self.lock:
            self._load_unlocked()
            report_item = {
                "id": str(uuid.uuid4()),
                "name": report_name,
                "filename": filename,
                "path": path,
                "created_at": datetime.utcnow().isoformat()
            }
            self.data["reports"].append(report_item)
            self._save_unlocked()
            return report_item

    def get_reports(self):
        with self.lock:
            self._load_unlocked()
            return self.data["reports"]

    def delete_report(self, rid):
        with self.lock:
            self._load_unlocked()
            new_reports = [r for r in self.data["reports"] if r["id"] != rid]
            if len(new_reports) < len(self.data["reports"]):
                # find file path to delete file
                deleted = [r for r in self.data["reports"] if r["id"] == rid]
                if deleted and os.path.exists(deleted[0]["path"]):
                    try:
                        os.remove(deleted[0]["path"])
                    except Exception:
                        pass
                self.data["reports"] = new_reports
                self._save_unlocked()
                return True
            return False

    # Settings methods
    def get_settings(self):
        with self.lock:
            self._load_unlocked()
            return self.data["settings"]

    def update_settings(self, settings_dict):
        with self.lock:
            self._load_unlocked()
            self.data["settings"].update(settings_dict)
            self._save_unlocked()
            return self.data["settings"]

db = JSONDatabase()
