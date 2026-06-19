import os
import json
import time
from datetime import datetime
from typing import Dict, Any

class InferenceLogger:
    def __init__(self, log_dir: str, max_size_mb: int = 100):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"inference_{datetime.now().strftime('%Y%m%d')}.log")
        self.max_size_mb = max_size_mb

    def _rotate_if_needed(self):
        if os.path.exists(self.log_file):
            size_mb = os.path.getsize(self.log_file) / (1024 * 1024)
            if size_mb > self.max_size_mb:
                # 重命名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = f"{self.log_file}.{timestamp}.bak"
                os.rename(self.log_file, backup)

    def log(self, entry: Dict[str, Any]):
        self._rotate_if_needed()
        entry['timestamp'] = datetime.now().isoformat()
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')