from typing import Dict, List
import json
import os
from datetime import datetime

class KnowledgeBase:
    def __init__(self, json_path: str = "data/knowledge_base.json"):
        self.json_path = json_path
        self.data = self._load_or_create()
        
    def _load_or_create(self) -> Dict:
        """Load existing or create new knowledge base"""
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "documents": []
        }

    def add_documents(self, documents: List[Dict]):
        """Add new documents to knowledge base"""
        for doc in documents:
            self.data["documents"].append({
                "id": len(self.data["documents"]),
                "content": doc["content"],
                "metadata": doc["metadata"],
                "created_at": datetime.now().isoformat()
            })
        self._save()
        
    def _save(self):
        """Save knowledge base to JSON file"""
        self.data["metadata"]["last_updated"] = datetime.now().isoformat()
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)