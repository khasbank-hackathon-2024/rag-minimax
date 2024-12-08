import pandas as pd
import json
from datetime import datetime
import os
from pathlib import Path

def create_knowledge_base(output_path="data/knowledge_base.json"):
    """Create knowledge base JSON file from CSV sources"""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    documents = []
    doc_id = 0
    
    # Load news data
    news_df = pd.read_csv(data_dir / 'khas_bank_news.csv')
    for _, row in news_df.iterrows():
        documents.append({
            "id": doc_id,
            "content": clean_text(row['Content']),
            "metadata": {
                "title": clean_text(row['Title']),
                "link": clean_text(row['Link']),
                "date": clean_text(row['Date']),
                "source_type": "news",
                "created_at": datetime.now().isoformat()
            }
        })
        doc_id += 1

    # Load products data
    products_df = pd.read_csv(data_dir / 'khas_bank_products.csv')
    for _, row in products_df.iterrows():
        documents.append({
            "id": doc_id,
            "content": f"{clean_text(row['Main Text'])}\n{clean_text(row['Side Menu Text'])}",
            "metadata": {
                "title": f"Product {row['Page ID']}",
                "link": clean_text(row['link']),
                "source_type": "product",
                "created_at": datetime.now().isoformat()
            }
        })
        doc_id += 1

    # Load pages data
    pages_df = pd.read_csv(data_dir / 'khas_bank_pages.csv')
    for _, row in pages_df.iterrows():
        documents.append({
            "id": doc_id,
            "content": f"{clean_text(row['Title Text'])}\n{clean_text(row['Main Text'])}",
            "metadata": {
                "title": clean_text(row['Title Text']),
                "link": clean_text(row['link']),
                "source_type": "page",
                "created_at": datetime.now().isoformat()
            }
        })
        doc_id += 1

    # Load branches data
    branches_df = pd.read_csv(data_dir / 'khas_bank_branches.csv')
    print("Branches CSV columns:", branches_df.columns.tolist())
    
    for _, row in branches_df.iterrows():
        # Create working hours string
        working_hours = f"Даваа-Баасан: {row['open_time']}-{row['close_time']}"
        if row['sat'] == 1:
            working_hours += f"\nБямба: {row['open_time']}-{row['close_time']}"
        if row['sun'] == 1:
            working_hours += f"\nНям: {row['open_time']}-{row['close_time']}"
            
        documents.append({
            "id": doc_id,
            "content": f"{clean_text(row['name'])}\n{clean_text(row['address'])}\n{working_hours}\n{clean_text(row['phone'])}",
            "metadata": {
                "title": clean_text(row['name']),
                "address": clean_text(row['address']),
                "working_hours": working_hours,
                "phone": clean_text(row['phone']),
                "type": clean_text(row['type']),
                "source_type": "branch",
                "created_at": datetime.now().isoformat()
            }
        })
        doc_id += 1

    # Create knowledge base structure
    knowledge_base = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "version": "1.0",
            "source": "Хас Банк"
        },
        "documents": documents
    }

    # Save to JSON file
    output_path = project_root / output_path
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, ensure_ascii=False, indent=2)

    print(f"Knowledge base created with {len(documents)} documents")
    print(f"Saved to: {output_path}")
    return knowledge_base

def clean_text(text):
    """Clean text content"""
    if pd.isna(text):
        return ""
    return str(text).strip()

if __name__ == "__main__":
    kb = create_knowledge_base() 