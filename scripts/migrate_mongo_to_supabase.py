"""Migrate all operational data from local MongoDB to cloud Supabase PostgreSQL.
Uses the Supabase Service Role Secret Key to bypass RLS during batch migration.
"""
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from supabase import create_client, Client

ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "app")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SECRET_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")

if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    print("Error: SUPABASE_URL and SUPABASE_SECRET_KEY must be set in backend/.env")
    sys.exit(1)


def clean_doc(doc: dict) -> dict:
    """Removes MongoDB _id and formats datetimes to ISO 8601 strings."""
    doc.pop("_id", None)
    cleaned = {}
    for k, v in doc.items():
        if isinstance(v, datetime):
            cleaned[k] = v.isoformat()
        elif isinstance(v, dict):
            cleaned[k] = clean_doc(v)
        elif isinstance(v, list):
            cleaned[k] = [clean_doc(item) if isinstance(item, dict) else item for item in v]
        else:
            cleaned[k] = v
    return cleaned


async def migrate():
    print(f"Connecting to MongoDB at {MONGO_URL} (db: {DB_NAME})...")
    mongo_client = AsyncIOMotorClient(MONGO_URL)
    mongodb = mongo_client[DB_NAME]

    print(f"Connecting to Supabase at {SUPABASE_URL}...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)

    # Collections to migrate in dependency order
    collections = [
        "users",
        "zones",
        "gateways",
        "sensors",
        "alerts",
        "simulations",
        "reports",
        "notifications",
    ]

    total_migrated = 0
    results = {}

    for col_name in collections:
        try:
            mongo_docs = await mongodb[col_name].find().to_list(None)
            count = len(mongo_docs)
            if count == 0:
                print(f"[-] {col_name}: 0 records found in MongoDB, skipping.")
                results[col_name] = 0
                continue

            print(f"[>] Migrating {count} records from MongoDB collection '{col_name}' to Supabase...")
            cleaned_batch = [clean_doc(d) for d in mongo_docs]

            # Upsert in chunks of 50 to respect payload limits
            chunk_size = 50
            for i in range(0, len(cleaned_batch), chunk_size):
                chunk = cleaned_batch[i : i + chunk_size]
                res = supabase.table(col_name).upsert(chunk).execute()

            # Verify count in Supabase
            verify_res = supabase.table(col_name).select("id", count="exact").execute()
            supabase_count = verify_res.count if verify_res.count is not None else len(verify_res.data or [])
            print(f"[✓] {col_name}: Successfully uploaded {count} rows (Supabase count: {supabase_count})")
            results[col_name] = count
            total_migrated += count

        except Exception as exc:
            print(f"[!] Error migrating {col_name}: {exc}")
            results[col_name] = f"FAILED: {exc}"

    print("\n" + "=" * 50)
    print("MIGRATION SUMMARY:")
    for col, stat in results.items():
        print(f"  - {col}: {stat}")
    print(f"Total records migrated: {total_migrated}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(migrate())
