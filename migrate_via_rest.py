import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import json
from datetime import datetime

load_dotenv()
mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
mongodb = client[os.environ["DB_NAME"]]

supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])

def serialize_dt(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError()

async def main():
    print("Extracting data from MongoDB...")
    zones = await mongodb.zones.find().to_list(None)
    sensors = await mongodb.sensors.find().to_list(None)
    alerts = await mongodb.alerts.find().to_list(None)
    stats = await mongodb.stats.find().to_list(None)
    print(f"Extracted {len(zones)} zones, {len(sensors)} sensors, {len(alerts)} alerts, {len(stats)} stats.")

    tables = [
        ("zones", zones),
        ("sensors", sensors),
        ("alerts", alerts),
        ("stats", stats)
    ]

    for table, docs in tables:
        print(f"Uploading {len(docs)} docs to {table}...")
        for doc in docs:
            doc.pop("_id", None)
            
            clean_doc = json.loads(json.dumps(doc, default=serialize_dt))
            
            res = supabase.table(table).upsert({"id": doc["id"], "data": clean_doc}).execute()
            if not res.data:
                print("Failed to upsert:", res)

    print("Migration complete!")

asyncio.run(main())
