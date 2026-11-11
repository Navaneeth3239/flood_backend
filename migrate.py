import asyncio
import asyncpg
import json
import uuid

from lib.db import db as mongodb

async def main():
    # 1. Connect to MongoDB to extract data
    print("Extracting data from MongoDB...")
    users = await mongodb.users.find().to_list(None)
    zones = await mongodb.zones.find().to_list(None)
    sensors = await mongodb.sensors.find().to_list(None)
    alerts = await mongodb.alerts.find().to_list(None)
    stats = await mongodb.stats.find().to_list(None)
    print(f"Extracted {len(users)} users, {len(zones)} zones, {len(sensors)} sensors, {len(alerts)} alerts, {len(stats)} stats.")

    # 2. Connect to Supabase Postgres
    print("Connecting to Supabase Postgres...")
    db_pw = os.environ.get("SUPABASE_DB_PASSWORD", "")
    host = os.environ.get("SUPABASE_DB_HOST", "db.odqmpcizoqatwxyrnlmj.supabase.co")
    if not db_pw:
        print("SUPABASE_DB_PASSWORD not set. Aborting.")
        return
    conn = await asyncpg.connect(
        user="postgres", 
        password=db_pw, 
        database="postgres", 
        host=host, 
        port=5432
    )

    # 3. Create Tables
    print("Creating tables...")
    tables = ["users", "zones", "sensors", "alerts", "stats", "custom_areas"]
    for t in tables:
        await conn.execute(f"CREATE TABLE IF NOT EXISTS {t} (id VARCHAR PRIMARY KEY, data JSONB);")
        await conn.execute(f"TRUNCATE TABLE {t};")

    # 4. Insert Data
    print("Inserting data...")
    for u in users:
        u.pop("_id", None)
        await conn.execute("INSERT INTO users (id, data) VALUES ($1, $2)", u["id"], json.dumps(u))
        
    for z in zones:
        z.pop("_id", None)
        await conn.execute("INSERT INTO zones (id, data) VALUES ($1, $2)", z["id"], json.dumps(z))
        
    for s in sensors:
        s.pop("_id", None)
        await conn.execute("INSERT INTO sensors (id, data) VALUES ($1, $2)", s["id"], json.dumps(s))
        
    for a in alerts:
        a.pop("_id", None)
        await conn.execute("INSERT INTO alerts (id, data) VALUES ($1, $2)", a["id"], json.dumps(a))
        
    for st in stats:
        st.pop("_id", None)
        await conn.execute("INSERT INTO stats (id, data) VALUES ($1, $2)", str(uuid.uuid4()), json.dumps(st))

    print("Migration complete!")
    await conn.close()

asyncio.run(main())
