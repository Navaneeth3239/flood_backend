import os
import asyncio
import asyncpg

async def main():
    pw = os.environ.get("SUPABASE_DB_PASSWORD", "")
    host = os.environ.get("SUPABASE_DB_HOST", "db.odqmpcizoqatwxyrnlmj.supabase.co")
    if not pw:
        print("SUPABASE_DB_PASSWORD not set")
        return
    try:
        conn = await asyncpg.connect(user="postgres", password=pw, database="postgres", host=host, port=5432)
        print("Connected via standard db host!")
        await conn.close()
    except Exception as e:
        print(f"Failed db connection: {e}")

asyncio.run(main())
