import asyncio
from lib.db import verify_database_connection, get_async_db

async def main():
    try:
        await verify_database_connection()
        print("SUCCESS: MongoDB Atlas connection verified!")
        db = get_async_db()
        count = await db.custom_areas.count_documents({})
        print(f"Total custom areas in MongoDB: {count}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(main())
