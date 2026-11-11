import asyncio
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv(".env")
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])

async def main():
    try:
        # Just check if we can query users
        res = supabase.table("users").select("*").limit(1).execute()
        print("Success:", res.data)
    except Exception as e:
        print("Error:", e)

asyncio.run(main())
