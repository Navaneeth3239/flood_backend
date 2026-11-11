import asyncio
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])

def main():
    email = "test2@gmail.com"
    password = "password123"
    try:
        user = supabase.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True
        })
        print("User created successfully!")
        
        # Insert into user_profiles
        supabase.table("user_profiles").insert({
            "id": user.user.id,
            "first_name": "Test",
            "last_name": "Admin",
            "phone": "9999999999",
            "organization": "Test Org",
            "designation": "Admin",
            "role": "admin",
            "state": "Delhi",
            "district": "New Delhi"
        }).execute()
        print(f"Profile created for {email}! Password: {password}")
    except Exception as e:
        print("Error:", e)

main()
