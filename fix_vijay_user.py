import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])

def main():
    email = "vijay6432e@gmail.com"
    
    # List users to find the existing one
    users = supabase.auth.admin.list_users()
    vijay = None
    for u in users:
        if u.email == email:
            vijay = u
            break
    
    if not vijay:
        print("User not found!")
        return
    
    print(f"Found user: {vijay.id}")
    
    # Confirm email so they can login
    supabase.auth.admin.update_user_by_id(vijay.id, {"email_confirm": True})
    
    # Reset password to a known value
    supabase.auth.admin.update_user_by_id(vijay.id, {"password": "Vijay@2008"})
    
    # Check if profile exists
    profile = supabase.table("user_profiles").select("*").eq("id", vijay.id).execute()
    if not profile.data:
        # Insert profile
        supabase.table("user_profiles").insert({
            "id": vijay.id,
            "first_name": "Vijay",
            "last_name": "Vijay",
            "phone": "9003899180",
            "organization": "none",
            "designation": "Forest Officer",
            "role": "admin",
            "state": "Tamil Nadu",
            "district": "pollachi"
        }).execute()
        print("Profile created!")
    else:
        print("Profile already exists!")
    
    print(f"Done! Login with: {email} / Vijay@2008")

main()
