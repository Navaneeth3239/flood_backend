"""Apply the additive report-data migration using the configured PostgreSQL connection.

Set SUPABASE_DB_HOST and SUPABASE_DB_USER for a pooler connection when required.
No credentials are printed. Does not touch existing reports or other tables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv


def main():
    import psycopg
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")
    project = os.environ["SUPABASE_URL"].split("//", 1)[1].split(".")[0]
    password = os.environ.get("SUPABASE_DB_PASSWORD")
    if not password:
        raise SystemExit("Set SUPABASE_DB_PASSWORD in backend/.env first")
    host = os.environ.get("SUPABASE_DB_HOST", f"db.{project}.supabase.co")
    user = os.environ.get("SUPABASE_DB_USER", f"postgres.{project}" if "pooler.supabase.com" in host else "postgres")
    try:
        with psycopg.connect(host=host, user=user, password=password, dbname="postgres",
                              port=int(os.environ.get("SUPABASE_DB_PORT", "5432")),
                              sslmode="require", connect_timeout=15) as connection:
            connection.execute((root / "migrations" / "20260917_simulation_reports.sql").read_text())
        print("Simulation report column is ready.")
    except psycopg.Error:
        raise SystemExit("Database connection/migration failed. Check SUPABASE_DB_HOST, SUPABASE_DB_USER and SUPABASE_DB_PASSWORD, or run the migration in the Supabase SQL editor.")


if __name__ == "__main__":
    main()
