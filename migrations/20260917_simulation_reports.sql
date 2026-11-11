ALTER TABLE public.reports ADD COLUMN IF NOT EXISTS simulation_report JSONB;
NOTIFY pgrst, 'reload schema';
