-- One complete snapshot per layer and exact boundary. Empty snapshots are valid.
CREATE TABLE IF NOT EXISTS public.area_map_layers (
    area_id uuid NOT NULL REFERENCES public.custom_areas(id) ON DELETE CASCADE,
    boundary_key text NOT NULL,
    layer text NOT NULL CHECK (layer IN ('roads', 'rivers', 'buildings')),
    geojson jsonb NOT NULL CHECK (jsonb_typeof(geojson->'features') = 'array'),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (area_id, boundary_key, layer)
);
ALTER TABLE public.area_map_layers ENABLE ROW LEVEL SECURITY;
GRANT ALL ON public.area_map_layers TO service_role;
-- Old feature rows have no foreign key. Clean up exact legacy keys in the same
-- transaction as the area deletion, including deletes from the Supabase UI.
CREATE OR REPLACE FUNCTION public.delete_legacy_area_networks() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
BEGIN
    DELETE FROM public.digital_twin_network_features
    WHERE area_key IN (OLD.id::text, 'dt-area-' || OLD.id::text);
    RETURN OLD;
END;
$$;
DROP TRIGGER IF EXISTS delete_legacy_area_networks ON public.custom_areas;
CREATE TRIGGER delete_legacy_area_networks AFTER DELETE ON public.custom_areas
FOR EACH ROW EXECUTE FUNCTION public.delete_legacy_area_networks();
NOTIFY pgrst, 'reload schema';
