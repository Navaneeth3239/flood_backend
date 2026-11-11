import re

with open("/Users/vijay/Documents/flash_flood/frontend/src/pages/DashboardPages.tsx", "r") as f:
    content = f.read()

# Add import
if 'import { supabase } from "@/lib/supabase"' not in content:
    content = content.replace(
        'import { Link, useLocation, Navigate } from "react-router-dom";',
        'import { Link, useLocation, Navigate } from "react-router-dom";\nimport { supabase } from "@/lib/supabase";'
    )

# Replace useState hook for customAreas with useEffect
new_state = """  const [customAreas, setCustomAreas] = useState<CustomArea[]>([]);

  import { useEffect } from "react";
  useEffect(() => {
    supabase.from('custom_areas').select('*').then(({ data, error }) => {
      if (data) {
        setCustomAreas(data.map(d => ({
          ...d.data,
          id: d.id,
          lat: d.data.lat,
          lng: d.data.lng,
          shape: d.data.shape,
          date: d.created_at
        })));
      }
    });
  }, []);
"""

content = content.replace("  const [customAreas, setCustomAreas] = useState<CustomArea[]>([]);", new_state)

# Replace handleSaveArea to insert into Supabase
handle_save_old = """  const handleSaveArea = () => {
    if (!currentBounds || !formData.name) return;
    const center = currentBounds.getCenter();
    const newArea: CustomArea = {
      id: Math.random().toString(36).substr(2, 9),
      ...formData,
      bounds: currentBounds,
      date: new Date().toLocaleDateString(),
      lat: Number(center.lat.toFixed(4)),
      lng: Number(center.lng.toFixed(4)),
      shape: "Rectangle" // Could be dynamic based on selection tool
    };
    setCustomAreas([newArea, ...customAreas]);
    setShowModal(false);
    setFormData({ name: "", district: "", type: "Forest", risk: "Medium", priority: "Normal", description: "" });
    setCurrentBounds(null);
  };"""

handle_save_new = """  const handleSaveArea = async () => {
    if (!currentBounds || !formData.name) return;
    const center = currentBounds.getCenter();
    const newArea: CustomArea = {
      id: Math.random().toString(36).substr(2, 9),
      ...formData,
      bounds: currentBounds,
      date: new Date().toLocaleDateString(),
      lat: Number(center.lat.toFixed(4)),
      lng: Number(center.lng.toFixed(4)),
      shape: "Rectangle" // Could be dynamic based on selection tool
    };
    
    const { data } = await supabase.auth.getSession();
    await supabase.from('custom_areas').insert({
      id: newArea.id,
      data: newArea
    });

    setCustomAreas([newArea, ...customAreas]);
    setShowModal(false);
    setFormData({ name: "", district: "", type: "Forest", risk: "Medium", priority: "Normal", description: "" });
    setCurrentBounds(null);
  };"""

content = content.replace(handle_save_old, handle_save_new)

with open("/Users/vijay/Documents/flash_flood/frontend/src/pages/DashboardPages.tsx", "w") as f:
    f.write(content)
