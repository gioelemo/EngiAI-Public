import { Component, ComponentArgs } from "@streamlit/component-v2-lib";

export type ComponentState = {
  export_data: string | null;
};

export type ComponentData = Record<string, never>;

// Track polling intervals per component instance
const instances: WeakMap<
  ComponentArgs["parentElement"],
  { intervalId: ReturnType<typeof setInterval>; lastSent: string | null }
> = new WeakMap();

const CanvasReceiver: Component<ComponentState, ComponentData> = (args) => {
  const { parentElement, setStateValue } = args;

  // Check if we already have an interval running for this instance
  if (!instances.has(parentElement)) {
    // Start polling localStorage for canvas exports
    const intervalId = setInterval(() => {
      try {
        const exportData = localStorage.getItem("excalidraw_pending_export");

        if (exportData) {
          // Always clear the localStorage item first
          localStorage.removeItem("excalidraw_pending_export");

          // Get the instance data
          const instanceData = instances.get(parentElement);

          // Only send if it's different from what we last sent
          if (instanceData && instanceData.lastSent !== exportData) {
            setStateValue("export_data", exportData);
            instanceData.lastSent = exportData;
            console.log(
              "Canvas export sent to Streamlit:",
              exportData.substring(0, 100) + "..."
            );
          }
        }
      } catch (error) {
        console.error("Error checking for canvas export:", error);
      }
    }, 500); // Check every 500ms

    instances.set(parentElement, { intervalId, lastSent: null });
  }

  // Cleanup function - called when component is unmounted
  return () => {
    const instanceData = instances.get(parentElement);
    if (instanceData) {
      clearInterval(instanceData.intervalId);
      instances.delete(parentElement);
    }
  };
};

export default CanvasReceiver;
