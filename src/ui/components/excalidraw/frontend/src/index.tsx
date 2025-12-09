import React, { useCallback, useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { Excalidraw, exportToBlob } from "@excalidraw/excalidraw";
import { Streamlit, RenderData } from "streamlit-component-lib";

interface ExcalidrawAppProps {
  height: number;
  triggerExport: boolean;
}

const ExcalidrawApp: React.FC<ExcalidrawAppProps> = ({ height, triggerExport }) => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const excalidrawApiRef = useRef<any>(null);
  const lastTriggerRef = useRef(false);

  // Notify Streamlit that we're ready
  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  const handleExport = useCallback(async () => {
    const api = excalidrawApiRef.current;
    if (!api) {
      console.error("Excalidraw not ready yet!");
      return;
    }

    const elements = api.getSceneElements();
    if (!elements || elements.length === 0) {
      // Send null to indicate empty canvas
      Streamlit.setComponentValue(null);
      return;
    }

    try {
      const blob = await exportToBlob({
        elements: elements,
        appState: api.getAppState(),
        files: api.getFiles(),
        mimeType: "image/png",
        quality: 0.95,
      });

      // Convert blob to base64
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64data = reader.result as string;
        // Send data back to Streamlit
        Streamlit.setComponentValue(base64data);
      };
      reader.readAsDataURL(blob);
    } catch (error) {
      console.error("Export failed:", error);
      Streamlit.setComponentValue(null);
    }
  }, []);

  // Watch for trigger_export prop changes
  useEffect(() => {
    if (triggerExport && !lastTriggerRef.current) {
      handleExport();
    }
    lastTriggerRef.current = triggerExport;
  }, [triggerExport, handleExport]);

  return (
    <div style={{ height: `${height}px`, width: "100%", position: "relative" }}>
      <Excalidraw
        excalidrawAPI={(api) => {
          excalidrawApiRef.current = api;
        }}
      />
    </div>
  );
};

// Main app that listens for Streamlit render events
const App: React.FC = () => {
  const [height, setHeight] = React.useState(650);
  const [triggerExport, setTriggerExport] = React.useState(false);

  useEffect(() => {
    const onRender = (event: CustomEvent<RenderData>) => {
      const data = event.detail;
      if (data.args.height) {
        setHeight(data.args.height);
      }
      if (data.args.trigger_export !== undefined) {
        setTriggerExport(data.args.trigger_export);
      }
      Streamlit.setFrameHeight(data.args.height || 650);
    };

    Streamlit.events.addEventListener(
      Streamlit.RENDER_EVENT,
      onRender as EventListener
    );
    Streamlit.setComponentReady();

    return () => {
      Streamlit.events.removeEventListener(
        Streamlit.RENDER_EVENT,
        onRender as EventListener
      );
    };
  }, []);

  return <ExcalidrawApp height={height} triggerExport={triggerExport} />;
};

// Mount the app
const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
