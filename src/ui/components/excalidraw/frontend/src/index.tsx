import React, { useCallback, useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { Excalidraw, exportToBlob } from "@excalidraw/excalidraw";
import { Streamlit, RenderData } from "streamlit-component-lib";

interface ExcalidrawAppProps {
  height: number;
}

const ExcalidrawApp: React.FC<ExcalidrawAppProps> = ({ height }) => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const excalidrawApiRef = useRef<any>(null);
  const [isExporting, setIsExporting] = useState(false);
  const [buttonText, setButtonText] = useState("📤 Send to Chat");

  // Notify Streamlit that we're ready
  useEffect(() => {
    Streamlit.setFrameHeight(height);
  }, [height]);

  const handleExport = useCallback(async () => {
    const api = excalidrawApiRef.current;
    if (!api) {
      alert("Excalidraw not ready yet!");
      return;
    }

    const elements = api.getSceneElements();
    if (!elements || elements.length === 0) {
      alert("Canvas is empty! Draw something first.");
      return;
    }

    setIsExporting(true);
    setButtonText("⏳ Exporting...");

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
        setButtonText("✅ Sent!");

        setTimeout(() => {
          setButtonText("📤 Send to Chat");
          setIsExporting(false);
        }, 2000);
      };
      reader.readAsDataURL(blob);
    } catch (error) {
      alert("Export failed: " + (error as Error).message);
      setButtonText("📤 Send to Chat");
      setIsExporting(false);
    }
  }, []);

  return (
    <div style={{ height: `${height}px`, width: "100%", position: "relative" }}>
      <div style={{ height: `${height - 50}px`, width: "100%" }}>
        <Excalidraw
          excalidrawAPI={(api) => {
            excalidrawApiRef.current = api;
          }}
        />
      </div>
      <button
        onClick={handleExport}
        disabled={isExporting}
        style={{
          position: "absolute",
          bottom: "10px",
          left: "50%",
          transform: "translateX(-50%)",
          padding: "10px 20px",
          background: isExporting ? "#999" : "#6965db",
          color: "white",
          border: "none",
          borderRadius: "8px",
          cursor: isExporting ? "not-allowed" : "pointer",
          fontSize: "14px",
          fontWeight: 600,
          boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
          zIndex: 1000,
        }}
      >
        {buttonText}
      </button>
    </div>
  );
};

// Main app that listens for Streamlit render events
const App: React.FC = () => {
  const [height, setHeight] = useState(650);

  useEffect(() => {
    const onRender = (event: CustomEvent<RenderData>) => {
      const data = event.detail;
      if (data.args.height) {
        setHeight(data.args.height);
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

  return <ExcalidrawApp height={height} />;
};

// Mount the app
const container = document.getElementById("root");
if (container) {
  const root = createRoot(container);
  root.render(<App />);
}
