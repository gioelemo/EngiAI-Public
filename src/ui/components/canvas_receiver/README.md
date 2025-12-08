# Canvas Receiver Component

Custom Streamlit component (v2) that receives Excalidraw canvas exports from localStorage.

Uses inline JavaScript - no build step required.

## Requirements

- Python >= 3.10
- Streamlit >= 1.51

## Usage

```python
from src.ui.components.canvas_receiver import canvas_receiver

# Returns base64 image data when available
export_data = canvas_receiver(key="canvas_export")

if export_data:
    # Process the exported canvas
    print("Canvas exported!")
```

## Development (file-backed JS)

The `frontend/` folder contains TypeScript source for building file-backed JS if needed in the future:

```bash
cd frontend
npm install
npm run build
```
