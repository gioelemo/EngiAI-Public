# Assets Folder

This folder contains static assets for the Engineer Assistant UI.

## Files

- `engiai_logo.jpg` - EngiAI logo (robot with gear icon)
  - Used in: Streamlit UI sidebar and page favicon
  - Dimensions: Variable
  - Source: Project branding

## Usage

The logo is automatically loaded by the Streamlit UI when the application starts.

To use the logo elsewhere:
```python
from PIL import Image
logo = Image.open("assets/engiai_logo.jpg")
```
