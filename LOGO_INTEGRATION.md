# 🎨 Logo Integration Instructions

## Logo Added to Streamlit UI

I've integrated the EngiAI logo into your Streamlit interface!

## Where to Place the Logo File

**Please save the logo image to:**
```
/Users/gioelemolinari/Desktop/engineer-assistant/assets/engiai_logo.jpg
```

The `assets/` folder has been created and is ready for the logo file.

## Logo Locations in the UI

### 1. **Sidebar** (Top)
- Logo displays at full width in the sidebar
- Below the logo: "Engineering Design Chatbot" subtitle
- Clean, professional header

### 2. **Main Page Header** (Centered)
- Logo displays centered at 200px width
- Below the logo: "Engineering Design Chatbot" subtitle
- Centered using a 3-column layout

### 3. **Page Icon/Favicon** (Browser Tab)
- Logo appears in the browser tab
- Fallback to 🤖 emoji if logo fails to load

## How to Add the Logo

### Method 1: Manual Copy
1. Save the attached image as `engiai_logo.jpg`
2. Place it in the `assets/` folder:
   ```bash
   cp path/to/your/logo.jpg assets/engiai_logo.jpg
   ```

### Method 2: Direct Save
1. Right-click the logo image
2. Save as `engiai_logo.jpg`
3. Move to: `/Users/gioelemolinari/Desktop/engineer-assistant/assets/`

## Features

✅ **Sidebar Logo** - Full-width display at top
✅ **Main Page Logo** - Centered, 200px width
✅ **Page Icon** - Shows in browser tab
✅ **Graceful Fallback** - Text title if logo not found
✅ **Error Handling** - Won't crash if logo missing
✅ **Professional Branding** - Consistent across UI

## Code Changes Made

### Updated Functions:
1. `render_sidebar()` - Added logo at top of sidebar
2. `main()` - Added centered logo in main area
3. `st.set_page_config()` - Uses logo as page icon

### Branding Updates:
- Changed title to "EngiAI"
- Added subtitle "Engineering Design Chatbot"
- Page title now: "EngiAI - Engineering Design Chatbot"

## Testing

Once the logo is in place:

```bash
streamlit run src/ui/streamlit_app.py
```

You should see:
- Logo in sidebar (full width)
- Logo in main area (centered, 200px)
- Logo in browser tab icon

## Customization

### Change Logo Size in Sidebar
Edit in `render_sidebar()`:
```python
st.image(logo, use_column_width=True)  # Full width (current)
st.image(logo, width=300)               # Fixed width
```

### Change Logo Size in Main Area
Edit in `main()`:
```python
st.image(logo, width=200)  # Current
st.image(logo, width=300)  # Larger
st.image(logo, width=150)  # Smaller
```

### Change Logo Position
Current: Centered with 3 columns `[1, 2, 1]`

For left-aligned:
```python
col1, col2 = st.columns([1, 3])
with col1:
    st.image(logo, width=200)
```

## File Structure

```
engineer-assistant/
├── assets/
│   ├── README.md
│   └── engiai_logo.jpg  ← Place logo here
├── src/
│   └── ui/
│       └── streamlit_app.py  ← Logo integration code
└── ...
```

## Fallback Behavior

If logo file is missing:
- Sidebar shows: "🤖 EngiAI" text title
- Main area shows: "💬 EngiAI" text title
- Browser tab shows: 🤖 emoji
- No errors or crashes

## Alternative Formats

The code currently expects `.jpg` format. To use PNG or other formats:

1. Save logo as desired format
2. Update filename in code (3 locations):
   ```python
   logo_path = project_root / "assets" / "engiai_logo.png"  # Change extension
   ```

## Next Steps

1. ✅ Save logo image to `assets/engiai_logo.jpg`
2. ✅ Launch UI: `streamlit run src/ui/streamlit_app.py`
3. ✅ Verify logo appears in sidebar, main area, and browser tab
4. ✅ Adjust sizing if needed (see Customization section)

---

**Ready!** Just add the logo file and the UI will automatically display it! 🎨
