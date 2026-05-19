# Streamlit UI for EngiAI

This directory contains the web-based user interface for the EngiAI chatbot.

## Features

- 💬 **Chat Interface**: Clean, intuitive chat interface for interacting with the multi-agent system
- 🔧 **Tool Visualization**: See which tools the agents are using in real-time
- 🖼️ **Image Gallery**: View all generated designs and visualizations in the sidebar
- 📸 **Inline Images**: Images mentioned in messages are automatically displayed
- 🎨 **3D Model Viewer**: Interactive STL file visualization with auto-rotate and zoom controls
- 🏛️ **3D Model Gallery**: Browse and interact with all 3D printable models in the sidebar
- � **File Downloads**: Download generated designs and outputs directly from the UI
- 🗑️ **Conversation Management**: Clear and restart conversations as needed
- 🎨 **Responsive Design**: Works on desktop and mobile browsers

## Quick Start

### Option 1: Using the Launch Script

```bash
make run-ui
```

### Option 2: Manual Launch

```bash
# Activate the conda environment
conda activate engiai

# Run Streamlit
streamlit run src/ui/streamlit_app.py
```

### Option 3: Using Python Module

```bash
# From the project root
python -m streamlit run src/ui/streamlit_app.py
```

The application will open automatically in your default browser at `http://localhost:8501`

## Usage

1. **Ask Questions**: Type your engineering questions in the chat input at the bottom
2. **View Responses**: See the agent's responses, including tool usage and results
3. **Download Outputs**: Use the sidebar to download any generated files (`.npy` designs)
4. **Clear Conversation**: Click the "Clear Conversation" button in the sidebar to start fresh

## Configuration

The UI automatically uses the configuration from `config.py`. Make sure your environment variables are set:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `GOOGLE_API_KEY`: Your Google API key (required)
- `TAVILY_API_KEY`: Tavily search API key (for search agent)

## Customization

You can customize the UI by editing `src/ui/streamlit_app.py`:

- **Styling**: Modify the Streamlit theme in `.streamlit/config.toml`
- **Layout**: Adjust the sidebar content and main chat area
- **Features**: Add file upload, visualization panels, etc.

## Troubleshooting

### Port Already in Use

If port 8501 is already in use, specify a different port:

```bash
streamlit run src/ui/streamlit_app.py --server.port 8502
```

### Module Import Errors

Make sure you're running from the project root directory and the conda environment is activated.

### Agent Errors

Check your API keys are properly set in your environment variables or `.env` file.
