# Web User Interface

EngiAI provides a Streamlit-based web interface for interactive use.

## Starting the UI

Launch the web interface:

```bash
# Using the run script
make run-ui

# Or directly with streamlit
streamlit run src/ui/streamlit_app.py
```

The interface will open in your browser at `http://localhost:8501`.

## Features

### 1. Chat Interface

Interactive chat with the AI assistant:

- Ask questions about engineering and design
- Request optimizations and simulations
- Get help with research papers
- Submit HPC jobs

### 2. File Upload

Upload files for processing:

- **Papers**: PDF papers for analysis and RAG
- **Designs**: NumPy arrays (.npy) or images
- **Data**: CSV, JSON, or other data formats

### 3. Visualization

View results in real-time:

- Design visualizations (2D/3D)
- Optimization progress plots
- Convergence curves
- STL model previews

### 4. Chat Management

Organize your conversations:

- Create new chats
- Save chat history
- Load previous conversations
- Export chat transcripts

### 5. Interactive Whiteboard (Excalidraw)

Draw and sketch designs directly in the chat:

- **Real-time Drawing**: Native Excalidraw canvas for visual design discussions
- **Canvas Export**: Export drawings as PNG images to chat
- **Custom Descriptions**: Add optional text descriptions when sending canvas images
- **Smart State Detection**: Send button disabled when canvas is empty
- **Keyboard Shortcuts**:
  - `Ctrl/Cmd + Enter`: Send canvas to chat
  - `Ctrl/Cmd + K`: Create new chat

The canvas automatically clears after sending, and visual indicators show the current state.

> **Note:** The Excalidraw component is a standalone Streamlit package ([gioelemo/streamlit-excalidraw](https://github.com/gioelemo/streamlit-excalidraw)) included as a git submodule at `src/ui/components/excalidraw`. Run `git submodule update --init` if the component directory is empty.

### 6. Voice Interaction

Speak to the assistant and hear responses:

- **Speech-to-Text (STT)**: Use your microphone to send voice messages
- **Text-to-Speech (TTS)**: Hear AI responses in natural voice
- **Dual Provider Support**:
  - **ElevenLabs**: High-quality, natural-sounding voices (6 voices: Rachel, Domi, Bella, Antoni, Josh, George)
  - **OpenAI**: Accurate Whisper STT + TTS (6 voices: alloy, echo, fable, onyx, nova, shimmer)
- **Per-Conversation Settings**: Voice provider and voice selection persist for each chat
- **Auto-play Option**: Automatically play AI responses
- **Microphone Toggle**: Enable/disable voice input as needed

**Voice Features:**
- 🎤 Click microphone icon to record voice messages
- 🔊 AI responses automatically generate audio when voice output is enabled
- Voice messages are transcribed and saved with audio for playback
- Switch between providers and voices in Settings without losing history

### 7. Settings

Configure the assistant:

- **API Keys**: Set LangChain, OpenAI, or other API keys
- **Model Selection**: Choose LLM models
- **Voice Interaction**:
  - Select voice provider (ElevenLabs or OpenAI)
  - Choose assistant voice
  - Enable/disable microphone input
  - Enable/disable voice responses
  - Toggle auto-play for responses
- **HPC Configuration**: Set cluster credentials
- **RAG Settings**: Configure MMORE service and retrieval
- **Output Management**: Clear generated files from the outputs folder with one click

### 8. Database Browser

Explore your knowledge base:

- Browse imported papers
- View document metadata
- Search MMORE documents
- Manage uploaded files

## Usage Examples

### Optimize a Design

1. Navigate to the chat interface
2. Type: "Optimize a beam design with 30% volume fraction"
3. View the optimization progress in real-time
4. Download the resulting design

### Analyze a Paper

1. Click "Upload File" in the sidebar
2. Select a PDF paper
3. Ask: "What are the key findings of this paper?"
4. The assistant will analyze and summarize

### Submit HPC Job

1. Type: "Submit an optimization job to the HPC cluster"
2. Provide job parameters when prompted
3. Monitor job status in the interface
4. Retrieve results when complete

### Export to STL

1. Upload or optimize a design
2. Type: "Export this design to STL for 3D printing"
3. Download the generated STL file
4. Preview the model in the viewer

## Keyboard Shortcuts

- **Ctrl/Cmd + Enter**: Send message
- **Ctrl/Cmd + N**: New chat
- **Ctrl/Cmd + S**: Save chat
- **Ctrl/Cmd + K**: Clear chat

## Configuration

### Theme

Change the appearance in Settings → Theme:

- Light mode
- Dark mode
- Auto (follows system)

### Layout

Customize the interface:

- Sidebar width
- Chat message density
- Code block theme

### Advanced Settings

Access advanced options in Settings → Advanced:

- Temperature for LLM responses
- Max tokens per message
- RAG retrieval parameters
- Agent timeout settings

## Persistence

The UI automatically saves:

- Chat history (in `data/chats/`)
- Uploaded files (in `data/uploads/`)
- User preferences (in `data/settings.json`)
- MMORE document tracking (in database)

## Troubleshooting

### Port Already in Use

If port 8501 is busy:

```bash
streamlit run src/ui/streamlit_app.py --server.port 8502
```

### Connection Issues

Clear the Streamlit cache:

```bash
streamlit cache clear
```

### Performance Issues

For better performance:

1. Reduce max tokens in settings
2. Limit chat history length
3. Use a smaller LLM model
4. Close unused browser tabs

## Development Mode

Enable development features:

```bash
export STREAMLIT_ENV=development
streamlit run src/ui/streamlit_app.py --server.runOnSave true
```

Features:
- Auto-reload on file changes
- Debug information
- Performance profiling

## Next Steps

- [Learn about agents](agents.md)
- [Tool reference](tools.md)
- [Configuration guide](../configuration.md)
