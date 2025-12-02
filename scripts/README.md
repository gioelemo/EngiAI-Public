# Scripts Directory

This directory contains utility scripts and examples for the Engineering Assistant project.

## Available Scripts

### Paper Import & Database Management

#### Local Paper Import (`import_local_papers.py`)

Import papers from a local directory or mounted network share.

**Setup:**

Add to your `.env` file:
```bash
PAPERS_SOURCE_DIR=/path/to/your/papers
PAPERS_STATE_FILE=data/local_import_state.json
```

**Usage:**
```bash
# Dry run (preview)
python import_local_papers.py --dry-run

# Import papers (uses .env configuration)
python import_local_papers.py

# Or specify path directly (overrides .env)
python import_local_papers.py /Volumes/Share/Papers
```

#### MMORE Document Inspector (`inspect_mmore.py`)

Inspect documents uploaded to MMORE RAG service:

```bash
# Show all uploaded documents
python inspect_mmore.py

# Get detailed stats
python inspect_mmore.py --stats

# List all document IDs
python inspect_mmore.py --list
```

**Note**: MMORE replaces the legacy ChromaDB-based vector store with a more powerful multimodal RAG service.

**See [Paper Import Guide](../docs/PAPER_IMPORT_GUIDE.md) for detailed documentation.**

---

### 1. 3D Heatmap to STL
TODO: Write description
---

### 2. 2D Heatmap to STL Converter (`2D_heatmap_to_stl.py`)

**Purpose**: Convert 2D heatmap data (NumPy arrays) into 3D STL files suitable for 3D printing or CAD applications.

---

### 3. 2D Heatmap to STL (Extruded) (`2D_heatmap_to_stl_extruded.py`)

Similar to script #2, but with different extrusion method.

---

### 4. Architecture Diagram Generator (`generate_architecture_diagram.py`)

**Purpose**: Generate visual architecture diagrams of the multi-agent system.

---

## Requirements

All required dependencies (`numpy-stl`, `torch`, `wandb`, `engibench`, `engiopt`) are automatically installed with the project. See the main README for setup instructions.

---

## 2D Heatmap to STL Converter - Detailed Documentation

## Features

- Convert 2D NumPy arrays to 3D STL mesh files
- Configurable height scaling and base dimensions
- Automatic generation of closed 3D surfaces with proper triangulation
- Command-line interface with flexible parameter control
- Input validation and error handling

## Requirements

- numpy-stl

## Usage
### Basic Usage

```bash
python 2D_heatmap_to_stl.py input.npy output.stl
```

### Advanced Usage with Parameters

```bash
python 2D_heatmap_to_stl.py input.npy output.stl --scale-z 20 --scale-xy 2 --thickness 2
```

### Command Line Arguments

- `input`: Input .npy file containing a 2D NumPy array
- `output`: Output .stl file path
- `-z, --scale-z`: Height scale factor (default: 10)
- `-xy, --scale-xy`: X/Y scale factor (default: 1)
- `-t, --thickness`: Base thickness (default: 1)

### Examples

```bash
# Basic conversion with default settings
python 2D_heatmap_to_stl.py beam2D_opt_design.npy output.stl

# Create a taller structure with scaled base
python 2D_heatmap_to_stl.py beam2D_opt_design.npy tall_beam.stl --scale-z 20 --scale-xy 2

# Create with custom base thickness
python 2D_heatmap_to_stl.py beam2D_opt_design.npy thick_base.stl -z 15 -xy 1.5 -t 2
```

## Input Data Format

The input file should be a NumPy array saved in `.npy` format containing:
- 2D array with numerical values
- Values typically normalized between 0 and 1 (but any range is supported)
- Array represents height values at each grid point

### Creating Input Data

```python
import numpy as np

# Create sample heatmap data
data = np.random.rand(50, 50)  # 50x50 grid of random values
np.save('sample_heatmap.npy', data)
```

## Output

The script generates an STL file containing:
- **Top surface**: Triangulated mesh based on heatmap values
- **Bottom surface**: Flat base at z=0
- **Side walls**: Connecting top and bottom surfaces to create a closed solid
- **Proper normals**: Correctly oriented triangles for 3D printing

## Technical Details

### Coordinate System
- X-axis: Corresponds to array columns (j index)
- Y-axis: Corresponds to array rows (i index), with Y flipped for standard orientation
- Z-axis: Height values scaled by `scale_z` parameter plus base thickness

### Mesh Generation
1. Creates vertices for top surface based on heatmap values
2. Creates vertices for bottom surface at z=0
3. Generates triangular faces for top and bottom surfaces
4. Creates side walls connecting perimeter vertices
5. Ensures proper winding order for correct surface normals

### Memory Considerations
For large arrays (e.g., 1000x1000), the script generates approximately:
- 2M vertices (2 × width × height)
- 4M triangular faces
- Consider memory usage for very large datasets

## Example Files

The repository includes example files:
- `beam2D_opt_design.npy`: Sample optimization result data
- `beam2D_opt_design.stl`: Generated STL output
- `beam2D_opt_design.png`: Visualization of the heatmap

## Error Handling

The script includes validation for:
- File existence checks
- Array dimensionality verification
- Data type validation
- Output path accessibility

## Applications

This tool is particularly useful for:
- Topology optimization visualization
- Heat map 3D visualization
- Converting simulation results to physical models
- Creating 3D printed representations of 2D data
- CAD integration of numerical analysis results

## Contributing

Feel free to contribute improvements such as:
- Additional mesh export formats
- Smoothing algorithms
- Color mapping support
- GUI interface
- Performance optimizations
