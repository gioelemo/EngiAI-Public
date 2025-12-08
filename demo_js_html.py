"""
Demo app showcasing Streamlit's new st.html feature for embedding JavaScript.

This demonstrates the new capability to run JavaScript directly within Streamlit
using the st.html block introduced in recent Streamlit versions.
"""

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Streamlit JS Demo",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Streamlit JavaScript Demo")
st.markdown("---")

# Introduction
st.markdown("""
This demo showcases the new **`st.html`** feature that allows embedding and executing
JavaScript directly in Streamlit apps!
""")

# Demo 1: Simple Hello World
st.header("1. Simple Hello World")
st.markdown("Basic JavaScript alert and console output:")

st.html(
    """
<div style="padding: 20px; background-color: #f0f8ff; border-radius: 10px; border: 2px solid #4169e1;">
    <h2 id="hello-target" style="color: #4169e1;">Waiting for JavaScript...</h2>
</div>

<script>
    // Simple hello world - modify the DOM
    document.getElementById('hello-target').innerText = '👋 Hello World from JavaScript!';
    console.log('JavaScript is running in Streamlit!');
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown("---")

# Demo 2: Simple Script Test
st.header("2. Basic JavaScript Execution Test")
st.markdown("Testing if JavaScript runs at all:")

st.html(
    """
<div style="padding: 20px; background-color: #f0fff0; border-radius: 10px; border: 2px solid #32cd32;">
    <h3 id="test-heading-2">Testing JavaScript...</h3>
</div>

<script>
    console.log('===== DEMO 2 SCRIPT STARTING =====');
    document.getElementById('test-heading-2').innerText = '✓ Demo 2 JavaScript is running!';
    console.log('===== DEMO 2 SCRIPT COMPLETE =====');
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown(
    "**If you see '✓ Demo 2 JavaScript is running!' above, then basic JS works. If not, there may be an issue with how Streamlit processes certain script blocks.**"
)

st.markdown("---")

# Demo 2b: Interactive Button
st.header("2b. Interactive Click Counter")
st.markdown("Now let's test if user interactions work:")

st.html(
    """
<div style="padding: 20px; background-color: #fff0f5; border-radius: 10px; border: 2px solid #ff69b4;">
    <button id="click-btn" style="
        padding: 15px 30px;
        font-size: 18px;
        background-color: #ff69b4;
        color: white;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s;
    ">Click Me!</button>
    <p id="click-count" style="font-size: 20px; margin-top: 15px; color: #ff1493;">
        Clicks: <strong>0</strong>
    </p>
</div>

<script>
    console.log('===== DEMO 2b: Setting up click handler =====');

    let clickCount = 0;
    const button = document.getElementById('click-btn');
    const countDisplay = document.getElementById('click-count');

    console.log('Button found:', button);
    console.log('Count display found:', countDisplay);

    if (button && countDisplay) {
        button.addEventListener('click', function() {
            console.log('BUTTON CLICKED!');
            clickCount++;
            countDisplay.innerHTML = 'Clicks: <strong>' + clickCount + '</strong>';

            // Change color
            const colors = ['#ff69b4', '#ff6347', '#4169e1', '#32cd32', '#ffd700'];
            button.style.backgroundColor = colors[clickCount % colors.length];
        });

        console.log('===== DEMO 2b: Click handler attached successfully =====');
    } else {
        console.error('===== DEMO 2b: Failed to find elements =====');
    }
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown(
    "**Check console:** You should see setup messages. Try clicking the button and see if clicks are logged!"
)

st.markdown("---")

# Summary section
st.header("Summary of JavaScript Capabilities")
st.markdown("""
Based on testing with `st.html(unsafe_allow_javascript=True)`:

**✅ What Works:**
- ✓ Basic DOM manipulation (changing text, colors, styles)
- ✓ Animations with `requestAnimationFrame()`
- ✓ Timers with `setInterval()` and `setTimeout()`
- ✓ Canvas drawing and animations
- ✓ Reading and modifying element properties
- ✓ ES6 module imports from CDN
- ✓ Mouse event listeners (click, mousedown, mousemove, etc.)
- ✓ Interactive canvas drawing

**Use Cases:**
- Custom animations and visualizations
- Interactive drawings and whiteboards (like Excalidraw)
- Real-time data displays (clocks, dashboards)
- Custom UI components
- Game-like interactions

**Note:** This feature enables embedding rich, interactive JavaScript content directly in Streamlit apps!
""")

st.markdown("---")

# Demo 2c: Interactive Drawing Canvas
st.header("2c. Interactive Drawing Canvas")
st.markdown("Click and drag to draw on the canvas:")

st.html(
    """
<div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 2px solid #6c757d; text-align: center;">
    <h4 style="margin-top: 0;">Drawing Canvas Demo</h4>
    <canvas id="drawing-canvas" width="600" height="400" style="
        border: 2px solid #333;
        cursor: crosshair;
        background: white;
        border-radius: 5px;
        display: inline-block;
    "></canvas>
    <div style="margin-top: 15px;">
        <button id="clear-btn" style="
            padding: 10px 20px;
            background-color: #dc3545;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        ">Clear Canvas</button>
        <button id="color-btn" style="
            padding: 10px 20px;
            background-color: #000;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        ">Change Color</button>
    </div>
</div>

<script>
    console.log('===== CANVAS: Initializing =====');

    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const colorBtn = document.getElementById('color-btn');

    console.log('Canvas found:', canvas);
    console.log('Clear button found:', clearBtn);
    console.log('Color button found:', colorBtn);

    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let currentColor = '#000000';
    const colors = ['#000000', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ff00ff', '#00ffff'];
    let colorIndex = 0;

    // Drawing functions
    function startDrawing(e) {
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        lastX = e.clientX - rect.left;
        lastY = e.clientY - rect.top;
        console.log('Started drawing at:', lastX, lastY);
    }

    function draw(e) {
        if (!isDrawing) return;

        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.strokeStyle = currentColor;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.stroke();

        lastX = x;
        lastY = y;
    }

    function stopDrawing() {
        isDrawing = false;
        console.log('Stopped drawing');
    }

    // Event listeners for drawing
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', stopDrawing);

    // Clear button
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            console.log('Clearing canvas');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        });
    }

    // Color change button
    if (colorBtn) {
        colorBtn.addEventListener('click', function() {
            colorIndex = (colorIndex + 1) % colors.length;
            currentColor = colors[colorIndex];
            colorBtn.style.backgroundColor = currentColor;
            console.log('Changed color to:', currentColor);
        });
    }

    console.log('===== CANVAS: Ready! =====');
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown("""
**Features:**
- Click and drag on the canvas to draw
- Click "Change Color" to cycle through different pen colors
- Click "Clear Canvas" to erase everything
""")

st.markdown("---")

# Demo 2d: ES Module Test
st.header("2d. ES Module Support Test")
st.markdown('Testing if Streamlit supports `<script type="module">`:')

st.html(
    """
<div style="padding: 20px; background-color: #fff3cd; border-radius: 10px; border: 2px solid #ffc107;">
    <h3 id="module-test-result">Testing ES modules...</h3>
</div>

<script>
    console.log('===== REGULAR SCRIPT: Running =====');
    document.getElementById('module-test-result').innerText = '✓ Regular script works';
</script>

<script type="module">
    console.log('===== MODULE SCRIPT: Running =====');
    const element = document.getElementById('module-test-result');
    if (element) {
        element.innerText = '✓✓ ES Module script works!';
        element.style.color = '#28a745';
    }
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown("""
**Expected behavior:**
- If you see "✓ Regular script works" → Regular scripts work but modules don't
- If you see "✓✓ ES Module script works!" → Both work (Excalidraw should be possible)

Check the browser console for log messages.
""")

st.markdown("---")

# Demo 2e: Script Execution Debugging
st.header("2e. Debugging Script Execution")
st.markdown("Testing what prevents the Excalidraw script from running:")

# Test 1: Minimal module script
st.subheader("Test 1: Minimal module script")
st.html(
    """
<div style="padding: 10px; background: #e3f2fd;">
    <p id="test1">Waiting...</p>
</div>
<script type="module">
    console.log('TEST 1: Module script running');
    document.getElementById('test1').innerText = 'Test 1: Module works!';
</script>
""",
    unsafe_allow_javascript=True,
)

# Test 2: With import map
st.subheader("Test 2: With import map")
st.html(
    """
<script type="importmap">
{
    "imports": {
        "react": "https://esm.sh/react@19.0.0"
    }
}
</script>
<div style="padding: 10px; background: #e3f2fd;">
    <p id="test2">Waiting...</p>
</div>
<script type="module">
    console.log('TEST 2: Module script with import map running');
    document.getElementById('test2').innerText = 'Test 2: Import map + module works!';
</script>
""",
    unsafe_allow_javascript=True,
)

# Test 3: With CSS link
st.subheader("Test 3: With CSS link")
st.html(
    """
<link rel="stylesheet" href="https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/dev/index.css" />
<div style="padding: 10px; background: #e3f2fd;">
    <p id="test3">Waiting...</p>
</div>
<script type="module">
    console.log('TEST 3: Module script with CSS link running');
    document.getElementById('test3').innerText = 'Test 3: CSS + module works!';
</script>
""",
    unsafe_allow_javascript=True,
)

# Test 4: All together
st.subheader("Test 4: All elements together")
st.html(
    """
<link rel="stylesheet" href="https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/dev/index.css" />
<script>
    window.EXCALIDRAW_ASSET_PATH = "https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/prod/";
</script>
<script type="importmap">
{
    "imports": {
        "react": "https://esm.sh/react@19.0.0"
    }
}
</script>
<div style="padding: 10px; background: #e3f2fd;">
    <p id="test4">Waiting...</p>
</div>
<script type="module">
    console.log('TEST 4: Module script with all elements running');
    document.getElementById('test4').innerText = 'Test 4: All elements + module works!';
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown("---")

# Demo 2f: Excalidraw Whiteboard (No Import Map)
st.header("2f. Excalidraw Whiteboard - No Import Map")
st.markdown("Using explicit full URLs to avoid import map conflicts:")

st.html(
    """
<link rel="stylesheet" href="https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/dev/index.css" />

<script>
    window.EXCALIDRAW_ASSET_PATH = "https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/prod/";
</script>

<div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px; border: 2px solid #6c757d;">
    <div id="excalidraw-app" style="height: 600px; width: 100%;"></div>
</div>

<script type="module">
    console.log('===== EXCALIDRAW (NO IMPORT MAP): Script executing =====');

    setTimeout(async () => {
        console.log('===== EXCALIDRAW (NO IMPORT MAP): Starting =====');

        try {
            console.log('Importing React with explicit URL...');
            const React = await import("https://esm.sh/react@19.0.0");
            console.log('✓ React loaded');

            console.log('Importing ReactDOM with explicit URL...');
            const ReactDOM = await import("https://esm.sh/react-dom@19.0.0/client");
            console.log('✓ ReactDOM loaded');

            console.log('Importing Excalidraw (will use external React)...');
            const ExcalidrawLib = await import('https://esm.sh/@excalidraw/excalidraw@0.18.0?deps=react@19.0.0,react-dom@19.0.0');
            console.log('✓ ExcalidrawLib loaded');

            console.log('All modules loaded successfully!');

            const App = () => {
                return React.createElement(
                    "div",
                    { style: { height: "600px", width: "100%" } },
                    React.createElement(ExcalidrawLib.Excalidraw)
                );
            };

            const container = document.getElementById("excalidraw-app");
            if (container) {
                console.log('Creating React root...');
                const root = ReactDOM.createRoot(container);
                console.log('Rendering Excalidraw...');
                root.render(React.createElement(App));
                console.log('===== EXCALIDRAW: Successfully initialized! =====');
            } else {
                console.error('Container not found!');
            }
        } catch (error) {
            console.error('===== EXCALIDRAW: ERROR =====');
            console.error('Message:', error.message);
            console.error('Stack:', error.stack);

            const container = document.getElementById('excalidraw-app');
            if (container) {
                container.innerHTML = `
                    <div style="display: flex; align-items: center; justify-content: center; height: 100%; background: #fff3cd; border-radius: 5px; padding: 20px;">
                        <div style="text-align: center;">
                            <h3 style="color: #856404;">⚠️ Excalidraw Failed</h3>
                            <p style="color: #856404;">${error.message}</p>
                            <p style="color: #666; font-size: 12px;">Check console for details</p>
                        </div>
                    </div>
                `;
            }
        }
    }, 100);
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown("""
**About Excalidraw:**
- Professional whiteboard for creating hand-drawn diagrams
- Full suite of drawing tools: selection, pen, shapes, arrows, text
- Collaborative features and export options
- Hand-drawn aesthetic with smooth interactions
- Uses React 19 and Excalidraw 0.18.0 from esm.sh CDN
""")

st.markdown("---")

# Demo 3: Animated Content
st.header("3. Animated Canvas")
st.markdown("JavaScript with HTML5 Canvas animation:")

st.html(
    """
<div style="padding: 20px; background-color: #fff5ee; border-radius: 10px; border: 2px solid #ff8c00;">
    <canvas id="animation-canvas" width="600" height="200" style="
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #000;
        display: block;
        margin: 0 auto;
    "></canvas>
</div>

<script>
    (function() {
        const canvas = document.getElementById('animation-canvas');
        const ctx = canvas.getContext('2d');

        let x = 0;
        let y = 100;
        let dx = 2;
        let dy = 1.5;
        const radius = 20;

        function drawBall() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Draw ball
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = '#00ff00';
            ctx.fill();
            ctx.strokeStyle = '#00ff00';
            ctx.stroke();
            ctx.closePath();

            // Add glow effect
            ctx.shadowBlur = 20;
            ctx.shadowColor = '#00ff00';

            // Update position
            x += dx;
            y += dy;

            // Bounce off walls
            if (x + radius > canvas.width || x - radius < 0) {
                dx = -dx;
            }
            if (y + radius > canvas.height || y - radius < 0) {
                dy = -dy;
            }

            requestAnimationFrame(drawBall);
        }

        drawBall();
    })();
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown("---")

# Demo 4: Real-time Clock
st.header("4. Real-time Clock")
st.markdown("JavaScript updating content continuously:")

st.html(
    """
<div style="padding: 20px; background-color: #f5f5f5; border-radius: 10px; border: 2px solid #333;">
    <div id="clock" style="
        font-size: 48px;
        font-family: 'Courier New', monospace;
        text-align: center;
        color: #333;
        font-weight: bold;
    "></div>
    <div id="date" style="
        font-size: 20px;
        text-align: center;
        color: #666;
        margin-top: 10px;
    "></div>
</div>

<script>
    function updateClock() {
        const now = new Date();

        // Format time
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');

        // Format date
        const options = { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' };
        const dateStr = now.toLocaleDateString('en-US', options);

        document.getElementById('clock').innerText = `${hours}:${minutes}:${seconds}`;
        document.getElementById('date').innerText = dateStr;
    }

    // Update immediately and then every second
    updateClock();
    setInterval(updateClock, 1000);
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown("---")

# Footer
st.info("""
**📚 Learn More:**
- [Streamlit st.html Documentation](https://docs.streamlit.io/develop/api-reference/text/st.html)
- This feature allows you to embed any HTML/CSS/JavaScript content directly in your Streamlit apps!
""")

st.success("✨ All JavaScript code is running directly in the browser!")
