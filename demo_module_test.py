"""
Ultra minimal module script test
"""

import streamlit as st

st.title("Module Script Test")

# Test 1: Absolutely minimal
st.subheader("Test 1: Minimal module")
st.html(
    """
<div id="test1" style="padding: 20px; background: #e3f2fd;">Waiting...</div>
<script type="module">
console.log('TEST 1: Module executing');
document.getElementById('test1').innerText = 'Test 1 works!';
</script>
""",
    unsafe_allow_javascript=True,
)

# Test 2: With CSS link
st.subheader("Test 2: With CSS link")
st.html(
    """
<link rel="stylesheet" href="https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/dev/index.css" />
<div id="test2" style="padding: 20px; background: #e3f2fd;">Waiting...</div>
<script type="module">
console.log('TEST 2: Module executing');
document.getElementById('test2').innerText = 'Test 2 works!';
</script>
""",
    unsafe_allow_javascript=True,
)

# Test 3: With CSS + regular script
st.subheader("Test 3: With CSS + regular script")
st.html(
    """
<link rel="stylesheet" href="https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/dev/index.css" />
<script>
console.log('TEST 3: Regular script executing');
window.EXCALIDRAW_ASSET_PATH = "https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/prod/";
</script>
<div id="test3" style="padding: 20px; background: #e3f2fd;">Waiting...</div>
<script type="module">
console.log('TEST 3: Module executing');
document.getElementById('test3').innerText = 'Test 3 works!';
</script>
""",
    unsafe_allow_javascript=True,
)

# Test 4: With setTimeout
st.subheader("Test 4: With setTimeout (no async)")
st.html(
    """
<link rel="stylesheet" href="https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/dev/index.css" />
<script>
window.EXCALIDRAW_ASSET_PATH = "https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/prod/";
</script>
<div id="test4" style="padding: 20px; background: #e3f2fd;">Waiting...</div>
<script type="module">
console.log('TEST 4: Module executing immediately');
setTimeout(() => {
    console.log('TEST 4: setTimeout callback');
    document.getElementById('test4').innerText = 'Test 4 works!';
}, 100);
</script>
""",
    unsafe_allow_javascript=True,
)

# Test 5: With async setTimeout
st.subheader("Test 5: With async setTimeout")
st.html(
    """
<link rel="stylesheet" href="https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/dev/index.css" />
<script>
window.EXCALIDRAW_ASSET_PATH = "https://esm.sh/@excalidraw/excalidraw@0.18.0/dist/prod/";
</script>
<div id="test5" style="padding: 20px; background: #e3f2fd;">Waiting...</div>
<script type="module">
console.log('TEST 5: Module executing immediately');
setTimeout(async () => {
    console.log('TEST 5: async setTimeout callback');
    document.getElementById('test5').innerText = 'Test 5 works!';
}, 100);
</script>
""",
    unsafe_allow_javascript=True,
)

st.markdown("---")
st.markdown("**Check console to see which tests execute their module scripts!**")
