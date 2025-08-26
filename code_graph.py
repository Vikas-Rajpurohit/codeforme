import streamlit as st
import os
import ast
import zipfile
import tempfile
import networkx as nx
from pyvis.network import Network

# ------------------------------
# Helper: extract imports from a file
# ------------------------------
def extract_imports(file_path):
    imports = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split('.')[0])
    except Exception as e:
        st.warning(f"Could not parse {file_path}: {e}")
    return imports


# ------------------------------
# Build dependency graph
# ------------------------------
def build_dependency_graph(base_path):
    G = nx.DiGraph()
    py_files = {}

    # Map filenames (without extension) to paths
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(".py"):
                file_name = os.path.splitext(file)[0]
                py_files[file_name] = os.path.join(root, file)

    # Build graph
    for module, path in py_files.items():
        G.add_node(module)
        imports = extract_imports(path)
        for imp in imports:
            if imp in py_files:  # only link if module is inside repo
                G.add_edge(module, imp)

    return G


# ------------------------------
# Render graph in Streamlit
# ------------------------------
def render_graph(G):
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", directed=True)
    net.from_nx(G)
    return net


# ------------------------------
# Streamlit App
# ------------------------------
st.title("📂 Project Dependency Visualizer")

uploaded_file = st.file_uploader("Upload your project (zip file)", type="zip")

if uploaded_file is not None:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "project.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        st.info("Building dependency graph... ⏳")
        G = build_dependency_graph(tmpdir)

        st.success("Graph built successfully! 🎉")

        net = render_graph(G)
        net.save_graph("graph.html")

        # Display inside Streamlit
        st.components.v1.html(open("graph.html", "r", encoding="utf-8").read(), height=650)
