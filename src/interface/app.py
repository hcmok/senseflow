import json
from pathlib import Path

import gradio as gr
import nltk
import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from huggingface_hub import hf_hub_download
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import KDTree

from src.pathfind.morph_a_star import morph_a_star_generator
from src.utils.load import load_model_for_inference

# Download NLTK resources
nltk.download("wordnet")
nltk.download("omw-1.4")


def ensure_assets_available(cfg):
    repo_id = cfg["hf"]["repo_id"]

    centroids_path = Path(cfg["data"]["dir"]) / cfg["data"]["centroids_path"]
    centroid_metadata_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
    )
    checkpoint_for_inference_path = (
        Path(cfg["checkpoints"]["dir"])
        / cfg["checkpoints"]["checkpoint_for_inference_path"]
    )
    semantic_manifold_path = (
        Path(cfg["data"]["dir"]) / cfg["data"]["semantic_manifold_path"]
    )

    if not centroids_path.exists():
        hf_hub_download(
            repo_id=repo_id,
            filename=cfg["data"]["centroids_path"],
            local_dir=cfg["data"]["dir"],
            repo_type="dataset",
        )
    if not centroid_metadata_path.exists():
        hf_hub_download(
            repo_id=repo_id,
            filename=cfg["data"]["centroid_metadata_path"],
            local_dir=cfg["data"]["dir"],
            repo_type="dataset",
        )
    if not checkpoint_for_inference_path.exists():
        hf_hub_download(
            repo_id=repo_id,
            filename=cfg["checkpoints"]["checkpoint_for_inference_path"],
            local_dir=cfg["checkpoints"]["dir"],
            repo_type="model",
        )
    if not semantic_manifold_path.exists():
        hf_hub_download(
            repo_id=repo_id,
            filename=cfg["data"]["semantic_manifold_path"],
            local_dir=cfg["data"]["dir"],
            repo_type="dataset",
        )


with open("./configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
ensure_assets_available(cfg)

centroids_path = Path(cfg["data"]["dir"]) / cfg["data"]["centroids_path"]
centroid_metadata_path = (
    Path(cfg["data"]["dir"]) / cfg["data"]["centroid_metadata_path"]
)
checkpoint_for_inference_path = (
    Path(cfg["checkpoints"]["dir"])
    / cfg["checkpoints"]["checkpoint_for_inference_path"]
)
semantic_manifold_path = (
    Path(cfg["data"]["dir"]) / cfg["data"]["semantic_manifold_path"]
)

with open(centroid_metadata_path, "r") as f:
    centroid_metadata = json.load(f)

word_to_centroid_indices = centroid_metadata["word_to_centroid_indices"]

index_to_word = {}
for word, indices in word_to_centroid_indices.items():
    for i, idx in enumerate(indices):
        index_to_word[idx] = word + f"_{i}"

d = centroid_metadata["dimension"]
centroid_count = centroid_metadata["centroid_count"]
centroids_memmap = np.memmap(
    centroids_path, dtype=np.float32, mode="r", shape=(centroid_count, d)
)


k_neighbors = cfg["pathfind"]["kdtree"]["k_neighbors"]
max_neighbor_distance = cfg["pathfind"]["kdtree"]["max_neighbor_distance"]
k_cutoff = cfg["pathfind"]["kdtree"]["k_cutoff"]
temperature = cfg["pathfind"]["kdtree"]["temperature"]
max_expansions = cfg["pathfind"]["astar"]["max_expansions"]
weights = cfg["pathfind"]["astar"]["weights"]
allow_immediate_reach = cfg["pathfind"]["astar"]["allow_immediate_reach"]

kdtree = KDTree(centroids_memmap)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model_for_inference(
    checkpoint_path=checkpoint_for_inference_path, cfg=cfg, device=device
)
lemmatizer = WordNetLemmatizer()

# Initialize semantic manifold
data = np.load(semantic_manifold_path)
coords_3d_sphere = data["coords_3d_sphere"]
density = data["density"]

centroid_count = centroid_metadata["centroid_count"]
word_to_centroid_indices = centroid_metadata["word_to_centroid_indices"]
centroid_labels = [""] * centroid_count

for word, indices in word_to_centroid_indices.items():
    for idx in indices:
        centroid_labels[idx] = word

shell_trace = go.Scatter3d(
    x=coords_3d_sphere[:, 0],
    y=coords_3d_sphere[:, 1],
    z=coords_3d_sphere[:, 2],
    mode="markers",
    text=centroid_labels,
    hoverinfo="text",
    marker=dict(size=1.5, opacity=0.7, color=density, colorscale="Inferno"),
)
path_trace = go.Scatter3d(
    mode="lines+markers+text",
    textposition="top center",
    # Explicitly define the font to prevent auto-scaling
    textfont=dict(family="Arial, sans-serif", size=12, color="white"),
    marker=dict(
        size=6,
        color="white",
        symbol="circle",
    ),
)
expanded_trace = go.Scatter3d(
    mode="markers", marker=dict(color="white", size=1.5, opacity=0.7)
)
base_fig = go.Figure()
base_fig.add_trace(shell_trace)
base_fig.add_trace(path_trace)
base_fig.add_trace(expanded_trace)
base_fig.update_layout(
    title="3D Semantic Manifold" if False else None,
    template="plotly_dark",
    scene=dict(
        xaxis=dict(visible=False, showbackground=False, showticklabels=False),
        yaxis=dict(visible=False, showbackground=False, showticklabels=False),
        zaxis=dict(visible=False, showbackground=False, showticklabels=False),
        aspectmode="cube",
    ),
    margin=dict(l=0, r=0, b=0, t=(30 if False else 0)),
    autosize=True,
    height=600,
    showlegend=False,
    uirevision="constant",
)


def generate_output(
    word_start,
    word_end,
    k_neighbors,
    max_neighbor_distance,
    k_cutoff,
    temperature,
    weights,
    max_expansions,
    allow_immediate_reach,
    live_update,
    dim_point_cloud,
):

    idx_a = word_to_centroid_indices.get(f"{word_start}")
    idx_b = word_to_centroid_indices.get(f"{word_end}")
    if idx_a is None:
        yield (
            f'⚠️ The word "{word_start}" is not in the vocabulary. Please try another word.',
            gr.update(value=None),
            None,
        )
        return
    if idx_b is None:
        yield (
            f'⚠️ The word "{word_end}" is not in the vocabulary. Please try another word.',
            gr.update(value=None),
            None,
        )
        return

    gen = morph_a_star_generator(
        word_start=word_start,
        word_end=word_end,
        model=model,
        lemmatizer=lemmatizer,
        k_neighbors=k_neighbors,
        max_neighbor_distance=max_neighbor_distance,
        k_cutoff=k_cutoff,
        temperature=temperature,
        weights=weights,
        max_expansions=max_expansions,
        allow_immediate_reach=allow_immediate_reach,
        centroids_path=centroids_path,
        centroid_metadata=centroid_metadata,
        device=device,
        yield_interval=100 if live_update else 0,
        kdtree=kdtree,
    )
    if not live_update:
        yield f"<div style='color: #888; font-style: italic; font-family: monospace;'>Searching...</div>", gr.update(
            value=None
        ), None
    for result in gen:
        if result["status"] == 1 or result["status"] == 2:
            expanded_word_senses = result["expanded_word_senses"]
            expanded_indices = result["expanded_indices"]

            word_sense_path = result["word_sense_path"]
            word_path = [x.split("_")[0] for x in word_sense_path]  # type: ignore
            centroid_path = result["centroid_path"]
            expansion_count = result["expansion_count"]
            dyanamic_path = [
                x.split("_")[0] if word_path.count(x.split("_")[0]) == 1 else x
                for x in word_sense_path  # type: ignore
            ]
            raw_display = " → ".join(dyanamic_path)
            visited_3d = coords_3d_sphere[expanded_indices]
            path_3d = coords_3d_sphere[centroid_path]

            fig = go.Figure(base_fig)  # Copy the layout and shell

            if dim_point_cloud:
                fig.data[0].opacity = 0.1

            fig.data[1].x = path_3d[:, 0]
            fig.data[1].y = path_3d[:, 1]
            fig.data[1].z = path_3d[:, 2]
            fig.data[1].text = word_path

            fig.data[2].x = visited_3d[:, 0]
            fig.data[2].y = visited_3d[:, 1]
            fig.data[2].z = visited_3d[:, 2]
            fig.data[2].text = expanded_word_senses

            if result["status"] == 1:
                # Searching
                display_str = f"<div style='color: #888; font-style: italic; font-family: monospace;'>Searching: {raw_display}</div>"
            else:
                # Success
                fig.data[1].line.color = "#00ffff"  # type: ignore
                display_str = f"<div style='color: #00ffff; font-weight: bold; font-size: 1.1em; font-family: monospace;'>Path Found: {raw_display}</div>"

            yield display_str, fig, f"{expansion_count} expanded node{'s' if expansion_count > 1 else ''}"  # type: ignore
        else:
            yield "⚠️ No path found. Please try other words or adjust the parameters.", gr.update(
                value=None
            ), None
            return


with gr.Blocks(
    css="""
#settings_acc {
    max-height: 46vh;
    overflow-y: auto;
}
"""
) as demo:

    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML(
                """
                     <h1 style="margin:0;margin-bottom: 1em;">🌀 SenseFlow</h1>
       <div style="display: flex; align-items: center; gap: 1em;">
        <h3 style="margin:0;font-weight:600 !important;">Semantic word morphing by flow-guided A*</h3>
        <a href="https://github.com/oakto/senseflow" target="_blank" style="display: flex;">
            <img src="https://img.shields.io/badge/View%20on%20GitHub-343432?logo=github&logoColor=white&style=for-the-badge" alt="View on GitHub">
        </a>
    </div>
    """
            )
            with gr.Row():
                word1 = gr.Textbox(label="Start Word", placeholder="e.g. book")
                word2 = gr.Textbox(label="Target Word", placeholder="e.g. cake")

            gr.Markdown("### Options")
            with gr.Row():
                live_update = gr.Checkbox(value=True, label="Live update")
                dim_point_cloud = gr.Checkbox(value=True, label="Dim Point Cloud")

            with gr.Accordion("Settings", open=False, elem_id="settings_acc"):
                with gr.Group():
                    gr.Markdown("&nbsp;Weights")

                    with gr.Row():
                        step_w = gr.Slider(0.0, 5.0, value=weights[0], label="STEP")
                        lemma_w = gr.Slider(0.0, 5.0, value=weights[1], label="LEMMA")
                        flow_w = gr.Slider(0.0, 5.0, value=weights[2], label="FLOW")
                with gr.Row():
                    max_expansions = gr.Number(
                        value=max_expansions,
                        precision=0,
                        minimum=1,
                        label="Max Expansions",
                    )
                    allow_immediate_reach = gr.Checkbox(
                        value=allow_immediate_reach, label="Allow Immediate Reach"
                    )

                with gr.Row():
                    k_neighbors = gr.Number(
                        value=k_neighbors,
                        minimum=1,
                        precision=0,
                        label="K Neighbors",
                        info="Number of nearest neighbors to collect at each step",
                    )
                    max_neighbor_distance = gr.Slider(
                        0.0,
                        2.0,
                        value=max_neighbor_distance,
                        label="Max Neighbor Distance",
                        info="Maximum distance to be considered a neighbor",
                    )
                with gr.Row():
                    k_cutoff = gr.Number(
                        value=k_cutoff,
                        minimum=1,
                        precision=0,
                        label="K Cutoff",
                        info="Number of nearest neighbors to sample from when temperature > 0",
                    )
                    temperature = gr.Slider(
                        0.0,
                        1.0,
                        value=temperature,
                        label="Temperature",
                        info="Controls randomness of neighbor sampling",
                    )

            run_btn = gr.Button("✨ Run Morph", variant="primary")

        with gr.Column(scale=3):
            gr.Markdown("### 🌐 3D Semantic Manifold")
            output_plot = gr.Plot(base_fig, show_label=False)
            output_text = gr.HTML(
                value=f"<div style='color: #999; font-family: monospace;'>e.g. book → publisher → bookstore → supermarket → bakery → cake (determinstic path)</div>"
            )
            output_expansion_label = gr.Markdown(None)

    def run(
        word1,
        word2,
        k_neighbors,
        max_neighbor_distance,
        k_cutoff,
        temperature,
        step_w,
        lemma_w,
        flow_w,
        max_expansions,
        allow_immediate_reach,
        live_update,
        dim_point_cloud,
    ):

        word1 = word1.lower().strip() if word1 else "book"
        word2 = word2.lower().strip() if word2 else "cake"

        yield from generate_output(
            word1,
            word2,
            k_neighbors,
            max_neighbor_distance,
            k_cutoff,
            temperature,
            [step_w, lemma_w, flow_w],
            max_expansions,
            allow_immediate_reach,
            live_update,
            dim_point_cloud,
        )

    run_btn.click(
        fn=run,
        inputs=[
            word1,
            word2,
            k_neighbors,
            max_neighbor_distance,
            k_cutoff,
            temperature,
            step_w,
            lemma_w,
            flow_w,
            max_expansions,
            allow_immediate_reach,
            live_update,
            dim_point_cloud,
        ],
        outputs=[output_text, output_plot, output_expansion_label],
        show_progress="minimal",
    )

demo.launch()
