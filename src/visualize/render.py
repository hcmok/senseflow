import numpy as np
import plotly.graph_objects as go


def render(
    word_path, centroid_path, semantic_manifold_path, centroid_metadata, show_info
):
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

    offset = 1.03
    path_3d = coords_3d_sphere[centroid_path] * offset

    path_trace = go.Scatter3d(
        x=path_3d[:, 0],
        y=path_3d[:, 1],
        z=path_3d[:, 2],
        mode="lines+markers+text",
        line=dict(color="cyan", width=6),
        marker=dict(size=6, color="white", symbol="circle"),
        text=word_path,
        textposition="top center",
        textfont=dict(color="white", size=16),
        name="Path",
    )

    fig = go.Figure(data=[shell_trace, path_trace])

    fig.update_layout(
        title="3D Semantic Manifold" if show_info else None,
        template="plotly_dark",
        scene=dict(
            xaxis=dict(range=[-1, 1], visible=False),
            yaxis=dict(range=[-1, 1], visible=False),
            zaxis=dict(range=[-1, 1], visible=False),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=(30 if show_info else 0)),
        autosize=True,
        height=600,
        showlegend=show_info,
    )

    return fig
