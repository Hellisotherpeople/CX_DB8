"""Interactive 3D visualization of text embeddings."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cx_db8.summarizer import SummaryResult


def visualize_3d(result: SummaryResult, use_umap: bool = True) -> None:
    """Show an interactive 3D scatter plot of embeddings colored by similarity.

    Requires the `viz` extra: `uv pip install cx-db8[viz]`
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import proj3d  # noqa: F401
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib. Install with: uv pip install cx-db8[viz]"
        )

    if result.embeddings is None:
        raise ValueError("No embeddings stored. Re-run summarize with want_embeddings=True.")

    embs = result.embeddings
    scores = np.array([s.score for s in result.spans])
    labels = [s.text for s in result.spans]

    if use_umap:
        try:
            import umap
        except ImportError:
            raise ImportError(
                "UMAP visualization requires umap-learn. Install with: uv pip install cx-db8[viz]"
            )
        reducer = umap.UMAP(n_components=3, random_state=42)
        coords = reducer.fit_transform(embs)
    else:
        coords = embs[:, :3]

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor("#1e1e2e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#1e1e2e")

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=scores,
        cmap="magma",
        s=40,
        alpha=0.85,
        depthshade=True,
        edgecolors="white",
        linewidths=0.3,
    )
    cbar = fig.colorbar(scatter, shrink=0.6, pad=0.1)
    cbar.set_label("Cosine Similarity to Query", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(
        f"CX_DB8 Embedding Space — \"{result.query[:60]}\"",
        color="white",
        fontsize=12,
        pad=20,
    )
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("grey")
        axis.label.set_color("white")
        for t in axis.get_ticklabels():
            t.set_color("grey")

    # Interactive annotation on hover
    annotation = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(15, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#313244", ec="#89b4fa", alpha=0.95),
        color="white",
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#89b4fa"),
    )
    annotation.set_visible(False)

    def on_motion(event):
        if event.inaxes != ax:
            annotation.set_visible(False)
            fig.canvas.draw_idle()
            return
        try:
            distances = []
            for i in range(len(coords)):
                x2, y2, _ = proj3d.proj_transform(
                    coords[i, 0], coords[i, 1], coords[i, 2], ax.get_proj()
                )
                x3, y3 = ax.transData.transform((x2, y2))
                d = np.sqrt((x3 - event.x) ** 2 + (y3 - event.y) ** 2)
                distances.append(d)
            idx = int(np.argmin(distances))
            if distances[idx] < 30:
                x2, y2, _ = proj3d.proj_transform(
                    coords[idx, 0], coords[idx, 1], coords[idx, 2], ax.get_proj()
                )
                annotation.xy = (x2, y2)
                label = labels[idx][:80]
                annotation.set_text(f"{label}\nscore: {scores[idx]:.3f}")
                annotation.set_visible(True)
            else:
                annotation.set_visible(False)
        except Exception:
            annotation.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    plt.tight_layout()
    plt.show()
