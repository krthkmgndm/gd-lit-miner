from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

from .utils.constants import (
    DEFAULT_FIGURE_HEIGHT,
    DEFAULT_FIGURE_WIDTH,
    DISEASE_NODE_COLOR,
    GENE_NODE_COLOR,
    REPORTS_DIR,
)


class ResultsVisualizer:
    def __init__(self):
        """Initialize visualizer."""
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(exist_ok=True)

    def create_network_graph(self, associations: List[Dict]) -> go.Figure:
        """Create interactive network visualization."""
        df = pd.DataFrame(associations)

        fig = go.Figure(
            data=[
                go.Network(
                    node=dict(
                        label=df["gene"].unique().tolist()
                        + df["disease"].unique().tolist(),
                        color=[GENE_NODE_COLOR] * len(df["gene"].unique())
                        + [DISEASE_NODE_COLOR] * len(df["disease"].unique()),
                    ),
                    link=dict(
                        source=df["gene"], target=df["disease"], value=df["confidence"]
                    ),
                )
            ]
        )

        fig.update_layout(
            width=DEFAULT_FIGURE_WIDTH,
            height=DEFAULT_FIGURE_HEIGHT,
            title="Gene-Disease Association Network",
        )

        return fig

    def generate_report(
        self, associations: List[Dict], filename: str = "association_report.csv"
    ):
        """Generate comprehensive report of findings."""
        output_path = self.reports_dir / filename
        df = pd.DataFrame(associations)
        df.to_csv(output_path, index=False)
        return df
