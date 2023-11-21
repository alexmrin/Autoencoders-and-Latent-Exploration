import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
import numpy as np
import torch
import torch.nn as nn

from models import conv_autoencoder

# Initialize the Dash app
app = dash.Dash(__name__)

matplotlib.use("Agg")

# Create a grid of invisible points for the overlay
grid_size = 0.005  # Adjust the grid size as needed
x_grid = np.arange(0, 1, grid_size)
y_grid = np.arange(0, 1, grid_size)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

device = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = "./weights/model_weights.pt"
model = conv_autoencoder()
checkpoint = torch.load(weights_path, map_location = torch.device(device))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.decoder
model.to(device)
model.eval()

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='interactive-graph',
            figure=go.Figure(
                data=[
                    go.Scattergl(
                        x=[0.5], y=[0.5],
                        mode='markers',
                        name='point'
                    ),
                    # Invisible grid overlay
                    go.Scattergl(
                        x=X_grid.ravel(), y=Y_grid.ravel(),
                        mode='markers',
                        marker=dict(size=5, opacity=0),  # Make points invisible
                        hoverinfo='none' 
                    ),
                ],
                layout=go.Layout(
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1]),
                    showlegend=False,
                    width=600,  # Adjust width and height to make the plot square
                    height=600,
                )
            ),
            style={'display': 'inline-block'}  # Inline style for side-by-side display
        ),
        # Div for the image
        html.Div(
            html.Img(id="plot-image"),
            style={'display': 'inline-block'}  # Inline style for side-by-side display
        )
    ], style={'display': 'flex', 'flex-direction': 'row'})  # Flexbox for aligning items in a row
])

# Callback to update text based on click location
@app.callback(
    [Output('interactive-graph', 'figure'), Output("plot-image", "src")],
    [Input('interactive-graph', 'clickData')],
    [State('interactive-graph', 'figure')]
)
def update_graph(clickData, figure):
    if clickData:
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        figure['data'][0]['x'] = [x]
        figure['data'][0]['y'] = [y]
        src = reconstruct(x, y)
        return figure, src
    return dash.no_update, dash.no_update

# reconstructs the 2d latent space through pretrained model
def reconstruct(x, y):
    input = torch.tensor([[x, y]]).to(device)
    output = model(input).view(28, 28)
    output = output.cpu().detach().numpy()
    plt.imshow(output, cmap="gray")
    plt.axis('off')
    plt.tight_layout()

    # Save the plot to a BytesIO buffer and encode it as a base64 string
    with io.BytesIO() as buffer:
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    image_src = f"data:image/png;base64,{encoded_image}"
    return image_src

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
