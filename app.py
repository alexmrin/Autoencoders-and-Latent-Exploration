import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import numpy as np
import torch
import torch.nn as nn

from models import conv_autoencoder

# Initialize the Dash app
app = dash.Dash(__name__)

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
model.eval()
print(model)

# Define the layout of the app
app.layout = html.Div([
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
                width=700,  # Adjust width and height to make the plot square
                height=700
            )
        )
    ),
    html.Div(id='click-info')
])

# Callback to update text based on click location
@app.callback(
    Output('interactive-graph', 'figure'),
    [Input('interactive-graph', 'clickData')],
    [State('interactive-graph', 'figure')]
)
def update_graph(clickData, figure):
    if clickData:
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
        figure['data'][0]['x'] = [x]
        figure['data'][0]['y'] = [y]
        return figure
    return dash.no_update

# reconstructs the 2d latent space through pretrained model
def reconstruct(x, y):
    output = model()



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
