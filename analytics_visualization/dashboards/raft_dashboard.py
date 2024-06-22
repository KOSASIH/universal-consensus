# raft_dashboard.py
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

class RaftDashboard:
    def __init__(self, raft_model):
        self.raft_model = raft_model
        self.app = dash.Dash(__name__)

    def create_layout(self):
        self.app.layout = html.Div([
            html.H1('Raft Dashboard'),
            html.Div([
                html.H2('Node Status'),
                dcc.Dropdown(
                    id='node-dropdown',
                    options=[{'label': node, 'value': node} for node in self.raft_model.nodes],
                    value=self.raft_model.node_id
                ),
                html.Div(id='node-status')
            ]),
            html.Div([
                html.H2('Log'),
                dcc.Graph(id='log-graph')
            ]),
            html.Div([
                html.H2('State Machine'),
                html.Div(id='state-machine')
            ])
        ])

    def create_callbacks(self):
        @self.app.callback(
            Output('node-status', 'children'),
            [Input('node-dropdown', 'value')]
        )
        def update_node_status(node_id):
            node_status = self.raft_model.get_node_status(node_id)
            return f'Node {node_id} is {node_status}'

        @self.app.callback(
            Output('log-graph', 'figure'),
            [Input('node-dropdown', 'value')]
        )
        def update_log_graph(node_id):
            log = self.raft_model.get_log(node_id)
            df = pd.DataFrame(log)
            fig = go.Figure(data=[go.Scatter(x=df['term'], y=df['value'])])
            fig.update_layout(title='Log', xaxis_title='Term', yaxis_title='Value')
            return fig

        @self.app.callback(
            Output('state-machine', 'children'),
            [Input('node-dropdown', 'value')]
        )
        def update_state_machine(node_id):
            state_machine = self.raft_model.get_state_machine(node_id)
            return f'State Machine: {state_machine}'

    def run(self):
        self.create_layout()
        self.create_callbacks()
        self.app.run_server()

# Example usage
raft_model = RaftModel('node1', ['node1', 'node2', 'node3'], 10, 5)
raft_dashboard = RaftDashboard(raft_model)
raft_dashboard.run()
