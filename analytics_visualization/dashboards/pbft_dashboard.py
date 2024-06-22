# pbft_dashboard.py
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

class PBFTDashboard:
    def __init__(self, pbft_model):
        self.pbft_model = pbft_model
        self.app = dash.Dash(__name__)

    def create_layout(self):
        self.app.layout = html.Div([
            html.H1('PBFT Dashboard'),
            html.Div([
                html.H2('Node ID:'),
                html.P(id='node-id')
            ]),
            html.Div([
                html.H2('View:'),
                html.P(id='view')
            ]),
            html.Div([
                html.H2('Sequence Number:'),
                html.P(id='seq-num')
            ]),
            html.Div([
                html.H2('Request Queue:'),
                dcc.Graph(id='request-queue-graph')
            ]),
            html.Div([
                html.H2('Pre-Prepare Queue:'),
                dcc.Graph(id='pre-prepare-queue-graph')
            ]),
            html.Div([
                html.H2('Prepare Queue:'),
                dcc.Graph(id='prepare-queue-graph')
            ]),
            html.Div([
                html.H2('Commit Queue:'),
                dcc.Graph(id='commit-queue-graph')
            ]),
            dcc.Interval(id='interval', interval=1000)
        ])

    def create_callbacks(self):
        @self.app.callback(
            Output('node-id', 'children'),
            [Input('interval', 'n_intervals')]
        )
        def update_node_id(n):
            return self.pbft_model.node_id

        @self.app.callback(
            Output('view', 'children'),
            [Input('interval', 'n_intervals')]
        )
        def update_view(n):
            return self.pbft_model.view

        @self.app.callback(
            Output('seq-num', 'children'),
            [Input('interval', 'n_intervals')]
        )
        def update_seq_num(n):
            return self.pbft_model.seq_num

        @self.app.callback(
            Output('request-queue-graph', 'figure'),
            [Input('interval', 'n_intervals')]
        )
        def update_request_queue_graph(n):
            request_queue_df = pd.DataFrame(self.pbft_model.request_queue)
            fig = go.Figure(data=[go.Bar(x=request_queue_df['key'], y=request_queue_df['value'])])
            fig.update_layout(title='Request Queue', xaxis_title='Key', yaxis_title='Value')
            return fig

        @self.app.callback(
            Output('pre-prepare-queue-graph', 'figure'),
            [Input('interval', 'n_intervals')]
        )
        def update_pre_prepare_queue_graph(n):
            pre_prepare_queue_df = pd.DataFrame(self.pbft_model.pre_prepare_queue)
            fig = go.Figure(data=[go.Bar(x=pre_prepare_queue_df['key'], y=pre_prepare_queue_df['value'])])
            fig.update_layout(title='Pre-Prepare Queue', xaxis_title='Key', yaxis_title='Value')
            return fig

        @self.app.callback(
            Output('prepare-queue-graph', 'figure'),
            [Input('interval', 'n_intervals')]
        )
        def update_prepare_queue_graph(n):
            prepare_queue_df = pd.DataFrame(self.pbft_model.prepare_queue)
            fig = go.Figure(data=[go.Bar(x=prepare_queue_df['key'], y=prepare_queue_df['value'])])
            fig.update_layout(title='Prepare Queue', xaxis_title='Key', yaxis_title='Value')
            return fig

        @self.app.callback(
            Output('commit-queue-graph', 'figure'),
            [Input('interval', 'n_intervals')]
        )
        def update_commit_queue_graph(n):
            commit_queue_df = pd.DataFrame(self.pbft_model.commit_queue)
            fig = go.Figure(data=[go.Bar(x=commit_queue_df['key'], y=commit_queue_df['value'])])
            fig.update_layout(title='Commit Queue', xaxis_title='Key', yaxis_title='Value')
            return fig

    def run(self):
        self.create_layout()
        self.create_callbacks()
        self.app.run_server()

# Example usage
pbft_model = PBFTModel('node1', ['node1', 'node2', 'node3'], 1)
pbft_dashboard = PBFTDashboard(pbft_model)
pbft_dashboard.run()
