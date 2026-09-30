"""
Social Network Analysis (SNA) Module
Constructs user interaction graphs, calculates degree centrality and graph metrics using NetworkX.
"""

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from src.data_preprocessing import extract_mentions_list


def build_social_network_graph(df: pd.DataFrame):
    """
    Constructs a weighted directed graph where:
        Nodes = Users (user_id)
        Edges = Interactions (user_id -> mentioned_user)
        Edge Weight = Frequency of mentions
    """
    G = nx.DiGraph()
    
    if df.empty:
        return G, {}
        
    for idx, row in df.iterrows():
        sender = str(row.get('user_id', '')).strip()
        if not sender or sender == 'nan':
            continue
            
        G.add_node(sender)
        
        mentions = extract_mentions_list(row.get('mentioned_user', ''))
        if not mentions and 'post' in df.columns:
            mentions = extract_mentions_list(row.get('post', ''))
            
        for target in mentions:
            target = str(target).strip()
            if not target or target == 'nan' or target == sender:
                continue
                
            G.add_node(target)
            if G.has_edge(sender, target):
                G[sender][target]['weight'] += 1
            else:
                G.add_edge(sender, target, weight=1)
                
    return G


def compute_network_metrics(G: nx.DiGraph):
    """
    Calculates key graph metrics: Node count, Edge count, Average Degree, Density, and Degree Centrality.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes == 0:
        return {
            'num_nodes': 0,
            'num_edges': 0,
            'avg_degree': 0.0,
            'density': 0.0
        }, pd.DataFrame()
        
    degrees = dict(G.degree())
    avg_degree = round(sum(degrees.values()) / max(1, num_nodes), 2)
    density = round(nx.density(G), 4)
    
    # Calculate Degree Centrality
    centrality = nx.degree_centrality(G)
    
    cent_list = []
    for node, cent_val in centrality.items():
        cent_list.append({
            'User': node,
            'Degree Centrality': round(cent_val, 4),
            'Number of Connections': degrees.get(node, 0)
        })
        
    cent_df = pd.DataFrame(cent_list)
    cent_df = cent_df.sort_values(by=['Degree Centrality', 'Number of Connections'], ascending=False).reset_index(drop=True)
    cent_df.insert(0, 'Rank', range(1, len(cent_df) + 1))
    
    metrics = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'density': density
    }
    
    return metrics, cent_df


def generate_interactive_network_fig(G: nx.DiGraph, top_n_nodes=40):
    """
    Generates a 2D interactive Plotly graph visualization of the social network.
    Node sizes proportional to degree centrality; edge thickness by interaction weight.
    """
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(title="No network connections found in dataset")
        return fig
        
    # Filter to top connected nodes if graph is very large
    if G.number_of_nodes() > top_n_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n_nodes]
        H = G.subgraph(top_nodes).copy()
    else:
        H = G.copy()
        
    pos = nx.spring_layout(H, seed=42, k=0.3)
    centrality = nx.degree_centrality(H)
    
    edge_x = []
    edge_y = []
    for edge in H.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#64748b'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    node_colors = []
    
    for node in H.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        cent = centrality.get(node, 0)
        deg = H.degree(node)
        node_text.append(f"User: {node}<br>Centrality: {round(cent, 3)}<br>Connections: {deg}")
        node_sizes.append(max(15, int(cent * 120) + 15))
        node_colors.append(deg)
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node if centrality.get(node, 0) > 0.05 else "" for node in H.nodes()],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title='Connections',
                xanchor='left'
            ),
            line=dict(width=2, color='#ffffff')
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Social Interaction Network Graph (Nodes: Users, Edges: Mentions)",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig
