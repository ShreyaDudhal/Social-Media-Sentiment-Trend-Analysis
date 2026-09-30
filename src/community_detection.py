"""
Community Detection Module
Performs community partitioning on user network graphs using Greedy Modularity Algorithms.
"""

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


def detect_communities(G: nx.DiGraph):
    """
    Detects communities in the social network graph using NetworkX Greedy Modularity.
    Returns:
        dict: Community metrics summary.
        pd.DataFrame: User-to-community mapping dataframe.
        list: Raw list of sets containing community nodes.
    """
    if G.number_of_nodes() == 0:
        return {'num_communities': 0, 'community_sizes': []}, pd.DataFrame(), []
        
    # Convert to undirected graph for community detection
    G_undirected = G.to_undirected()
    
    try:
        communities = list(nx.community.greedy_modularity_communities(G_undirected))
    except Exception:
        # Fallback to connected components if modularity fails
        communities = [set(c) for c in nx.connected_components(G_undirected)]
        
    num_communities = len(communities)
    community_sizes = [len(c) for c in communities]
    
    user_comm_map = []
    for comm_id, comm_nodes in enumerate(communities, start=1):
        for node in comm_nodes:
            deg = G.degree(node)
            user_comm_map.append({
                'User': node,
                'Community ID': f"Community {comm_id}",
                'Degree': deg
            })
            
    comm_df = pd.DataFrame(user_comm_map)
    comm_df = comm_df.sort_values(by=['Community ID', 'Degree'], ascending=[True, False]).reset_index(drop=True)
    
    summary = {
        'num_communities': num_communities,
        'community_sizes': community_sizes,
        'largest_community_size': max(community_sizes) if community_sizes else 0,
        'smallest_community_size': min(community_sizes) if community_sizes else 0
    }
    
    return summary, comm_df, communities


def generate_community_network_fig(G: nx.DiGraph, communities: list, top_n_nodes=40):
    """
    Generates a Plotly network graph with node colors representing detected communities.
    """
    if G.number_of_nodes() == 0 or not communities:
        fig = go.Figure()
        fig.update_layout(title="No community data available")
        return fig
        
    G_undirected = G.to_undirected()
    
    if G.number_of_nodes() > top_n_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:top_n_nodes]
        H = G_undirected.subgraph(top_nodes).copy()
    else:
        H = G_undirected.copy()
        
    pos = nx.spring_layout(H, seed=42, k=0.35)
    
    # Assign community ID to each node
    node_community = {}
    for comm_id, comm_nodes in enumerate(communities, start=1):
        for node in comm_nodes:
            node_community[node] = comm_id
            
    edge_x = []
    edge_y = []
    for edge in H.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#94a3b8'),
        hoverinfo='none',
        mode='lines'
    )
    
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    
    color_palette = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
    
    for node in H.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        c_id = node_community.get(node, 1)
        deg = H.degree(node)
        node_text.append(f"User: {node}<br>Assigned: Community {c_id}<br>Connections: {deg}")
        node_colors.append(color_palette[(c_id - 1) % len(color_palette)])
        
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node for node in H.nodes()],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            size=22,
            color=node_colors,
            line=dict(width=2, color='#ffffff')
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="User Community Network (Color-coded by Greedy Modularity Partition)",
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
