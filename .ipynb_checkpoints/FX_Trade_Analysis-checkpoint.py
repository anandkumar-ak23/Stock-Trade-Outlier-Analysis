# import streamlit as st
# import pandas as pd
# from py2neo import Graph, Node, Relationship  # Import Relationship class
# import plotly.express as px
# import networkx as nx
# from pyvis.network import Network
# import streamlit.components.v1 as components
# import numpy as np
# # from py2neo import Graph, Node, Relationship

# # Function to create trade nodes and edges in the Neo4j graph
# def create_trade_graph(fx_data):
#     previous_trade_node = None  # To keep track of the previous trade for consecutive edges
    
#     for _, row in fx_data.iterrows():
#         # Create a trade node
#         trade_node = Node("Trade", 
#                           date=row['Date'],
#                           open=row['Open'],
#                           high=row['High'],
#                           low=row['Low'],
#                           close=row['Close'],
#                           change_pips=row['Change_Pips'],
#                           change_percent=row['Change_Percent'],
#                           volume=row['Volume'])
        
#         # Create the node in Neo4j
#         graph.create(trade_node)
        
#         # Create an edge for consecutive trades
#         if previous_trade_node:
#             consecutive_edge = Relationship(previous_trade_node, "CONSECUTIVE", trade_node)
#             graph.create(consecutive_edge)
        
#         # Update the previous node
#         previous_trade_node = trade_node
        
#     # Create edges for similar trades based on price or volume criteria
#     for i in range(len(fx_data)):
#         for j in range(i+1, len(fx_data)):
#             if abs(fx_data.iloc[i]['Change_Percent'] - fx_data.iloc[j]['Change_Percent']) < 0.1:
#                 # Create an edge between similar trades
#                 node_i = Node("Trade", date=fx_data.iloc[i]['Date'])
#                 node_j = Node("Trade", date=fx_data.iloc[j]['Date'])
#                 similar_edge = Relationship(node_i, "SIMILAR", node_j)
#                 graph.create(similar_edge)

# # Function to visualize the graph from Neo4j using NetworkX and PyVis
# def visualize_graph():
#     query = """
#     MATCH (t1:Trade)-[r]->(t2:Trade)
#     RETURN t1, t2, type(r) as relationship
#     """
#     result = graph.run(query)
    
#     # Create a NetworkX graph
#     G = nx.Graph()

#     for record in result:
#         node1 = record['t1']['date']
#         node2 = record['t2']['date']
#         relationship = record['relationship']

#         # Add nodes and edges to NetworkX graph
#         G.add_node(node1, label=node1)
#         G.add_node(node2, label=node2)
#         G.add_edge(node1, node2, label=relationship)

#     # Use PyVis to visualize the NetworkX graph
#     net = Network(notebook=True)
#     net.from_nx(G)
    
#     # Save the graph as an HTML file
#     net.show("graph.html")
    
#     # Display the graph in Streamlit using components.html
#     HtmlFile = open("graph.html", 'r', encoding='utf-8')
#     source_code = HtmlFile.read() 
#     components.html(source_code, height=500)

# # Connect to Neo4j
# graph = Graph("bolt://localhost:7687", auth=("neo4j", "pass-word"))

# st.title("Stock Trade Outlier Analysis")

# # Upload CSV file
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
# if uploaded_file:
#     fx_data = pd.read_csv(uploaded_file)
#     st.write("Data Preview:")
#     st.write(fx_data.head())
    
#     # Calculate z-scores to detect outliers
#     fx_data['z_score'] = (fx_data['Change_Pips'] - fx_data['Change_Pips'].mean()) / fx_data['Change_Pips'].std()
#     outliers = fx_data[fx_data['z_score'].abs() > 3]
    
#     # Display outliers
#     st.write("Outliers:")
#     st.write(outliers)
    
#     # Create synthetic trade volume data (if missing)
#     if 'Volume' not in fx_data.columns:
#         fx_data['Volume'] = np.random.randint(1000, 10000, size=len(fx_data))
    
#     # Create trade graph in Neo4j
#     if st.button("Create Trade Graph in Neo4j"):
#         create_trade_graph(fx_data)
#         st.success("Graph created successfully!")
    
#     # Visualization using Plotly
#     st.write("Change (Pips) Over Time:")
#     fig = px.line(fx_data, x='Date', y='Change_Pips', title='FX Trade Data - Change (Pips) Over Time')
#     st.plotly_chart(fig)

#     # Visualize the trade graph from Neo4j
#     if st.button("Visualize Trade Graph"):
#         visualize_graph()
import streamlit as st
import pandas as pd
from py2neo import Graph, Node, Relationship  # Import Relationship class
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import numpy as np

# Function to create trade nodes and edges in the Neo4j graph
def create_trade_graph(fx_data):
    previous_trade_node = None  # To keep track of the previous trade for consecutive edges
    
    for _, row in fx_data.iterrows():
        # Create a trade node
        trade_node = Node("Trade", 
                          date=row['Date'],
                          open=row['Open'],
                          high=row['High'],
                          low=row['Low'],
                          close=row['Close'],
                          change_pips=row['Change_Pips'],
                          change_percent=row['Change_Percent'],
                          volume=row['Volume'])
        
        # Create the node in Neo4j
        graph.create(trade_node)
        
        # Create an edge for consecutive trades
        if previous_trade_node:
            consecutive_edge = Relationship(previous_trade_node, "CONSECUTIVE", trade_node)
            graph.create(consecutive_edge)
        
        # Update the previous node
        previous_trade_node = trade_node
        
    # Create edges for similar trades based on price or volume criteria
    for i in range(len(fx_data)):
        for j in range(i+1, len(fx_data)):
            if abs(fx_data.iloc[i]['Change_Percent'] - fx_data.iloc[j]['Change_Percent']) < 0.1:
                # Create an edge between similar trades
                node_i = Node("Trade", date=fx_data.iloc[i]['Date'])
                node_j = Node("Trade", date=fx_data.iloc[j]['Date'])
                similar_edge = Relationship(node_i, "SIMILAR", node_j)
                graph.create(similar_edge)

# Function to visualize the graph from Neo4j using NetworkX and PyVis
def visualize_graph():
    query = """
    MATCH (t1:Trade)-[r]->(t2:Trade)
    RETURN t1, t2, type(r) as relationship
    """
    result = graph.run(query)
    
    # Create a NetworkX graph
    G = nx.Graph()

    for record in result:
        node1 = record['t1']['date']
        node2 = record['t2']['date']
        relationship = record['relationship']

        # Add nodes and edges to NetworkX graph
        G.add_node(node1, label=node1)
        G.add_node(node2, label=node2)
        G.add_edge(node1, node2, label=relationship)

    # Use PyVis to visualize the NetworkX graph
    net = Network(notebook=True)
    net.from_nx(G)
    
     # Add nodes and edges to PyVis network, with different colors for outliers
    for node in G.nodes():
        if node in outlier_dates:
            net.add_node(node, label=node, color="red")  # Highlight outliers with red
        else:
            net.add_node(node, label=node, color="blue")  # Normal nodes in blue

    for edge in G.edges(data=True):
        net.add_edge(edge[0], edge[1])
    
    # Save the graph as an HTML file
    net.show("graph.html")
    
    # Display the graph in Streamlit using components.html
    HtmlFile = open("graph.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    components.html(source_code, height=500)

# Function to compare the actual trade graph with predefined guidelines
def compare_with_guidelines(fx_data):
    # Define the expected guideline
    expected_rule = "Price difference between consecutive trades should not exceed 5%."
    
    query = """
    MATCH (t1:Trade)-[r:CONSECUTIVE]->(t2:Trade)
    RETURN t1, t2
    """
    result = graph.run(query)
    
    violations = []
    
    # Iterate through the consecutive trades and compare against the expected guidelines
    for record in result:
        trade1 = record['t1']
        trade2 = record['t2']
        
        price1 = float(trade1['close'])
        price2 = float(trade2['close'])
        
        price_diff_percent = abs((price2 - price1) / price1) * 100
        
        # Compare the price difference to the expected rule (5% in this case)
        if price_diff_percent > 5:
            violations.append({
                "Trade 1 Date": trade1['date'],
                "Trade 2 Date": trade2['date'],
                "Price Difference (%)": price_diff_percent
            })
    
    if violations:
        st.write("Violations found:")
        violations_df = pd.DataFrame(violations)
        st.write(violations_df)
    else:
        st.write("No violations found. All trades meet the guidelines.")

# Connect to Neo4j
graph = Graph("bolt://localhost:7687", auth=("neo4j", "pass-word"))

st.title("Stock Trade Outlier Analysis")

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    fx_data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(fx_data.head())
    
    # Calculate z-scores to detect outliers
    fx_data['z_score'] = (fx_data['Change_Pips'] - fx_data['Change_Pips'].mean()) / fx_data['Change_Pips'].std()
    outliers = fx_data[fx_data['z_score'].abs() > 3]
    
    # Display outliers
    outlier_dates = outliers['Date'].tolist()
    st.write("Outliers:")
    st.write(outliers)
    
    # Create synthetic trade volume data (if missing)
    if 'Volume' not in fx_data.columns:
        fx_data['Volume'] = np.random.randint(1000, 10000, size=len(fx_data))
    
    # Create trade graph in Neo4j
    if st.button("Create Trade Graph in Neo4j"):
        create_trade_graph(fx_data)
        st.success("Graph created successfully!")
    
    # Visualize the trade graph from Neo4j
    if st.button("Visualize Trade Graph"):
        visualize_graph()
    
    # Visualization using Plotly
    st.write("Change (Pips) Over Time:")
    fig = px.line(fx_data, x='Date', y='Change_Pips', title='FX Trade Data - Change (Pips) Over Time')
    st.plotly_chart(fig)

    
    

    # Compare the trade graph with expected guidelines
    if st.button("Compare Trade Graph with Guidelines"):
        compare_with_guidelines(fx_data)
