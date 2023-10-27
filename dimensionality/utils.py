def visualize_dict(data_dict):

    import networkx as nx
    assert type(data_dict) is dict

    def generate_graph(d, G: nx.classes.graph.Graph, label_dict={}, root="0"):

        for k, v in d.items():

            label_dict[str(root)+k] = k
            k = str(root)+k
            G.add_node(k)
            G.add_edge(root, k)

            if isinstance(v, dict):
                generate_graph(v, G, label_dict, k)
            else:
                pass

        return label_dict

    G = nx.DiGraph()
    label_dict = generate_graph(data_dict, G)

    options = {"font_size": 7,
               "node_size": 1,
               "node_color": "blue",
               "edgecolors": "black",
               "linewidths": 1,
               "width": 1,
               # "pos": nx.drawing.nx_agraph.graphviz_layout(G, prog='dot'),
               "pos": nx.drawing.nx_agraph.graphviz_layout(G, prog="twopi", args=""),
               "arrows": False,
               "alpha": 1
               }

    nx.draw(G, labels=label_dict, with_labels=True, **options)

    return G


def visualize_dict_layer(data_dict, layer_root=None, layer_name=None, ax=None):

    import networkx as nx
    import matplotlib.pyplot as plt
    assert type(data_dict) is dict

    def generate_graph(d, G: nx.classes.graph.Graph, label_dict={}, root="0"):

        for k, v in d.items():

            if (layer_root in k and layer_name in k) or (layer_root not in k) or (layer_name is None):
                label_dict[str(root)+k] = k
                k = str(root)+k
                G.add_node(k)
                G.add_edge(root, k)

                if isinstance(v, dict):
                    generate_graph(v, G, label_dict, k)
                else:
                    pass

        return label_dict

    G = nx.DiGraph()
    label_dict = generate_graph(data_dict, G)

    options = {"font_size": 7,
               "node_size": 1,
               "node_color": "blue",
               "edgecolors": "black",
               "linewidths": 1,
               "width": 1,
               # "pos": nx.drawing.nx_agraph.graphviz_layout(G, prog='dot'),
               "pos": nx.drawing.nx_agraph.graphviz_layout(G, prog="twopi", args=""),
               "arrows": False,
               "alpha": 1
               }

    if ax is not None:
        nx.draw(G, labels=label_dict, with_labels=True, ax=ax, **options)
    else:
        nx.draw(G, labels=label_dict, with_labels=True, **options)
        plt.show()

    # import pprint
    # pprint.pprint(label_dict)

    return G
