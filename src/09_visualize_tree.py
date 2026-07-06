#!/usr/bin/env python3
"""
09_visualize_tree.py
====================
Provides qualitative analysis by visualizing a specific sentence's
gold dependency tree side-by-side with the attention-induced MST tree.
Requires `networkx`.

Usage
-----
    python src/09_visualize_tree.py
"""

import sys
import os
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.conllu_utils import read_jsonl

# Simple hierarchical layout
def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    if not nx.is_tree(G):
        raise TypeError("cannot use hierarchy_pos on a graph that is not a tree")

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root)) if not isinstance(G, nx.DiGraph) else list(G.successors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children) != 0:
            dx = width / len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def build_nx_graph(tokens, heads):
    G = nx.DiGraph()
    for i, t in enumerate(tokens):
        # 1-indexed as ROOT is 0
        G.add_node(i+1, label=t)
    G.add_node(0, label="ROOT")
    
    # Edges from head to dependent
    for i, h in enumerate(heads):
        dep = i + 1
        G.add_edge(h, dep)
    return G

def draw_tree(ax, G, title, color="lightblue"):
    pos = hierarchy_pos(G, 0)
    labels = nx.get_node_attributes(G, 'label')
    
    # Draw edges first so they go under the text boxes
    # We pass node_size=1500 so the arrows stop at the edge of the text box
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True, arrowstyle="-|>", arrowsize=15, node_size=1500)
    
    # Draw labels with clean bounding boxes
    for node, (x, y) in pos.items():
        text = labels[node]
        ax.text(x, y, text, ha='center', va='center', 
                bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.4', alpha=0.9),
                fontsize=11, fontweight='bold')
                
    ax.set_title(title, fontweight="bold", pad=20, fontsize=13)
    ax.axis('off')

def main():
    print("Generating Qualitative Tree Vision...")
    gold_path = Path("data/processed/en.jsonl")
    pred_path = Path("results/predicted_trees/en_trees.jsonl")
    output_path = Path("results/figures/fig5_dependency_tree.pdf")
    
    if not gold_path.exists() or not pred_path.exists():
        print("Data files not found.")
        return

    gold_records = read_jsonl(str(gold_path))
    pred_records = read_jsonl(str(pred_path))
    gold_by_id = {r["sent_id"]: r for r in gold_records}
    
    target_sid = None
    target_gold = None
    target_pred = None
    
    # Find a good visualizable sentence (length 7-10) with exactly 1 or 2 mistakes
    for prec in pred_records:
        sid = prec["sent_id"]
        gold = gold_by_id[sid]
        
        if 7 <= gold["sent_len"] <= 9:
            pred_heads = prec["predictions"]["5-8"]
            gold_heads = gold["gold_heads"]
            
            # Count mistakes
            mistakes = sum(1 for p, g in zip(pred_heads, gold_heads) if p != g)
            if mistakes == 1:  # Perfect to show a minor cross-clausal error
                target_sid = sid
                target_gold = gold
                target_pred = pred_heads
                break

    if target_sid is None:
        print("No suitable sentence found.")
        return

    tokens = target_gold["tokens"]
    gold_heads = target_gold["gold_heads"]
    print(f"Visualizing sentence: {' '.join(tokens)}")

    G_gold = build_nx_graph(tokens, gold_heads)
    
    try:
        G_pred = build_nx_graph(tokens, target_pred)
        if not nx.is_tree(G_pred):
            print("Predicted graph is not a perfect tree structurally, visualization might be messy.")
    except Exception as e:
        print("Error forming predicted graph:", e)
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    draw_tree(ax1, G_gold, "Gold Standard MST", color="#a8e6cf")
    draw_tree(ax2, G_pred, "Attention Induced MST (L5-H8)", color="#ff8b94")
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    
    # Save a PNG version too so it shows up in the markdown
    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved qualitative visualization to {output_path}")

if __name__ == "__main__":
    main()
