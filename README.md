# Junction-Tree-Inference-Engine
Exact inference engine implementing triangulation, maximal clique extraction, junction-tree construction, and sum-product/max-product algorithms for marginals, partition function, and top-k MAP assignments. Done as an assignment for CS726: Advanced Machine Leaning in Spring 2025 at IITB.
## 🔧 Function Overview (Brief)
Below is a short summary of what each major function in `template.py` does.

### **Triangulation & Cliques**
- **_is_simplicial_in_graph()** : Checks if a node’s neighbors already form a clique.
- **_min_fillin_vertex()** : Selects the vertex requiring the fewest fill-in edges.
- **_complete_neighbors()** : Adds fill-in edges to maintain chordality.
- **_bron_kerbosch()** : Finds all maximal cliques using the Bron–Kerbosch algorithm.
- **triangulate_and_get_cliques()** : Runs minimum fill-in triangulation and extracts maximal cliques.

### **Junction Tree Construction**
- **get_junction_tree()** : Builds a junction tree using separator sizes and a maximum-weight spanning tree (Kruskal).

### **Assigning Potentials**
- **assign_potentials_to_cliques()** : Expands input potentials to clique domains and multiplies them into clique factors.

### **Sum-Product Inference**
- **pass_messages()** : Sends sum-product messages between cliques.
- **compute_message()** : Computes a message over the separator by marginalizing sender-only variables.
- **compute_beliefs()** : Forms clique beliefs by multiplying local potentials and incoming messages.
- **get_z_value()** : Computes the partition function \(Z\).

### **Marginals**
- **compute_marginals()** : Computes normalized marginal distribution for each variable.

### **Top-k MAP**
- **compute_top_k()** : Computes the top-k most probable assignments via max-product message passing.
- **compute_message_k()** : Top-k version of message computation using heap-based candidate merging.

### **Utility**
- **Get_Input_and_Check_Output** : Loads input JSON, runs the full pipeline, and writes output.
