import json
import itertools
import heapq
import math

########################################################################

# Do not install any external packages. You can only use Python's default libraries such as:
# json, math, itertools, collections, functools, random, heapq, etc.

########################################################################

class Inference:
    def __init__(self , data):
        """
        data : dict
            The input data containing the graphical model details, such as variables,
            cliques, potentials, and k value.
        
        What to do here:
        ----------------
        - Parse the input data and store necessary attributes (e.g., variables, cliques,
        potentials, k value).
        - Initialize any data structures required for triangulation, junction tree creation,
        and message passing.

        Refer to the sample test case for the structure of the input data.
            """

        self.test_case_number = data.get("TestCaseNumber")
        self.num_variables = data.get("VariablesCount", 0)
        self.num_potentials = data.get("Potentials_count", 0)
        self.clique_potentials = data.get("Cliques and Potentials", [])
        self.k = data.get("k value (in top k)", 1)
        
        self.graph = {v: set() for v in range(self.num_variables)}
        for clique_info in self.clique_potentials:
            clique_nodes = clique_info.get("cliques", [])
            for i in range(len(clique_nodes)):
                for j in range(i + 1, len(clique_nodes)):
                    a, b = clique_nodes[i], clique_nodes[j]
                    self.graph[a].add(b)
                    self.graph[b].add(a)

                    
        self.triangulated_graph = None  
        self.cliques = []                 
        self.messages = {}
        self.marginals = {}
        self.top_k_assignments = []
        self.z_value = None
        self.top_k_messages = {}

    def _is_simplicial_in_graph(self, v, graph):
        neighbors = graph[v]
        from itertools import combinations
        for u, w in combinations(neighbors, 2):
            if w not in graph[u]:
                return False
        return True

    def _min_fillin_vertex(self, graph):
        from itertools import combinations
        best_v = None
        min_missing = float('inf')
        for v in graph:
            nbrs = list(graph[v])
            n = len(nbrs)
            total_possible = n * (n - 1) // 2
            existing = 0
            for u, w in combinations(nbrs, 2):
                if w in graph[u]:
                    existing += 1
            missing = total_possible - existing
            if missing < min_missing:
                min_missing = missing
                best_v = v
        return best_v

    def _complete_neighbors(self, v, working_graph):
        nbrs = list(working_graph[v])
        for u, w in itertools.combinations(nbrs, 2):
            if w not in working_graph[u]:
                working_graph[u].add(w)
                working_graph[w].add(u)
                self.graph[u].add(w)
                self.graph[w].add(u)

    def _bron_kerbosch(self, R, P, X, cliques, graph):
        if not P and not X:
            cliques.append(sorted(list(R)))
            return
        for v in list(P):
            new_R = R.union({v})
            new_P = P.intersection(graph[v])
            new_X = X.intersection(graph[v])
            self._bron_kerbosch(new_R, new_P, new_X, cliques, graph)
            P.remove(v)
            X.add(v)
    
    def triangulate_and_get_cliques(self):
        working_graph = {v: set(neighbors) for v, neighbors in self.graph.items()}
        elimination_order = [] 
        while working_graph:
            found_simplicial = False
            for v in list(working_graph.keys()):
                if self._is_simplicial_in_graph(v, working_graph):
                    elimination_order.append(v)
                    for u in working_graph[v]:
                        working_graph[u].remove(v)
                    del working_graph[v]
                    found_simplicial = True
                    break  
            
            if not found_simplicial:
                v = self._min_fillin_vertex(working_graph)
                self._complete_neighbors(v, working_graph)
                elimination_order.append(v)
                for u in working_graph[v]:
                    working_graph[u].remove(v)
                del working_graph[v]
        
        self.triangulated_graph = {v: set(neighbors) for v, neighbors in self.graph.items()}
        all_nodes = set(self.triangulated_graph.keys())
        cliques = []
        self._bron_kerbosch(set(), all_nodes, set(), cliques, self.triangulated_graph)
        self.cliques = cliques

    def get_junction_tree(self):
        """
        Construct the junction tree from the maximal cliques.
        
        What to do here:
        ----------------
        - Create a junction tree using the maximal cliques obtained from the triangulated graph.
        - Ensure the junction tree satisfies the running intersection property.
        - Store the junction tree for later use in message passing.

        Refer to the problem statement for details on junction tree construction.
        """
        separator = {}
        separator_sizes = {}
        num_cliques = len(self.cliques)
        for i in range(num_cliques):
            for j in range(i + 1, num_cliques):
                intersection = set(self.cliques[i]) & set(self.cliques[j])
                separator[(i, j)] = intersection
                separator_sizes[(i, j)] = len(intersection)

        sorted_edges = sorted(separator_sizes.items(), key=lambda item: item[1], reverse=True)

        parent = list(range(num_cliques))
        rank = [0] * num_cliques

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if rank[root_i] > rank[root_j]:
                    parent[root_j] = root_i
                elif rank[root_i] < rank[root_j]:
                    parent[root_i] = root_j
                else:
                    parent[root_j] = root_i
                    rank[root_i] += 1

        junction_tree = {i: {} for i in range(num_cliques)}

        for ((i, j), weight) in sorted_edges:
            if find(i) != find(j):
                union(i, j)
                junction_tree[i][j] = separator[(i, j)]
                junction_tree[j][i] = separator[(i, j)] 

        self.junction_tree = {
            "cliques": {i: {"vars": sorted(clique), "potential": None} for i, clique in enumerate(self.cliques)},
            "adjacency_list": junction_tree
        }


    def assign_potentials_to_cliques(self):
        """
        Assign each input potential to a junction tree clique that subsumes its variables.
        
        For each clique in the junction tree (assumed stored in self.junction_tree["cliques"]),
        we initialize a factor (potential table) with the multiplicative identity (1.0's).
        Then, for each input potential (from self.clique_potentials), we look for a
        clique whose variables include all variables of the potential. We "expand" the input
        potential from its original variables to the clique’s domain and multiply it in.
        """
        updated_cliques = {}

        for clique_id, clique_data in self.junction_tree.get("cliques", {}).items():
            sorted_vars = sorted(clique_data["vars"])  
            n = len(sorted_vars)
            updated_cliques[clique_id] = {
                "vars": sorted_vars,
                "potential": [1.0] * (2 ** n)
            }

        def index_to_assignment(index, n):
            return [(index >> (n - i - 1)) & 1 for i in range(n)]

        def assignment_to_index(assignment):
            return sum(bit * (2 ** i) for i, bit in enumerate(reversed(assignment)))

        def expand_and_multiply(current_factor, clique_vars, input_factor, input_vars):
            n = len(clique_vars)
            new_factor = [0.0] * (2 ** n)
            mapping = [clique_vars.index(v) for v in input_vars]
            for i in range(2 ** n):
                full_assign = index_to_assignment(i, n)
                sub_assign = [full_assign[pos] for pos in mapping]
                sub_i = assignment_to_index(sub_assign)
                new_factor[i] = current_factor[i] * input_factor[sub_i]

            return new_factor

        for cp in self.clique_potentials:
            input_vars = cp.get("cliques", [])
            input_pot = cp.get("potentials", [])
            assigned = False

            for clique_id, clique_data in updated_cliques.items():
                clique_vars = clique_data["vars"]
                if set(input_vars).issubset(set(clique_vars)): 
                    updated_cliques[clique_id]["potential"] = expand_and_multiply(
                        clique_data["potential"], clique_vars, input_pot, input_vars
                    )
                    assigned = True
                    break

            if not assigned:
                print(f"Warning: Could not assign potential for variables {input_vars}")

        self.junction_tree["cliques"] = updated_cliques

    def pass_messages(self):
        message_queue = []
        received = {clique: set() for clique in self.junction_tree["cliques"]}
        for clique, neighbors in self.junction_tree["adjacency_list"].items():
            if len(neighbors) == 1:  
                neighbor = next(iter(neighbors))
                message_queue.append((clique, neighbor))

        while message_queue:
            sender, receiver = message_queue.pop(0)
            message = self.compute_message(sender, receiver)
            self.messages[(sender, receiver)] = message
            received[receiver].add(sender)
            for neighbor in self.junction_tree["adjacency_list"][receiver]:
                if neighbor not in received[receiver]:  
                    if received[receiver] == set(self.junction_tree["adjacency_list"][receiver]) - {neighbor}:
                        message_queue.append((receiver, neighbor))  

        for sender, neighbors in self.junction_tree["adjacency_list"].items():
            for receiver in neighbors:
                if (sender, receiver) not in self.messages:
                    message_queue.append((sender, receiver))

        while message_queue:
            sender, receiver = message_queue.pop(0)
            message = self.compute_message(sender, receiver)
            self.messages[(sender, receiver)] = message


    def compute_message(self, sender, receiver):
        message = {}
        sender_clique = self.junction_tree["cliques"][sender]
        sender_vars = sender_clique["vars"]
        sender_potential = sender_clique["potential"]
        receiver_clique = self.junction_tree["cliques"][receiver]
        receiver_vars = receiver_clique["vars"]

        separator_vars = sorted(set(sender_vars) & set(receiver_vars))
        num_assignments = 2 ** len(sender_vars)
        incoming_product = [1] * num_assignments
        for neighbor in self.junction_tree["adjacency_list"][sender]:
            if neighbor != receiver:
                neighbor_clique = self.junction_tree["cliques"][neighbor]
                common_vars = sorted(set(neighbor_clique["vars"]) & set(sender_vars))
                for idx in range(num_assignments):
                    assignment = {}
                    temp = idx
                    for var in reversed(sender_vars):
                        assignment[var] = temp % 2
                        temp //= 2
                    key = tuple(assignment[var] for var in common_vars)
                    incoming_product[idx] *= self.messages[(neighbor, sender)].get(key, 1)
        
        for idx in range(num_assignments):
            assignment = {}
            temp = idx
            for var in reversed(sender_vars):
                assignment[var] = temp % 2
                temp //= 2
            contribution = sender_potential[idx] * incoming_product[idx]
            sep_assignment = tuple(assignment[var] for var in separator_vars)
            if sep_assignment not in message:
                message[sep_assignment] = 0
            message[sep_assignment] += contribution

        return message
    
    def compute_beliefs(self):
        self.beliefs = {}
        
        for clique_id, clique in self.junction_tree["cliques"].items():
            clique_vars = clique["vars"]
            clique_potential = clique["potential"]
            num_assignments = 2 ** len(clique_vars)
            belief = clique_potential[:]

            for neighbor in self.junction_tree["adjacency_list"].get(clique_id, {}):
                if (neighbor, clique_id) in self.messages:
                    separator_vars = sorted(set(clique_vars) & set(self.junction_tree["cliques"][neighbor]["vars"]))
                    incoming_message = self.messages[(neighbor, clique_id)]
                    for idx in range(num_assignments):
                        assignment = {}
                        temp = idx
                        for var in reversed(clique_vars):
                            assignment[var] = temp % 2 
                            temp //= 2 
                        
                        sep_assignment = tuple(assignment[var] for var in separator_vars)
                        if sep_assignment in incoming_message:
                            belief[idx] *= incoming_message[sep_assignment]
            
            self.beliefs[clique_id] = belief

        Z = sum(self.beliefs[next(iter(self.beliefs))])  
        self.z_value = Z
        for clique_id in self.beliefs:
            self.beliefs[clique_id] = [b /Z for b in self.beliefs[clique_id]]

    def get_z_value(self):
        """
        Compute the partition function (Z value) using two-pass sum-product message passing.

        Steps:
        1. Compute upward messages from leaves to root.
        2. Compute downward messages from root to leaves.
        3. Compute Z at the root clique.
        """
        self.pass_messages()
        self.compute_beliefs()
        return int(self.z_value)
        
    
    def compute_marginals(self):
        self.marginals = [[] for _ in range(self.num_variables)] 
        remaining_vars = set(range(self.num_variables))  
        
        for clique_id, clique in self.junction_tree["cliques"].items():
            if not remaining_vars: 
                break

            belief = self.beliefs[clique_id]
            clique_vars = clique["vars"]
            num_assignments = 2 ** len(clique_vars)
            
            for var in clique_vars:
                if var in remaining_vars:
                    marginal = [0, 0]  
                    for idx in range(num_assignments):
                        assignment = {}
                        temp = idx
                        for v in reversed(clique_vars):
                            assignment[v] = temp % 2
                            temp //= 2
                        
                        marginal[assignment[var]] += belief[idx]
                    total = sum(marginal)
                    marginal = [p / total for p in marginal]
                    
                    self.marginals[var] = marginal 
                    remaining_vars.remove(var)

        return self.marginals
        
    def compute_top_k(self):
        """
        Compute the top-k most probable assignments in the graphical model using max-product message passing.

        Returns:
            A list of dictionaries, where each dictionary contains:
            - "assignment": The variable assignment (list of binary values).
            - "probability": The corresponding unnormalized probability.
        """
        k = self.k  # Number of top assignments to find
        cliques = self.junction_tree["cliques"]

        # Step 1: Initialize message-passing structures
        message_queue = []
        received = {clique: set() for clique in range(len(cliques))}
        senders = []

        # Add leaf nodes to the message queue
        for clique, neighbors in self.junction_tree["adjacency_list"].items():
            if len(neighbors) == 1:  # Leaf node
                neighbor = next(iter(neighbors))
                message_queue.append((clique, neighbor))
                senders.append(clique)

        # Step 2: Perform max-product message passing
        while message_queue:
            sender, receiver = message_queue.pop(0)
            message = self.compute_message_k(sender, receiver, k)
            self.top_k_messages[(sender, receiver)] = message
            received[receiver].add(sender)

            # Check if all messages have been received by the receiver
            for neighbor in self.junction_tree["adjacency_list"][receiver]:
                if neighbor not in received[receiver] and neighbor not in senders:
                    if received[receiver] == set(self.junction_tree["adjacency_list"][receiver]) - {neighbor}:
                        message_queue.append((receiver, neighbor))
                        senders.append(receiver)

        # Step 3: Compute top-k assignments from the root clique
        last_receiver = max(self.top_k_messages.keys(), key=lambda x: x[1])[1]
        receiver_clique = cliques[last_receiver]
        receiver_potential = receiver_clique["potential"]
        receiver_vars = receiver_clique["vars"]
        num_assignments = 2 ** len(receiver_vars)

        total_product = [[] for _ in range(num_assignments)]

        # Initialize with root clique potential
        for idx in range(num_assignments):
            heapq.heappush(total_product[idx], (receiver_potential[idx], {}))

        # Merge messages into root clique's belief
        for (sender, receiver), message in self.top_k_messages.items():
            if receiver == last_receiver:
                sender_clique = cliques[sender]
                common_vars = sorted(set(sender_clique["vars"]) & set(receiver_vars))

                for idx in range(num_assignments):
                    assignment = {var: (idx >> i) & 1 for i, var in enumerate(reversed(receiver_vars))}
                    key = tuple(assignment[var] for var in common_vars)

                    if key in message:
                        best_values = message[key]

                        candidates = []
                        for best_value, best_assignment in best_values:
                            for j in range(len(total_product[idx])):
                                new_prob = best_value * total_product[idx][j][0]
                                new_assignment = dict(total_product[idx][j][1])
                                new_assignment.update(best_assignment) 
                                candidates.append((new_prob, frozenset(new_assignment.items())))

                        total_product[idx] = heapq.nlargest(k, candidates, key=lambda x: x[0])

        final_assignments = []
        for idx in range(num_assignments):
            for prob, assignment in total_product[idx]:
                assignment_dict = dict(assignment)
                assignment_dict.update({var: (idx >> i) & 1 for i, var in enumerate(reversed(receiver_vars))})
                heapq.heappush(final_assignments, (prob, frozenset(assignment_dict.items())))
                if len(final_assignments) > k:
                    heapq.heappop(final_assignments)

        final_assignments = sorted(final_assignments, reverse=True, key=lambda x: x[0])
        
        return [{"assignment": [dict(assignment).get(var, 0) for var in range(self.num_variables)],
                "probability": prob / self.z_value} 
                for prob, assignment in final_assignments]


    def compute_message_k(self, sender, receiver, k):
        """
        Compute the top-k max-product message from sender clique to receiver clique.

        Args:
            sender: Index of the sender clique.
            receiver: Index of the receiver clique.
            k: Number of top assignments to compute.

        Returns:
            A dictionary mapping separator assignments to their top-k max-product values.
        """
        sender_clique = self.junction_tree["cliques"][sender]
        sender_vars = sender_clique["vars"]
        sender_potential = sender_clique["potential"]
        
        separator_vars = sorted(set(sender_vars) & set(self.junction_tree["cliques"][receiver]["vars"]))
        
        num_sender_assignments = 2 ** len(sender_vars)
        
        incoming_product = [[] for _ in range(num_sender_assignments)]
        
        # Initialize incoming product heaps
        for idx in range(num_sender_assignments):
            heapq.heappush(incoming_product[idx], (1, {}))

        # Multiply incoming messages from neighbors (excluding receiver)
        for neighbor in self.junction_tree["adjacency_list"][sender]:
            if neighbor != receiver:
                common_vars = sorted(set(self.junction_tree["cliques"][neighbor]["vars"]) & set(sender_vars))

                for idx in range(num_sender_assignments):
                    assignment = {var: (idx >> i) & 1 for i, var in enumerate(reversed(sender_vars))}
                    key = tuple(assignment[var] for var in common_vars)
                    neighbor_messages = self.top_k_messages[(neighbor, sender)].get(key, [])

                    candidates = []
                    for best_prob, best_assignment in neighbor_messages:
                        for j in range(len(incoming_product[idx])):
                                new_prob = best_prob * incoming_product[idx][j][0]
                                new_assignment = dict(incoming_product[idx][j][1])
                                new_assignment.update(best_assignment) 
                                candidates.append((new_prob, frozenset(new_assignment.items())))

                    incoming_product[idx] = heapq.nlargest(k, candidates , key = lambda x: x[0])

        message = {}
        
        for idx in range(num_sender_assignments):
            for base_prob, base_assignment in incoming_product[idx]:
                assignment_dict = dict(base_assignment)
                temp_idx = idx

                for i, var in enumerate(reversed(sender_vars)):
                    assignment_dict[var] = temp_idx % 2
                    temp_idx //= 2

                contribution_prob = sender_potential[idx] * base_prob
                
                sep_assignment_key = tuple(assignment_dict[var] for var in separator_vars)
                
                if sep_assignment_key not in message:
                    message[sep_assignment_key] = []

                heapq.heappush(message[sep_assignment_key], (contribution_prob,
                                                            frozenset(assignment_dict.items())))

                if len(message[sep_assignment_key]) > k:
                    heapq.heappop(message[sep_assignment_key])

        for sep_assignment_key in message:
            message[sep_assignment_key] = heapq.nlargest(k, message[sep_assignment_key], key=lambda x: x[0])

        return message


########################################################################

# Do not change anything below this line

########################################################################

class Get_Input_and_Check_Output:
    def __init__(self, file_name):
        with open(file_name, 'r') as file:
            self.data = json.load(file)

    def get_output(self):
        n = len(self.data)
        output = []
        for i in range(n):
            inference = Inference(self.data[i]['Input'])
            inference.triangulate_and_get_cliques()
            inference.get_junction_tree()
            inference.assign_potentials_to_cliques()
            z_value = inference.get_z_value()
            marginals = inference.compute_marginals()
            top_k_assignments = inference.compute_top_k()
            output.append({
                'Marginals': marginals,
                'Top_k_assignments': top_k_assignments,
                'Z_value' : z_value
            })
        
        self.output = output

    def write_output(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.output, file, indent=4)

if __name__ == '__main__':
    evaluator = Get_Input_and_Check_Output('Sample_Testcase.json')
    evaluator.get_output()
    evaluator.write_output('Sample_Testcase_Output.json')
