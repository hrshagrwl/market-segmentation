import igraph
import sys
import numpy as np
import pandas as pd


# Function to return cosine similarity between 2 vectors
def similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Get Attribute vector of a vertex v of Graph g
def get_vertex_attri(G, v):
  return list(G.vs[v].attributes().values())

# Get similarity between 2 vertices of a graph
def get_sim_between(G, i, j):
  return round(similarity(get_vertex_attri(G, i), get_vertex_attri(G, j)), 2)

# Find the cluster the vertex currently belongs to
def find_cluster(vertex, C):
  for c in C:
    if vertex in c:
      return c
  return []

# Computes the modularity gain when a vertes is added to a cluster
# Equation 4 in the paper
def compute_mod_gain(G, vertex, cluster, sim_matrix, alpha):
  # Number of edges
  E = len(G.es)

  cluster = list(set(cluster))
  G_ij = 0
  Q_attr = 0

  for member in cluster:
    Q_attr += sim_matrix[vertex][member]
    if G.are_connected(vertex, member):
      edge = G.get_eid(vertex, member)
      G_ij += G.es['weight'][edge]

  # print(G_ij)
  d_i = sum(G.degree(cluster))
  d_x = G.degree(vertex)
  Q_newman = (G_ij - ((d_i * d_x) / (2 * E))) / (2 * E)

  # Normalize
  Q_attr = Q_attr / len(cluster) / len(cluster)
  delta_Q = alpha * Q_newman + (1 - alpha) * Q_attr
  return delta_Q


def phase_one(G, sim_matrix, C, alpha):
  changed = False
  V = G.vcount()

  for v in range(V):
    # cluster of Vertex v
    C_i = find_cluster(v, C)

    # Max Modularity Gain
    max_mod_gain = -1
    C_new = []

    # c -> cluster
    # C -> Set of all Clusters
    for c in C:
      # Compute for a different cluster
      if set(c) != set(C_i):
        # Compute the modularity gain for cluster c when vertex v is added
        gain = compute_mod_gain(G, v, c, sim_matrix, alpha)
        if gain > 0 and gain > max_mod_gain:
          max_mod_gain = gain
          C_new = c

    if max_mod_gain > 0 and set(C_i) != set(C_new):
      # print(v, C_i, C_new)
      # Remove from older cluster
      C_i.remove(v)
      # Add to new cluster
      C_new.append(v)
      # Mark the change (count the number of changes)
      changed = True
      # if a cluster becomes empty, remove it from the set C
      if len(C_i) == 0:
        C.remove([])

  return changed

# Runs the phase 1 of the algorithm
def run_phase_one(G, alpha, epochs):
  i = 0
  change = True
  # Vertices
  V = G.vcount()
  # Initial Clusters
  C = [[x] for x in range(V)]
  # Similarity Matrix
  sim_matrix = np.array([[get_sim_between(G, i, j)
                          for j in range(V)] for i in range(V)])

  while change and i < epochs:
    change = phase_one(G, sim_matrix, C, alpha)
    print("Number of Clusters after iteration {} - {}".format(i, len(C)))
    i += 1

  return C

# Runs the phase 2 of the algorithm
def run_phase_two(G, C, alpha, epochs):
  V = G.vcount()
  new_vertex_map = [0 for x in range(V)]

  for idx, cluster in enumerate(C):
    for vertex in cluster:
      new_vertex_map[vertex] = idx

  # https://igraph.org/python/doc/igraph.GraphBase-class.html#contract_vertices
  G.contract_vertices(new_vertex_map, combine_attrs="mean")
  # https://igraph.org/python/doc/igraph.GraphBase-class.html#simplify
  G.simplify(combine_edges=sum)

  print('Vertex count after contracting the graph -> ', G.vcount())

  new_clusters = run_phase_one(G, alpha, epochs)
  return new_clusters

# Write the output to file as required
def writeToFile(C):
  file = open("communities.txt", 'w+')
  for c in C:
    file.write(','.join(str(x) for x in c))
    file.write('\n')

  file.close()

# Calculate the composite modularity of the graph after clustering
def composite_modularity(G, C):
  V = G.vcount()
  membership = np.zeros((V, ), dtype = int)
  for idx, c in enumerate(C):
    for vertex in c:
      membership[vertex] = idx
  return G.modularity(membership, weights='weight')

# After phase 2, reinsert the node values into the array using p1 clusters
def map_p2_cluster(p1_clusters, p2_clusters):
  mapped_p2_clusters = []
  for cluster in p2_clusters:
    temp = []
    for vertex in cluster:
      temp += p1_clusters[vertex]
    mapped_p2_clusters.append(temp)
  return mapped_p2_clusters

def main(alpha):
    # Attributes
    df = pd.read_csv('./data/fb_caltech_small_attrlist.csv')

    # Edges
    file = open('./data/fb_caltech_small_edgelist.txt')
    E = []
    for line in file:
      split = line.split(" ")
      E.append(tuple(map(int, split)))

    # Vertices
    V = len(df)

    # Graph
    G = igraph.Graph()
    G.add_vertices(V)
    G.add_edges(E)
    G.es['weight'] = np.ones((V, ), dtype=int)
    for col in df.keys():
      G.vs[col] = df[col]

    # Number of iterations
    epochs = 15
    # Phase 1
    p1_clusters = run_phase_one(G, alpha, epochs)
    # Get modularity
    p1_mod = composite_modularity(G, p1_clusters)

    # Phase 2
    p2_clusters = run_phase_two(G, p1_clusters, alpha, epochs)
    # Get Modularity
    p2_mod = composite_modularity(G, p2_clusters)

    # Map the clusters for writing to file
    p2_clusters = map_p2_cluster(p1_clusters, p2_clusters)
    
    print('Modularity of clusters after p1 -> ', p1_mod)
    print('Modularity of clusters after p2 -> ', p2_mod)

    if (p1_mod > p2_mod):
      writeToFile(p1_clusters)
    else:
      writeToFile(p2_clusters)


if __name__ == '__main__':
  if (len(sys.argv) != 2):
    print("Missing / Invalid Parameter")
    sys.exit(1)
  else:
    alpha = float(sys.argv[1])
    main(alpha)
