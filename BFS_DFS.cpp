// #include <iostream>
// #include <vector>
// #include <queue>
// #include <chrono>  // For measuring time
// #include <omp.h>

// using namespace std;

// // Add an edge to the graph
// void addEdge(vector<vector<int>>& adj, int v, int w) {
//     adj[v].push_back(w);
// }

// // Sequential Depth-First Search
// void sequentialDFS(int v, vector<vector<int>>& adj, vector<bool>& visited) {
//     visited[v] = true;
//     cout << v << " ";
//     for (int i = 0; i < adj[v].size(); ++i) {
//         int n = adj[v][i];
//         if (!visited[n]) {
//             sequentialDFS(n, adj, visited);
//         }
//     }
// }

// // Parallel Depth-First Search
// void parallelDFSUtil(int v, vector<vector<int>>& adj, vector<bool>& visited) {
//     visited[v] = true;
//     cout << v << " ";

//     #pragma omp parallel for
//     for (int i = 0; i < adj[v].size(); ++i) {
//         int n = adj[v][i];
//         if (!visited[n]) {
//             parallelDFSUtil(n, adj, visited);
//         }
//     }
// }

// void parallelDFS(int startVertex, vector<vector<int>>& adj, int V) {
//     vector<bool> visited(V, false);
//     parallelDFSUtil(startVertex, adj, visited);
// }

// // Sequential Breadth-First Search
// void sequentialBFS(int startVertex, vector<vector<int>>& adj, int V) {
//     vector<bool> visited(V, false);
//     queue<int> q;

//     visited[startVertex] = true;
//     q.push(startVertex);

//     while (!q.empty()) {
//         int v = q.front();
//         q.pop();
//         cout << v << " ";

//         for (int i = 0; i < adj[v].size(); ++i) {
//             int n = adj[v][i];
//             if (!visited[n]) {
//                 visited[n] = true;
//                 q.push(n);
//             }
//         }
//     }
// }

// // Parallel Breadth-First Search
// void parallelBFS(int startVertex, vector<vector<int>>& adj, int V) {
//     vector<bool> visited(V, false);
//     queue<int> q;

//     visited[startVertex] = true;
//     q.push(startVertex);

//     while (!q.empty()) {
//         int v = q.front();
//         q.pop();
//         cout << v << " ";

//         #pragma omp parallel for
//         for (int i = 0; i < adj[v].size(); ++i) {
//             int n = adj[v][i];
//             if (!visited[n]) {
//                 visited[n] = true;
//                 q.push(n);
//             }
//         }
//     }
// }

// int main() {
//     int V = 7;  // Number of vertices
//     vector<vector<int>> adj(V);  // Adjacency list

//     // Adding edges
//     addEdge(adj, 0, 1);
//     addEdge(adj, 0, 2);
//     addEdge(adj, 1, 3);
//     addEdge(adj, 1, 4);
//     addEdge(adj, 2, 5);
//     addEdge(adj, 2, 6);

//     // Measure time for sequential DFS
//     auto start = chrono::high_resolution_clock::now();
//     cout << "Sequential Depth-First Search (DFS): ";
//     vector<bool> visited(V, false);
//     sequentialDFS(0, adj, visited);
//     auto end = chrono::high_resolution_clock::now();
//     chrono::duration<double> seqDFS_duration = end - start;
//     cout << "\nTime taken for sequential DFS: " << seqDFS_duration.count() << " seconds\n\n";

//     // Measure time for parallel DFS
//     start = chrono::high_resolution_clock::now();
//     cout << "Parallel Depth-First Search (DFS): ";
//     parallelDFS(0, adj, V);
//     end = chrono::high_resolution_clock::now();
//     chrono::duration<double> parDFS_duration = end - start;
//     cout << "\nTime taken for parallel DFS: " << parDFS_duration.count() << " seconds\n\n";

//     // Measure time for sequential BFS
//     start = chrono::high_resolution_clock::now();
//     cout << "Sequential Breadth-First Search (BFS): ";
//     sequentialBFS(0, adj, V);
//     end = chrono::high_resolution_clock::now();
//     chrono::duration<double> seqBFS_duration = end - start;
//     cout << "\nTime taken for sequential BFS: " << seqBFS_duration.count() << " seconds\n\n";

//     // Measure time for parallel BFS
//     start = chrono::high_resolution_clock::now();
//     cout << "Parallel Breadth-First Search (BFS): ";
//     parallelBFS(0, adj, V);
//     end = chrono::high_resolution_clock::now();
//     chrono::duration<double> parBFS_duration = end - start;
//     cout << "\nTime taken for parallel BFS: " << parBFS_duration.count() << " seconds\n\n";

//     return 0;
// }

// // ##### CODE TO RUN FILES ######
// // g++ -fopenmp filename.cpp -o output
// // ./output


#include <iostream>
#include <vector>
#include <queue>
#include <chrono>  // For measuring time
#include <omp.h>

using namespace std;

// Add an edge to the graph
void addEdge(vector<vector<int>>& adj, int v, int w) {
    adj[v].push_back(w);
}

// Sequential Depth-First Search
void sequentialDFS(int v, vector<vector<int>>& adj, vector<bool>& visited) {
    visited[v] = true;
    cout << v << " ";
    for (int i = 0; i < adj[v].size(); ++i) {
        int n = adj[v][i];
        if (!visited[n]) {
            sequentialDFS(n, adj, visited);
        }
    }
}

// Parallel Depth-First Search (Fixed with OpenMP locks)
void parallelDFSUtil(int v, vector<vector<int>>& adj, vector<bool>& visited, omp_lock_t* locks) {
    omp_set_lock(&locks[v]);
    if (visited[v]) {
        omp_unset_lock(&locks[v]);
        return;
    }
    visited[v] = true;
    cout << v << " ";
    omp_unset_lock(&locks[v]);

    #pragma omp parallel for
    for (int i = 0; i < adj[v].size(); ++i) {
        int n = adj[v][i];
        if (!visited[n]) {
            parallelDFSUtil(n, adj, visited, locks);
        }
    }
}

void parallelDFS(int startVertex, vector<vector<int>>& adj, int V) {
    vector<bool> visited(V, false);
    omp_lock_t locks[V];
    for (int i = 0; i < V; ++i) {
        omp_init_lock(&locks[i]);
    }
    parallelDFSUtil(startVertex, adj, visited, locks);
    for (int i = 0; i < V; ++i) {
        omp_destroy_lock(&locks[i]);
    }
}

// Sequential Breadth-First Search
void sequentialBFS(int startVertex, vector<vector<int>>& adj, int V) {
    vector<bool> visited(V, false);
    queue<int> q;

    visited[startVertex] = true;
    q.push(startVertex);

    while (!q.empty()) {
        int v = q.front();
        q.pop();
        cout << v << " ";

        for (int i = 0; i < adj[v].size(); ++i) {
            int n = adj[v][i];
            if (!visited[n]) {
                visited[n] = true;
                q.push(n);
            }
        }
    }
}

// Parallel Breadth-First Search (Fixed with OpenMP locks)
void parallelBFS(int startVertex, vector<vector<int>>& adj, int V) {
    vector<bool> visited(V, false);
    queue<int> q;

    omp_lock_t locks[V];

    for (int i = 0; i < V; ++i) {
        omp_init_lock(&locks[i]);
    }

    visited[startVertex] = true;
    q.push(startVertex);

    while (!q.empty()) {
        int v = q.front();
        q.pop();
        cout << v << " ";

        #pragma omp parallel for
        for (int i = 0; i < adj[v].size(); ++i) {
            int n = adj[v][i];

            omp_set_lock(&locks[n]);

            if (!visited[n]) {
                visited[n] = true;
                #pragma omp critical
                q.push(n);
            }
            
            omp_unset_lock(&locks[n]);
        }
    }

    for (int i = 0; i < V; ++i) {
        omp_destroy_lock(&locks[i]);
    }
}

int main() {
    int V = 7;  // Number of vertices
    vector<vector<int>> adj(V);  // Adjacency list

    // Adding edges
    addEdge(adj, 0, 1);
    addEdge(adj, 0, 2);
    addEdge(adj, 1, 3);
    addEdge(adj, 1, 4);
    addEdge(adj, 2, 5);
    addEdge(adj, 2, 6);

    // Measure time for sequential DFS
    auto start = chrono::high_resolution_clock::now();
    cout << "Sequential Depth-First Search (DFS): ";
    vector<bool> visited(V, false);
    sequentialDFS(0, adj, visited);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> seqDFS_duration = end - start;
    cout << "\nTime taken for sequential DFS: " << seqDFS_duration.count() << " seconds\n\n";

    // Measure time for parallel DFS
    start = chrono::high_resolution_clock::now();
    cout << "Parallel Depth-First Search (DFS): ";
    parallelDFS(0, adj, V);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> parDFS_duration = end - start;
    cout << "\nTime taken for parallel DFS: " << parDFS_duration.count() << " seconds\n\n";

    // Measure time for sequential BFS
    start = chrono::high_resolution_clock::now();
    cout << "Sequential Breadth-First Search (BFS): ";
    sequentialBFS(0, adj, V);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> seqBFS_duration = end - start;
    cout << "\nTime taken for sequential BFS: " << seqBFS_duration.count() << " seconds\n\n";

    // Measure time for parallel BFS
    start = chrono::high_resolution_clock::now();
    cout << "Parallel Breadth-First Search (BFS): ";
    parallelBFS(0, adj, V);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> parBFS_duration = end - start;
    cout << "\nTime taken for parallel BFS: " << parBFS_duration.count() << " seconds\n\n";

    return 0;
}

// ##### CODE TO RUN FILES ######
// g++ -fopenmp filename.cpp -o output
// ./output
