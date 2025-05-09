#include<iostream>
#include<vector>
#include<queue>
#include<stack>
#include<omp.h>
#include<chrono>
#include<cstdlib>

using namespace std;



class Graph{
    int V;
    vector<vector<int>> adj;

    public:

        Graph(int V){
            this->V=V;
            adj.resize(V);
        }

        void addEdge(int u, int v){
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        void sequentialBFS(int start, int n);
        void parallelBFS(int start, int n);
        void sequentialDFS(int start, int n);
        void parallelDFS(int start, int n);

};

void Graph:: sequentialBFS(int start, int n){
    vector<bool> visited(n, false);
    queue<int> q;
    visited[start]=true;
    q.push(start);

    while(!q.empty()){
        int node=q.front();
        q.pop();
        for(int i=0;i<adj[node].size();i++){
            int neighbour=adj[node][i];
            if(visited[neighbour]==false){
                q.push(neighbour);
                visited[neighbour]=true;
            }
        }
    }
}

void Graph::parallelBFS(int start, int n){
    vector<bool> visited(n, false);
    queue<int> q;
    visited[start]=true;
    q.push(start);


    while(!q.empty()){
        #pragma omp parallel for
        for(int i=0;i<q.size();i++){
            int node;
            #pragma omp critical
            {
                node=q.front();
                q.pop();
            }
            

            for(int j=0;j<adj[node].size();j++){
                int neighbour=adj[node][j];
                #pragma omp critical
                {
                    if(visited[neighbour]==false){
                        visited[neighbour]=true;
                        q.push(neighbour);
                    }
                }
                
            }
        }
        
    }
}

void Graph::sequentialDFS(int start, int n){
    vector<bool> visited(n, false);
    stack<int> s;
    visited[start]=true;
    s.push(start);

    while(!s.empty()){
        int node=s.top();
        s.pop();

        for(int j=0;j<adj[node].size();j++){
            int neighbour=adj[node][j];
            if(visited[neighbour]==false){
                visited[neighbour]=true;
                s.push(neighbour);
            }
        }
    }
}

void Graph::parallelDFS(int start, int n){
    vector<bool> visited(n, false);
    stack<int> s;
    visited[start]=true;
    s.push(start);


    while(!s.empty()){
        int node;
        #pragma omp critical
        {
            node=s.top();
            s.pop(); 
        }

        #pragma omp parallel for
        for(int i=0;i<adj[node].size();i++){
            int neighbour=adj[node][i];
            #pragma omp critical
            {
                if(visited[neighbour]==false){
                    visited[neighbour]=true;
                    s.push(neighbour);
                }
            }
            
        }
    }


}
int main(){


    int n=7896;

    Graph g(n);

    for(int i=0;i<n;i++){
        for(int j=0;j<5;j++){
            int x=rand()%n;
            if(x!=i){
                g.addEdge(i, x);
            }
        }
    }

    auto start1=chrono::high_resolution_clock::now();
    g.sequentialBFS(0,n);
    auto end1=chrono::high_resolution_clock::now();
    chrono::duration<double> seq_bfs_duration=end1-start1;
    cout<<"\nSequential BFS time: "<<seq_bfs_duration.count();


    auto start2=chrono::high_resolution_clock::now();
    g.parallelBFS(0,n);
    auto end2=chrono::high_resolution_clock::now();
    chrono::duration<double> par_bfs_duration=end2-start2;
    cout<<"\nParallel BFS time: "<<par_bfs_duration.count();

    cout<<"\n\nSpeedup factor for BFS: "<<seq_bfs_duration.count()/par_bfs_duration.count();

    auto start3=chrono::high_resolution_clock::now();
    g.sequentialDFS(0,n);
    auto end3=chrono::high_resolution_clock::now();
    chrono::duration<double> seq_dfs_duration=end3-start3;
    cout<<"\nSequential DFS time: "<<seq_dfs_duration.count();


    auto start4=chrono::high_resolution_clock::now();
    g.parallelDFS(0,n);
    auto end4=chrono::high_resolution_clock::now();
    chrono::duration<double> par_dfs_duration=end4-start4;
    cout<<"\nParallel DFS time: "<<par_dfs_duration.count();

    cout<<"\n\nSpeedup factor for DFS: "<<seq_dfs_duration.count()/par_dfs_duration.count();






    return 0;
}
