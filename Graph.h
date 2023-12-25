//
// Created by zhang on 2023/10/30.
//

#ifndef MWRFC_GRAPH_H
#define MWRFC_GRAPH_H

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <stack>
#include <utility>
#include <iostream>
#include <cstring>

using namespace std;

class Graph{
private:
    int n, m;
    int attr_size;
    int max_color;
    int max_result;
    int max_weight;
    int max_WRFCMax;
    int lower_bound;
    int parts_len;
    long long branches;
    int delta;
    int result;
    string dir;
    int *offset;
    int *edge_list;
    int *attribute;
    int *weight;
    int *peeling_idx;
    int *pend;
    int *color;
    int *WRFCMax;
    int *nvis;
    int **index_value;

    vector<int> left;
    vector<int> component;
    vector<vector<int>> divided_vertices;
    vector<int> parts_size;
    vector<int> index1;
    vector<int> index2;
    vector<int> two;
    int ****groups;

public:
    Graph(const string &_dir, int attrs, int delta);
    ~Graph();
    void ReadGraph();
    void SetColor();
    void CalculateWRFCMax();
    void ReduceGraph();
    void BronKerbosch(vector<int>&, vector<int>&);
    void MWRFCSearch(vector<int> &, vector<int> &, int, int, int, int);
    void GetConnectedComponent(int root, int *vis);
    int CalculateResult(vector<int> &V);
    void HeuristicAlgorithm();
    void DivideParts();
    void MergeParts();
    int GetUpperBoundSpeed(int, vector<int> &);
    int GetUpperBoundNormal(int, vector<int> &);
    void prepareSearch();

    void Baseline();
    void Middle();
    void Last();

    // common
    void SortWeight(vector<int> &);
};

#endif //MWRFC_GRAPH_H
