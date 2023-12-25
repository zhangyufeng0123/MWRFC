//
// Created by zhang on 2023/10/30.
//
#include "Graph.h"

Graph::Graph(const string &_dir, int attrs, int delta) {
    dir = string(_dir);
    n = 0;
    m = 0;
    result = 0;
    attr_size = attrs;
    max_color = 0;
    max_result = 0;
    branches = 0;
    lower_bound = 0;
    max_weight = 10;
    max_WRFCMax = 0;
    this->delta = delta;
    parts_len = 0;

    offset = nullptr;
    edge_list = nullptr;
    attribute = nullptr;
    weight = nullptr;
    peeling_idx = nullptr;
    pend = nullptr;
    color = nullptr;
    WRFCMax = nullptr;
    nvis = nullptr;
    groups = nullptr;

    left.clear();
    component.clear();
    parts_size.resize(attrs);
    index1.clear();
    index2.clear();
    two.clear();
}

Graph::~Graph() {
    if (offset != nullptr) {
        delete[] offset;
        offset = nullptr;
    }
    if (edge_list != nullptr) {
        delete[] edge_list;
        edge_list = nullptr;
    }
    if (attribute != nullptr) {
        delete[] attribute;
        attribute = nullptr;
    }
    if (weight != nullptr) {
        delete[] weight;
        weight = nullptr;
    }
    if (peeling_idx != nullptr) {
        delete[] peeling_idx;
        peeling_idx = nullptr;
    }
    if (pend != nullptr) {
        delete[] pend;
        pend = nullptr;
    }
    if (color != nullptr) {
        delete[] color;
        color = nullptr;
    }
    if (WRFCMax != nullptr) {
        delete[] WRFCMax;
        WRFCMax = nullptr;
    }
    if (nvis != nullptr) {
        delete[] nvis;
        nvis = nullptr;
    }
}

void Graph::ReadGraph() {
    ifstream infile;
    infile.open(dir);
    if (!infile) {
        printf("Can not open the graph file !\n\n");
        exit(1);
    }
    infile >> n >> m;
    vector<vector<int>> edges(n + 1);
    int from, to;
    for (int i = 0; i < m; i++) {
        infile >> from >> to;
        if (from == to) continue;   // 取消自环
        edges[from].emplace_back(to);
        edges[to].emplace_back(from);
    }

    for (int i = 1; i <= n; i++) {
        unordered_map<int, int> hash;
        int t = 0;
        for (int j = 0; j < edges[i].size() - t; j++) {
            if (hash[edges[i][j]]) {
                swap(edges[i][j], edges[i].back());
                edges[i].pop_back();
                j--;
            } else hash[edges[i][j]] = 1;
        }
    }

    printf("< Number of nodes = %d, number of edges = %d >\n", n, m);

    edge_list = new int[m * 2 + 1];
    offset = new int[n + 1];
    pend = new int[n + 1];
    attribute = new int[n];
    weight = new int[n];
    nvis = new int[n];
    index1.resize(n);
    index2.resize(n);
    offset[0] = 0;
    for (int i = 0; i <= n - 1; i++) {
        offset[i + 1] = offset[i];
        for (int u: edges[i + 1]) {
            edge_list[offset[i + 1]++] = u - 1;
        }
    }

    for (int i = 1; i <= n; i++) {
        infile >> attribute[i - 1];
    }

    for (int i = 1; i <= n; i++) {
        infile >> weight[i - 1];
        weight[i - 1]++;
    }

    infile.close();
}

void Graph::SetColor() {
    int max_degree = 0;
    int *cvis = new int[n];
    color = new int[n];
    int *degree = new int[n];
    int *head = new int[n];
    int *nxt = new int[n];

    for (int i = 0; i < n; i++) {
        degree[i] = offset[i + 1] - offset[i];
        head[i] = n;
    }

    for (int i = 0; i < n; i++) {
        nxt[i] = head[degree[i]];
        head[degree[i]] = i;
        if (degree[i] > max_degree) max_degree = degree[i];
    }

    delete[] degree;

    for (int i = 0; i < n; i++) cvis[i] = 0;
    for (int i = 0; i < n; i++) color[i] = n;
    max_color = 0;
    for (int ii = max_degree; ii >= 1; ii--) {
        for (int jj = head[ii]; jj != n; jj = nxt[jj]) {
            int u = jj;
            for (int j = offset[u]; j < offset[u + 1]; j++) {
                int c = color[edge_list[j]];
                if (c != n) {
                    cvis[c] = 1;
                }
            }
            for (int j = 0;; j++) {
                if (!cvis[j]) {
                    color[u] = j;
                    if (j > max_color) max_color = j;
                    break;
                }
            }
            for (int j = offset[u]; j < offset[u + 1]; j++) {
                int c = color[edge_list[j]];
                if (c != n) cvis[c] = 0;
            }
        }
    }

    max_color++;

    delete[] head;
    delete[] nxt;
    delete[] cvis;
}

void Graph::CalculateWRFCMax() {
    int *colorful_d = new int [n];
    int **colorful_r = new int *[n];
    WRFCMax = new int[n];

    for (int i = 0; i < n; i++) {
        colorful_r[i] = new int[max_color];
    }

    vector<vector<int>> nums(attr_size, vector<int>(max_weight + 1, 0));
    for (int i = 0; i < n; i++) {

        for (int j = 0; j < attr_size; j++) colorful_d[i] = 0;
        for (int j = 0; j < attr_size; j++) {
            for (int jj = 0; jj < max_color; jj++) {
                colorful_r[j][jj] = 0;
            }
        }
        for (int j = 0; j < attr_size; j++) {
            fill(nums[j].begin(), nums[j].end(), 0);
        }

        colorful_d[attribute[i]] = 1;
        colorful_r[attribute[i]][color[i]] = weight[i];
        nums[attribute[i]][weight[i]]++;
        for (int j = offset[i]; j < offset[i + 1]; j++) {
            int neighbor = edge_list[j];
            if (colorful_r[attribute[neighbor]][color[neighbor]] == 0) {
                colorful_d[attribute[neighbor]]++;
            }
            if (colorful_r[attribute[neighbor]][color[neighbor]] < weight[neighbor]) {
                nums[attribute[neighbor]][colorful_r[attribute[neighbor]][color[neighbor]]]--;
                nums[attribute[neighbor]][weight[neighbor]]++;
                colorful_r[attribute[neighbor]][color[neighbor]] = weight[neighbor];
            }
        }

        int min_colorful_degree = colorful_d[0];
        for (int j = 1; j < attr_size; j++) {
            min_colorful_degree = min(min_colorful_degree, colorful_d[j]);
        }
        WRFCMax[i] = 0;
        for (int j = 0; j < attr_size; j++) {
            int number_vertices = min(min_colorful_degree + delta, colorful_d[j]);
            for (int w = max_weight; w >= 1; w--) {
                if (number_vertices > 0) {
                    WRFCMax[i] += w * min(number_vertices, nums[j][w]);
                    number_vertices -= nums[j][w];
                } else {
                    break;
                }
            }
        }
        max_WRFCMax = max(max_WRFCMax, WRFCMax[i]);
    }
    delete[] colorful_d;
    delete[] colorful_r;
}

void Graph::ReduceGraph() {
    int *head = new int[max_WRFCMax - lower_bound + 1];
    int *nxt = new int[n];
    int *colorful_d = new int[attr_size];
    int **colorful_r = new int *[attr_size];
    left.clear();

    for (int i = 0; i < attr_size; i++) {
        colorful_r[i] = new int[max_color];
    }

    for (int i = 0; i < max_WRFCMax + 1; i++) head[i] = n;
    for (int i = 0; i < n; i++) {
        nxt[i] = head[WRFCMax[i]];
        head[WRFCMax[i]] = i;
    }

    vector<vector<int>> nums(attr_size, vector<int>(max_weight + 1, 0));
    for (int ii = lower_bound + 1; ii <= max_WRFCMax; ii++) {
        for (int jj = head[ii]; jj != n; jj = nxt[jj]) {
            int u = jj;

            for (int i = 0; i < attr_size; i++) colorful_d[i] = 0;
            for (int i = 0; i < attr_size; i++) {
                for (int j = 0; j < max_color; j++) colorful_r[i][j] = 0;
            }
            for (int i = 0; i < attr_size; i++) fill(nums[i].begin(), nums[i].end(), 0);

            colorful_d[attribute[u]] = 1;
            colorful_r[attribute[u]][color[u]] = weight[u];
            nums[attribute[u]][weight[u]]++;

            for (int i = offset[u]; i < offset[u + 1]; i++) {
                int neighbor = edge_list[i];
                if (WRFCMax[neighbor] <= lower_bound) continue;
                if (colorful_r[attribute[neighbor]][color[neighbor]] == 0) {
                    colorful_d[attribute[neighbor]]++;
                }
                if (colorful_r[attribute[neighbor]][color[neighbor]] < weight[neighbor]) {
                    nums[attribute[neighbor]][colorful_r[attribute[neighbor]][color[neighbor]]]--;
                    nums[attribute[neighbor]][weight[neighbor]]++;
                    colorful_r[attribute[neighbor]][color[neighbor]] = weight[neighbor];
                }
            }

            int min_colorful_degree = colorful_d[0];
            for (int i = 1; i < attr_size; i++) {
                min_colorful_degree = min(min_colorful_degree, colorful_d[i]);
            }
            WRFCMax[u] = 0;
            for (int i = 0; i < attr_size; i++) {
                int number_vertices = min(min_colorful_degree + delta, colorful_d[i]);
                for (int w = max_weight; w > 0; w--) {
                    if (number_vertices > 0) {
                        WRFCMax[i] += w * min(number_vertices, nums[i][w]);
                        number_vertices -= nums[i][w];
                    } else {
                        break;
                    }
                }
            }
            if (WRFCMax[u] > lower_bound) left.emplace_back(u);
        }
    }

    int start_pos = 0;
    for (int i = 0; i < n; i++) {
        if (WRFCMax[i] > lower_bound) {
            int offset_start = start_pos;
            for (int j = offset[i]; j < offset[i + 1]; j++) {
                if (WRFCMax[edge_list[j]] > lower_bound) {
                    edge_list[start_pos++] = edge_list[j];
                }
            }
            offset[i] = offset_start;
            pend[i] = start_pos;
        } else {
            offset[i] = pend[i] = 0;
        }
    }
    delete[] colorful_r;
    delete[] colorful_d;
    delete[] head;
    delete[] nxt;
}

void Graph::Baseline() {
    clock_t TimeBegin, TimeEnd;
    double TimeAll = 0;
    printf("THERE IS BASELINE METHOD \n");
    printf("当前属性的维度d=%d, delta=%d\n", attr_size, delta);

    // 读取数据
    TimeBegin = clock();
    ReadGraph();
    TimeEnd = clock();
    printf("读取数据消耗时间为%lf\n", double(TimeEnd - TimeBegin) / CLOCKS_PER_SEC);

    vector<int> R, V(n);
    for (int i = 0; i < n; i++) {
        V[i] = i;
    }

    branches = 0;
    TimeBegin = clock();
    BronKerbosch(R, V);
    TimeEnd = clock();
    printf("BASELINE METHOD max result is %d\n", max_result);
    printf("BASELINE METHOD 消耗的时间为%lf\n", double(TimeEnd - TimeBegin) / CLOCKS_PER_SEC);
    printf("BASELINE METHOD 搜索节点数量为%lld\n\n", branches);
}


void Graph::BronKerbosch(vector<int> &R, vector<int> &C) {
    branches++;
    if (C.empty()) {
        int temp = CalculateResult(R);
        if (temp > max_result) {
            for (auto r : R) {
                printf("%d %d %d \n", r, attribute[r], weight[r]);
            }
            printf("\n");
        }
        max_result = max(max_result, CalculateResult(R));
        return;
    }

    int len = int(C.size());
    vector<int> newR, newP;
    newR.clear();
    newP.clear();

    for (auto r: R) newR.emplace_back(r);

    for (int i = 0; i < len; i++) {
        int cur = C[i];
        newP.clear();
        newR.emplace_back(cur);
        for (int j = i + 1; j < len; j++) {
            nvis[C[j]] = 1;
        }

        for (int j = offset[cur]; j < offset[cur + 1]; j++) {
            if (nvis[edge_list[j]]) newP.emplace_back(edge_list[j]);
        }

        for (int j = i + 1; j < len; j++) {
            nvis[C[j]] = 0;
        }

        BronKerbosch(newR, newP);
        newR.pop_back();
    }
}

void Graph::MWRFCSearch(vector<int> &R, vector<int> &C, int aIdx, int CMax, int flag, int N) {
    if (aIdx == 0) N++;
    if (C.empty() || (N - flag) >= delta && flag != -1) {
        result = max(result, CMax);
        return;
    }

    int len = int(C.size()), idx = 0;
    vector<int> newR(R), newP;
    newR.clear();
    newP.clear();
    for (int i = 0; i < len - idx; i++) {
        if (attribute[C[i]] == aIdx) {
            swap(C[i], C[len - idx - 1]);
            idx++;
            i--;
        }
    }

    if (idx == 0) {
        if (flag == -1) flag = N;
        MWRFCSearch(R, C, (aIdx + 1) % attr_size, CMax, flag, N);
    } else {
        for (int i = len - 1; i >= len - idx; i--) {
            int v = C[i];
            newR.emplace_back(v);
            newP.clear();

            for (int j = offset[v]; j < pend[v]; j++) {
                nvis[edge_list[j]] = 1;
            }

            for (int j = i - 1; j >= 0; j--) {
                if (nvis[C[j]]) {
                    newP.emplace_back(C[j]);
                }
            }
            for (int j = offset[v]; j < pend[v]; j++) {
                nvis[edge_list[j]] = 0;
            }

            int upper_bound = GetUpperBoundSpeed(aIdx, newP);

            if (max_result > upper_bound + CMax + weight[v]) {
                newR.pop_back();
                continue;
            }

            MWRFCSearch(newR, newP, (aIdx + 1) % attr_size, CMax + weight[v], flag, N);
            newR.pop_back();
        }
    }
}

int Graph::GetUpperBoundSpeed(int attr, vector<int> &P) {
    for (int i = 0; i < attr_size; i++) {
        memset(index_value[i], 0, sizeof(index_value[i]));
    }
    for (auto v: P) {
        index_value[attribute[v]][index1[v]] += index2[v];
    }

    vector<int> attr_len(attr_size, 0);
    for (int i = 0; i <= attr; i++) attr_len[i] = 1;
    for (int i = 0; i < attr_size; i++) {
        for (int j = 0; j < parts_size[i]; j++) {
            if (index_value[i][j]) attr_len[i] += groups[i][j][index_value[i][j]][0];
        }
    }

    int length = attr_len[0];
    for (int i = 1; i < attr_size; i++) {
        length = min(length, attr_len[0]);
    }
    int upper_bound = 0;
    for (int i = 0; i < attr_size; i++) {
        int len = min(length + delta, attr_len[i]);
        if (i <= attr) len--;
        if (!len) continue;
        for (int j = 0; j < parts_size[i]; j++) {
            if (index_value[i][j]) {
                if (len > groups[i][j][index_value[i][j]][0]) {
                    upper_bound += groups[i][j][index_value[i][j]][groups[i][j][index_value[i][j]][0]];
                    len -= groups[i][j][index_value[i][j]][0];
                } else {
                    upper_bound += groups[i][j][index_value[i][j]][len];
                    break;
                }
            }
        }
    }
    return upper_bound;
}

int Graph::GetUpperBoundNormal(int attr, vector<int> &P) {
    vector<vector<int>> W(attr_size);
    for (int i = 0; i <= attr; i++) {
        W[i].emplace_back(0);
    }

    for (auto v : P) {
        W[attribute[v]].emplace_back(weight[v]);
    }

    for (int i = 0; i < attr_size; i++) {
        if (i <= attr) {
            sort(W[i].begin() + 1, W[i].end(), [&] (int &A, int &B) {
                return A > B;
            });
        } else {
            sort(W[i].begin(), W[i].end(), [&] (int &A, int &B) {
                return A > B;
            });
        }
    }

    int length = int(W[0].size());
    for (int i = 1; i < attr_size; i++) {
        length = min(length, int(W[i].size()));
    }

    int upper_bound = 0;
    for (int i = 0; i < attr_size; i++) {
        for (int j = 0; j < min(length + delta, int(W[i].size())); j++) {
            upper_bound += W[i][j];
        }
    }

    return upper_bound;
}

void Graph::GetConnectedComponent(int root, int *vis) {
    stack<int> s;
    while (!s.empty()) s.pop();
    s.push(root);
    vis[root] = 0;
    while (!s.empty()) {
        int cur = s.top();
        s.pop();
        component.emplace_back(cur);
        for (int i = offset[cur]; i < pend[cur]; i++) {
            if (vis[edge_list[i]]) {
                s.push(edge_list[i]);
                vis[edge_list[i]] = 0;
            }
        }
    }

    int max_temp = 0;
    for (auto u: component) {
        max_temp = max(max_temp, WRFCMax[u]);
    }
    int *head = new int[max_temp + 1];
    int *nxt = new int[n];

    for (int i = 0; i < max_temp + 1; i++) head[i] = n;

    for (int i: component) {
        nxt[i] = head[WRFCMax[i]];
        head[WRFCMax[i]] = i;
    }
    delete[] head;
    delete[] nxt;
}

int Graph::CalculateResult(vector<int> &V) {
    int current_result = 0;

    vector<vector<int>> vertices(attr_size);
    for (auto v: V) {
        vertices[attribute[v]].emplace_back(v);
    }

    for (auto &vertex: vertices) {
        sort(vertex.begin(), vertex.end(), [&](int a, int b) {
            return weight[a] > weight[b];
        });
    }

    int min_threshold = int(vertices[0].size());
    for (const auto &vertex: vertices) {
        min_threshold = min(min_threshold, int(vertex.size()));
    }

    for (const auto &vertex: vertices) {
        for (int i = 0; i < min(min_threshold, int(vertex.size())); i++) {
            current_result += weight[vertex[i]];
        }
    }

    return current_result;
}

void Graph::HeuristicAlgorithm() {
    int *head = new int[max_WRFCMax + 1];
    int *nxt = new int[n];
    int *Flag = new int[n];

    int *hhead = new int[max_WRFCMax + 1];
    int *nnxt = new int[n];

    for (int i = 0; i < max_WRFCMax + 1; i++) head[i] = n;
    for (int i = 0; i < max_WRFCMax + 1; i++) hhead[i] = n;
    for (int i = 0; i < n; i++) Flag[i] = -1;

    for (int i = 0; i < n; i++) {
        nxt[i] = head[WRFCMax[i]];
        head[WRFCMax[i]] = i;
    }

    int index = 0;
    for (int ii = max_WRFCMax; ii >= 1; ii--) {
        if (ii <= lower_bound) break;
        for (int jj = head[ii]; jj != n; jj = nxt[jj]) {
            index++;
            int u = jj;
            Flag[u] = index;
            for (int j = offset[u]; j < offset[u + 1]; j++) {
                int v = edge_list[j];
                if (WRFCMax[v] <= lower_bound) continue;
                nnxt[v] = hhead[WRFCMax[v]];
                hhead[WRFCMax[v]] = v;
            }
            vector<vector<int>> P(attr_size);
            vector<int> t(attr_size, 0);
            for (int i = ii; i >= 1; i--) {
                for (int j = hhead[i]; j != n; j = nnxt[j]) {
                    P[attribute[j]].emplace_back(j);
                }
            }
            for (int j = 1; j <= ii; j++) {
                hhead[j] = n;
            }
            int MTmp = 1, idxx = attribute[u], wTmp = weight[u];
            int T = 1, mi = -1;
            while (T++) {
                if (mi != -1 && T - mi >= delta) break;
                int f = 0;
                for (int i = idxx + (T == 2 ? 1 : 0); i < idxx + attr_size; i++) {
                    int j = i % attr_size;
                    while (t[j] < P[j].size()) {
                        int F = 0;
                        int v = P[j][t[j]];
                        int CNums = 0;
                        for (int k = offset[v]; k < offset[v + 1]; k++) {
                            if (Flag[edge_list[k]] == index) CNums++;
                            if (CNums == MTmp) {
                                f++;
                                F++;
                                MTmp++;
                                wTmp += weight[v];
                                Flag[v] = index;
                                break;
                            }
                        }
                        t[j]++;
                        if (F) break;
                    }
                    if (mi == -1 && t[j] == P[j].size()) mi = T - 1;
                }
                if (!f) break;
            }
            lower_bound = max(lower_bound, wTmp);
        }
    }
}

void Graph::SortWeight(vector<int> &nums) {
    vector<int> res_tmp;
    res_tmp.assign(nums.begin(), nums.end());
    int *head = new int[max_weight + 1];
    int *nxt = new int[int(nums.size())];

    for (int i = max_weight; i >= 0; i--) head[i] = n;
    for (int i = 0; i < int(nums.size()); i++) {
        nxt[i] = head[weight[nums[i]]];
        head[weight[i]] = i;
    }

    int index = 0;
    for (int w = max_weight; w >= 0; w--) {
        for (int jj = head[w]; jj != n; jj = nxt[jj]) {
            int u = jj;
            res_tmp[index++] = u;
        }
    }
    nums = std::move(res_tmp);
}

void Graph::DivideParts() {
    divided_vertices.resize(attr_size);

    for (auto v: component) {
        divided_vertices[attribute[v]].emplace_back(v);
    }

    for (int i = 0; i < attr_size; i++) {
        SortWeight(divided_vertices[i]);
    }

    if (int(component.size()) < 100) parts_len = 20;
    else if (int(component.size()) < 1000) parts_len = 15;
    else parts_len = 10;

    two.resize(parts_len + 1);
    two[0] = 1;
    for (int i = 1; i < parts_len + 1; i++) {
        two[i] = two[i - 1] * 2;
    }

    vector<int> sum(parts_len + 1, 0);
    for (int i = 0; i < attr_size; i++) {
        parts_size[i] = -1;
        int k = parts_len;
        for (int j = 0; j < divided_vertices[i].size(); j++) {
            int u = divided_vertices[i][j];
            if (k == parts_len) {
                index1[u] = ++parts_size[i];
                index2[u] = 1;
                sum[1] = weight[u];
                k = 1;
            } else if (k == 1) {
                index1[u] = parts_size[i];
                index2[u] = two[k];
                sum[k + 1] = sum[k] + weight[u];
                k++;
            } else {
                sum[k + 1] = sum[k] + weight[u];
                int flag = 0;
                for (int l = 2; l <= (k - 1) / 2; l++) {
                    if (sum[l - 1] >= sum[k + 1] - sum[k - l + 1]) {
                        flag = 1;
                        break;
                    }
                }
                if (!flag) {
                    index1[u] = parts_size[i];
                    index2[u] = two[k];
                    k++;
                } else {
                    index1[u] = ++parts_size[i];
                    index2[u] = 1;
                    sum[1] = weight[u];
                    k = 1;
                }
            }
        }
        parts_size[i]++;
    }
    index_value = new int*[attr_size];
    for (int i = 0; i < attr_size; i++) {
        index_value[i] = new int[parts_size[i] + 1];
    }
}

void Graph::MergeParts() {
    if (groups != nullptr) delete[] groups;

    groups = new int ***[attr_size];
    for (int i = 0; i < attr_size; i++) {
        groups[i] = new int **[divided_vertices[i].size()];
        for (int j = 0; j < int(divided_vertices[i].size()); j++) {
            groups[i][j] = new int *[two[parts_len]];
        }
    }

    int len = int(component.size());
    vector<int> f(len + 1, 0);
    for (int i = 0; i < attr_size; i++) {
        for (int v: divided_vertices[i]) {
            int tmp_index1 = index1[v], tmp_index2 = index2[v];
            f[v] = 1;
            int tmp = 0;
            for (int k = offset[v]; k < pend[v]; k++) {
                int u = edge_list[k];
                if (attribute[u] == attribute[v] && index1[u] == index1[v] && f[u]) tmp += index2[u];
            }
            groups[i][tmp_index1][tmp_index2] = new int[parts_len];

            groups[i][tmp_index1][tmp_index2][0] = 1;
            groups[i][tmp_index1][tmp_index2][1] = weight[v];

            for (int j = 1; j < tmp_index2; j++) {
                int mtmp = j + index2[v];
                int commonNeighbor = tmp & j;
                if (commonNeighbor == 0) {
                    // 说明没有公共邻居
                    for (int e = 0; e <= groups[i][tmp_index1][j][0]; e++) {
                        groups[i][tmp_index1][mtmp][e] = groups[i][tmp_index1][j][e];
                    }
                } else {
                    for (int e = 0; e <= groups[i][tmp_index1][j][0]; e++) {
                        groups[i][tmp_index1][mtmp][e + 1] = groups[i][tmp_index1][j][e];
                    }
                    int tmp_len = groups[i][tmp_index1][commonNeighbor][0];
                    for (int e = 1; e < tmp_len; e++) {
                        groups[i][tmp_index1][mtmp][e + 1] = max(weight[v] + groups[i][tmp_index1][commonNeighbor][e],
                                                                 groups[i][tmp_index1][j][e + 1]);
                    }
                    if (groups[i][tmp_index1][j][0] > tmp_len) {
                        groups[i][tmp_index1][mtmp][tmp_len + 1] = max(
                                weight[v] + groups[i][tmp_index1][commonNeighbor][tmp_len],
                                groups[i][tmp_index1][j][tmp_len + 1]);
                    } else {
                        groups[i][tmp_index1][j][0] = tmp_len + 1;
                        groups[i][tmp_index1][mtmp][tmp_len + 1] =
                                weight[v] + groups[i][tmp_index1][commonNeighbor][tmp_len];
                    }
                }
            }
        }
    }
}

void Graph::Middle() {
    // 读取数据
    ReadGraph();

    // 上色
    SetColor();

    //


}

void Graph::prepareSearch() {
    if (left.empty()) return;
    int *vis = new int[n];
    memset(vis, 0, sizeof (&vis));
    for (int i : left) {
        vis[i] = 1;
    }
    max_result = lower_bound;
    vector<int> R;
    for (int i = 0; i < n; i++) {
        if (vis[i]) {
            component.clear();
            GetConnectedComponent(i, vis);
            DivideParts();
            MergeParts();
            R.clear();
            MWRFCSearch(R, component, 0, 0, -1, 0);


            for (int j = 0; j < attr_size; j++) {
                delete[] index_value[i];
            }
            delete[] index_value;
            index_value = nullptr;
        }
    }
}

void Graph::Last() {
    // 读取数据
    clock_t time_begin, time_end;
    double time_all = 0;

    cout << "There is speedup version" << endl;
    cout << "======================speedup version================" << endl;
    time_begin = clock();
    ReadGraph();
    time_end = clock();
    cout << "read data cost " << double(time_end - time_begin) / CLOCKS_PER_SEC << "s" << endl;

    // 上色
    time_begin = clock();
    SetColor();
    time_end = clock();
    time_all += double(time_end - time_begin) / CLOCKS_PER_SEC;
    cout << "color graph cost " << double(time_end - time_begin) / CLOCKS_PER_SEC << "s" << endl;

    // 计算WRFCMax
    time_begin = clock();
    CalculateWRFCMax();
    time_end = clock();
    time_all += double(time_end - time_begin) / CLOCKS_PER_SEC;
    cout << "calculate WRFCMax cost " << double(time_end - time_begin) / CLOCKS_PER_SEC << "s" << endl;

    // 贪心求下限
    time_begin = clock();
    HeuristicAlgorithm();
    time_end = clock();
    time_all += double(time_end - time_begin) / CLOCKS_PER_SEC;
    cout << "heuristic to get the lower bound of MWRFC cost " << double(time_end - time_begin) / CLOCKS_PER_SEC << "s" << endl;
    cout << "the result of heuristic is " << lower_bound << endl;

    // 缩减图大小
    time_begin = clock();
    ReduceGraph();
    time_end = clock();
    time_all += double(time_end - time_begin) / CLOCKS_PER_SEC;
    cout << "remove vertices cost " << double(time_end - time_begin) / CLOCKS_PER_SEC << "s" << endl;
    cout << "after removing vertices, the left number is " << int(left.size()) << endl;

    // 搜索
    time_begin = clock();
    max_result = lower_bound;
    prepareSearch();
    time_end = clock();
    time_all += double(time_end - time_begin) / CLOCKS_PER_SEC;
    cout << "search cost " << double (time_end - time_begin) / CLOCKS_PER_SEC << "s" << endl;

    cout << "result of speedup version is " << max_result << endl;
    cout << "all time cost " << time_all << endl;
}