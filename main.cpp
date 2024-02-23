#include "Graph.h"

int main(int argv, char** argc) {
//    system("chcp 65001");
    if (argv != 3) {
        cout << "参数数量有误" << endl;
        return 0;
    }
    if (strcmp(argc[2], "base") != 0 and strcmp(argc[2], "middle") != 0 and strcmp(argc[2], "last") != 0) {
        cout << "参数流程有误" << endl;
        return 0;
    }
    string dir = string("../datasets/") + argc[1];
    for (int delta = 5; delta <= 5; delta++) {
        for (int attr_size = 4; attr_size <= 4; attr_size++) {
            cout << "delta = " << delta << " , d = " << attr_size << endl;
            string base_dir = dir + '-' + to_string(attr_size) +  ".in";
//            string base_dir = dir;
            Graph *graph;
            graph = new Graph(base_dir, attr_size, delta);
            if (strcmp(argc[2], "base") == 0) {
                graph->Baseline();
            }else if (strcmp(argc[2], "middle") == 0) {
                graph->Middle();
            }else if (strcmp(argc[2], "last") == 0) {
                graph->Last();
            }
            delete graph;
        }
    }
    return 0;
}