// gcc -shared -o alpha_AffinityMatrix_efficient.dll -fPIC alpha_AffinityMatrix_efficient.c
// gcc -shared -o alpha_AffinityMatrix_efficient.so -fPIC alpha_AffinityMatrix_efficient.c
// 不仅依照输入的亲和矩阵、还同时alpha进行重启随机游走

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void affinity_matrix_walk_with_restart(
    float *M, int M_size, int *nodes, int num_nodes,
    int length_walk, int num_walks, float alpha, int **walks_result)
{
    srand(616); // 设置随机种子

    for (int i = 0; i < num_walks; ++i)
    {
        for (int j = 0; j < num_nodes; ++j)
        {
            int node = nodes[j];
            int *walk = (int *)malloc(length_walk * sizeof(int));
            walk[0] = node;

            int current_node = node;
            for (int k = 1; k < length_walk; ++k)
            {
                // 以概率 alpha 返回初始节点
                float r_alpha = ((float)rand() / RAND_MAX);
                if (r_alpha < alpha)
                {
                    current_node = node;
                }
                else
                {
                    // 以 (1 - alpha) 的概率按照亲和矩阵进行游走
                    float *weights = &M[current_node * M_size];
                    float sum = 0.0;
                    for (int l = 0; l < M_size; ++l)
                    {
                        sum += weights[l];
                    }

                    float r = ((float)rand() / RAND_MAX) * sum;
                    float cum_sum = 0.0;
                    int next_node = 0;
                    for (next_node = 0; next_node < M_size; ++next_node)
                    {
                        cum_sum += weights[next_node];
                        if (r <= cum_sum)
                        {
                            break;
                        }
                    }

                    current_node = next_node;
                }

                walk[k] = current_node;
            }

            walks_result[i * num_nodes + j] = walk;

            // 进度输出
            int step = i * num_nodes + j + 1;
            int total_steps = num_walks * num_nodes;
            if (step % 1000 == 0 || step == total_steps)
            {
                printf("\rProgress: %d/%d", step, total_steps);
                fflush(stdout);
            }
        }
    }
    printf("\n");
}