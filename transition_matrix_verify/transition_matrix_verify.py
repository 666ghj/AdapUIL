import numpy as np

def read_matrix(file_path):
    with open(file_path, 'r') as file:
        matrix = []
        for line in file:
            row = list(map(float, line.strip().split()))
            matrix.append(row)
        return np.array(matrix)

def find_nonzero_columns(matrix):
    nonzero_info = []
    for row in matrix:
        nonzero_indices = np.nonzero(row)[0]
        nonzero_count = len(nonzero_indices)
        nonzero_info.append((nonzero_indices, nonzero_count))
    return nonzero_info

def print_transition_columns(trans_matrix, nonzero_info):
    for row_idx, (indices, count) in enumerate(nonzero_info):
        if row_idx < 5:  # Only print for the first five rows
            print(f"Row {row_idx}: Non-zero columns = {indices}, Count = {count}")
            print("Corresponding columns in transition matrix for the same row:")
            for col_idx in indices:
                print(f"Column {col_idx}: {trans_matrix[row_idx, col_idx]}")

def main():
    adjacency_matrix_file = 'adjacency.txt'
    transition_matrix_file = 'transition_matrix.txt'

    adj_matrix = read_matrix(adjacency_matrix_file)
    trans_matrix = read_matrix(transition_matrix_file)

    nonzero_info = find_nonzero_columns(adj_matrix)
    print_transition_columns(trans_matrix, nonzero_info)

if __name__ == "__main__":
    main()