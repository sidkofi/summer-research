import csv


def load_matrix_from_csv(file_path):
    matrix = []
    with open(file_path, "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            matrix.append(row)
    return matrix
