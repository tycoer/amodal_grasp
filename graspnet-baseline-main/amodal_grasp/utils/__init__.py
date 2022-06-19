def workspace_lines(size):
    return [
        [0.0, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, 0.0, 0.0],
        [size, size, 0.0],
        [size, size, 0.0],
        [0.0, size, 0.0],
        [0.0, size, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, size],
        [size, 0.0, size],
        [size, size, size],
        [size, size, size],
        [0.0, size, size],
        [0.0, size, size],
        [0.0, 0.0, size],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, size],
        [size, 0.0, 0.0],
        [size, 0.0, size],
        [size, size, 0.0],
        [size, size, size],
        [0.0, size, 0.0],
        [0.0, size, size],
    ]

# from .libmesh import check_mesh_contains
# from .libmesh import triangle_hash