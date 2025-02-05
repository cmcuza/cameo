from libcpp.unordered_map cimport unordered_map


cdef struct Node:
    double value
    int ts
    int left
    int right

ctypedef Node* NodePtr
ctypedef Heap* HeapPtr
ctypedef unordered_map[int, int]& MapPtr

cdef struct Heap:
    unsigned int c_size
    # unsigned int c_size
    NodePtr values
    unsigned int m_size

cdef void initialize_from_np(HeapPtr heap, MapPtr map_node_to_heap, double[:] x)
cdef void initialize_from_pointer(HeapPtr heap, MapPtr map_node_to_heap, double *x, int n)
cdef inline int parent(int n)
cdef Node pop(HeapPtr heap, MapPtr map_node_to_heap)

cdef void initialize_for_turning_point(HeapPtr heap, MapPtr map_node_to_heap, double *import_tp, int *selected_tp, int n)

cdef void reheap(HeapPtr heap, MapPtr map_node_to_heap, Node& node)

cdef void update_left_right(HeapPtr heap, MapPtr map_node_to_heap, const int& left, const int& right)

cdef void get_update_left_right(HeapPtr heap, MapPtr map_node_to_heap, const Node& node, Node& left, Node& right)

cdef void release_memory(HeapPtr heap)