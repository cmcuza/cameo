from libcpp.unordered_map cimport unordered_map


cdef struct Segment:
    double merge_cost
    int id
    int seg_start
    int seg_end
    int left_seg
    int right_seg

ctypedef Segment* SegmentPtr

cdef struct Heap:
    unsigned int c_size
    # unsigned int c_size
    SegmentPtr values
    unsigned int m_size

ctypedef Heap* HeapPtr
ctypedef unordered_map[int, int]& MapPtr

cdef void initialize(HeapPtr heap, MapPtr map_node_to_heap, int n)

cdef inline int parent(int n)

cdef Segment pop(HeapPtr heap, MapPtr map_node_to_heap)

cdef void reheap(HeapPtr heap, MapPtr map_node_to_heap, Segment& node)

cdef void update_left_right(HeapPtr heap, MapPtr map_node_to_heap, const int& left, const int& right)

cdef void get_update_left_right(HeapPtr heap, MapPtr map_node_to_heap, const Segment& node, Segment& left, Segment& right)

cdef void release_memory(HeapPtr heap)