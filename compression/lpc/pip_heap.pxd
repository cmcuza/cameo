ctypedef PIPNode* NodePtr
ctypedef PIPNode** NodeDbPtr
ctypedef PIPHeap* HeapPtr

cdef struct PIPNode:
    NodePtr left
    NodePtr right
    HeapPtr parent
    int ts
    int index
    int order
    double value
    double cache

cdef struct PIPHeap:
    NodePtr head
    NodePtr tail
    NodeDbPtr values
    int size
    int m_size
    int global_order

cdef double vertical_distance(NodePtr left, NodePtr node, NodePtr right)

cdef void init_node(NodePtr node, HeapPtr heap, int i)

cdef void update_cache(NodePtr node)

cdef NodePtr put_after(NodePtr node, NodePtr tail)

cdef PIPNode recycle(NodePtr node)

cdef PIPNode clear(NodePtr node)

cdef void init_heap(HeapPtr heap, int size)

cdef NodePtr acquire_item(HeapPtr heap, int ts, double value)

cdef void add(HeapPtr heap, int ts, double value)

cdef  int notify_change(HeapPtr heap, int index)

cdef  int min(HeapPtr heap, int i, int j, int k)

cdef PIPNode remove_at(HeapPtr heap,  int index)

cdef  int bubble_up(HeapPtr heap,  int n)

cdef  int bubble_down(HeapPtr heap,  int n)

cdef  int swap(HeapPtr heap,  int i,  int j)

cdef bint less(HeapPtr heap,  int i,  int j)

cdef bint i_smaller_than_j(HeapPtr heap,  int i,  int j)

cdef void iterate(HeapPtr heap)

cdef void deinit_heap(HeapPtr heap)