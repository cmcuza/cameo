from libc.stdlib cimport malloc, free
from numpy.math cimport INFINITY
import cython
cdef int NODE_TYPE_ROOT = 0
cdef int NODE_TYPE_INVALID = -1
cdef Segment FAKE = Segment(INFINITY, -1, -1, -1, -1, -1)


cdef inline int parent(int n):
    return (n - 1) >> 1


cdef inline bint less(Segment n1, Segment n2):
    return n1.merge_cost < n2.merge_cost


cdef void initialize(HeapPtr heap, MapPtr map_node_to_heap, int n):
    cdef int i
    heap.c_size = n
    heap.m_size = n
    map_node_to_heap.reserve(heap.m_size)

    heapify(heap)

    for i in range(heap.m_size):
        map_node_to_heap[heap.values[i].id] = i

cdef inline Segment top(HeapPtr heap):
    return heap.values[0]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef Segment pop(HeapPtr heap, MapPtr map_node_to_heap):
    cdef Segment res
    if heap.c_size != 1:
        res = top(heap)
        heap.c_size -= 1
        clear_heap_index(map_node_to_heap, res.id)
        heap.values[NODE_TYPE_ROOT] = heap.values[heap.c_size]
        heap.values[heap.c_size] = FAKE
        bubble_down(heap, map_node_to_heap, NODE_TYPE_ROOT)
        return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void set_heap_index(MapPtr map_node_to_heap, int v, int i):
    map_node_to_heap[v] = i


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void insert(HeapPtr heap, MapPtr map_node_to_heap, Segment node):
    heap.values[heap.c_size] = node
    heap.c_size += 1
    bubble_up(heap, map_node_to_heap, heap.c_size-1)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void shiftup(HeapPtr heap, int start, int end):
    cdef int parent_idx, child
    while end > start:
        child = end
        parent_idx = (child-1)>>1
        if heap.values[child].merge_cost < heap.values[parent_idx].merge_cost:
            heap.values[child], heap.values[parent_idx] = heap.values[parent_idx], heap.values[child]
            end = parent_idx
        else:
            break


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void shiftdown(HeapPtr heap, int start):
    cdef int iend, istart, ichild, iright
    iend = heap.c_size
    istart = start
    ichild = 2 * istart + 1
    while ichild < iend:
        iright = ichild + 1
        if iright < iend and heap.values[ichild].merge_cost > heap.values[iright].merge_cost:
            ichild = iright
        heap.values[ichild], heap.values[istart] = heap.values[istart], heap.values[ichild]
        istart = ichild
        ichild = 2 * istart + 1

    shiftup(heap, start, istart)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int bubble_down(HeapPtr heap, MapPtr map_node_to_heap, int n):
    cdef unsigned int left, right, smallest
    while True:
        left = 2 * n + 1
        right = 1 + left
        smallest = n
        if left < heap.c_size and heap.values[left].merge_cost < heap.values[smallest].merge_cost:
                smallest = left
        if right < heap.c_size and heap.values[right].merge_cost < heap.values[smallest].merge_cost:
                smallest = right
        if smallest != n:
            heap.values[smallest], heap.values[n] = heap.values[n], heap.values[smallest]
            map_node_to_heap[heap.values[n].id] = n
            n = smallest
        else:
            map_node_to_heap[heap.values[n].id] = n
            return n

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int bubble_down_exp(HeapPtr heap, MapPtr map_node_to_heap, int n):
    cdef unsigned int left, right, end
    end = heap.c_size
    left = 2 * n + 1
    while left < end:
        right = left + 1
        if right < end and heap.values[right].merge_cost < heap.values[left].merge_cost:
            left = right

        heap.values[left], heap.values[n] = heap.values[n], heap.values[left]
        map_node_to_heap[heap.values[n].id] = n
        n = left
        left = 2 * n + 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void heapify(HeapPtr heap):
    cdef int i = heap.m_size//2
    while i >= 0:
        shiftdown(heap, i)
        i -= 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int bubble_up(HeapPtr heap, MapPtr map_node_to_heap, int n):
    cdef unsigned int parent_idx = (n - 1) >> 1
    while n != NODE_TYPE_ROOT and heap.values[n].merge_cost < heap.values[parent_idx].merge_cost:
        heap.values[n], heap.values[parent_idx] = heap.values[parent_idx], heap.values[n]
        map_node_to_heap[heap.values[n].id] = n
        n = parent_idx
        parent_idx = (n - 1) >> 1

    map_node_to_heap[heap.values[n].id] = n
    return n


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void clear_heap_index(MapPtr map_node_to_heap, int node):
    map_node_to_heap.erase(node)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_update_left_right(HeapPtr heap, MapPtr map_node_to_heap,
                                const Segment& node, Segment& left, Segment& right):
    cdef int i

    if node.left_seg > 0:
        i = map_node_to_heap[node.left_seg]
        heap.values[i].right = node.right_seg
        left = heap.values[i]
    else:
        left.id = -1
    if node.right_seg < heap.m_size-1:
        i = map_node_to_heap[node.right_seg]
        heap.values[i].left = node.left_seg
        right = heap.values[i]
    else:
        right.id = -1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_left_right(HeapPtr heap, MapPtr map_node_to_heap, const int& left, const int& right):
    cdef int i

    if left >= 0:
        i = map_node_to_heap[left]
        heap.values[i].right = right
    if right < heap.m_size:
        i = map_node_to_heap[right]
        heap.values[i].left = left


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void reheap(HeapPtr heap, MapPtr map_node_to_heap, Segment& node):
    cdef int heap_idx = map_node_to_heap[node.id]
    heap.values[heap_idx] = node
    bubble_down(heap, map_node_to_heap, bubble_up(heap, map_node_to_heap, heap_idx))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint empty(HeapPtr heap):
    return heap.c_size == 2


cdef void release_memory(HeapPtr heap):
    free(heap.values)
    free(heap)

