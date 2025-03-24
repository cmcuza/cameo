from numpy.math cimport INFINITY
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double vertical_distance(NodePtr left, NodePtr node, NodePtr right):
    cdef:
        double EPSILON = 1e-06, result
        int a_x, b_x, c_x
        double a_y, b_y, c_y

    a_x = left.ts
    b_x = node.ts
    c_x = right.ts
    a_y = left.value
    b_y = node.value
    c_y = right.value

    if (fabs(a_x - b_x) < EPSILON) or (fabs(b_x - c_x) < EPSILON):
        result = 0
    elif (c_x - a_x) == 0:
        result = INFINITY
    else:
        result = fabs((a_y + (c_y - a_y) * (b_x - a_x) / (c_x - a_x) - b_y))

    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void init_node(NodePtr node, HeapPtr heap,  int i):
    node.index = i
    node.parent = heap
    node.cache = INFINITY
    node.ts = 0
    node.value = 0
    node.order = 0
    node.left = NULL
    node.right = NULL



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void update_cache(NodePtr node):
    if node.left is not NULL and node.right is not NULL:
        node.cache = vertical_distance(node.left, node, node.right)
    else:
        node.cache = INFINITY

    notify_change(node.parent, node.index)



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef NodePtr put_after(NodePtr node, NodePtr tail):
    if tail is not NULL:
        tail.right = node
        update_cache(tail)
    node.left = tail
    update_cache(node)
    return node



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef PIPNode recycle(NodePtr node):
    if node.left is NULL:
        node.parent.head = node.right
    else:
        node.left.right = node.right
        update_cache(node.left)

    if node.right is NULL:
        node.parent.tail = node.left
    else:
        node.right.left = node.left
        update_cache(node.right)

    return clear(node)



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef PIPNode clear(NodePtr node):
    cdef PIPNode returned_node
    cdef NodePtr left, right

    returned_node.left = node.left
    returned_node.right = node.right
    returned_node.value = node.value
    returned_node.ts = node.ts

    node.left = NULL
    node.right = NULL
    node.parent = NULL
    node.cache = INFINITY
    free(node)
    return returned_node
    # return ret, left, right



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void init_heap(HeapPtr heap,  int size):
    cdef  int i
    cdef NodePtr node

    heap.head = NULL
    heap.tail = NULL
    heap.size = 0
    heap.m_size = size
    heap.global_order = 0

    heap.values = <NodeDbPtr> malloc(size*sizeof(NodePtr))
    for i in range(size):
        node = <NodePtr> malloc(sizeof(PIPNode))
        init_node(node, heap, i)
        heap.values[i] = node



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef NodePtr acquire_item(HeapPtr heap,  int ts, double value):
    node = heap.values[heap.size]
    node.ts = ts
    node.value = value

    heap.size += 1
    heap.global_order += 1
    node.order = heap.global_order
    return node



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void add(HeapPtr heap,  int ts, double value):
    cdef NodePtr node
    node = acquire_item(heap, ts, value)
    heap.tail = put_after(node, heap.tail)
    if heap.head is NULL:
        heap.head = heap.tail


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef  int notify_change(HeapPtr heap,  int index):
    return bubble_down(heap, bubble_up(heap, index))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef  int min(HeapPtr heap,  int i,  int j,  int k):
    cdef  int result

    if k != -1:
        result = min(heap, i, min(heap, j, k, -1), -1)
    else:
        result = i if less(heap, i, j) else j

    return result



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef PIPNode remove_at(HeapPtr heap,  int index):
    cdef NodePtr node

    heap.size -= 1
    swap(heap, index, heap.size)
    bubble_down(heap, index)

    node = heap.values[heap.size]
    return recycle(node)



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef  int bubble_up(HeapPtr heap,  int n):
    while (n != 0) and less(heap, n, (n-1)//2):
        n = swap(heap, n, (n-1)//2)
    return n


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef  int bubble_down(HeapPtr heap,  int n):
    cdef  int k

    k = min(heap, n, n*2+1, n*2+2)
    while (k != n) and (k < heap.size):
        n = swap(heap, n, k)
        k = min(heap, n, n*2+1, n*2+2)
    return n


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef  int swap(HeapPtr heap,  int i,  int j):
    heap.values[i].index, heap.values[j].index = j, i
    heap.values[i], heap.values[j] = heap.values[j], heap.values[i]
    return j



@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint less(HeapPtr heap,  int i,  int j):
    return (i < heap.size) and (j >= heap.size or i_smaller_than_j(heap, i, j))


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint i_smaller_than_j(HeapPtr heap,  int i,  int j):
    cdef bint result

    if heap.values[i].cache != heap.values[j].cache:
        result = heap.values[i].cache < heap.values[j].cache
    else:
        result = heap.values[i].order < heap.values[j].order

    return result


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void iterate(HeapPtr heap):
    current = heap.head
    while current is not NULL:
        # yield current.value
        current = current.right


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void deinit_heap(HeapPtr heap):
    cdef NodePtr current = heap.head
    cdef NodePtr next

    while current is not NULL:
        # print("Current", current.ts, current.value, current.index, current.cache)
        next = current.right
        free(current)
        current = next



