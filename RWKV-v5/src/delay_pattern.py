import numpy

def calculate_offsets(size, n_group):
    group_size = size // n_group
    return [(i % group_size) + (i // group_size) for i in range(size)]

def apply(arr, n_group, padding_idx, cut_end=False):
    # shape = (length, channels)
    orig_type = type(arr)
    arr = numpy.array(arr)
    offsets = calculate_offsets(arr.shape[1], n_group)
    arr = numpy.pad(
        arr,
        ((0, max(offsets)), (0, 0)),
        constant_values=padding_idx,
    )
    arr = arr.T
    arr = numpy.stack([
        numpy.roll(i[1], offsets[i[0]])
        for i in enumerate(arr)
    ])
    arr = arr.T
    if cut_end:
        arr = arr[:(-max(offsets) if max(offsets) else None)]
    
    if orig_type == list:
        return arr.tolist()
    elif orig_type == numpy.ndarray:
        return arr
    else:
        return orig_type(arr)

def unapply(arr, n_group, padding_idx, cut_start=True, cut_end=True):
    # shape = (length, channels)
    orig_type = type(arr)
    arr = numpy.array(arr)
    offsets = calculate_offsets(arr.shape[1], n_group)
    arr = numpy.pad(
        arr,
        ((max(offsets), 0), (0, 0)),
        constant_values=padding_idx,
    )
    arr = arr.T
    arr = numpy.stack([
        numpy.roll(i[1], -offsets[i[0]])
        for i in enumerate(arr)
    ])
    arr = arr.T
    if cut_start:
        arr = arr[numpy.argmax(numpy.all(arr != padding_idx, axis=1)):]
    if cut_end:
        arr = arr[:len(arr) - numpy.argmax(numpy.all(arr[::-1] != padding_idx, axis=1))]
    
    if orig_type == list:
        return arr.tolist()
    elif orig_type == numpy.ndarray:
        return arr
    else:
        return orig_type(arr)