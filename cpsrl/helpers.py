from cpsrl.errors import ShapeError


def check_shape(arrays, shapes, shape_dict=None, keep_dict=False):
    
    if (type(arrays) in [list, tuple]) and \
       (type(shapes) in [list, tuple]):
        
        shape_dict = {} if shape_dict is None else shape_dict
        
        if len(arrays) != len(shapes):
            raise ShapeError(f"Got number of tesors/arrays {len(arrays)}, "
                             f"and number of shapes {len(shapes)}.")
            
        for array, shape in zip(arrays, shapes):
            array, shape_dict = _check_shape(array,
                                             shape,
                                             shape_dict=shape_dict)
            
        if keep_dict:
            return arrays, shape_dict
        
        else:
            return arrays
            
    
    else:
        return _check_shape(arrays, shapes)[0]
        


def _check_shape(array, shape, shape_dict=None):
    
    array_shape = array.shape
    check_string_names = shape_dict is not None
    
    # Check if array shape and shape have same length (i.e. are comparable)
    if len(array_shape) != len(shape):
        raise ShapeError(f"Tensor/Array shape {array_shape}, "
                         f"check shape {shape}")
    
    # Check if shapes are compatible
    for s1, s2 in zip(array.shape, shape):
        
        # Try to convert s2 to int
        try:
            # If s2 == '-1', any shape passes test
            if int(s2) == -1:
                continue

            elif s1 != int(s2):

                raise ShapeError(f"Tensor/Array shape {array_shape}, "
                                 f"check shape {shape}")
                
        # If s2 string found, try to match against dict
        except ValueError:
            
            if not (s2 in shape_dict):
                shape_dict[s2] = s1
                
            elif shape_dict[s2] != s1:
                raise ShapeError(f"Tensor/Array shape had {s2} axis size {s1}, "
                                 f"expected axis size {shape_dict[s2]}.")
            
    if check_string_names:
        return array, shape_dict
    
    else:
        return array