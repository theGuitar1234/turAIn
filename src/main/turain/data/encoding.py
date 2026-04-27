from ..utilities import core_method


class OneHotCoding:
    
    @core_method
    @staticmethod
    def one_hot_encode(Y, classes, backend):
        xp = backend.xp

        if type(Y) is not xp.ndarray:
            return None
        if type(classes) is not int:
            return None
        try:
            Y = Y.flatten()
            one_hot = xp.eye(classes)[Y]
            return one_hot
        except Exception:
            return None

    @core_method
    @staticmethod
    def one_hot_decode(one_hot, backend):
        xp = backend.xp

        if type(one_hot) is not xp.ndarray or len(one_hot.shape) != 2:
            return None
        vector = one_hot.argmax(axis=1)
        return vector



    
