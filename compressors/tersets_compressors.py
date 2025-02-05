# import compressors.tersets as tersets
import numpy as np
import os

class TerseTSCompressors:
    coefficients = [15, 20, 25, 30]
    uncompressed = None

    def __init__(self):
        pass
        

class SimPiece(TerseTSCompressors):
    def __init__(self):
        super().__init__()

    def compress(self, uncompressed, error_bound, fh=None, index=None):
        return np.asarray(tersets.compress(uncompressed, tersets.Method.SimPiece, error_bound))
        
    def decompress(self, compressed):
        if isinstance(compressed, np.ndarray):
            return tersets.decompress(compressed.tolist())
        return tersets.decompress(compressed)
        

class PMC(TerseTSCompressors):
    def __init__(self):
        super().__init__()

    def compress(self, uncompressed, error_bound,  fh=None, index=None):
        return np.asarray(tersets.compress(uncompressed, tersets.Method.PoorMansCompressionMean, error_bound))
        
        
    def decompress(self, compressed):
        if isinstance(compressed, np.ndarray):
            return tersets.decompress(compressed.tolist())
        return tersets.decompress(compressed)
    
class SWING(TerseTSCompressors):
    def __init__(self):
        super().__init__()

    def compress(self, uncompressed, error_bound,  fh=None, index=None):
        return np.asarray(tersets.compress(uncompressed, tersets.Method.SwingFilter, error_bound))
        
    def decompress(self, compressed):
        if isinstance(compressed, np.ndarray):
            return tersets.decompress(compressed.tolist())
        return tersets.decompress(compressed)
    

    