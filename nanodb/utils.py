#!/usr/bin/env python3
from clip_trt.utils import (
    AttributeDict, 
    torch_dtype, 
    tqdm_redirect_stdout,
)

from faiss_lite import (
    cudaAllocMapped,
    assert_cuda,
)
