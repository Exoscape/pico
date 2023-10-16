class PipelineOptions:
    def __init__(self, max_concurrency: int = 1, is_xformers_enabled: bool = False):
        self.MaxConcurrency = max_concurrency
        self.IsXformersEnabled = is_xformers_enabled