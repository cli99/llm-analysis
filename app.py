from llm_analysis.ui import main

try:
    import torch

    # https://github.com/VikParuchuri/marker/issues/442
    torch.classes.__path__ = []
except ImportError:
    pass

if __name__ == "__main__":
    main()
