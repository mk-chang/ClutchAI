"""ClutchAI Agent package."""

# Import agent first (most important)
from .agent import ClutchAIAgent

# Optional imports - only import if needed
try:
    from .rag.vectorstore import VectorstoreManager
    from .rag.data_class import YouTubeVideo
    __all__ = ['ClutchAIAgent', 'VectorstoreManager', 'YouTubeVideo']
except ImportError:
    # If rag imports fail, at least export the agent
    __all__ = ['ClutchAIAgent']

