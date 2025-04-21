from setuptools import setup, find_packages

setup(
    name="raiden_agents",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "litellm",
        "python-dotenv",
        "requests",
        "numpy",
        "upstash-vector",
        "upstash-redis",
        "PyPDF2",
        "mss",
        "Pillow",
        "duckduckgo-search",
        "firecrawl-py",
        "sentence-transformers",
        "matplotlib",
        "seaborn",
        "PyGithub",
    ],
    author="RaidenAI",
    author_email="raiden@example.com",
    description="Core agent functionality for Raiden AI",
) 