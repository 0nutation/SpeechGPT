from setuptools import setup, find_namespace_packages
import platform

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


setup(
    name="speechgpt",
    version="1.0.0.dev1",
    author="Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, Xipeng Qiu",
    description="SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="Speech, Multimodal LLM, Multimodal, LLM, Generative AI, Deep Learning, Library, PyTorch",
    license="Apache-2.0 license",
    packages=find_namespace_packages(include="speechgpt.*"),
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.7.0",
    include_package_data=True,
    dependency_links=DEPENDENCY_LINKS,
    zip_safe=False,
)