from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="cpools",
    ext_modules=[
        CppExtension("top_pool", ["top_pool.cpp"]),
        CppExtension("bottom_pool", ["bottom_pool.cpp"])
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)