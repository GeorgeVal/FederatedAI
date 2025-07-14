# python3 setup.py build_ext --inplace
from setuptools import setup, Extension
from Cython.Build import cythonize


extensions = [
    Extension("FL_cpp_server", ["fl_cpp_backbone/PythonServerWrapper.pyx", "fl_cpp_backbone/server.cpp", "fl_cpp_backbone/net_lib.cpp", "fl_cpp_backbone/shared_buffer.cpp", "fl_cpp_backbone/barrier.cpp"],
              language='c++',
              extra_compile_args=["-std=c++11"]),
              
    Extension("FL_cpp_client", ["fl_cpp_backbone/PythonClientWrapper.pyx", "fl_cpp_backbone/client.cpp", "fl_cpp_backbone/net_lib.cpp", "fl_cpp_backbone/shared_buffer.cpp"],
              language='c++',
              extra_compile_args=["-std=c++11"]),

]

setup(
    name="cython_fl_framework",
    ext_modules=cythonize(extensions, language_level="3"),
)
print("Successfully built.")
