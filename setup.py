from setuptools import setup

__VERSION__ = '0.0.2'

setup(name='adabound',
      version=__VERSION__,
      description='AdaBound optimization algorithm, build on PyTorch.',
      keywords='machine learning deep learning',
      url='https://github.com/Luolc/AdaBound',
      author='Liangchen Luo',
      author_email='luolc.witty@gmail.com',
      license='Apache',
      packages=['adabound'],
      install_requires=[
          'torch>=0.4.0',
      ],
      zip_safe=False,
      python_requires='>=3.6.0')
